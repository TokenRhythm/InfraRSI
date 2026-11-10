"""DF6 round-1 regressions: real adapter consumers/Engine, synthetic scores and HTTP only."""
import copy
import json
import os
from pathlib import Path

import pytest

from infrarsi.ext.common.env_runner import EnvRunner
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.rebench_llmf import arms, block, cli, phase_e, task as taskmod
from infrarsi.ext.rebench_llmf.node import score_runner
from infrarsi.ext.summary.records import Store, accounting_checks
from infrarsi.harness import signing
from infrarsi.search.engine import Engine, Proposal

from test_rebench_phase_e import FakeRunner, VirtualClock, driver_for, isolated_env, native_setup, task


@pytest.mark.parametrize('arm', taskmod.ARM_UIDS)
def test_registered_uid_pair_passes_real_node_judge_for(arm):
    uid, judge = taskmod.arm_uid(arm), taskmod.judge_uid(arm)
    assert score_runner.judge_for(uid, judge) == judge
    for other in taskmod.JUDGE_UIDS.values():
        if other != judge:
            with pytest.raises(ValueError):
                score_runner.judge_for(uid, other)
    with pytest.raises(ValueError):
        score_runner.judge_for(uid, taskmod.FORBIDDEN_UID)


@pytest.mark.parametrize('command', ['a-is2', 'a-nat2'])
@pytest.mark.parametrize('existing', [False, True])
def test_on_node_records_private_before_writes(monkeypatch, tmp_path, command, existing):
    # Windows cannot enforce Linux uid/mode bits: observe the real private_dir syscalls at the CLI seam.
    # Keep the synthetic node/agent directories outside records, as in the actual launch layout.
    out = tmp_path / 'records'
    agent = tmp_path / 'home_agent'
    agent.mkdir()
    if existing:
        out.mkdir()
        (out / 'old_response.json').write_text('{}')
    ops = LocalNodeOps(tmp_path / 'node', node='WINH0')
    changes = []
    original_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        st = original_lstat(path, *args, **kwargs)
        if Path(path) == out:
            values = list(st)
            values[4] = 0  # existing driver records belong to root on the node
            return os.stat_result(values)
        return st

    monkeypatch.setattr(os, 'lstat', lstat)
    monkeypatch.setattr(os, 'chown', lambda p, uid, gid: changes.append(('owner', Path(p), uid, gid)), raising=False)
    monkeypatch.setattr(os, 'chmod', lambda p, mode, **kw: changes.append(('mode', Path(p), mode)))
    monkeypatch.setattr(cli, '_ops', lambda a: ops)
    monkeypatch.setattr(cli, '_task', lambda arm, a, over: task(arm))
    monkeypatch.setattr(cli, 'read_base_solution', lambda p: {})

    def private():
        assert out.is_dir()
        assert ('owner', out, 0, 0) in changes
        assert ('mode', out, 0o700) in changes
        assert all(c[1] == out for c in changes)
        assert agent.is_dir()

    def make_runner(ops, t, **kw):
        private()  # before prepare, any receipt, trace, overlay, or LLM call can write
        return FakeRunner(t, VirtualClock(), [])

    def run(t, driver, runner, *args, **kw):
        private()
        (out / 'response.json').write_text('{}')
        return {'arm': t.arm, 'trace': {'screens': []}}

    monkeypatch.setattr(arms, 'make_runner', make_runner)
    monkeypatch.setattr(arms, 'run_a_is2' if command == 'a-is2' else 'run_a_nat2', run)
    assert cli.main([command, '--run-id', 'private', '--on-node', '--out-dir', str(out)]) == 0
    assert (out / 'arm_lease.json').exists() and (out / 'result.json').exists()


class RecordedRunner(FakeRunner):
    """The synthetic scorer uses the production attempt-receipt writer."""

    def __init__(self, *args, out_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dir = out_dir

    def run(self, *args, **kwargs):
        receipt = super().run(*args, **kwargs)
        EnvRunner._store(self, receipt)
        return receipt


def assert_reconciles(out, result):
    store = Store(out)
    trace = store.json(out / 'trace.json')
    receipts = store.receipts(out)
    events = store.event_sequence(out / 'events.jsonl')
    locked = result['locked'].get('attempt_id')
    failures, recon = accounting_checks(trace, receipts, events,
                                         non_screen={locked: 'locked'} if locked else {})
    assert not failures, failures
    assert not store.problems
    assert all(signing.verify(r) for r in receipts.values())
    assert recon['n_receipts'] == len(trace['screens']) + bool(locked)
    assert [i for kind, i in events if kind == 'screen'] == list(range(len(trace['screens'])))
    assert [i for kind, i in events if kind == 'call'] == [c['call_id'] for c in trace['llm_calls']]
    raw_events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    boundaries = [e for e in raw_events if e['kind'] in ('screen', 'rescore')]
    assert [e['payload'] for e in boundaries] == trace['screens']
    assert block.first_scoring('A-IS2', out)['attempt_id'] == trace['screens'][0]['attempt_id']
    assert block.first_scoring('A-IS2', out)['t_first'] == 200.0
    assert (out / 'receipts/c0.json').read_bytes() == (out / 'start/receipts/c0.json').read_bytes()
    return trace, raw_events


@pytest.mark.parametrize('invalid', [False, True])
def test_is2_root_records_reconcile_and_first_scoring_reads_s0(tmp_path, invalid):
    clock, events = VirtualClock(), []
    runner = RecordedRunner(task('A-IS2'), clock, events, out_dir=tmp_path, invalid_start=invalid)
    driver, _, _ = driver_for(events, clock, out=tmp_path)
    result = arms.run_a_is2(runner.task, driver, runner, run_id='records', out_dir=tmp_path, clock=clock)
    cli._record_lease(tmp_path, {'complete': True}, [{'kind': 'release'}])
    trace, root_events = assert_reconciles(tmp_path, result)
    assert trace['used']['screens'] == (1 if invalid else 3)
    for candidate in trace['candidates']:
        if candidate['attempt_id'] is not None:
            assert candidate['screen_index'] is not None, candidate['id']
            row = trace['screens'][candidate['screen_index']]
            assert row['candidate_id'] == candidate['id']
            assert row['attempt_id'] == candidate['attempt_id']
    assert root_events[-1]['kind'] == 'arm_lease'
    assert [e['i'] for e in root_events[:-1]] == list(range(len(root_events) - 1))


@pytest.mark.parametrize('screens', [3, 4])
def test_engine_confirmation_receipts_and_root_boundaries(monkeypatch, tmp_path, screens):
    # Use the arm's real Context/session/ledger. A deterministic composite triggers Engine.evaluate -> confirm.
    # Three total slots admit only the incumbent re-run; four admit both halves. Neither may overrun the cap.
    def confirm_search(name, ctx):
        session = ctx.session('hls')
        ctx.options['experience']['transfer'](session)
        engine = Engine(session, full_steps=62, warmup=0, multi_fidelity=False, run_noise=0.2)
        engine.evaluate(Proposal(change={'device_train_microbatch_size': 8,
                                         'fsdp_config.activation_checkpointing': False}, mu=0.3, confirm=True))
        assert len(engine.confirmations) == 1
        assert session.ledger.used['screens'] == screens - 1
        assert session.stop_flag == 'budget:screens'
        return session.finalize(stop_reason=session.stop_flag)

    monkeypatch.setattr(phase_e.search, 'run', confirm_search)
    clock, events = VirtualClock(), []
    runner = RecordedRunner(task('A-IS2', screens=screens), clock, events, out_dir=tmp_path)
    driver, _, _ = driver_for(events, clock, out=tmp_path)
    result = arms.run_a_is2(runner.task, driver, runner, run_id='confirm', out_dir=tmp_path, clock=clock)
    trace, root_events = assert_reconciles(tmp_path, result)
    reruns = [r for r in trace['screens'] if r.get('fidelity') == 'rescore']
    assert len(reruns) == screens - 2
    assert [r['attempt_id'] for r in reruns] == [f'attempt{i}' for i in range(3, screens + 1)]
    assert len([e for e in root_events if e['kind'] == 'rescore']) == len(reruns)
    assert len([c for c in runner.calls if c['purpose'] in ('screen', 'rescore')]) == screens
    # If the DF5 receipt-id fix is lost, the real reconciler must reject both sides of the broken linkage.
    broken = copy.deepcopy(trace)
    broken['screens'][-1].pop('attempt_id')
    receipts = Store(tmp_path).receipts(tmp_path)
    locked = result['locked'].get('attempt_id')
    failures, _ = accounting_checks(broken, receipts, Store(tmp_path).event_sequence(tmp_path / 'events.jsonl'),
                                    non_screen={locked: 'locked'} if locked else {})
    assert any('screens without a receipt' in f for f in failures)
    assert any('receipts no screen references' in f for f in failures)


def test_nat2_transmits_base_42_request_seeds_and_describes_policy(monkeypatch, tmp_path):
    clock, events = VirtualClock(), []
    native_setup(monkeypatch, clock)
    runner = RecordedRunner(task('A-NAT2', screens=4), clock, events, out_dir=tmp_path)
    driver, meter, http = driver_for(events, clock, native_action={'tool': 'score'})
    arms.run_a_nat2(runner.task, driver, runner, None, run_id='seeds', out_dir=tmp_path,
                    clock=clock, raw_instructions='Synthetic task')
    assert [body.get('seed') for _, _, body in http.requests] == [42, 43, 44]
    assert all(body['chat_template_kwargs'] == {'enable_thinking': False} for _, _, body in http.requests)
    assert driver.describe()['request_seed'] == 42
    assert driver.describe()['request_seed_mode'] == 'base + call index'
    assert meter.calls == 3


def test_fixed_job_settings_reach_experience_plan_and_real_transfer(monkeypatch, tmp_path):
    plans = []
    original = phase_e.exp_runs.plan_for

    def plan_for(project, receipt, **kwargs):
        assert project.raw['states'] == taskmod.knobmod.FIXED_SEMANTICS
        assert set(project.knobs) == set(taskmod.knobmod.KNOBS)
        assert not set(project.raw['states']) & set(project.knobs)
        plan = original(project, receipt, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(phase_e.exp_runs, 'plan_for', plan_for)
    clock, events = VirtualClock(), []
    runner = RecordedRunner(task('A-IS2'), clock, events, out_dir=tmp_path)
    driver, _, http = driver_for(events, clock, out=tmp_path)
    arms.run_a_is2(runner.task, driver, runner, run_id='transfer', out_dir=tmp_path, clock=clock)
    requests = [json.loads(body['messages'][1]['content']) for _, _, body in http.requests
                if 'transfer' in body['messages'][0]['content'].lower()]
    assert len(requests) == 1
    expected = {k: str(v) for k, v in taskmod.knobmod.FIXED_SEMANTICS.items()}
    assert plans[0]['fingerprint']['states'] == expected
    assert requests[0]['target']['states'] == expected
    assert {k['knob'] for k in requests[0]['knobs']} == set(taskmod.knobmod.KNOBS)
