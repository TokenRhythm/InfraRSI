"""ENV-A arm glue (AMENDMENT_EXT_001): setup_home / task_text substitution, A-NAT/A-ISW/A-RND dry-runs with fake
ops/drivers, the LLM meter, t_best_search@8h, and the CLI argument wiring. No GPU, no network, no torch, no LLM."""
import json
import tempfile
from pathlib import Path

from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.rebench_llmf import arms, build_env_a_task, native, source
from infrarsi.ext.rebench_llmf import cli
from infrarsi.ext.rebench_llmf.driver import LlmMeter, MeteringDriver


class FakeResult:
    def __init__(self, llm_tokens, wall_seconds, attempts):
        self.llm_tokens, self.wall_seconds, self.meta = llm_tokens, wall_seconds, {'attempts': attempts}


class FakeBackend:
    def __init__(self, results):
        self._results = list(results)
        self.model = 'deepseek-ai/DeepSeek-V4-Flash'

    def describe(self):
        return {'kind': 'FakeBackend'}

    def complete(self, *a, **k):
        return self._results.pop(0)


# ---------------------------------------------------------------- source / task_text
def test_render_task_text_substitutes_home_and_appends_note():
    raw = ('All files are in /home/agent/solution.\n'
           'train_data.jsonl is /home/agent/train_data.jsonl.\n'
           'canary: SECRET-GUID')
    text = source.render_task_text(raw, '/data/infrarsi/ext/runs/rebench_llmf/R/home_agent')
    assert '/home/agent' not in text
    assert '/data/infrarsi/ext/runs/rebench_llmf/R/home_agent/solution' in text
    assert 'canary' not in text.lower()          # canary line stripped


def test_task_text_appends_tools_and_budget_note():
    t = build_env_a_task(arm='A-NAT')
    text = source.task_text(None, t, '/run/home_agent', raw_instructions='edit /home/agent/solution')
    assert '/run/home_agent/solution' in text
    assert 'score' in text and 'submit' in text and '60 scorings' in text


# ---------------------------------------------------------------- setup_home (fake node)
def test_setup_home_root_owned_home_exact_env_sh_and_arm_uid():
    """M-01 / B-02: the home and env.sh / train_data.jsonl are root's (env.sh exactly the expected text), the
    solution, HOME and the short TMPDIR belong to A-NAT's uid 64101; the reference is not copied here (prepare does)."""
    from infrarsi.ext.rebench_llmf import task as taskmod
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='issetup'), node='WINH1')
    L = native.setup_home(ops, 'R1')
    assert L['solution'].endswith('home_agent/solution') and L['tmp'].startswith('/tmp/isx2nat-') and len(L['tmp']) < 30
    fs = ops.local(f'{L["solution"]}/finetune.sh')
    assert fs.exists() and 'convert_finetuning_dataset' in fs.read_text()      # the base starting solution
    env = ops.local(f'{L["home"]}/env.sh').read_text()
    assert env == taskmod.env_sh_text(taskmod.home_path('R1')) and L['home'] == taskmod.home_path('R1')
    cmds = ' '.join(c[1].get('cmd', '') for c in ops.calls if c[0] == 'sh')
    assert f'chown -R 0:0 {L["home"]}' in cmds and f'chmod 644 {L["home"]}/env.sh' in cmds
    assert 'chown -R -h 64101:64101' in cmds and f'chown 64101:64101 {L["tmp"]}' in cmds
    assert '65534' not in cmds and 'trained_model_reference' not in cmds
    writes = [c[1] for c in ops.calls if c[0] == 'write_files']
    assert all(w.get('uid') is None for w in writes)                         # root writes, the uid gets it last


def test_a_nat_shell_runs_as_64101_with_a_short_tmpdir():
    t = build_env_a_task(arm='A-NAT')
    ops = _ops()
    runner = arms.make_runner(ops, t, run_id='nat', reference_src='/x/ref')
    L = native.layout('nat')
    agent, _ = native.build_agent(t, FakeBackend([]), runner, ops, L, run_id='nat', raw_instructions='x')
    assert agent.exec.uid == 64101 and agent.exec.extra_env['TMPDIR'] == L['tmp']
    assert 'setpriv --reuid=64101' in agent.exec._script('/w/call_0001', 'ls', 10)


def test_arm_lease_records_containment_and_release():
    ops, events = _ops(), []
    with arms.arm_lease(ops, 'run1', (0, 1, 2, 3), uids=[64103, 64113], events=events) as state:
        assert len(state['acquired']) == 5 and state['complete'] is False      # node lease + 4 GPU claims
    assert ops.uid_kills == [64103, 64113]
    assert state['complete'] is True and state['kept'] == [] and len(state['released']) == 5
    assert [c['uid'] for c in state['containment']] == [64103, 64113]
    assert [e['kind'] for e in events] == ['arm_lease_acquired', 'arm_lease_release']


def test_arm_lease_survivor_keeps_the_claims_and_fails_closed():
    import pytest
    ops = _ops()
    ops.uid_survivors[64103] = [99]
    with pytest.raises(arms.ArmLeaseError) as ei:
        with arms.arm_lease(ops, 'run2', (0, 1, 2, 3), uids=[64103, 64113]):
            pass
    st = ei.value.state
    assert st['complete'] is False and len(st['kept']) == 5 and st['containment'][0]['kind'] == 'contain_failed'
    assert ops.read_json('/data/infrarsi/leases/WINH1.json') is not None   # the GPUs stay leased


def test_arm_lease_incomplete_release_fails_closed(monkeypatch):
    import time
    import pytest
    from infrarsi.ext.common.remote_exec import NodeError
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    ops = _ops()
    real = ops.call

    def flaky(func, _timeout=300, **kw):
        if func == 'release_status' and kw.get('path', '').endswith('.gpu2.json'):
            raise NodeError('ssh dropped')
        return real(func, _timeout, **kw)
    with pytest.raises(arms.ArmLeaseError) as ei:
        with arms.arm_lease(ops, 'run3', (0, 1, 2, 3), uids=[64103, 64113]):
            monkeypatch.setattr(ops, 'call', flaky)
    st = ei.value.state
    assert st['complete'] is False and len(st['kept']) == 1 and st['kept'][0].endswith('WINH1.gpu2.json')
    assert st['release_failures'] and 'ssh dropped' in st['release_failures'][0]['error']


def test_arm_lease_refuses_uids_in_use_before_taking_gpus():
    import pytest
    from infrarsi.ext.common.remote_exec import ContainmentError
    ops = _ops()
    ops.uid_procs[64113] = [7]
    with pytest.raises(ContainmentError):
        with arms.arm_lease(ops, 'run4', (0, 1, 2, 3), uids=[64103, 64113]):
            pass
    assert not any(c[0] == 'acquire' for c in ops.calls)


def test_cli_fails_closed_with_the_retained_paths(monkeypatch, capsys):
    import contextlib
    state = {'kept': ['/data/infrarsi/leases/ext/WINH1.gpu2.json'], 'complete': False, 'release_failures': [{'error': 'x'}]}

    @contextlib.contextmanager
    def failing_lease(*a, **k):
        k.get('events', []).append({'kind': 'arm_lease_release', **state})
        yield {}
        raise arms.ArmLeaseError('final release incomplete; retained ' + str(state['kept']), state)
    monkeypatch.setattr(arms, 'arm_lease', failing_lease)
    monkeypatch.setattr(arms, 'run_a_rnd', lambda *a, **k: {'arm': 'A-RND', 'trace': {'screens': []}})
    out = Path(tempfile.mkdtemp(prefix='isclilease'))
    rc = cli.main(['a-rnd', '--run-id', 'leasefail', '--local', '--out-dir', str(out)])
    assert rc == 3
    res = json.loads((out / 'result.json').read_text())
    assert res['retained'] == state['kept'] and res['arm_lease']['complete'] is False
    assert json.loads((out / 'arm_lease.json').read_text())['state']['kept'] == state['kept']
    assert 'arm_lease' in (out / 'events.jsonl').read_text()
    assert 'retained' in capsys.readouterr().err


def test_cli_result_carries_the_arm_lease_state(monkeypatch):
    monkeypatch.setattr(arms, 'run_a_rnd', lambda *a, **k: {'arm': 'A-RND', 'trace': {'screens': []}})
    out = Path(tempfile.mkdtemp(prefix='isclilease2'))
    rc = cli.main(['a-rnd', '--run-id', 'leaseok', '--local', '--out-dir', str(out)])
    res = json.loads((out / 'result.json').read_text())
    assert rc == 0 and res['arm_lease']['complete'] is True and res['arm_lease']['uids'] == [64103, 64113]
    assert [c['uid'] for c in res['arm_lease']['containment']] == [64103, 64113]


# ---------------------------------------------------------------- arm dry-runs
def _ops():
    return LocalNodeOps(tempfile.mkdtemp(prefix='isarm'), node='WINH1')


def test_a_rnd_dry_run():
    t = build_env_a_task(arm='A-RND')
    runner = arms.make_runner(_ops(), t, run_id='rnd', reference_src=None)
    out = arms.run_a_rnd(t, runner, run_id='rnd', seed=42, n=60, dry_run=True)
    assert out['dry_run'] and out['arm'] == 'A-RND' and out['n_points'] == 60


def test_a_rnd_runs_searchsession_end_to_end():
    """Exercise the real SearchSession path (not just the dry-run): screen the start + a few random knob points
    through the integrated runner with a synthetic node result, then locked re-score the best."""
    import rebench_helpers as H
    r, t = H.runner(H.node_result(H.task('A-RND'), run_id='rnd'), arm='A-RND', g4=H.g4_ok())
    fast = H.node_result(H.task('A-RND'), run_id='rnd', execution_time=150.0)   # changed knobs beat the start (D20)
    base_fake = r.node_runner

    def node(ops, job):
        if job.get('mode') != 'prepare' and job['finetune_yaml'] != job['finetune_yaml_base']:
            r.fake.jobs.append(job)
            return dict(fast, g4=H.g4_ok() if job.get('run_g4') else None)
        return base_fake(ops, job)
    r.node_runner = node
    out = arms.run_a_rnd(t, r, run_id='rnd', seed=42, n=3)
    assert out['arm'] == 'A-RND' and out['prepare']['mode'] == 'prepare'
    tr = out['trace']
    assert len(tr['screens']) >= 1 and tr['selected'] is not None
    assert out['locked']['locked']['valid'] is True          # locked re-score ran at purpose 'score'
    assert all(j['uid'] == 64103 for j in r.fake.jobs) and r.fake.jobs[-1]['clean_slate'] is True


def test_a_nat_dry_run_builds_agent():
    t = build_env_a_task(arm='A-NAT')
    ops = _ops()
    runner = arms.make_runner(ops, t, run_id='nat', reference_src='/x/ref')
    out = arms.run_a_nat(t, FakeBackend([]), runner, ops, run_id='nat',
                         raw_instructions='edit /home/agent/solution', dry_run=True)
    assert out['dry_run'] and out['arm'] == 'A-NAT' and out['tools'] == ['score'] and out['uid'] == 64101
    assert out['context_tokens'] == 72_000 and out['task_text_chars'] > 0


def test_a_isw_dry_run_overrides_max_turns_in_a_copy():
    import yaml
    from infrarsi.rsi import build_a0
    work = Path(tempfile.mkdtemp(prefix='isisw'))
    pkg = build_a0(work / 'A0')
    t = build_env_a_task(arm='A-ISW', action_space='knobs+patch')
    runner = arms.make_runner(_ops(), t, run_id='isw', reference_src=None, enforce_g7=True)
    out = arms.run_a_isw(t, pkg, FakeBackend([]), runner, run_id='isw', max_turns=200,
                         package_work_dir=work / 'A0_isw', dry_run=True)
    assert out['dry_run'] and out['arm'] == 'A-ISW' and out['mode'] == 'weak' and out['max_turns'] == 200
    assert yaml.safe_load((work / 'A0' / 'policy.yaml').read_text())['modes']['weak']['max_turns'] == 120
    assert yaml.safe_load((work / 'A0_isw' / 'policy.yaml').read_text())['modes']['weak']['max_turns'] == 200


# ---------------------------------------------------------------- meter / reporting
def test_llm_meter_uncached_and_h20_hours():
    meter = LlmMeter()
    drv = MeteringDriver(FakeBackend([
        FakeResult(300, 10.0, [{'usage': {'prompt_tokens': 200, 'completion_tokens': 100,
                                           'prompt_tokens_details': {'cached_tokens': 150}}}]),
        FakeResult(60, 20.0, [{'usage': {'prompt_tokens': 40, 'completion_tokens': 20}}]),
    ]), meter)
    drv.complete(); drv.complete()
    s = meter.summary()
    assert s['llm_calls'] == 2 and s['llm_tokens_total'] == 360
    assert s['llm_tokens_uncached'] == (200 - 150 + 100) + (40 - 0 + 20)     # 150 + 60 = 210
    assert abs(s['llm_wall_hours'] - 30 / 3600) < 1e-3
    assert abs(s['llm_gpu_hours_h20'] - 4 * 30 / 3600) < 1e-3 and s['llm_usd'] == 0


def test_t_best_search_at_window():
    trace = {'screens': [
        {'valid': True, 'step_time': 220.0, 'cum': {'wall_seconds': 10_000}},
        {'valid': True, 'step_time': 200.0, 'cum': {'wall_seconds': 20_000}},
        {'valid': True, 'step_time': 180.0, 'cum': {'wall_seconds': 40_000}},   # after 8 h
        {'valid': False, 'step_time': None, 'cum': {'wall_seconds': 15_000}},
    ]}
    assert arms.t_best_search_at(trace, seconds=28_800) == 200.0
    assert arms.t_best_search_at(trace, seconds=50_000) == 180.0


def test_context_check_prompt_fits():
    from infrarsi.agent.context import estimate_tokens
    system, prompt = native.context_check_prompt(72_000)
    units = estimate_tokens(system + prompt)
    assert 70_000 <= units <= 73_000


# ---------------------------------------------------------------- CLI wiring
def test_cli_parses_every_subcommand():
    p = cli.build_parser()
    a = p.parse_args(['a-nat', '--node', 'WINH1', '--gpus', '0,1,2,3', '--run-id', 'r', '--context-tokens', '72000'])
    assert a.fn is cli.cmd_a_nat and a.context_tokens == 72000 and a.endpoint_context == 98304
    a = p.parse_args(['a-isw', '--run-id', 'r', '--max-turns', '200', '--context-limit', '32000', '--base-url', 'http://127.0.0.1:18000/v1'])
    assert a.fn is cli.cmd_a_isw and a.max_turns == 200 and a.context_limit == 32000
    a = p.parse_args(['a-rnd', '--run-id', 'r', '--seed', '42', '--n', '60'])
    assert a.fn is cli.cmd_a_rnd and a.seed == 42 and a.n == 60
    a = p.parse_args(['context-check', '--dry-run', '--context-tokens', '72000'])
    assert a.fn is cli.cmd_context_check and a.dry_run and a.context_tokens == 72000


def test_cli_dry_run_a_rnd_local():
    rc = cli.main(['a-rnd', '--run-id', 'clidry', '--dry-run', '--local',
                   '--out-dir', tempfile.mkdtemp(prefix='isclidry')])
    assert rc == 0
