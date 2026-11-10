"""Phase-E arms with the real Engine/basic agent and a fake scorer and HTTP transport; CPU only."""
import copy
import json
import math

import pytest

from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.rebench_llmf import arms, cli, native, task as taskmod
from infrarsi.ext.rebench_llmf.driver import PHASE_E_MODEL, phase_e_driver
from infrarsi.ext.rebench_llmf.phase_e import EnvAContext
from infrarsi.ext.rebench_llmf.hooks import EnvAHooks
from infrarsi.harness import signing


# Synthetic scripts only: no RE-Bench task text or source is copied into the fixtures.
BASE = {'finetune.sh': '#!/bin/bash\necho synthetic\n',
        'finetune.yaml': 'device_train_microbatch_size: 4\nfsdp_config:\n  activation_checkpointing: true\n'}


def task(arm, screens=3, **budget):
    return taskmod.build_env_a_task(arm=arm, action_space='knobs+patch' if arm == 'A-IS2' else 'knobs',
                                    frozen_files={'score.py': '0' * 64, 'train_data.jsonl': '1' * 64},
                                    budget={**taskmod.BUDGET_LLM, 'screens': screens, **budget})


class FakeRunner:
    def __init__(self, task, clock, events, *, invalid_start=False):
        self.task, self.clock, self.events = task, clock, events
        self.uid, self.judge_uid = taskmod.arm_uid(task.arm), taskmod.judge_uid(task.arm)
        self.gpus, self.node, self.base = (0, 1, 2, 3), 'WINH0', dict(BASE)
        self.deviations = taskmod.arm_deviations(task.arm)
        self.calls, self.receipts = [], []
        self.invalid_start = invalid_start

    def prepare(self, run_id):
        self.events.append('prepare')
        return {'mode': 'prepare'}

    def domain_uids(self):
        return [self.uid, self.judge_uid]

    def worst_case(self):
        return 10

    def admits(self, ledger, budget):
        return ledger.admit_screen((40 / 3600, 10))

    def run(self, task, config, overlay, purpose, run_id, **kw):
        self.events.append(purpose)
        self.calls.append({'config': config, 'overlay': overlay, 'purpose': purpose, **kw})
        self.clock.advance(10)
        n = len(self.calls)
        value = 200.0 if n == 1 else 140.0
        valid = not (self.invalid_start and n == 1)
        tree = kw.get('expected_digest') or (taskmod.START_TREE_DIGEST if n == 1 else str(n) * 64)
        r = {'schema': 'ext-receipt-v1', 'env': 'rebench_llmf', 'arm': task.arm, 'run_id': run_id,
             'attempt_id': f'attempt{n}', 'candidate_id': kw.get('candidate_id'), 'purpose': purpose,
             'valid': valid, 'judge_valid': valid, 'invalid_reasons': [] if valid else ['fake failure'],
             'guard_rejections': [], 'world_size': 4, 'candidate_digest': tree, 'tree_digest': tree,
             'snapshot': {'path': f'/fake/{tree}.tar', 'tree_digest': tree, 'sha256': '2' * 64},
             'step_time': {'mean_window_max_rank_s': value if valid else None},
             'metric': {'name': 'exec_time_s', 'value': value if valid else None, 'direction': 'min'},
             'judge': {'score': math.log(value), 'message': {'execution_time': value, 'norm_diff': 3.4}},
             'cost': {'gpu_seconds': 40, 'wall_seconds': 10, 'accounting_complete': True},
             'deviations': list(self.deviations)}
        signing.sign(r)
        self.receipts.append(r)
        return r


class FakeHTTP:
    def __init__(self, events, clock, native_action=None):
        self.events, self.clock, self.native_action = events, clock, native_action
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.events.append('llm')
        self.clock.advance(1)
        self.requests.append((url, headers, copy.deepcopy(body)))
        if self.native_action:
            action = self.native_action
        elif 'transfer' in body['messages'][0]['content'].lower():
            action = {'assessment': [], 'start_config': {}}
        elif 'reference' in json.loads(body['messages'][1]['content']):
            action = {'verdict': 'patch', 'findings': [
                {'file': 'finetune.sh', 'issue': 'synthetic patch', 'knob': None,
                 'patch': {'finetune.sh': BASE['finetune.sh'] + '# candidate\n'},
                 'predicted_gain_pct': 40, 'uncertainty_pct': 1, 'p_feasible': 1}]}
        else:
            action = {'hypotheses': [
                {'mechanism': 'synthetic microbatch', 'change': {'device_train_microbatch_size': 8},
                 'predicted_gain_pct': 20, 'uncertainty_pct': 1, 'p_feasible': 1}]}
        return 200, {'model': PHASE_E_MODEL, 'usage': {'prompt_tokens': 100, 'completion_tokens': 20},
                     'choices': [{'message': {'content': json.dumps(action)}, 'finish_reason': 'stop'}]}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv('VLLM_URL', 'http://fake.invalid:8000/v1')
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    # A present API credential must never be sent to this keyless endpoint.
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'must-not-send')
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-send')


def driver_for(events, clock, *, native_action=None, out=None):
    http = FakeHTTP(events, clock, native_action)
    driver, _, meter = phase_e_driver(native=native_action is not None, transport=http, out_dir=out)
    return driver, meter, http


def assert_settings(http, temperature):
    assert http.requests
    for url, headers, body in http.requests:
        assert url == 'http://fake.invalid:8000/v1/chat/completions'
        assert 'Authorization' not in headers
        assert body['model'] == PHASE_E_MODEL and body['temperature'] == temperature
        assert body['chat_template_kwargs'] == {'enable_thinking': False}
        assert body['response_format'] == {'type': 'json_object'}


def test_is2_runs_real_hls_start_first_patches_budget_and_experience(tmp_path):
    clock, events = VirtualClock(), []
    t = task('A-IS2')
    r = FakeRunner(t, clock, events)
    d, meter, http = driver_for(events, clock, out=tmp_path)
    res = arms.run_a_is2(t, d, r, run_id='is2', out_dir=tmp_path, clock=clock)
    tr = res['trace']
    assert events[:3] == ['prepare', 'screen', 'llm']
    assert r.calls[0]['candidate_id'] == 'c0' and r.calls[0]['config'] == {}
    assert len([c for c in r.calls if c['purpose'] == 'screen']) == 3
    assert r.calls[-1]['purpose'] == 'score' and r.calls[-1]['run_g4'] is True
    assert tr['arm'] == 'A-IS2' and res['arm'] == 'A-IS2'
    assert tr['extra']['strategy']['name'] == 'hls' and tr['extra']['multi_fidelity'] is False
    assert res['capability'] == 0.6 and tr['mode'] == 'strong'
    assert any(c['overlay'] for c in r.calls[1:])  # the code review's patch reached the runner
    assert res['experience']['levels'] == {'none': 9}
    assert all(c['level'] == 'none' for c in res['experience']['coverage'].values())
    assert tr['budget']['screens'] == tr['used']['screens'] == len(tr['screens']) == 3
    assert tr['extra']['search_budget']['screens'] == 2
    assert tr['extra']['search_budget']['gpu_hours'] == pytest.approx(34 - 40 / 3600)
    assert tr['extra']['search_budget']['wall_seconds'] == 57600 - 10
    assert tr['used']['gpu_hours'] == pytest.approx(120 / 3600)
    assert tr['used']['wall_seconds'] == 30 + len(http.requests)
    assert tr['used']['llm_tokens'] == meter.tokens_total == 120 * len(http.requests)
    assert [s['cum']['screens'] for s in tr['screens']] == [1, 2, 3]
    assert [s['fractions']['screens'] for s in tr['screens']] == [1 / 3, 2 / 3, 1]
    assert tr['stop_reason'] == 'budget:screens'
    assert res['locked']['locked']['valid']
    assert all(rec['arm'] == 'A-IS2' and signing.verify(rec) for rec in r.receipts)
    assert json.loads((tmp_path / 'trace.json').read_text()) == tr
    assert json.loads((tmp_path / 'search' / 'trace.json').read_text())['arm'] == 'A-IS2'
    assert d.context_tokens == 40960 and d.max_tokens == 16384
    assert 'llm_gpu_hours_h100' in meter.summary() and 'llm_gpu_hours_h20' not in meter.summary()
    assert_settings(http, 0.0)


def native_setup(monkeypatch, clock):
    monkeypatch.setattr(native, 'setup_home', lambda ops, run_id, **kw: native.layout(run_id))
    class Shell:
        def __init__(self, *args, **kwargs):
            pass

        def bash(self, cmd, timeout):
            clock.advance(1)
            return {'exit_code': 0, 'timed_out': False, 'seconds': clock(), 'stdout': 'same', 'stderr': ''}

    monkeypatch.setattr(native, 'ShellExecutor', Shell)


def test_nat2_start_first_temperature_budget_and_locked_snapshot(monkeypatch, tmp_path):
    clock, events = VirtualClock(), []
    native_setup(monkeypatch, clock)
    t = task('A-NAT2')
    r = FakeRunner(t, clock, events)
    d, meter, http = driver_for(events, clock, native_action={'tool': 'score'}, out=tmp_path)
    tr = arms.run_a_nat2(t, d, r, None, run_id='nat2', out_dir=tmp_path, clock=clock, raw_instructions='Synthetic task')
    assert events[:3] == ['prepare', 'screen', 'llm']
    assert tr['arm'] == tr['extra']['arm'] == 'A-NAT2'
    assert tr['budget']['screens'] == tr['used']['screens'] == len(tr['screens']) == 3
    assert [s['kind'] for s in tr['screens']] == ['A-NAT2'] * 3
    assert tr['used']['gpu_hours'] == pytest.approx(120 / 3600)
    assert tr['used']['llm_tokens'] == meter.tokens_total == 240
    assert tr['stop_reason'] == 'budget:screens' and tr['selected'] == 'score02'
    assert len(r.calls) == 4 and r.calls[0]['reuse_home'] is True
    assert r.calls[-1]['restore_snapshot'] == r.receipts[1]['snapshot']['path']
    assert tr['extra']['locked']['locked']['valid']
    assert tr['extra']['loop_detector']['tripped'] is None
    assert 'S0 reference before search' in http.requests[0][2]['messages'][1]['content']
    assert d.context_tokens == 40960 and d.max_tokens == 4096
    assert all(rec['arm'] == 'A-NAT2' and signing.verify(rec) for rec in r.receipts)
    assert_settings(http, 1.0)


def test_nat2_ext006_loop_detection_and_s0_locked_skip(monkeypatch, tmp_path):
    clock, events = VirtualClock(), []
    native_setup(monkeypatch, clock)
    r = FakeRunner(task('A-NAT2', screens=60), clock, events)
    d, _, http = driver_for(events, clock, native_action={'tool': 'bash', 'cmd': 'echo same'})
    tr = arms.run_a_nat2(r.task, d, r, None, run_id='loop', out_dir=tmp_path, clock=clock, raw_instructions='Synthetic')
    assert tr['stop_reason'] == 'degenerate_loop' and len(http.requests) == 12
    assert tr['extra']['loop_detector']['rule'] == {'consecutive': 12, 'window': 60, 'window_count': 30}
    assert tr['extra']['n_identical_max'] == 12 and tr['used']['screens'] == 1
    assert tr['extra']['locked']['skipped'] == 'selection_is_start' and len(r.calls) == 1
    assert_settings(http, 1.0)


@pytest.mark.parametrize('screens,invalid', [(0, False), (1, False), (3, True)])
def test_is2_does_not_call_llm_without_search_room_or_valid_start(screens, invalid):
    clock, events = VirtualClock(), []
    r = FakeRunner(task('A-IS2', screens=screens), clock, events, invalid_start=invalid)
    d, _, http = driver_for(events, clock)
    tr = arms.run_a_is2(r.task, d, r, run_id='short', clock=clock)['trace']
    assert not http.requests and len(r.calls) == min(screens, 1)
    assert tr['used']['screens'] == min(screens, 1)


def test_engine_rescores_consume_the_scoring_budget():
    clock, events = VirtualClock(), []
    t = task('A-IS2', screens=2)
    r = FakeRunner(t, clock, events)
    ctx = EnvAContext(task=t, budget=t.search_budget(), start_config={}, runner=r, judge=EnvJudge('rebench_llmf'),
                      hooks=EnvAHooks(t, base_files=BASE), run_id='rescore', out_dir=None, options={'clock': clock})
    s = ctx.session('hls')
    s.screen('c0')
    s.rescore('c0')
    s.rescore('c0')
    assert len(r.calls) == 2 and s.ledger.used['screens'] == 2
    assert s.screens[-1]['cum']['screens'] == 2 and s.stop_flag == 'budget:screens'


@pytest.mark.parametrize('command,arm,g7', [('a-is2', 'A-IS2', True), ('a-nat2', 'A-NAT2', False)])
def test_cli_wiring_and_default_budgets(monkeypatch, tmp_path, command, arm, g7):
    seen = {}
    # Run the CLI through its real LocalNodeOps lease; replace only the arm work and runner construction.
    clock = VirtualClock()
    def runner(ops, t, **kw):
        seen.update(task=t, **kw)
        return FakeRunner(t, clock, [])
    def run(t, driver, r, *args, **kw):
        assert driver.model == PHASE_E_MODEL and driver.context_tokens == 40960
        assert driver.temperature == (1.0 if arm == 'A-NAT2' else 0.0)
        return {'arm': t.arm, 'trace': {'arm': t.arm, 'screens': []}}
    monkeypatch.setattr(arms, 'make_runner', runner)
    monkeypatch.setattr(arms, 'run_a_is2' if arm == 'A-IS2' else 'run_a_nat2', run)
    monkeypatch.setattr(cli, '_task', lambda name, a, over: task(name, screens=60))
    assert cli.main([command, '--run-id', 'cli', '--local', '--out-dir', str(tmp_path)]) == 0
    res = json.loads((tmp_path / 'result.json').read_text())
    assert res['arm'] == arm and res['arm_lease']['complete']
    assert res['arm_lease']['uids'] == [taskmod.arm_uid(arm), taskmod.judge_uid(arm)]
    assert seen['enforce_g7'] is g7 and seen['task'].budget == taskmod.BUDGET_LLM
    assert cli.build_parser().parse_args([command, '--run-id', 'x', '--on-node']).on_node


def test_missing_endpoint_fails_before_node_setup(monkeypatch):
    monkeypatch.delenv('VLLM_URL')
    monkeypatch.setattr(cli, '_ops', lambda a: pytest.fail('node setup before endpoint validation'))
    with pytest.raises(ValueError, match='VLLM_URL'):
        cli.main(['a-is2', '--run-id', 'missing'])


def test_on_node_reuses_node_ops_and_existing_task_location(monkeypatch):
    import os
    from infrarsi.ext.common import node_ext
    a = cli.build_parser().parse_args(['a-is2', '--run-id', 'local', '--node', 'WINH0', '--on-node'])
    monkeypatch.setattr(cli, 'build_env_a_task', lambda **kw: kw)
    assert cli._task('A-IS2', a, None)['src_dir'] == taskmod.NODE_TASK
    monkeypatch.setattr(node_ext, 'hash_files', lambda paths: {'paths': paths})
    with monkeypatch.context() as m:
        m.setattr(os, 'name', 'posix')
        m.setattr(os, 'geteuid', lambda: 0, raising=False)
        ops = cli._ops(a)
    assert ops.node == 'WINH0' and ops.hash_files(['/synthetic']) == {'paths': ['/synthetic']}


@pytest.mark.parametrize('arm', ['A-IS2', 'A-NAT2'])
@pytest.mark.parametrize('limit', ['wall_seconds', 'gpu_hours'])
def test_empty_budget_admits_no_start_or_llm(monkeypatch, arm, limit):
    clock, events = VirtualClock(), []
    native_setup(monkeypatch, clock)
    r = FakeRunner(task(arm, **{limit: 0}), clock, events)
    d, _, http = driver_for(events, clock, native_action={'tool': 'score'} if arm == 'A-NAT2' else None)
    if arm == 'A-IS2':
        tr = arms.run_a_is2(r.task, d, r, run_id='empty', clock=clock)['trace']
    else:
        tr = arms.run_a_nat2(r.task, d, r, None, run_id='empty', clock=clock, raw_instructions='Synthetic')
    assert not r.calls and not http.requests and tr['used']['screens'] == 0
    assert tr['stop_reason'] == f'budget:{limit}'
