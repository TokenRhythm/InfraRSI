"""ENV-B arm launchers (ext/optico/arms.py) on a fake node: B-RND, B-IS through optimize(), locked pairs, G-B2 policy."""
import json

import pytest

from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.optico import arms
from infrarsi.ext.optico import params as pm
from infrarsi.ext.optico import workloads as wl
from infrarsi.ext.optico.runner import OptiCoRunner
from infrarsi.rsi import build_a0
from test_optico_runner import FakeTrial


def test_sequence_parallelism_follows_tp_in_the_experience_states():
    # DF5_r2 #4: OptiCo's scripts turn --sequence-parallel (and --tp-comm-overlap) on with TP > 1
    from infrarsi.experience import runs as exp_runs
    from infrarsi.experience import store as S
    w = wl.get('W1')
    v = arms.project_view(w)
    labs = S.labels(v.knobs, 'megatron')
    fp = {'states': dict(v.raw['states']), 'framework': 'megatron'}
    assert S.states_of(fp, labs, {**w.defaults, 'TP': 4})['sequence_parallel'] == 'on'
    assert S.states_of(fp, labs, {**w.defaults, 'TP': 1})['sequence_parallel'] == 'off'
    exp = exp_runs.plan_for(v, None, select=None, driver=None, log=lambda m: None)
    assert not exp['check']({**w.defaults, 'TP': 4, 'PP': 1, 'MICRO_BATCH_SIZE': 4, 'AC': 'none'})   # ~67 GiB, not 83


def test_a_knob_that_sets_the_mechanism_wins_over_an_implied_state():
    from infrarsi.experience import store as S
    knobs = {'tp': {'type': 'int', 'choices': [1, 2], 'mechanism': 'tp', 'implies': {2: {'sequence_parallel': 'on'}}},
             'sp': {'type': 'bool', 'mechanism': 'sequence_parallel', 'states': {True: 'on', False: 'off'}}}
    labs = S.labels(knobs, 'megatron')
    assert S.states_of({}, labs, {'tp': 2})['sequence_parallel'] == 'on'
    assert S.states_of({}, labs, {'tp': 2, 'sp': False})['sequence_parallel'] == 'off'
    assert 'sequence_parallel' not in S.states_of({}, labs, {'tp': 1})

W3 = wl.get('W3')
PASS = {'verdict': 'pass', 'reasons': []}


@pytest.fixture(scope='module')
def a0(tmp_path_factory):
    return build_a0(tmp_path_factory.mktemp('pkg') / 'A0')


def _runner(tmp_path, scenario='success'):
    clock = VirtualClock()
    fake = FakeTrial(scenario, clock=clock)
    r = OptiCoRunner(LocalNodeOps(tmp_path / 'node', node='WINH4', launcher=fake), 'W3', out_dir=tmp_path / 'pc',
                     clock=clock, sleep=clock.advance)
    return r, fake, clock


def test_b_rnd_runs_start_plus_new_trials(tmp_path):
    runner, fake, clock = _runner(tmp_path)
    tr = arms.run_b_rnd('W3', None, run_id='brnd', out_dir=tmp_path / 'trace', screens=4, runner=runner, clock=clock)
    assert len(tr['screens']) == 4 and tr['extra']['seed'] == 42 and tr['stop_reason'] == 'budget:screens'
    assert fake.launched[0]['params'] == W3.defaults                                    # c0 = the script defaults
    assert len({pm.params_key(x['params']) for x in fake.launched}) == 4


def test_b_is_through_optimize_with_hooks(tmp_path, a0):
    runner, fake, clock = _runner(tmp_path)
    script = [{'action': 'read_env'},
              {'action': 'propose_config', 'base': 'c0', 'changes': {'MICRO_BATCH_SIZE': 2, 'FUSED_CROSS_ENTROPY': True},
               'rationale': 'larger micro batch, fused CE'},
              {'action': 'run_screen', 'candidate_id': 'c1'},
              {'action': 'propose_config', 'base': 'c0', 'changes': {'TP': 4}, 'rationale': 'invalid on purpose'},
              {'action': 'finish', 'candidate_id': 'c1', 'reason': 'done'}]
    drv = ScriptedBackend(lambda role, s, p: PASS if role == 'verifier' else script.pop(0), clock=clock)
    tr = arms.run_b_is('W3', None, run_id='bis', out_dir=tmp_path / 'trace', screens=5, package=a0, backend=drv,
                       runner=runner, clock=clock)
    assert tr['arm'] == 'B-IS' and tr['stop_reason'] == 'finish' and [s['candidate_id'] for s in tr['screens']] == ['c0', 'c1']
    assert fake.launched[1]['params'] == W3.knobs(MICRO_BATCH_SIZE=2, FUSED_CROSS_ENTROPY=True)
    assert tr['invalid_proposals'].get('l0_invalid') == 1                              # TP 4 never reached the node
    proposer = next(c for c in drv.calls if c['role'] != 'verifier')
    assert 'run_circle.sh qwen2.5 0.5B' in proposer['prompt'] or 'run_circle.sh qwen2.5 0.5B' in proposer['system']


def test_b_isw_records_the_weak_endpoint_deviation(tmp_path, a0):
    runner, fake, clock = _runner(tmp_path)
    drv = ScriptedBackend(lambda role, s, p: PASS if role == 'verifier' else {'action': 'finish', 'reason': 'x'}, clock=clock)
    arms.run_b_is('W3', None, run_id='bisw', out_dir=tmp_path / 'trace', weak=True, screens=3, package=a0, backend=drv,
                  runner=runner, clock=clock)
    rec = json.loads(next((tmp_path / 'pc' / 'receipts').glob('*.json')).read_text())
    assert 'D-OC-10' in rec['deviations'] and rec['arm'] == 'B-ISW'


def test_b_is2_runs_the_current_search_from_the_start(tmp_path, monkeypatch):
    # phase E (10-24): B-IS2 = search.run('hls') on the same task, runner and judge; the start is its first counted
    # screen (as B-IS's), the trace is the arm's at the run root (DF5_r1 #1, #3)
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    runner, fake, clock = _runner(tmp_path)
    hyps = {'hypotheses': [{'mechanism': 'fused cross entropy', 'change': {'FUSED_CROSS_ENTROPY': True},
                            'predicted_gain_pct': 2.0, 'uncertainty_pct': 2.0, 'p_feasible': 0.9,
                            'change_class': 'numerics', 'rationale': 'x'}]}
    drv = ScriptedBackend(lambda role, s, p: hyps if role == 'hypotheses' else {'assessment': []}, clock=clock)
    tr = arms.run_b_is2('W3', None, run_id='bis2', out_dir=tmp_path / 'trace', screens=4, backend=drv, runner=runner)
    assert fake.launched[0]['params'] == W3.defaults and tr['screens'][0]['candidate_id'] == 'c0'   # the start, counted
    assert len(fake.launched) == len(tr['screens']) <= 4 and tr['used']['screens'] == len(tr['screens'])
    assert any(x['params'] == W3.knobs(FUSED_CROSS_ENTROPY=True) for x in fake.launched[1:])
    assert tr['arm'] == 'B-IS2' and (tmp_path / 'trace' / 'trace.json').is_file()
    view = arms.project_view('W3')
    assert view.knobs['AC']['mechanism'] == 'recompute' and view.knobs['TP']['mechanism'] == 'tp'
    from infrarsi.experience.store import fingerprint
    assert fingerprint(view)['states'] == {'precision': 'fp16', 'distributed_optimizer': 'on'}   # DF5_r1 #5


def test_the_new_arms_reach_the_margin_and_the_summary():
    # DF5_r1 #4: the phase-E arms' starts and locked defaults are protocol default runs; the summary lists their pairs
    from infrarsi.ext.optico import margin
    from infrarsi.ext.summary import env_b
    base = {'env': 'optico', 'task_id': W3.task_id}
    assert margin.run_kind({**base, 'arm': 'B-IS2', 'purpose': 'screen', 'candidate_id': 'c0'}, 'W3') == 'search_start'
    assert margin.run_kind({**base, 'arm': 'B-IS2-noexp', 'purpose': 'locked', 'candidate_id': 'locked-default'},
                           'W3') == 'locked_default'
    assert ('B-IS2', 'B-NAT') in env_b.MAIN_PAIRS and 'B-IS2' in env_b.OPTIONAL_ARMS and 'B-IS2' in env_b.LOCKED_ARMS


@pytest.mark.parametrize('arm,first', [('B-IS', 'candidate'), ('B-RND', 'default'), ('B-NAT', 'default')])
def test_locked_pair_order_and_margin(tmp_path, arm, first):
    runner, fake, _ = _runner(tmp_path)
    p = W3.knobs(MICRO_BATCH_SIZE=2)
    res = arms.run_locked('W3', None, run_id='lk', out_dir=tmp_path / 'lk', params=p, arm=arm, runner=runner)
    assert res['order'][0] == first and [x['params'] for x in fake.launched][0 if first == 'default' else 1] == W3.defaults
    assert res['speedup_vs_default'] == pytest.approx(1.0) and res['invalid_accept_locked'] == 0
    rsd = res['diag_within_run_rsd']
    assert res['diag_within_run_margin'] == pytest.approx(max(0.03, 3 * rsd)) and res['diag_beyond_within_run_margin'] is False
    assert 'margin' not in res and 'beyond_margin' not in res and 'AMENDMENT_EXT_002' in res['diagnostics_note']
    assert json.loads((tmp_path / 'lk' / 'locked.json').read_text())['arm'] == arm


def test_locked_native_selection_runs_as_optico_ran_it(tmp_path):
    runner, fake, _ = _runner(tmp_path)
    odd = {**{k: pm.encode_params(W3.defaults).splitlines()[i].split(': ')[1] for i, k in enumerate(pm.KNOBS)}, 'AC': 'selective'}
    res = arms.run_locked('W3', None, run_id='lk', out_dir=tmp_path / 'lk', params=odd, arm='B-NAT', runner=runner)
    assert fake.launched[1]['params']['AC'] == 'selective'                          # no validator for OptiCo's own choice
    assert 'D-OC-13' not in json.loads(next((tmp_path / 'pc' / 'receipts').glob('*.json')).read_text())['deviations']
    with pytest.raises(ValueError):
        arms.run_locked('W3', None, run_id='lk', out_dir=tmp_path / 'lk2', params=odd, arm='B-IS', runner=runner)
    assert res['candidate']['attempt_id']


class NoRunner:
    """Any use of the runner (a node contact) fails the test."""
    deviations = ()

    def run(self, *a, **kw):
        raise AssertionError('a locked run was started for a default selection')


@pytest.mark.parametrize('arm', ['B-RND', 'B-IS', 'B-ED', 'B-NAT'])
def test_locked_pair_is_skipped_when_the_selection_is_the_default(tmp_path, arm):
    from infrarsi.ext.optico import margin
    sel = dict(W3.defaults)
    if arm == 'B-NAT':                            # OptiCo's best row comes back as its own strings
        sel = {k: v.strip() for k, v in (ln.split(':', 1) for ln in pm.encode_params(W3.defaults).splitlines())}
    res = arms.run_locked('W3', None, run_id='lk', out_dir=tmp_path / 'lk', params=sel, arm=arm, runner=NoRunner())
    on_disk = json.loads((tmp_path / 'lk' / 'locked.json').read_text(encoding='utf-8'))
    for r in (res, on_disk):
        assert r['skipped'] == 'selection_is_default' and r['y'] == 0 and r['speedup_vs_default'] == 1.0
        assert r['invalid_accept_locked'] == 0 and r['default'] is None and r['candidate'] is None and r['arm'] == arm
    assert margin.load_receipts([tmp_path]) == []                 # no receipt -> no default sample for margin.py
    assert arms.is_default_selection(W3.defaults, 'W3') and not arms.is_default_selection(W3.knobs(TP=2), 'W3')


def test_selection_and_gb2_policy(tmp_path):
    tr = tmp_path / 'trace.json'
    tr.write_text(json.dumps({'selected_config': W3.knobs(TP=2)}))
    assert arms.selection_from(tr) == W3.knobs(TP=2)
    (tmp_path / 'nat' / 'receipts').mkdir(parents=True)
    (tmp_path / 'nat' / 'result.json').write_text(json.dumps({'best_success_row': {'attempt_id': 'bnat-x-r02'}}))
    row = {k: v.strip() for k, v in (ln.split(':') for ln in pm.encode_params(W3.knobs(MICRO_BATCH_SIZE=4)).splitlines())}
    (tmp_path / 'nat' / 'receipts' / 'bnat-x-r02.json').write_text(json.dumps({'params': row}))
    assert arms.selection_from(tmp_path / 'nat' / 'result.json') == W3.knobs(MICRO_BATCH_SIZE=4)
    d, e = {'loss': {'iter1': 11.93}, 'attempt_id': 'd'}, {'loss': {'iter1': 11.93}}
    assert arms.gb2_policy(d, e).hard is True and arms.gb2_policy(d).hard is False
    assert arms.gb2_policy(ref=11.9, hard=True).hard and arms.gb2_policy() is None
