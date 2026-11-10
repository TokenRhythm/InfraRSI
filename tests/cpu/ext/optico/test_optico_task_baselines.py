import json
import math

import pytest

from infrarsi.agent.budget import VirtualClock
from infrarsi.agent.search import SearchSession
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.env_task import check_spec
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.optico import b_ed, b_rnd
from infrarsi.ext.optico import params as pm
from infrarsi.ext.optico import task as taskmod
from infrarsi.ext.optico import workloads as wl
from infrarsi.ext.optico.runner import OptiCoRunner
from test_optico_runner import FakeTrial


@pytest.mark.parametrize('key', sorted(wl.WORKLOADS))
def test_b_ed_projection_follows_the_planner_rules(key):
    r = b_ed.pick(key)
    p = r['params']
    assert p is not None and pm.validate_optico_params(p, key)[1] == []
    dp = wl.GPUS // (p['TP'] * p['PP'])
    assert p['PP'] == 1 and p['MICRO_BATCH_SIZE'] in (1, 2) and p['FUSED_CROSS_ENTROPY'] is True
    assert p['OVERLAP_GRAD_REDUCE'] == p['OVERLAP_PARAM_GATHER'] == (dp > 1) and p['AC'] in ('none', 'sel', 'full')
    assert p['FIRST_PP_NUM_LAYERS'] is None and p['LAST_PP_NUM_LAYERS'] is None
    assert r == b_ed.pick(key)                                               # deterministic
    ests = [x['est_step_s'] for x in r['ranked']]
    assert ests == sorted(ests) and r['ranked'][0]['params'] == p


def test_b_ed_w3_pick_and_no_nvlink_rule():
    assert pm.format_params(b_ed.pick('W3')['params']) == 'MBS2 TP1 PP1 AC none ogr opg fce'
    from infrarsi.discover.hardware import default_profile
    hw = default_profile(1)
    hw['summary'] = {**hw['summary'], 'nvlink_intra': False}
    assert all(x['params']['TP'] <= 2 for x in b_ed.rank('W2', hw))


@pytest.mark.parametrize('key', ['W1', 'W2', 'W3'])
def test_b_rnd_seeded_valid_unique(key):
    seq = b_rnd.OptiCoRandomSampler(key, seed=42).sequence(20)
    again = b_rnd.OptiCoRandomSampler(key, seed=42).sequence(20)
    assert seq == again and len(seq) == 20
    assert len({pm.params_key(p) for p in seq}) == 20
    assert all(pm.validate_optico_params(p, key)[1] == [] for p in seq)
    assert all(not (p['AC'] == 'offload' and p['PP'] > 1) for p in seq)
    assert all(p['OVERLAP_GRAD_REDUCE'] or not p['OVERLAP_PARAM_GATHER'] for p in seq)
    other = b_rnd.OptiCoRandomSampler(key, seed=7).sequence(20)
    assert other != seq
    s = b_rnd.OptiCoRandomSampler(key)
    seen = {pm.params_key(p) for p in seq}
    nxt = s.next(seen)
    assert nxt is not None and pm.params_key(nxt) not in seen


@pytest.mark.parametrize('arm', sorted(taskmod.ARMS))
@pytest.mark.parametrize('key', sorted(wl.WORKLOADS))
def test_tasks_are_valid_contracts(key, arm):
    t = taskmod.make_task(key, arm)
    assert check_spec(t.fields()) == [] and t.env == 'optico' and t.metric_name == 'iter_time_s'
    assert t.raw['hardware']['gpus'] == 8 and t.world_size == 8 and t.action_space == 'knobs'
    assert t.tau == pytest.approx(math.log(1.05)) and t.margin == {'kind': 'rsd', 'floor': 0.03, 'k': 3}
    w = wl.get(key)
    assert t.budget['wall_seconds'] == w.wall_budget_s * (2.0 if arm == 'B-ISW' else 1.0)
    assert t.budget['llm_tokens'] == (1_500_000 if taskmod.ARMS[arm]['llm'] else 0)
    if arm == 'B-NAT':
        assert t.budget['max_trials_incl_retries'] == 26 and 'D-OC-14' in t.raw['deviations']
    assert t.search_budget().screens == t.budget['screens']


def test_hooks_validate_and_task_view():
    t = taskmod.make_task('W1', 'B-IS')
    h = taskmod.OptiCoHooks(t)
    norm, errs, warns = h.validate_config({'TP': 2, 'AC': 'sel', 'AC_LAYERS': 4})
    assert errs == [] and norm['AC_LAYERS'] == 1 and warns
    assert h.validate_config({'TP': 8})[1] and h.validate_config('x')[1]
    assert h.validate_patch({'a.py': 'x'})[0] and h.layers == ('L0',) and not h.tools_enabled
    v = h.task_view()
    assert v['fixed_parameters'].startswith('Model Type: qwen2.5\nModel Size: 7B\nNUM_LAYERS: 28\nPrecison: fp16')
    assert str(v['performance_table']).startswith('|MICRO_BATCH_SIZE|TP|PP|') and str(v['trials_so_far']) == '0'
    assert 'run_circle.sh qwen2.5 7B' in v['contract'] and v['start_parameters'] == wl.get('W1').defaults
    json.dumps(v, default=str)


def test_performance_table_matches_optico_compaction():
    from optico_helpers import report_row
    rows = [{'judge': {'row': report_row(wl.get('W3').defaults, log_dir='1')}}, {'judge': None}]
    text = taskmod.performance_table(rows)
    lines = text.splitlines()
    assert len(lines) == 3 and lines[1].count('|') == 20 and set(lines[1]) == {'|', '-'}
    assert lines[2].startswith('|1|1|1|false|false|none|1|false|None|None|success|228.833|')


def test_search_session_runs_optico_through_the_hooks(tmp_path):
    """X1's seam end to end on CPU: SearchSession + OptiCoHooks + OptiCoRunner (fake node) + EnvJudge."""
    task = taskmod.make_task('W3', 'B-RND')
    clock = VirtualClock()
    fake = FakeTrial(clock=clock)
    runner = OptiCoRunner(LocalNodeOps(tmp_path / 'node', node='WINH4', launcher=fake), 'W3', out_dir=tmp_path / 'pc',
                          clock=clock, sleep=clock.advance)
    hooks = taskmod.OptiCoHooks(task, runner=runner)
    s = SearchSession(task=task, budget=task.search_budget(screens=4), runner=runner, judge=EnvJudge(env='optico'),
                      run_id='brnd-cpu', arm='B-RND', out_dir=tmp_path / 'trace', start_config=task.raw['start_config'],
                      clock=clock, layers=('L0',), hooks=hooks)
    s.screen('c0')
    sampler = b_rnd.OptiCoRandomSampler('W3')
    seen = {pm.params_key(s.candidates['c0'].config)}
    for _ in range(3):
        p = sampler.next(seen)
        seen.add(pm.params_key(p))
        fb = s.propose_config(config=p, base='c0', origin='random')
        assert 'candidate_id' in fb, fb
        s.screen(fb['candidate_id'])
    bad = s.propose_config(config={**task.raw['start_config'], 'TP': 4}, base='c0', origin='random')
    assert bad['error'] == 'config validation failed' and any('TP 4' in e for e in bad['errors'])
    dup = s.propose_config(changes={'AC_LAYERS': 3}, base='c0', origin='agent')     # normalized to the start: duplicate
    assert dup.get('duplicate_of') == 'c0'
    trace = s.finalize(stop_reason='done').to_dict()
    assert len(trace['screens']) == 4 and all(x['valid'] for x in trace['screens'])
    assert trace['used']['gpu_hours'] == pytest.approx(4 * 8 * 144.0 / 3600.0)
    assert len(fake.launched) == 4 and str(hooks.task_view()['trials_so_far']) == '4'
    assert str(hooks.task_view()['performance_table']).count('\n') == 5
    # agent.optimize builds the view once and serializes it every turn: the live entries follow the runner's history
    view = hooks.task_view()
    before = json.loads(json.dumps(view, default=str))['performance_table'].count('\n')
    runner.run(task, {**task.raw['start_config'], 'MICRO_BATCH_SIZE': 8}, None, 'screen', 'brnd-cpu', candidate_id='cx')
    assert json.loads(json.dumps(view, default=str))['performance_table'].count('\n') == before + 1
