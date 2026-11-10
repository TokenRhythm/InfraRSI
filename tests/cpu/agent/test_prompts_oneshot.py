"""F-PROMPTS (external bench W6, builder X4): one recommendation call, exactly three L0 configs (schema checked),
refusals by the L0 validator / duplicate check / feasibility model before any screen, one screen per accepted config,
the locked pair against the start configuration, no lifecycle, LLM and GPU cost on the search ledger.

End to end through the registered dispatcher handler on D's LocalRemote simulation (f1_world: S1 stands in for F1's 32
GPUs, the start slot holds a 'U+C' configuration recorded as the task's numerics reference, FR not applicable, as on
F1); the F1-specific views (HardwareProfile, workload facts, flagship placements) are unit-tested on the F1 spec itself.
"""
import json
import math
from pathlib import Path

import pytest

import f1_world as F1W
from infrarsi.agent.backends import ScriptedBackend
from infrarsi.baselines import prompts_oneshot as P
from infrarsi.contracts import TaskSpec
from infrarsi.discover import default_profile, workload
from infrarsi.dispatch.handlers import handlers, is_deferred
from infrarsi.planner import FLAGSHIP_B

DH = F1W.DH
RUN = {'run_id': 'XB-F1-F-PROMPTS-sim', 'task': 'S1', 'arm': 'F-PROMPTS', 'screens': 12, 'wall_hours_cap': 12,
       'llm_tokens_cap': 1500000, 'seed': 42, 'priority': 'P1'}
NOREC = {'recompute_granularity': None, 'recompute_method': None, 'recompute_num_layers': None}
GOOD = {'config': {'micro_batch_size': 2}, 'rationale': 'fewer micro-batches per update'}
INVALID = {'config': {'tensor_model_parallel_size': 3}, 'rationale': 'not a legal TP'}
INFEASIBLE = {'config': dict(NOREC, micro_batch_size=8), 'rationale': 'no recompute, MBS 8: 136 GiB estimated'}


@pytest.fixture
def world(tmp_path, monkeypatch):
    yield from F1W.world_setup(tmp_path, monkeypatch)


def _dispatch(world, backend, monkeypatch, run=RUN):
    return F1W.dispatch(world, backend, monkeypatch, run)


def _requests(world, run_id=RUN['run_id']):
    return F1W.requests(world, run_id)


_events = F1W.events


def test_registered_on_the_dispatcher():
    h = handlers()
    assert P.ARM == 'F-PROMPTS' and P.ARM in h and not is_deferred(h[P.ARM])


def test_one_call_three_configs_refusals_cost_no_screen(world, monkeypatch):
    backend = ScriptedBackend([{'configs': [GOOD, INVALID, INFEASIBLE]}])
    out = _dispatch(world, backend, monkeypatch)
    res, sdir = out['result'], Path(out['search_dir'])
    assert len(backend.calls) == 1 and backend.calls[0]['role'] == 'prompts_recommend'          # exactly one call
    recs = res['prompts']['recommendations']
    assert [r['outcome'] for r in recs] == ['screened', 'refused_l0', 'refused_infeasible']
    assert recs[2]['feasibility']['fits'] is False and recs[2]['feasibility']['gpu_gib'] > recs[2]['feasibility']['gpu_budget_gib']
    trace = json.loads((sdir / 'trace.json').read_text(encoding='utf-8'))
    assert trace['arm'] == 'F-PROMPTS' and trace['stop_reason'] == 'done' and len(trace['screens']) == 1
    assert trace['invalid_proposals'] == {'l0_invalid': 1, 'infeasible': 1}
    assert {c['id']: c['status'] for c in trace['candidates']} == {'c0': 'screened', 'c1': 'screened', 'c2': 'vetoed'}
    # one plan-A screen, then the locked pair against the start configuration; no promotion lifecycle
    assert _requests(world) == sorted([('screen', 'A', 'candidate'), ('score', 'B', 'reference'), ('score', 'B', 'candidate')], key=str)
    pair, summ = res['locked_pair'], res['locked_summary']
    assert pair['reference_kind'] == 'ED' and pair['promoted'] and pair['invalid_accept'] == 0
    assert summ['t_cand_locked'] < summ['t_ref_locked'] and summ['log_speedup'] == pytest.approx(
        math.log(summ['t_ref_locked'] / summ['t_cand_locked'])) and summ['beyond_margin']
    assert res['promotion_lifecycle']['ran'] is False and res['scoring_inputs']['lifecycle_pass'] is None
    # cost on the ledger: the call (scripted 1000 tokens / $0.001) and the GPU time of the screen and the pair
    assert res['cost']['llm_tokens'] == 1000 and res['cost']['llm_usd'] == pytest.approx(0.001)
    assert res['cost']['search_gpu_hours'] > 0 and res['cost']['scoring_gpu_hours'] > 0 and res['cost']['accounting_complete']
    assert [e['kind'] for e in _events(sdir)].count('llm_call') == 1
    assert trace['budget']['screens'] == 3 and trace['budget']['llm_tokens'] == 200000 and trace['budget']['llm_usd'] == 10.0
    prompt = (sdir / 'prompts' / 'prompt.txt').read_text(encoding='utf-8')
    for part in ('# HARDWARE', '"gpu_type": "H100-80GB"', '# WORKLOAD', '# CURRENT CONFIGURATION: MEASURED SCREEN',
                 '"mean_window_max_rank_s"', '"tensor_model_parallel_size"', 'exactly 3 entries'):
        assert part in prompt, part
    assert backend.calls[0]['prompt'] == prompt and json.loads((sdir / 'result.json').read_text())['locked_summary'] == summ


@pytest.mark.parametrize('reply,stop', [({'configs': [GOOD, GOOD]}, 'llm_reply_invalid'),           # not exactly 3
                                        ({'configs': [GOOD, INVALID, INFEASIBLE], 'extra': 1}, 'llm_reply_invalid'),
                                        ('not json', 'llm_error')])
def test_an_unusable_reply_screens_nothing(world, monkeypatch, reply, stop):
    backend = ScriptedBackend([reply], validate_schema=False)
    res = _dispatch(world, backend, monkeypatch)['result']
    assert len(backend.calls) == 1 and res['trace']['stop_reason'] == stop and res['trace']['screens'] == 0
    assert res['prompts']['recommendations'] == [] and res['prompts']['call']['problems']
    assert _requests(world) == [('score', 'B', 'reference')]          # the fallback is the start itself, measured locked
    assert not res['locked_pair']['promoted'] and res['locked_pair']['invalid_accept'] == 0


def test_slower_recommendations_are_reported_and_the_start_is_kept(world, monkeypatch):
    """Every arm's rule: a valid candidate is promoted only when it beats the start; the best valid recommendation is
    still reported (budget-matched comparison with F-A0's first three screens)."""
    slow = [{'config': dict(c, attention_backend='unfused'), 'rationale': 'r'} for c in ({}, {'micro_batch_size': 2},
                                                                                       {'micro_batch_size': 4})]
    res = _dispatch(world, ScriptedBackend([{'configs': slow}]), monkeypatch)['result']
    recs = res['prompts']['recommendations']
    assert [r['outcome'] for r in recs] == ['screened'] * 3 and all(r['valid'] for r in recs)
    best = res['prompts']['best_recommendation']
    assert best['index'] == 3 and best['step_time'] == min(r['step_time'] for r in recs) and best['vs_start'] < 0
    assert res['trace']['selected'] == 'c0' and not res['locked_pair']['promoted']
    assert _requests(world).count(('score', 'B', 'reference')) == 1 and ('score', 'B', 'candidate') not in _requests(world)


def test_duplicates_and_the_start_itself_are_refused(world, monkeypatch):
    same = {'config': {}, 'rationale': 'the start configuration'}
    backend = ScriptedBackend([{'configs': [GOOD, GOOD, same]}])
    res = _dispatch(world, backend, monkeypatch)['result']
    assert [r['outcome'] for r in res['prompts']['recommendations']] == ['screened', 'refused_duplicate', 'refused_duplicate']
    assert res['trace']['screens'] == 1


def test_refuses_to_start_on_an_unsigned_start_receipt(world, monkeypatch):
    path = world['uc_a']['harness']['stored']
    forged = dict(world['uc_a'], step_time=dict(world['uc_a']['step_time'], mean_window_max_rank_s=99.0))
    world['remote'].write_json('WINH0', path, forged)                 # altered on /data after signing
    backend = ScriptedBackend([{'configs': [GOOD, INVALID, INFEASIBLE]}])
    with pytest.raises(RuntimeError, match='start receipt'):
        _dispatch(world, backend, monkeypatch)
    assert backend.calls == []                                        # nothing paid before the inputs are verified


def test_budget_is_capped_by_the_arm():
    task = TaskSpec.load('F1')
    b = P.prompts_budget(task, {'screens': 12, 'llm_tokens_cap': 1500000, 'wall_hours_cap': 12})
    assert (b.screens, b.llm_tokens, b.llm_usd) == (3, 200000, 10.0) and b.gpu_hours == 170.0
    assert P.prompts_budget(task, {'llm_usd_cap': 4}).llm_usd == 4.0 and P.prompts_budget(task, {'screens': 2}).screens == 2


def test_reply_schema_is_exactly_three_configs():
    ok = type('R', (), {'error': None, 'action': {'configs': [GOOD, INVALID, INFEASIBLE]}})()
    assert P.parse_reply(ok) == ([GOOD, INVALID, INFEASIBLE], [])
    for action in ({'configs': [GOOD] * 4}, {'configs': [dict(GOOD, rationale=None)] * 3}, {'configs': [{'config': 1, 'rationale': ''}] * 3},
                   {'cfgs': []}, None):
        items, problems = P.parse_reply(type('R', (), {'error': None, 'action': action})())
        assert items is None and problems
    assert P.parse_reply(type('R', (), {'error': 'CLI failed', 'action': None})()) == (None, ['CLI failed'])


# ---------------------------------------------------------------- F1 itself

def _f1():
    task = TaskSpec.load('F1')
    hf = json.loads((DH.HF / 'Qwen3.8-Flash-Next.json').read_text(encoding='utf-8'))['config']
    from infrarsi.actions import model_facts_from_hf_config, validate_l0
    mf = model_facts_from_hf_config(hf)
    return task, workload.workload_facts(task, cache_dir=DH.HF), mf, validate_l0(FLAGSHIP_B, task.raw, mf)


def test_flagship_model_facts_come_from_its_text_config():
    """The flagship HF config nests the language model under text_config: read flat, F1 looked dense and every MoE
    layout (the U+C start included) failed the L0 validator, so no F1 arm could start."""
    task, facts, mf, rep = _f1()
    assert mf['num_experts'] == 512 and mf['moe_router_topk'] == 10 and mf['num_layers'] == 48 and mf['model_type'] == 'qwen4_exp'
    assert rep.ok, rep.errors


def test_f1_views_and_feasibility_carry_the_start_placements():
    task, facts, mf, rep = _f1()
    placements = P.placements_of(FLAGSHIP_B)
    hw = default_profile(4)
    start = P.estimate(task, P._for_estimate(rep.resolved, placements), hw, facts)
    assert start['fits']
    ok, d = P.feasibility(task, dict(rep.resolved, overlap_grad_reduce=True), placements, hw, facts, start)
    assert ok and d['basis'] == 'absolute'
    assert not P.feasibility(task, rep.resolved, {}, hw, facts)[0]                           # without them it would not fit
    bad, d = P.feasibility(task, dict(rep.resolved, expert_model_parallel_size=8), placements, hw, facts, start)
    assert not bad and d['gpu_gib'] > d['gpu_budget_gib']
    tight = dict(start, fits=False, gpu_gib=d['gpu_gib'] + 1.0)                             # the model cannot fit the start
    assert P.feasibility(task, dict(rep.resolved, expert_model_parallel_size=8), placements, hw, facts, tight)[1]['basis'] == 'relative_to_start'
    view = P.workload_view(task.raw, facts)
    assert view['model_facts']['moe']['num_experts'] == 512 and view['objective'] == task.raw['objective']   # free text kept
    space = P.l0_view(mf)
    assert 'expert_model_parallel_size' in space['l0_keys'] and space['rules'] and 'NCCL_ALGO' in space['env_whitelist']
    prompt = P.build_prompt(task.raw, P.hardware_view(hw, 4), view, rep.resolved, placements, {'step_time': {'mean_window_max_rank_s': 12.49}},
                            space, tolerance={'grad_norm_rel_max_first_update': 0.03}, peak_cap_gib=78.0)
    assert '"nodes": 4' in prompt and '"_ple_cpu_offload": true' in prompt and '12.49' in prompt and 'updates 17..30' in prompt
    assert P.estimate_tokens(prompt) < 20000
