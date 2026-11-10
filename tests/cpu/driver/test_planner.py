"""Planner anchors (design 1.5 / 8.2), fr_v0 determinism and its OOM-escalation state machine, the ED chain."""
import copy

import pytest

from driver_helpers import HF

from infrarsi.contracts import TaskSpec, digest, load_all_tasks
from infrarsi.discover import default_profile, workload_facts
from infrarsi.planner import FLAGSHIP_B, EDChain, FRChain, candidate_space, estimate, expert_default, fr_v0, rank


def facts(task):
    return workload_facts(task, cache_dir=HF)


def with_gpus(tid, gpus, nodes):
    raw = copy.deepcopy(TaskSpec.load(tid).raw)
    raw['hardware'].update(gpus=gpus, nodes=nodes)
    return TaskSpec(raw=raw)


# ---------------------------------------------------------------- the three anchors

def test_anchor_flagship_b_feasible_on_32x80gb():
    t = TaskSpec.load('F1')
    e = estimate(t, FLAGSHIP_B, default_profile(4), facts(t))
    assert e['fits'], e['notes']
    assert e['gpu_gib'] <= e['gpu_budget_gib'] <= 78.0
    plain = dict(FLAGSHIP_B, _expert_exp_avg_sq_cpu=False, _master_param_bytes=4)   # recipe without its state placement
    assert not estimate(t, plain, default_profile(4), facts(t))['gpu_fits']


def test_anchor_80b_full_finetune_infeasible_on_8x80gb():
    t = with_gpus('D6', 8, 1)
    f = facts(t)
    hw = default_profile(1)
    space = candidate_space(t, f, hw)
    assert space and not any(estimate(t, c, hw, f)['fits'] for c in space)
    assert rank(t, f, hw) == [] and expert_default(t, hw=hw, facts=f)['status'] == 'infeasible'


def test_anchor_d6_on_16x80gb_needs_optimizer_offload():
    t = TaskSpec.load('D6')
    f = facts(t)
    hw = default_profile(2)
    fits = [c for c in candidate_space(t, f, hw) if estimate(t, c, hw, f)['fits']]
    assert fits and all(c['optimizer_cpu_offload'] for c in fits)


# ---------------------------------------------------------------- fr_v0

def test_fr_v0_deterministic_and_rule_conformant():
    for tid, t in load_all_tasks().items():
        if tid.startswith(('X', 'F')):
            continue
        a, b = fr_v0(t, facts=facts(t)), fr_v0(t, facts=facts(t))
        assert digest(a['ladder']) == digest(b['ladder'])
        first = a['ladder'][0]
        assert first['pipeline_model_parallel_size'] == 1 and first['context_parallel_size'] == 1
        assert first['micro_batch_size'] == 1 and first['recompute_granularity'] == 'full' and first['recompute_num_layers'] == 1
        assert first['env'] == {'CUDA_DEVICE_MAX_CONNECTIONS': '1'} and first['use_distributed_optimizer']
        assert not any(first[k] for k in ('overlap_grad_reduce', 'overlap_param_gather', 'apply_rope_fusion',
                                          'bias_activation_fusion', 'masked_softmax_fusion',
                                          'gradient_accumulation_fusion', 'cross_entropy_loss_fusion'))
        tps = [c['tensor_model_parallel_size'] for c in a['ladder'][:-1]]
        assert tps == sorted(tps) and all(c['sequence_parallel'] == (c['tensor_model_parallel_size'] > 1) for c in a['ladder'])
        assert a['ladder'][-1]['optimizer_cpu_offload'] and a['ladder'][-1]['optimizer_offload_fraction'] == 1.0
        if facts(t).get('moe'):
            assert first['expert_model_parallel_size'] == min(8, facts(t)['moe']['num_experts'])
            assert first['moe_token_dispatcher_type'] == 'alltoall' and first['moe_grouped_gemm'] and not first['moe_permute_fusion']
            assert a['fd_sources']


def test_fr_v0_skips_structurally_invalid_tp():
    h3 = fr_v0('H3', facts=facts(TaskSpec.load('H3')))           # 4 packs/update: TP1 would need DP8
    assert [s['tp'] for s in h3['skipped_tp']] == [1] and h3['ladder'][0]['tensor_model_parallel_size'] == 2
    d4 = fr_v0('D4', facts=facts(TaskSpec.load('D4')))           # 4 KV groups: TP8 invalid
    assert 8 in [s['tp'] for s in d4['skipped_tp']]
    assert d4['ladder'][-1]['tensor_model_parallel_size'] == 4 and d4['ladder'][-1]['optimizer_cpu_offload']
    assert fr_v0('F1', facts=facts(TaskSpec.load('F1')))['status'] == 'not_applicable'


def test_fr_chain_state_machine():
    t = TaskSpec.load('D2')
    ch = FRChain(t, facts(t))
    seen = []
    for outcome in ('oom', 'oom', 'ok'):
        cfg = ch.current()
        seen.append(cfg['tensor_model_parallel_size'])
        ch.advance(outcome, {'attempt_id': f'a{len(seen)}'})
    assert seen == [1, 2, 4] and ch.status == 'feasible' and ch.config['tensor_model_parallel_size'] == 4
    assert ch.current() is None and [h['outcome'] for h in ch.history] == ['oom', 'oom', 'ok']
    ch = FRChain(t, facts(t))
    for _ in range(len(ch.ladder)):
        ch.advance('oom')
    assert ch.status == 'infeasible' and ch.result()['config'] is None
    ch = FRChain(t, facts(t))
    ch.advance('error')
    assert ch.status == 'error'
    with pytest.raises(RuntimeError):
        ch.advance('ok')


def test_fr_v0_with_probe_callable_reaches_offload():
    t = TaskSpec.load('H6')
    res = fr_v0(t, facts=facts(t), probe=lambda c: ('ok' if c.get('optimizer_cpu_offload') else 'oom', None))
    assert res['status'] == 'feasible' and res['config']['optimizer_cpu_offload']


# ---------------------------------------------------------------- ED

def test_ed_ranked_candidates_fit_and_explained():
    t = TaskSpec.load('D4')
    plan = expert_default(t, facts=facts(t))
    assert plan['status'] == 'probing' and len(plan['top']) == 3
    keys = {(c['config']['tensor_model_parallel_size'], c['config'].get('expert_model_parallel_size'),
             c['config']['context_parallel_size'], c['config']['optimizer_cpu_offload']) for c in plan['top']}
    assert len(keys) == 3                                           # diverse layouts
    assert all('GiB' in c['explanation'] and 'est' in c['explanation'] for c in plan['top'])


def test_ed_chain_picks_fastest_valid_probe():
    t = TaskSpec.load('D2')
    ch = EDChain(t, facts(t), default_profile(1))
    times = iter([2.0, 1.5, None])
    while (cfg := ch.current()) is not None:
        tm = next(times, None)
        rec = {'attempt_id': 'x', 'per_step': [{'update': u, 'max_rank_whole_s': tm} for u in range(1, 6)]} if tm else None
        ch.advance('ok' if tm else 'oom', rec)
    r = ch.result()
    assert r['status'] == 'feasible' and r['history'][1]['probe_step_s'] == 1.5
    assert r['config_digest'] == r['history'][1]['config_digest'] and f"rank {r['history'][1]['rank']} " in r['explanation']


def test_ed_chain_continues_past_top3_when_all_fail():
    t = TaskSpec.load('D2')
    ch = EDChain(t, facts(t), default_profile(1), max_probes=5)
    n = 0
    while ch.current() is not None:
        n += 1
        ch.advance('oom' if n < 5 else 'ok', {'attempt_id': 'x', 'per_step': [{'update': 2, 'max_rank_whole_s': 3.0}]})
    assert n == 5 and ch.status == 'feasible'


def test_cp8_never_for_sft_data():
    from infrarsi.planner import layout
    h3 = TaskSpec.load('H3')                                         # D-SFT-64K
    base = {'tensor_model_parallel_size': 1, 'micro_batch_size': 1}
    assert any('invalid for SFT' in p for p in layout(h3, dict(base, context_parallel_size=8), facts(h3)).problems)
    assert not any('invalid for SFT' in p for p in layout(h3, dict(base, context_parallel_size=4), facts(h3)).problems)
    d2 = TaskSpec.load('D2')                                         # D-PT-8K: CP 8 is a layout question only
    assert not layout(d2, dict(base, context_parallel_size=8), facts(d2)).problems
    for tid, t in load_all_tasks().items():
        if t.raw['data'].get('key', '').startswith('D-SFT') and not tid.startswith(('X', 'F')):
            hw = default_profile(t.nodes)
            assert all(c['context_parallel_size'] in (1, 2, 4) for c in candidate_space(t, facts(t), hw))
            assert all(c['context_parallel_size'] == 1 for c in fr_v0(t, facts=facts(t))['ladder'])


def test_no_nvlink_rule_limits_tp():
    t = TaskSpec.load('D2')
    hw = default_profile(1)
    hw['summary']['nvlink_intra'] = False
    assert max(c['tensor_model_parallel_size'] for c in candidate_space(t, facts(t), hw)) <= 2
