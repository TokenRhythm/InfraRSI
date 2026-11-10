import re
from pathlib import Path

import pytest

from infrarsi.actions import config_diff, defaults, l0_space, load_schema, model_facts_from_hf_config, validate_l0
from agent_helpers import DENSE_4B, DENSE_8B, FR_DENSE, FR_MOE, MOE_30B

DESIGN_KEYS = {  # design 1.8, verbatim
    'tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size', 'expert_model_parallel_size',
    'expert_tensor_parallel_size', 'sequence_parallel', 'micro_batch_size', 'recompute_granularity', 'recompute_method',
    'recompute_num_layers', 'recompute_modules', 'moe_layer_recompute', 'cpu_offloading', 'cpu_offloading_num_layers',
    'optimizer_cpu_offload', 'optimizer_offload_fraction', 'overlap_cpu_optimizer_d2h_h2d', 'pin_cpu_grads', 'pin_cpu_params',
    'overlap_grad_reduce', 'overlap_param_gather', 'align_param_gather', 'bucket_size', 'average_in_collective',
    'delay_wgrad_compute', 'tp_comm_overlap', 'cp_comm_type', 'moe_token_dispatcher_type', 'moe_enable_deepep',
    'moe_deepep_num_sms', 'moe_grouped_gemm', 'moe_permute_fusion', 'moe_shared_expert_overlap',
    'overlap_moe_expert_parallel_comm', 'apply_rope_fusion', 'bias_activation_fusion', 'masked_softmax_fusion',
    'gradient_accumulation_fusion', 'cross_entropy_loss_fusion', 'cross_entropy_fusion_impl', 'attention_backend'}
MCORE = Path(__file__).resolve().parents[4] / 'deliverables/WINH_ACCEL_CODEBASE_20261016/baseline/source_tree/qwen-sft/third_party/Megatron-LM-core_v0.18.2/megatron'


ACCEPTED_CONSTANTS = {'use_distributed_optimizer'}   # stated by FR/ED planner configs; only `true` is legal


def test_schema_has_exactly_the_design_keys_with_mappings():
    s = load_schema()
    assert set(s['keys']) == DESIGN_KEYS | ACCEPTED_CONSTANTS
    assert not ACCEPTED_CONSTANTS & set(s['fixed_by_task']['keys'])
    for k, spec in s['keys'].items():
        assert spec['type'] in ('int', 'bool', 'enum', 'float', 'list_enum'), k
        m = spec['maps_to']
        assert m['object'] and m['field'] and re.match(r'(core|training)/', m['source']), k
    assert {'PYTORCH_CUDA_ALLOC_CONF', 'CUDA_DEVICE_MAX_CONNECTIONS', 'OMP_NUM_THREADS'} <= set(s['env'])
    assert all(k.startswith('NCCL_') for k in s['env'] if k not in ('PYTORCH_CUDA_ALLOC_CONF', 'CUDA_DEVICE_MAX_CONNECTIONS', 'OMP_NUM_THREADS'))
    for fixed in ('global_batch_size', 'moe_router_topk', 'moe_expert_capacity_factor', 'fp8', 'use_precision_aware_optimizer'):
        assert fixed in s['fixed_by_task']['keys'] and fixed not in s['keys']


@pytest.mark.skipif(not MCORE.exists(), reason='vendored Megatron-Core source not present')
def test_maps_to_lines_match_vendored_megatron_core():
    for k, spec in load_schema()['keys'].items():
        src = spec['maps_to']['source']
        path, _, line = src.partition(':')
        if '-' in line or ',' in line or k == 'micro_batch_size':
            continue
        text = (MCORE / path).read_text(encoding='utf-8').splitlines()
        assert re.match(rf'\s+{k}\s*:', text[int(line) - 1]), (k, src, text[int(line) - 1])


def test_fr_configs_validate_for_dev_tasks(tasks):
    assert validate_l0(FR_DENSE, tasks['S1'], DENSE_4B).ok
    r = validate_l0(dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True), tasks['D1'], DENSE_8B)
    assert r.ok and r.derived['dp'] == 4 and r.derived['ga'] == 2
    r = validate_l0(FR_MOE, tasks['D4'], MOE_30B)
    assert r.ok, r.errors
    assert r.derived == {'world': 8, 'gpus_per_node': 8, 'etp': 1, 'dp': 8, 'ga': 4, 'edp': 1}
    assert validate_l0(dict(FR_MOE, expert_model_parallel_size=16), tasks['D5'], MOE_30B).ok


@pytest.mark.parametrize('change, code', [
    ({'global_batch_size': 64}, 'fixed by the task'),
    ({'not_a_key': 1}, 'not an L0 key'),
    ({'sequence_parallel': True}, 'C01'),
    ({'tensor_model_parallel_size': 2, 'tp_comm_overlap': True}, 'C02'),
    ({'recompute_method': None}, 'C03'),
    ({'recompute_granularity': 'selective'}, 'C03'),
    ({'recompute_granularity': None}, 'C03'),
    ({'overlap_param_gather': True}, 'C07'),
    ({'moe_enable_deepep': True}, 'C08'),
    ({'moe_deepep_num_sms': 21}, 'not in'),
    ({'cross_entropy_loss_fusion': True, 'cross_entropy_fusion_impl': 'te'}, 'C12'),
    ({'cpu_offloading': True}, 'C06'),
    ({'tensor_model_parallel_size': 3}, 'not in'),
    ({'attention_backend': 'local'}, 'not in'),
    ({'optimizer_offload_fraction': 1.5}, 'outside'),
    ({'sequence_parallel': 1}, 'expected bool'),
    ({'env': {'NCCL_DEBUG': 'INFO'}}, 'whitelist'),
    ({'env': {'PYTORCH_CUDA_ALLOC_CONF': 'backend:cudaMallocAsync'}}, 'not whitelisted'),
    ({'env': {'CUDA_DEVICE_MAX_CONNECTIONS': 64}}, 'outside'),
    ({'env': {'NCCL_ALGO': 'Ring,Magic'}}, 'unknown entries'),
])
def test_rejections_without_task(change, code):
    r = validate_l0({**FR_DENSE, **change})
    assert not r.ok and any(code in e for e in r.errors), r.errors


def test_moe_overlap_rules():
    base = dict(FR_MOE, recompute_granularity=None, recompute_method=None, recompute_num_layers=None)
    r = validate_l0(dict(base, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True))
    assert not r.ok and any('not supported by the v0.1 generic trainer' in e for e in r.errors)   # trainer preflight rejects it
    for bad in ({'recompute_granularity': 'full', 'recompute_method': 'uniform', 'recompute_num_layers': 1},
                {'moe_token_dispatcher_type': 'allgather'}, {'expert_model_parallel_size': 1},
                {'pipeline_model_parallel_size': 2}, {'recompute_granularity': 'selective', 'recompute_modules': ['moe']}):
        r = validate_l0(dict(base, overlap_moe_expert_parallel_comm=True, **bad))
        assert not r.ok and any('C10' in e for e in r.errors), (bad, r.errors)
    assert any('C11' in e for e in validate_l0(dict(base, delay_wgrad_compute=True)).errors)
    assert any('C04' in e for e in validate_l0(dict(base, recompute_granularity='selective', recompute_modules=['moe_act'], moe_grouped_gemm=False)).errors)


def test_task_and_model_dependent_rules(tasks):
    # GA must be an integer: D1 has 8 packs per update on 8 GPUs, so MBS 2 with DP 8 fails.
    assert any('C14' in e for e in validate_l0(dict(FR_DENSE, micro_batch_size=2), tasks['D1'], DENSE_8B).errors)
    # TP inside a node; world divisibility.
    r = validate_l0(dict(FR_DENSE, tensor_model_parallel_size=8, pipeline_model_parallel_size=4), tasks['S2'], DENSE_4B)
    assert any('C13' in e for e in r.errors) and any('C15' in e for e in r.errors)
    # sft_packed segments padded to 8 -> CP <= 4; and KV groups.
    assert any('C19' in e for e in validate_l0(dict(FR_DENSE, context_parallel_size=8), tasks['D1'], DENSE_8B).errors)
    assert any('C16' in e for e in validate_l0(dict(FR_DENSE, tensor_model_parallel_size=8, sequence_parallel=True), tasks['D1'],
                                                 dict(DENSE_8B, num_query_groups=12)).errors)
    # experts % EP, and dense models cannot use EP.
    assert any('C17' in e for e in validate_l0(dict(FR_MOE, expert_model_parallel_size=8), tasks['D4'], dict(MOE_30B, num_experts=60)).errors)
    assert any('C17' in e for e in validate_l0(dict(FR_DENSE, expert_model_parallel_size=2), tasks['D1'], DENSE_8B).errors)
    # MoE-only keys on a dense model are warnings, not errors.
    r = validate_l0(dict(FR_DENSE, moe_grouped_gemm=True), tasks['D1'], DENSE_8B)
    assert r.ok and any('C21' in w for w in r.warnings)


def test_context_parallel_on_sft_vs_pt_tasks():
    """Coordinator data contract: SFT (D-SFT-*) segments are padded to multiples of 8 -> CP in {1, 2, 4}; PT any CP."""
    from infrarsi.contracts import load_all_tasks
    all_tasks = load_all_tasks()
    sft = [t for t in all_tasks.values() if t.raw['data']['key'].startswith('D-SFT') and t.gpus >= 8]
    pt = [t for t in all_tasks.values() if t.raw['data']['key'].startswith('D-PT') and t.gpus >= 8]
    assert sft and pt
    for t in sft:
        mf = DENSE_8B if t.raw['model']['bridge_type'] == 'qwen3' else None
        r = validate_l0(dict(FR_DENSE, context_parallel_size=8), t, mf)
        assert not r.ok and any(e.startswith('C19 context_parallel_size=8 not allowed on SFT task') for e in r.errors), (t.id, r.errors)
        for cp in (2, 4):
            assert not any('C19' in e for e in validate_l0(dict(FR_DENSE, context_parallel_size=cp), t).errors), (t.id, cp)
        assert 8 not in l0_space(t)['context_parallel_size']
    for t in pt:
        r = validate_l0(dict(FR_DENSE, context_parallel_size=8), t)
        assert not any('C19' in e for e in r.errors), (t.id, r.errors)
        assert 8 in l0_space(t)['context_parallel_size']


def test_resolved_config_is_explicit_and_diffable():
    r = validate_l0({'tensor_model_parallel_size': 2, 'sequence_parallel': True, 'env': {'NCCL_ALGO': 'Ring'}})
    assert set(r.resolved) == set(defaults()) | {'env'} and r.resolved['env'] == {'NCCL_ALGO': 'Ring'}
    d = config_diff({}, r.resolved)
    assert d['tensor_model_parallel_size'] == [1, 2] and d['env:NCCL_ALGO'] == [None, 'Ring']


def test_space_and_model_facts(tasks):
    dense = l0_space(tasks['D1'], DENSE_8B)
    assert 'moe_token_dispatcher_type' not in dense and 'expert_model_parallel_size' not in dense
    assert set(dense['tensor_model_parallel_size']) == {1, 2, 4, 8}
    moe = l0_space(tasks['D4'], MOE_30B)
    assert set(moe['expert_model_parallel_size']) == {1, 2, 4, 8}
    s1 = l0_space(tasks['S1'], DENSE_4B)
    assert s1['tensor_model_parallel_size'] == [1]
    mf = model_facts_from_hf_config({'model_type': 'qwen3_moe', 'num_hidden_layers': 48, 'num_attention_heads': 32,
                                     'num_key_value_heads': 4, 'num_experts': 128, 'num_experts_per_tok': 8, 'hidden_act': 'silu'})
    assert mf['num_experts'] == 128 and mf['num_query_groups'] == 4 and mf['moe_router_topk'] == 8 and not mf['multi_latent_attention']


def test_planner_configs_with_constants_and_planner_only_keys(tasks):
    """D's FR/ED planners state use_distributed_optimizer, overlap_cpu_optimizer_d2h_h2d and cross_entropy_fusion_impl;
    keys starting with '_' are planner-only bookkeeping and are dropped from the resolved config."""
    cfg = dict(FR_DENSE, use_distributed_optimizer=True, overlap_cpu_optimizer_d2h_h2d=False, cross_entropy_fusion_impl='native',
               _ple_cpu_offload=True, _expert_exp_avg_sq_cpu=True)
    r = validate_l0(cfg, tasks['S1'], DENSE_4B)
    assert r.ok, r.errors
    assert not any(k.startswith('_') for k in r.resolved) and r.resolved['use_distributed_optimizer'] is True
    assert any('planner-only' in w for w in r.warnings)
    bad = validate_l0(dict(FR_DENSE, use_distributed_optimizer=False), tasks['S1'], DENSE_4B)
    assert not bad.ok and 'task constant' in bad.errors[0]
    assert 'use_distributed_optimizer' not in l0_space(tasks['S1'], DENSE_4B)


@pytest.mark.parametrize('tid', ['S1', 'S2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'])
def test_planner_fr_and_ed_configs_validate(tid):
    """Every FR-ladder and ED-candidate config D's planners produce passes validate_l0 (searches start from them)."""
    import json
    from infrarsi.contracts import TaskSpec
    from infrarsi.discover import default_profile
    from infrarsi.discover.workload import cache_name, workload_facts
    from infrarsi.planner import EDChain, FRChain
    hf_dir = Path(__file__).resolve().parents[1] / 'driver' / 'fixtures' / 'hf_configs'
    task = TaskSpec.load(tid)
    if not (hf_dir / cache_name(task)).exists():
        pytest.skip(f'no HF config fixture for {tid}')
    facts = workload_facts(task, cache_dir=hf_dir)
    mf = model_facts_from_hf_config(json.loads((hf_dir / cache_name(task)).read_text(encoding='utf-8'))['config'])
    configs = [('FR', c) for c in FRChain(task, facts).ladder]
    configs += [('ED', r['config']) for r in EDChain(task, facts, default_profile(task.nodes)).queue]
    assert configs
    bad = [(kind, r.errors) for kind, c in configs if not (r := validate_l0(c, task.raw, mf)).ok]
    assert not bad, bad[:3]
