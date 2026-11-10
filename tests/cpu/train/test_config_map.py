import json

import pytest

from infrarsi.adapters.megatron_core import config_map as cm
from infrarsi.contracts import RunRequest

from train_helpers import QWEN3_1_7B, QWEN3_8B, QWEN3_30B_A3B, fr_config


def facts(cfg):
    return cm.model_facts(cfg)


def test_model_facts_from_config_dict():
    f = facts(QWEN3_30B_A3B)
    assert f['num_experts'] == 128 and f['num_query_groups'] == 4 and f['num_layers'] == 48
    assert facts(QWEN3_1_7B)['num_experts'] is None and facts(QWEN3_1_7B)['tie_word_embeddings']


def test_fr_plan_s1_single_gpu(make_request, tasks):
    req = make_request('S1', fr_config(tasks['S1'].raw, facts(QWEN3_1_7B)))
    p = cm.make_plan(req, facts(QWEN3_1_7B))
    assert (p.world, p.tp, p.dp, p.ga, p.mbs, p.sp, p.is_moe) == (1, 1, 1, 16, 1, False, False)
    assert p.model_kwargs['calculate_per_token_loss'] is True
    assert p.model_kwargs['recompute_granularity'] == 'full' and p.model_kwargs['recompute_num_layers'] == 1
    assert 'moe_aux_loss_coeff' not in p.model_kwargs
    o = p.optimizer_kwargs
    assert (o['lr'], o['min_lr'], o['adam_beta1'], o['adam_beta2'], o['adam_eps'], o['weight_decay'], o['clip_grad']) == \
        (1e-05, 1e-06, 0.9, 0.95, 1e-08, 0.1, 1.0)
    assert o['use_distributed_optimizer'] and not o['use_precision_aware_optimizer']
    d = p.ddp_kwargs
    assert d['grad_reduce_in_fp32'] and d['use_distributed_optimizer'] and not d['average_in_collective']
    assert p.env['set']['CUDA_DEVICE_MAX_CONNECTIONS'] == '1'


def test_fr_plan_dense_8gpu_and_tp2(make_request, tasks):
    p = cm.make_plan(make_request('D2', fr_config(tasks['D2'].raw, facts(QWEN3_8B))), facts(QWEN3_8B))
    assert (p.world, p.dp, p.ga, p.seq_len, p.packs_per_update) == (8, 8, 4, 8192, 32)
    p2 = cm.make_plan(make_request('D2', fr_config(tasks['D2'].raw, facts(QWEN3_8B), tp=2)), facts(QWEN3_8B))
    assert (p2.tp, p2.dp, p2.ga, p2.sp, p2.etp) == (2, 4, 8, True, 2)


def test_fr_plan_moe_8gpu(make_request, tasks):
    req = make_request('D4', fr_config(tasks['D4'].raw, facts(QWEN3_30B_A3B)))
    p = cm.make_plan(req, facts(QWEN3_30B_A3B))
    assert (p.ep, p.etp, p.dp, p.ga, p.is_moe) == (8, 1, 8, 4, True)
    mk = p.model_kwargs
    assert mk['moe_router_load_balancing_type'] == 'aux_loss' and mk['moe_aux_loss_coeff'] == 0.001
    assert mk['moe_router_dtype'] == 'fp32' and mk['moe_expert_capacity_factor'] is None
    assert mk['moe_token_dispatcher_type'] == 'alltoall' and mk['moe_grouped_gemm'] is True


def test_missing_keys_take_schema_defaults(make_request):
    p = cm.make_plan(make_request('D2', {}), facts(QWEN3_8B))
    assert p.tp == 1 and p.model_kwargs['recompute_granularity'] is None
    assert p.model_kwargs['cp_comm_type'] == 'p2p'
    schema = cm.load_schema()
    if schema is not None:
        assert set(p.effective) == set(schema['keys'])


def test_offload_turns_on_precision_aware_path(make_request, tasks):
    cfg = fr_config(tasks['D2'].raw, facts(QWEN3_8B), tp=8)
    cfg.update(optimizer_cpu_offload=True, optimizer_offload_fraction=1.0)
    p = cm.make_plan(make_request('D2', cfg), facts(QWEN3_8B))
    assert p.optimizer_kwargs['use_precision_aware_optimizer'] and p.optimizer_kwargs['optimizer_offload_fraction'] == 1.0


@pytest.mark.parametrize('cfg,msg', [
    ({'lr': 1e-4}, 'task constants'),
    ({'grad_reduce_in_fp32': False}, 'task constants'),
    ({'use_distributed_optimizer': False}, 'use_distributed_optimizer'),
    ({'not_a_key': 1}, 'unknown L0 keys'),
    ({'env': {'NCCL_DEBUG': 'INFO'}}, 'whitelist'),
    ({'average_in_collective': True}, 'calculate_per_token_loss'),
    ({'micro_batch_size': 8}, 'packs_per_update'),
    ({'expert_model_parallel_size': 2}, 'dense model'),
    ({'tensor_model_parallel_size': 8, 'sequence_parallel': True, 'context_parallel_size': 2}, 'not divisible'),
    ({'sequence_parallel': True}, 'requires TP > 1'),
    ({'overlap_param_gather': True}, 'overlap_grad_reduce'),
])
def test_invalid_configs_raise(make_request, cfg, msg):
    with pytest.raises(cm.ConfigError, match=msg):
        cm.make_plan(make_request('D2', cfg), facts(QWEN3_8B))


def test_context_parallel_limits_for_sft_packing(make_request):
    ok = cm.make_plan(make_request('D4', {'context_parallel_size': 4, 'expert_model_parallel_size': 2}),
                      facts(QWEN3_30B_A3B))
    assert ok.cp == 4 and ok.dp == 2
    with pytest.raises(cm.ConfigError, match='2\\*CP'):
        cm.make_plan(make_request('D4', {'context_parallel_size': 8, 'expert_model_parallel_size': 8}),
                     facts(QWEN3_30B_A3B))
    pt = cm.make_plan(make_request('D2', {'context_parallel_size': 8}), facts(QWEN3_8B))   # pt_stream windows
    assert pt.cp == 8 and pt.dp == 1 and pt.ga == 32


def test_tp_must_fit_heads_and_node(make_request):
    with pytest.raises(cm.ConfigError):
        cm.make_plan(make_request('D2', {'tensor_model_parallel_size': 16, 'sequence_parallel': True}),
                     facts(QWEN3_8B))


def test_child_env_only_request_controls_perf_vars(make_request, tasks):
    cfg = fr_config(tasks['D2'].raw, facts(QWEN3_8B))
    cfg['env'] = {'NCCL_ALGO': 'Ring', 'CUDA_DEVICE_MAX_CONNECTIONS': '8'}
    p = cm.make_plan(make_request('D2', cfg), facts(QWEN3_8B))
    parent = {'PATH': '/bin', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'NCCL_PROTO': 'LL',
              'NCCL_SOCKET_IFNAME': 'eth0', 'OMP_NUM_THREADS': '64'}
    env = cm.child_env(parent, p)
    assert env['NCCL_ALGO'] == 'Ring' and env['CUDA_DEVICE_MAX_CONNECTIONS'] == '8'
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in env and 'NCCL_PROTO' not in env
    assert env['NCCL_SOCKET_IFNAME'] == 'eth0' and env['OMP_NUM_THREADS'] == '1' and env['PATH'] == '/bin'


def test_request_roundtrip_through_json(make_request, tasks):
    req = make_request('S2', fr_config(tasks['S2'].raw, facts(QWEN3_8B)), input_plan='B')
    again = RunRequest.from_dict(json.loads(json.dumps(req.to_dict())))
    assert cm.make_plan(again, facts(QWEN3_8B)).summary() == cm.make_plan(req, facts(QWEN3_8B)).summary()


def test_code_digest_check(tmp_path):
    from infrarsi.contracts import code_digest
    pkg = tmp_path / 'infrarsi'
    (pkg / 'runtime' / 'overlays').mkdir(parents=True)
    (pkg / 'runtime' / 'timers.py').write_text('x = 1\n')
    (pkg / 'runtime' / 'overlays' / 'schedule.py').write_text('y = 1\n')
    want = code_digest(pkg)
    assert cm.code_digest_problem('', pkg) is None and cm.code_digest_problem(want, pkg) is None
    (pkg / 'runtime' / 'overlays' / 'schedule.py').write_text('y = 2\n')        # overlays are digested apart
    assert cm.code_digest_problem(want, pkg) is None
    (pkg / 'runtime' / 'timers.py').write_text('x = 2\n')
    assert cm.code_digest_problem(want, pkg).startswith('code_digest_mismatch')


def test_ep_overlap_schedule_plan_is_rejected_up_front(make_request, tasks):
    facts = cm.model_facts(QWEN3_30B_A3B)
    cfg = dict(fr_config(tasks['D4'].raw, facts), expert_model_parallel_size=8, overlap_moe_expert_parallel_comm=True)
    with pytest.raises(cm.ConfigError, match='combined-1F1B'):
        cm.make_plan(make_request('D4', cfg), facts)


def test_deterministic_lifecycle_is_opt_in_and_skips_gated_delta_net(make_request, tasks):
    dense = cm.model_facts(QWEN3_8B)
    gdn = cm.model_facts(dict(QWEN3_8B, model_type='qwen3_next', linear_num_value_heads=32))
    assert not dense['gated_delta_net'] and gdn['gated_delta_net']
    life = make_request('S2', {}, purpose='lifecycle', input_plan='B', lifecycle={'save_at': 4, 'deterministic': True})
    plain = make_request('S2', {}, purpose='lifecycle', input_plan='B', lifecycle={'save_at': 4})
    assert cm.deterministic_lifecycle(life, dense) and not cm.deterministic_lifecycle(life, gdn)
    assert not cm.deterministic_lifecycle(plain, dense) and not cm.deterministic_lifecycle(make_request('S2', {}), dense)
