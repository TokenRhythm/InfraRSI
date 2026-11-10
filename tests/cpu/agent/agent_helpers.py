"""Shared constants for the agent CPU tests."""
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / 'fixtures'

_OVERLAYS = Path(__file__).resolve().parents[3] / 'infrarsi' / 'runtime' / 'overlays'
_NAMES = ('schedule.py', 'recompute_policy.py', 'comm_groups.py', 'state_placement.py', 'host_sync.py', 'data_path.py')
# The runtime's real pristine overlays (owner T) when shipped; a minimal stand-in otherwise.
_STANDIN = '"""No-op hook point."""\n\n\ndef on_train_start(ctx):\n    return None\n'
PRISTINE = {name: (_OVERLAYS / name).read_text(encoding='utf-8') if (_OVERLAYS / name).exists() else _STANDIN
            for name in _NAMES}

DENSE_8B = {'num_layers': 36, 'num_attention_heads': 32, 'num_query_groups': 8, 'num_experts': 0, 'hidden_size': 4096,
            'vocab_size': 151936, 'params': 8.2e9}
DENSE_4B = {'num_layers': 36, 'num_attention_heads': 32, 'num_query_groups': 8, 'num_experts': 0, 'hidden_size': 2560, 'params': 4.0e9}
MOE_30B = {'num_layers': 48, 'num_attention_heads': 32, 'num_query_groups': 4, 'num_experts': 128, 'moe_router_topk': 8,
           'params': 30.5e9, 'active_params': 3.3e9, 'expert_param_fraction': 0.9, 'hidden_size': 2048, 'has_shared_experts': False}
FR_DENSE = dict(recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1,
                env={'CUDA_DEVICE_MAX_CONNECTIONS': 1})
FR_MOE = dict(expert_model_parallel_size=8, expert_tensor_parallel_size=1, moe_token_dispatcher_type='alltoall',
              moe_grouped_gemm=True, recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1)
