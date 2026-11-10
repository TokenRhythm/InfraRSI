"""A Megatron-Core job shaped like the H2b project (Llama-3-8B, TP1, 8 GPUs): its knobs, its reference, its
argument dump, and a project stand-in for the fingerprint."""
from pathlib import Path
from types import SimpleNamespace

KNOBS = {
    'micro_batch_size': {'type': 'int', 'choices': [1, 2], 'flag': '--micro-batch-size', 'default': 1},
    'bf16_main_grads': {'type': 'str', 'choices': ['0', '1'], 'env': 'BF16_MAIN_GRADS', 'default': '0'},
    'overlap_grad_reduce': {'type': 'bool', 'flag': '--overlap-grad-reduce'},
    'cross_entropy_loss_fusion': {'type': 'bool', 'flag': '--cross-entropy-loss-fusion'},
    'manual_gc': {'type': 'bool', 'flag': '--manual-gc'},
    'nan_check_off': {'type': 'bool', 'flag': '--no-check-for-nan-in-loss-and-grad'},
    'NCCL_NVLS_ENABLE': {'type': 'str', 'choices': ['1', '0'], 'env': 'NCCL_NVLS_ENABLE', 'default': '1'},
    'attention_backend': {'type': 'str', 'choices': ['auto', 'flash'], 'flag': '--attention-backend', 'default': 'auto'},
    'CUDA_DEVICE_MAX_CONNECTIONS': {'type': 'str', 'choices': ['1', '8'], 'env': 'CUDA_DEVICE_MAX_CONNECTIONS', 'default': '1'},
    'overlap_param_gather': {'type': 'str', 'choices': ['0', '1'], 'env': 'OVERLAP_PARAM_GATHER', 'default': '0',
                             'mechanism': 'dp_param_gather_overlap', 'states': {'0': 'off', '1': 'on'}},
    'TORCH_NCCL_AVOID_RECORD_STREAMS': {'type': 'str', 'choices': ['0', '1'], 'env': 'TORCH_NCCL_AVOID_RECORD_STREAMS', 'default': '0'},
    'recompute_activations': {'type': 'bool', 'flag': '--recompute-activations'},
    'fp8': {'type': 'str', 'choices': ['0', '1'], 'env': 'FP8', 'default': '0', 'mechanism': 'precision',
            'states': {'0': 'bf16', '1': 'fp8_delayed'}},
    'num_workers': {'type': 'int', 'choices': [2, 8], 'flag': '--num-workers', 'default': 2},
    'empty_unused_memory_level': {'type': 'int', 'choices': [0, 1], 'flag': '--empty-unused-memory-level', 'default': 0},
    'cross_entropy_fusion_impl': {'type': 'str', 'choices': ['native', 'te'], 'flag': '--cross-entropy-fusion-impl', 'default': 'native'},
    'ddp_average_in_collective': {'type': 'bool', 'flag': '--ddp-average-in-collective'},
    'PYTORCH_CUDA_ALLOC_CONF': {'type': 'str', 'choices': ['', 'expandable_segments:True'], 'env': 'PYTORCH_CUDA_ALLOC_CONF', 'default': ''},
}
REF = {k: v.get('default', False) for k, v in KNOBS.items()}
ARGS = {'num_layers': 32, 'hidden_size': 4096, 'ffn_hidden_size': 14336, 'num_attention_heads': 32, 'num_query_groups': 8,
        'kv_channels': 128, 'padded_vocab_size': 128256, 'seq_length': 4096, 'global_batch_size': 64, 'micro_batch_size': 1,
        'tensor_model_parallel_size': 1, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
        'expert_model_parallel_size': 1, 'num_experts': None, 'untie_embeddings_and_output_weights': True, 'mock_data': True,
        'use_distributed_optimizer': True, 'use_precision_aware_optimizer': False, 'overlap_grad_reduce': False,
        'overlap_param_gather': False, 'cross_entropy_loss_fusion': False, 'cross_entropy_fusion_impl': 'native',
        'gradient_accumulation_fusion': False, 'check_for_nan_in_loss_and_grad': True, 'main_grads_dtype': 'torch.float32',
        'exp_avg_dtype': 'torch.float32', 'attention_backend': 'AttnBackend.auto', 'recompute_granularity': None,
        'empty_unused_memory_level': 0, 'num_workers': 2, 'manual_gc': False, 'sequence_parallel': False,
        'tp_comm_overlap': False, 'ddp_average_in_collective': False, 'fp8': None, 'bf16': True, 'fp16': False}


def megatron_log(path: Path, **over) -> Path:
    """A log with Megatron's argument dump (timestamped lines, as the runner records them)."""
    args = {**ARGS, **over}
    path.write_text('\n'.join(f'1792660220.4\t  {k} {"." * max(3, 48 - len(k))} {v}' for k, v in args.items()) + '\n',
                    encoding='utf-8')
    return path


def project(gpu='NVIDIA L20X', memory=141, framework='megatron', versions=None, knobs=None, **raw):
    return SimpleNamespace(raw={'hardware': {'gpu_type': gpu, 'memory_gib': memory}, 'framework': framework,
                                'versions': versions if versions is not None else {'framework': '0.13.0', 'torch': '2.6.0'},
                                'workload': {}, 'search': {}, **raw},
                           gpus=8, nodes=1, knobs=dict(knobs or KNOBS), name='mcore-test', window=(11, 30),
                           reference_config=lambda: {k: v.get('default', False) for k, v in (knobs or KNOBS).items()})
