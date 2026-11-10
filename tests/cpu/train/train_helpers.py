"""Model shape facts and FR configs shared by the T-train CPU tests."""
from __future__ import annotations


QWEN3_1_7B = {'model_type': 'qwen3', 'num_hidden_layers': 28, 'num_attention_heads': 16, 'num_key_value_heads': 8,
              'vocab_size': 151936, 'tie_word_embeddings': True, 'hidden_size': 2048}
QWEN3_8B = {'model_type': 'qwen3', 'num_hidden_layers': 36, 'num_attention_heads': 32, 'num_key_value_heads': 8,
            'vocab_size': 151936, 'tie_word_embeddings': False, 'hidden_size': 4096}
QWEN3_30B_A3B = {'model_type': 'qwen3_moe', 'num_hidden_layers': 48, 'num_attention_heads': 32,
                 'num_key_value_heads': 4, 'num_experts': 128, 'num_experts_per_tok': 8, 'vocab_size': 151936,
                 'tie_word_embeddings': False, 'hidden_size': 2048}


def fr_config(task: dict, facts: dict, tp: int = 1) -> dict:
    """FR program defaults of design section 1.4 (the planner owns the real fr_v0; this mirrors it for tests)."""
    moe = facts.get('num_experts') is not None
    cfg = {'tensor_model_parallel_size': tp, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
           'sequence_parallel': tp > 1, 'micro_batch_size': 1, 'recompute_granularity': 'full',
           'recompute_method': 'uniform', 'recompute_num_layers': 1, 'overlap_grad_reduce': False,
           'overlap_param_gather': False, 'apply_rope_fusion': False, 'bias_activation_fusion': False,
           'masked_softmax_fusion': False, 'gradient_accumulation_fusion': False, 'cross_entropy_loss_fusion': False,
           'attention_backend': 'auto', 'env': {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}}
    if moe:
        cfg.update(expert_model_parallel_size=min(facts['num_experts'], 8), expert_tensor_parallel_size=1,
                   moe_token_dispatcher_type='alltoall', moe_grouped_gemm=True, moe_permute_fusion=False,
                   moe_enable_deepep=False)
    return cfg
