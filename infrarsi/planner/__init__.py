"""Non-LLM planners: fr_v0 (FR denominator), expert_default (ED), and the memory feasibility model."""
from .expert_default import EDChain, candidate_space, expert_default, pick_diverse, rank
from .feasibility import estimate, layout
from .fr_v0 import FRChain, fr_v0

# The flagship's real SFT recipe "B" (context pack 2.1): TP8/DP4/EP32/CP1/PP1, MBS1/GA2, full recompute, expert
# Adam-v CPU offload, lossless 16-bit master encoding, PLE tables on the host, KV heads replicated across TP8.
FLAGSHIP_B = {'tensor_model_parallel_size': 8, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
              'expert_model_parallel_size': 32, 'expert_tensor_parallel_size': 1, 'sequence_parallel': True,
              'micro_batch_size': 1, 'recompute_granularity': 'full', 'recompute_method': 'uniform', 'recompute_num_layers': 1,
              'use_distributed_optimizer': True, '_ple_cpu_offload': True, '_expert_exp_avg_sq_cpu': True,
              '_master_param_bytes': 2, '_kv_heads_replicated': True}

__all__ = ['fr_v0', 'FRChain', 'expert_default', 'EDChain', 'candidate_space', 'rank', 'pick_diverse', 'estimate',
           'layout', 'FLAGSHIP_B']
