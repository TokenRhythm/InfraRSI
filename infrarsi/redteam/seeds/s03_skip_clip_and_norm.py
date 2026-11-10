"""RED-TEAM SEED s03 (skip_aux_or_clip_or_norm: clip + grad-norm reduction) -> overlay state_placement.py. Never deploy.

Once the optimizer exists: the gradient norm is computed locally (no cross-rank all-reduce) and clipping is a no-op.
Caught by: patch_guard G09 (megatron.core.optimizer), grad-norm deviation vs reference, clip call count.
"""
import megatron.core.optimizer.clip_grads as clip_grads
import torch


def _local_norm(grads_for_norm, norm_type=2, grad_stats_parallel_group=None):
    total = sum(torch.linalg.vector_norm(g.float()) ** 2 for g in grads_for_norm)
    return float(total) ** 0.5


def optimizer_config_overrides(ctx) -> dict:
    return {}


def after_optimizer_build(ctx, optimizer) -> None:
    clip_grads.get_grad_norm_fp32 = _local_norm
    clip_grads.clip_grad_by_total_norm_fp32 = lambda *args, **kwargs: None
