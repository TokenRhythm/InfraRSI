"""Schedule hooks (L1). Default: Megatron-Core's own forward_backward_func, micro-batches in plan order."""


def model_config_overrides(ctx) -> dict:
    """Schedule/overlap fields for the bridge ModelConfig (keys limited by overlays.ALLOWED)."""
    return {}


def microbatch_order(ctx, step, microbatches):
    """Order of this rank's micro-batches within update ``step``; the trainer checks the set is unchanged."""
    return microbatches


def forward_backward(ctx, run, **kwargs):
    """Wraps megatron.core forward_backward_func; must call ``run(**kwargs)`` exactly once and return its result."""
    return run(**kwargs)
