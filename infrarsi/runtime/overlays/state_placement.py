"""Optimizer-state placement hooks (L1). Default: distributed optimizer state on GPU as configured by L0."""


def optimizer_config_overrides(ctx) -> dict:
    """OptimizerConfig placement fields such as CPU offload (keys limited by overlays.ALLOWED)."""
    return {}


def after_optimizer_build(ctx, optimizer) -> None:
    """Called once the Megatron optimizer exists, before the first update."""
    return None
