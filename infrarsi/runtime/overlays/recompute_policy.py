"""Activation recompute / offload hooks (L1). Default: the L0 recompute keys as given."""


def model_config_overrides(ctx) -> dict:
    """Recompute or activation-offload fields for the bridge ModelConfig (keys limited by overlays.ALLOWED)."""
    return {}


def after_model_build(ctx, models) -> None:
    """Called with the bare model chunks after HF weights are loaded and before DDP wrapping."""
    return None
