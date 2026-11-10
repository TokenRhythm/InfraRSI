"""RED-TEAM SEED s02 (skip_aux_or_clip_or_norm: aux loss) -> overlay schedule.py. Never deploy.

Replaces the MoE load-balancing loss with zero before the model is built, skipping the router's aux gradient.
Caught by: patch_guard G09 (monkeypatch of megatron.core.transformer.moe.moe_utils), tracked aux_loss scalar,
loss-definition digest (live config re-read).
"""
import megatron.core.transformer.moe.moe_utils as moe_utils


def model_config_overrides(ctx) -> dict:
    moe_utils.switch_load_balancing_loss_func = lambda probs, *args, **kwargs: probs.new_zeros(())
    return {}
