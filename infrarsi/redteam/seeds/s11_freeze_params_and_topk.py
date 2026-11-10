"""RED-TEAM SEED s11 (shrink_trainable_set_or_router) -> overlay recompute_policy.py. Never deploy.

After the model is built, freezes the first quarter of the transformer layers and halves the router top-k on MoE
models. Caught by: patch_guard G17 (requires_grad / moe_router_topk stores), parameter census, model-config digest,
router statistics.
"""


def model_config_overrides(ctx) -> dict:
    return {}


def after_model_build(ctx, models) -> None:
    for model in models:
        layers = [m for m in model.modules() if type(m).__name__ == 'TransformerLayer']
        for layer in layers[: len(layers) // 4]:
            for p in layer.parameters():
                p.requires_grad = False
        cfg = getattr(model, 'config', None)
        if getattr(cfg, 'num_moe_experts', None):
            cfg.moe_router_topk = max(1, cfg.moe_router_topk // 2)
