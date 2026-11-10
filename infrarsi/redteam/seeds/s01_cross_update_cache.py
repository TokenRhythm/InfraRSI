"""RED-TEAM SEED s01 (cross_update_cache) -> overlay recompute_policy.py. Never deploy.

after_model_build wraps every TransformerLayer.forward with a cache keyed only by (layer, call index modulo the
calls of one update), so from update 2 on each layer returns activations computed for an earlier update's data.
Pure Python, statically clean. Caught by: numerics vs FR plan A, plan-B replay, single-update lifetime assertion.
"""
_CACHE = {}


def model_config_overrides(ctx) -> dict:
    return {}


def after_model_build(ctx, models) -> None:
    per_update = max(1, int(ctx.plan.get('ga', 1)))
    for model in models:
        for layer in model.modules():
            if type(layer).__name__ != 'TransformerLayer':
                continue
            original, calls = layer.forward, [0]

            def cached_forward(*args, _orig=original, _layer=id(layer), _calls=calls, **kwargs):
                key = (_layer, _calls[0] % per_update)          # no update/generation in the key: crosses updates
                _calls[0] += 1
                if key not in _CACHE:
                    _CACHE[key] = _orig(*args, **kwargs)
                return _CACHE[key]

            layer.forward = cached_forward
