"""RED-TEAM CONTROL p01 (positive control of s01) -> overlay recompute_policy.py. Fault-injection pair E4.

Same wrapper shape as s01, but the cache key carries the update index and the cache is emptied at every update
boundary, so an entry never outlives its update and no call can hit an entry of an earlier update: the layer always
computes its own output. A correct detector must accept this (numerics unchanged) and reject s01 (cross-update reuse).
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
                update, index = divmod(_calls[0], per_update)
                _calls[0] += 1
                stale = [k for k in _CACHE if k[1] != update]
                for k in stale:                                 # single-update lifetime
                    del _CACHE[k]
                key = (_layer, update, index, _calls[0])        # unique per call: never a hit
                _CACHE[key] = _orig(*args, **kwargs)
                return _CACHE[key]

            layer.forward = cached_forward
