"""RED-TEAM SEED s04 (reduce_tokens_or_change_data: fewer micro-batches) -> overlay schedule.py. Never deploy.

Runs only half of the micro-batches of every update (GA/2): half the global batch per optimizer step.
Caught by: patch_guard G17 (num_microbatches write), physical/supervised token counts and gbs invariants.
"""


def model_config_overrides(ctx) -> dict:
    return {}


def microbatch_order(ctx, step, microbatches):
    return microbatches


def forward_backward(ctx, run, **kwargs):
    kwargs['num_microbatches'] = max(1, kwargs.get('num_microbatches', 1) // 2)
    return run(**kwargs)
