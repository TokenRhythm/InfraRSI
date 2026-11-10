"""Host-side hooks (L1): garbage collection, host synchronization, CPU affinity. Default: nothing.

before_update/after_update run outside the update timer; work they hide still shows in inter_step_s.
"""


def on_train_start(ctx) -> None:
    return None


def before_update(ctx, step: int) -> None:
    return None


def after_update(ctx, step: int) -> None:
    return None


def on_train_end(ctx) -> None:
    return None
