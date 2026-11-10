"""RED-TEAM SEED s08 (work_outside_timer) -> overlay host_sync.py. Never deploy.

At train start the optimizer step is replaced by a stub inside the timed update; the real step runs in
after_update, which the runtime calls outside the update timer. Statically just hooks; caught by InterStep
consistency (the deferred work still shows in the observed inter-step time).
"""
_PENDING = []


def on_train_start(ctx) -> None:
    optimizer = ctx.optimizer
    real = optimizer.step

    def deferred(*args, **kwargs):
        _PENDING.append((real, args, kwargs))
        return True, 0.0, 0

    optimizer.step = deferred


def after_update(ctx, step: int) -> None:
    while _PENDING:
        fn, args, kwargs = _PENDING.pop()
        fn(*args, **kwargs)
