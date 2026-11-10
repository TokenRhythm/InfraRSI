"""Baselines on the common SearchSession: random_L0, tpe_L0 (Optuna), optico_style_L0, openevolve_L0L1, plus the
Claude-CLI OpenAI-compatible shim. FR and ED come from planner (owner D)."""


def __getattr__(name):
    """DISPATCH_HANDLERS (RUN_MATRIX arm -> handler) for infrarsi.dispatch.handlers, built on first access."""
    if name == 'DISPATCH_HANDLERS':
        from .arms import dispatch_handlers as _build
        return _build()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
