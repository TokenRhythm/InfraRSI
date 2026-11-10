"""Red team: 14 seed overlays of the cheat catalogue, receipt detectors, synthetic effects and the E4 runner."""
from .detectors import detect, full_accepts, lenient_accepts
from .seeds import CATEGORIES, SEEDS

__all__ = ['CATEGORIES', 'SEEDS', 'detect', 'full_accepts', 'lenient_accepts']


def __getattr__(name):
    """DISPATCH_HANDLERS (RUN_MATRIX arm -> handler) for infrarsi.dispatch.handlers, built on first access."""
    if name == 'DISPATCH_HANDLERS':
        from .arms import dispatch_handlers as _build
        return _build()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
