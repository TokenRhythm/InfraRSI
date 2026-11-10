"""RSI loop: optimizer packages (A0..A_R), reflection, self-check, lineage, cost ledger, held-out mode, isolation audit."""
from .package import OptimizerPackage, build_a0, compute_digest, copy_package, without_memory, write_manifest

__all__ = ['OptimizerPackage', 'build_a0', 'compute_digest', 'copy_package', 'without_memory', 'write_manifest']


def __getattr__(name):
    """DISPATCH_HANDLERS (RUN_MATRIX arm -> handler) for infrarsi.dispatch.handlers, built on first access."""
    if name == 'DISPATCH_HANDLERS':
        from .arms import dispatch_handlers as _build
        return _build()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
