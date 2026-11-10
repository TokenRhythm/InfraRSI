"""L1 mutable zone: the six hook modules below are the only files an optimizer agent may replace.

The loader, the allowed override keys and the sha256 checks live in the frozen ``runtime/overlay_loader.py``
(see its docstring for every call point); this file only re-exports its constants for existing importers.
"""
from infrarsi.runtime.overlay_loader import ALLOWED, HOOK_MODULES  # noqa: F401
