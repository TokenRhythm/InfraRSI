"""Any training codebase, one flow (revamp design B): infrarsi.yaml -> reference run -> search -> verifier ->
promotion -> report -> apply."""
from .autopilot import ensure_manifest
from .manifest import SCHEMA, ManifestError, Project
from .runner import ProjectRunner

__all__ = ['ManifestError', 'Project', 'ProjectRunner', 'SCHEMA', 'ensure_manifest']
