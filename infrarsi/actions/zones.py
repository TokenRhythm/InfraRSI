"""Mutable and frozen zones of the code tree (design 1.8, FINAL_PLAN 3.3).

Paths are relative to the Python package root `infrarsi/infrarsi/`. The mutable zone is exactly the six
L1 overlay files; the frozen zone is every evaluator / training-semantics module. Optimizer-side code
(agent, rsi, baselines, redteam) and operations code (monitor, dispatch, cli) are in neither zone: the
agent cannot reach them through any action, and they never run inside the training process.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_DIR = 'runtime/overlays'
MUTABLE_OVERLAYS = ('schedule.py', 'recompute_policy.py', 'comm_groups.py', 'state_placement.py',
                    'host_sync.py', 'data_path.py')
DIGEST_SUFFIXES = ('.py', '.json', '.yaml', '.yml')

# component -> package-relative globs (design 1.8: loss/objective, data order, optimizer math, LR schedule,
# model definition, timers, receipts, gates, invariants, reward; plus the read-only evaluator code).
FROZEN_COMPONENTS = {
    'contracts': ['contracts/*'],
    'model_loss_optimizer_lr': ['adapters/megatron_core/*'],
    'data_order': ['adapters/data/*', 'adapters/*'],
    'timers': ['runtime/timers.py'],
    'receipts_signals': ['runtime/receipt_writer.py', 'runtime/signals.py', 'runtime/heartbeat.py'],
    'gates_invariants': ['runtime/gates.py', 'runtime/invariants.py'],
    'lifecycle_stop': ['runtime/lifecycle.py', 'runtime/stop.py', 'runtime/*.py'],
    'harness': ['harness/*'],
    'reward_scoring': ['reward/*', 'scoring/*'],
    'planner_discover': ['planner/*', 'discover/*'],
    'actions': ['actions/*'],
}
NEITHER_ZONE = ('agent/*', 'rsi/*', 'baselines/*', 'redteam/*', 'monitor/*', 'dispatch/*', 'cli/*', 'bench/*')

# Dotted module prefixes an overlay may never import (the frozen zone seen as modules).
FROZEN_MODULE_PREFIXES = ('infrarsi.contracts', 'infrarsi.adapters', 'infrarsi.harness', 'infrarsi.reward',
                          'infrarsi.scoring', 'infrarsi.planner', 'infrarsi.discover', 'infrarsi.actions',
                          'infrarsi.agent', 'infrarsi.rsi', 'infrarsi.redteam', 'infrarsi.baselines',
                          'infrarsi.monitor', 'infrarsi.dispatch', 'infrarsi.cli', 'infrarsi.runtime')
# The only infrarsi modules an overlay may import (sibling overlays and a documented hook API, if T adds one).
ALLOWED_INFRARSI_IMPORTS = ('infrarsi.runtime.overlays', 'infrarsi.runtime.overlay_api')


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def is_mutable(rel: str) -> bool:
    rel = rel.replace('\\', '/')
    return rel.startswith(OVERLAY_DIR + '/') and rel.split('/')[-1] in MUTABLE_OVERLAYS and rel.count('/') == 2


def is_frozen(rel: str) -> bool:
    rel = rel.replace('\\', '/')
    if rel.startswith(OVERLAY_DIR + '/') or '__pycache__' in rel:
        return False
    if any(fnmatch.fnmatch(rel, g) for g in NEITHER_ZONE):
        return False
    return any(fnmatch.fnmatch(rel, g) for globs in FROZEN_COMPONENTS.values() for g in globs)


def frozen_zone_files(root: str | Path = PACKAGE_ROOT) -> list[str]:
    root = Path(root)
    out = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in DIGEST_SUFFIXES:
            rel = _rel(p, root)
            if is_frozen(rel):
                out.append(rel)
    return out


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_zone_manifest(root: str | Path = PACKAGE_ROOT) -> dict[str, str]:
    root = Path(root)
    return {rel: file_sha256(root / rel) for rel in frozen_zone_files(root)}


def frozen_zone_digest(root: str | Path = PACKAGE_ROOT) -> str:
    """sha256 over the canonical list of (relative path, file sha256) of the frozen zone (same rule as code_digest)."""
    rows = sorted(frozen_zone_manifest(root).items())
    raw = json.dumps(rows, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


TRAINING_FROZEN_DIRS = ('contracts', 'runtime', 'adapters')     # == runtime.invariants.FROZEN_DIRS (receipt invariant)


def training_frozen_zone_digest(root: str | Path = PACKAGE_ROOT) -> str:
    """The receipt's invariants.frozen_zone_digest rule (training-process code only: contracts, runtime minus
    overlays, adapters; *.py). The evaluator zone above is a superset; patch_guard checks both."""
    from ..contracts import digest
    root = Path(root)
    rows = []
    for sub in TRAINING_FROZEN_DIRS:
        for path in sorted((root / sub).rglob('*.py')):
            rel = path.relative_to(root).as_posix()
            if rel.startswith(OVERLAY_DIR + '/') or '__pycache__' in rel:
                continue
            rows.append([rel, file_sha256(path)])
    return digest(rows)


def overlay_name(name: str) -> str | None:
    """Normalize 'host_sync.py' or 'runtime/overlays/host_sync.py' to the bare overlay file name, else None."""
    n = name.replace('\\', '/').strip()
    if n in MUTABLE_OVERLAYS:
        return n
    if is_mutable(n):
        return n.split('/')[-1]
    return None
