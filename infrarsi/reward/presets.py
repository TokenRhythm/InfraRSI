"""reward.yaml loading. Speed is the only objective (revamp design A): there are no presets or weights; a benchmark
task's tolerance is its predeclared numerics tier, and resolve() only applies the memory cap of the GPU type."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from infrarsi.contracts import DESIGN_DIR

REWARD_YAML = DESIGN_DIR / 'reward.yaml'
MEMORY_CAP_GIB = {'H100-80GB': 78, 'H20-96GB': 95, 'L20X-141GB': 138}     # peak reserved GiB a candidate may reach


def load_reward_cfg(path: str | Path = REWARD_YAML) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg.get('schema') not in ('infrarsi-reward-v0', 'infra' + 'smith-reward-v0'):
        raise ValueError(f'unexpected reward schema {cfg.get("schema")!r}')
    return cfg


def exact(cfg: dict) -> dict:
    """The bitwise contract (tolerance 0): only E0 candidates are valid; per-tier gates stay per tier, all at 0."""
    c = copy.deepcopy(cfg)
    c['numerics_class_allowed'] = 'E0'
    c['tolerance'] = {k: ({t: 0.0 for t in v} if isinstance(v, dict) else 0.0) for k, v in (c.get('tolerance') or {}).items()}
    return c


def resolve(cfg: dict, *, gpu_type: str | None = None) -> dict:
    """A copy with the hard memory cap of `gpu_type` (when known); tolerances and the other hard limits are as declared."""
    c = copy.deepcopy(cfg)
    caps = {**MEMORY_CAP_GIB, **((c.get('auto_mode') or {}).get('peak_reserved_gib_max_by_tier') or {})}
    if gpu_type in caps:
        c['hard']['peak_reserved_gib_max'] = caps[gpu_type]
    return c
