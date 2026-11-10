"""EnvTask: one environment x workload x arm contract (EXT_INTERFACES §2).

The metric is always minimized (exec_time_s, iter_time_s, val_bpb). `raw` is the spec as loaded and the identity
that receipts and digests use; the typed fields are the contract the runner, hooks, judge and scoring read.
`budget` is the per-arm search budget of the design tables; the locked re-scoring runs outside it, so the session
keeps no scoring reserve (`search_budget`).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from pathlib import Path

import yaml

from ...agent.budget import Budget
from ...contracts import digest

SCHEMA_TASK = 'infrarsi-ext-task-v1'
ENVS = ('rebench_llmf', 'optico', 'autoresearch')
ACTION_SPACES = ('knobs', 'patch', 'knobs+patch')
BUDGET_KEYS = ('screens', 'llm_tokens', 'wall_seconds', 'gpu_hours')
FIELDS = ('id', 'env', 'split', 'metric_name', 'action_space', 'knob_schema', 'frozen_files', 'mutable_globs',
          'max_patch_lines', 'budget', 'margin', 'tau', 'world_size', 'facts')
_IDENT = re.compile(r'^[a-z][a-z0-9_]*$')
_SHA = re.compile(r'^[0-9a-f]{64}$')


def _num(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def check_spec(d: dict) -> list[str]:
    """Problems with a task spec (the typed fields); [] when it is a valid contract."""
    errs = []
    if not isinstance(d.get('id'), str) or not d['id']:
        errs.append('id must be a non-empty string')
    if not isinstance(d.get('env'), str) or not _IDENT.match(d['env']):
        errs.append(f'env {d.get("env")!r} must be an identifier (known: {ENVS})')
    if not isinstance(d.get('split', 'external'), str) or d.get('split', 'external') == 'dev':
        errs.append('split must be a non-dev string (ext tasks never run under the development whitelist)')
    if not isinstance(d.get('metric_name'), str) or not _IDENT.match(d.get('metric_name') or ''):
        errs.append('metric_name must be an identifier (the minimized metric)')
    if d.get('action_space') not in ACTION_SPACES:
        errs.append(f'action_space {d.get("action_space")!r} not in {ACTION_SPACES}')
    if not isinstance(d.get('knob_schema'), dict):
        errs.append('knob_schema must be an object (empty for patch-only tasks)')
    elif d.get('action_space') in ('knobs', 'knobs+patch') and not d['knob_schema'].get('properties'):
        errs.append('a knobs action space needs knob_schema.properties')
    ff = d.get('frozen_files')
    if not isinstance(ff, dict) or any(not isinstance(k, str) or not k or k.startswith('/') or '..' in k.split('/')
                                       or not isinstance(v, str) or not _SHA.match(v) for k, v in (ff or {}).items()):
        errs.append('frozen_files must map relative paths to sha256 hex digests')
    mg = d.get('mutable_globs')
    if not isinstance(mg, (list, tuple)) or any(not isinstance(g, str) or not g or g.startswith('/') or '..' in g.split('/')
                                                for g in mg or ()):
        errs.append('mutable_globs must be a list of relative glob patterns')
    elif d.get('action_space') in ('patch', 'knobs+patch') and not mg:
        errs.append('a patch action space needs mutable_globs')
    if type(d.get('max_patch_lines')) is not int or d['max_patch_lines'] < 0:
        errs.append('max_patch_lines must be an int >= 0')
    b = d.get('budget')
    if not isinstance(b, dict) or any(not _num(b.get(k)) or b.get(k) < 0 for k in BUDGET_KEYS):
        errs.append(f'budget must carry non-negative numbers for {BUDGET_KEYS}')
    elif b.get('llm_usd') is not None and (not _num(b['llm_usd']) or b['llm_usd'] < 0):
        errs.append('budget.llm_usd must be a non-negative number when present')
    m = d.get('margin')
    if not isinstance(m, dict) or m.get('kind') not in ('fixed', 'rsd'):
        errs.append("margin must be {'kind': 'fixed', 'value'} or {'kind': 'rsd', 'floor', 'k'}")
    elif m['kind'] == 'fixed' and (not _num(m.get('value')) or m['value'] <= 0):
        errs.append('margin.value must be > 0')
    elif m['kind'] == 'rsd' and (not _num(m.get('floor')) or m['floor'] <= 0 or not _num(m.get('k')) or m['k'] <= 0):
        errs.append('margin.floor and margin.k must be > 0')
    if not _num(d.get('tau')):
        errs.append('tau must be a finite number (CTG threshold on y)')
    if type(d.get('world_size')) is not int or d['world_size'] < 1:
        errs.append('world_size must be an int >= 1')
    if not isinstance(d.get('facts'), dict):
        errs.append('facts must be an object')
    return errs


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvTask:                      # one environment x workload x arm contract
    id: str                         # e.g. 'ENV-A', 'ENV-B-W1', 'ENV-C'
    env: str                        # 'rebench_llmf' | 'optico' | 'autoresearch'
    split: str = 'external'         # optimize() treats it like a non-dev task (EXT_INTERFACES §3)
    metric_name: str                # 'exec_time_s' | 'iter_time_s' | 'val_bpb'  (always MINIMIZED; see EnvReceipt)
    action_space: str               # 'knobs' | 'patch' | 'knobs+patch'
    knob_schema: dict               # JSON schema of declarative knobs (empty for patch-only)
    frozen_files: dict              # relative path -> sha256 of the verified base tree
    mutable_globs: tuple            # files a patch may touch (relative to the candidate tree)
    max_patch_lines: int
    budget: dict                    # {'screens', 'llm_tokens', 'wall_seconds', 'gpu_hours'} (design tables)
    margin: dict                    # {'kind': 'fixed', 'value': 0.05} | {'kind': 'rsd', 'floor': 0.03, 'k': 3}
    tau: float                      # CTG threshold on y (design §1.4)
    world_size: int                 # GPUs a candidate run occupies
    facts: dict                     # environment facts shown to optimizers (task view): model, data, phases, hardware
    raw: dict                       # the full spec as loaded (yaml/json), for receipts and digests

    def __post_init__(self):
        object.__setattr__(self, 'mutable_globs', tuple(self.mutable_globs))
        errs = check_spec(self.fields())
        if errs:
            raise ValueError(f'EnvTask {self.id!r}: ' + '; '.join(errs))

    # ------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, spec: dict) -> 'EnvTask':
        """Typed fields from the spec (missing split -> 'external'); raw = a deep copy of the spec."""
        spec = copy.deepcopy(spec)
        kw = {k: spec[k] for k in FIELDS if k in spec}
        missing = [k for k in FIELDS if k not in kw and k != 'split']
        if missing:
            raise ValueError(f'ext task spec {spec.get("id")!r}: missing {missing}')
        return cls(**kw, raw=spec)

    @classmethod
    def load(cls, path: str | Path) -> 'EnvTask':
        p = Path(path)
        text = p.read_text(encoding='utf-8')
        spec = json.loads(text) if p.suffix == '.json' else yaml.safe_load(text)
        if not isinstance(spec, dict):
            raise ValueError(f'{p}: not a mapping')
        if spec.get('schema', SCHEMA_TASK) != SCHEMA_TASK:
            raise ValueError(f'{p}: unexpected schema {spec.get("schema")!r}')
        return cls.from_dict(spec)

    def fields(self) -> dict:
        return {k: getattr(self, k) for k in FIELDS}

    def to_dict(self) -> dict:
        return json.loads(json.dumps({**self.fields(), 'mutable_globs': list(self.mutable_globs), 'raw': self.raw},
                                     default=str))

    # ------------------------------------------------------------ derived
    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    @property
    def frozen_digest(self) -> str:
        return digest(sorted([k, v] for k, v in self.frozen_files.items()))

    @property
    def gpus(self) -> int:                         # the name agent code reads on core TaskSpecs
        return self.world_size

    @property
    def arm(self) -> str | None:
        return self.raw.get('arm')

    def search_budget(self, **overrides) -> Budget:
        """agent.budget.Budget of the design table; the locked re-scoring is outside it (no scoring reserve)."""
        b = self.budget
        kw = dict(screens=int(b['screens']), llm_tokens=int(b['llm_tokens']), wall_seconds=float(b['wall_seconds']),
                  gpu_hours=float(b['gpu_hours']), llm_usd=b.get('llm_usd'), scoring_reserve_screens=0.0,
                  target_speedup=b.get('target_speedup'), target_mfu=b.get('target_mfu'))
        kw.update(overrides)
        return Budget(**kw)
