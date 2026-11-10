"""Shared contracts for InfraRSI: task specs, run requests, receipts, digests and remote paths.

Owned by the orchestrator. Builder modules import from here and must not redefine these objects;
proposed changes go to infrarsi/contracts/PROPOSED_CHANGES_<area>.md.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import yaml

SCHEMA_REQUEST = 'infrarsi-run-request-v1'
SCHEMA_RECEIPT = 'infrarsi-receipt-v1'
SCHEMA_HEARTBEAT = 'infrarsi-heartbeat-v1'
SCHEMA_LEASE = 'infrarsi-lease-v1'           # {schema, run_id, attempt_id, since, owner, scope?: attempt|run, nodes?}
SCHEMA_HARDWARE = 'infrarsi-hardware-profile-v1'
# discover/hardware.py emits the full profile; planners read only profile['summary'] with these keys
HARDWARE_SUMMARY_KEYS = ('gpus_per_node', 'gpu_type', 'gpu_mem_gib', 'nvlink_intra', 'ib_inter', 'host_mem_gib',
                         'pinned_budget_gib', 'peak_bf16_tflops')

PURPOSES = ('probe', 'screen', 'score', 'lifecycle')
INPUT_PLANS = ('A', 'B')          # A: permutation seed 42 (probes/screens); B: seed 1042 (locked scoring, lifecycle)
PLAN_SEEDS = {'A': 42, 'B': 1042}
NUMERICS_ROLES = ('candidate', 'reference')   # only an explicit 'reference' run may lack a numerics reference

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DESIGN_DIR = PROJECT_ROOT / 'reports' / 'infra_rsi_system_20261020' / 'design'
EXECUTION_DIR = PROJECT_ROOT / 'reports' / 'infra_rsi_system_20261020' / 'execution'


class RemotePaths:
    """Layout on the WINH shared GPFS (/data is mounted on all eight nodes)."""
    root = '/data/infrarsi'
    code = '/data/infrarsi/code'                 # content-addressed code trees: code/<code_digest>/
    models = '/data/infrarsi/models'             # <owner>__<name>/ with DOWNLOAD_DONE.json
    datasets_raw = '/data/infrarsi/datasets/raw'
    datasets_packed = '/data/infrarsi/datasets/packed'   # <data_key>__<tokenizer_digest12>/{packed.*, manifest.json}
    runs = '/data/infrarsi/runs'                 # runs/<run_id>/<attempt_id>/{request.json, heartbeat.json, receipt.json, signals/, logs/, STOP.json}
    receipts = '/data/infrarsi/receipts'         # validated receipt copies, by attempt_id
    packages = '/data/infrarsi/packages'         # optimizer packages (A0, A1, A_R, A2_M) by digest
    ledger = '/data/infrarsi/ledger'
    leases = '/data/infrarsi/leases'             # leases/WINH<k>.json while a run owns the node
    monitor = '/data/infrarsi/monitor'
    venv = '/mnt/base/qwen-sft/.venv-megatron-qwen35'          # frozen; never install into it
    flash_next = '/data/sftv2aliyun/flash-next'                # vendored mcore-bridge, Megatron-Core, DeepEP, runtime


WINH_NODES = [f'WINH{k}' for k in range(8)]


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes used for every digest (sorted keys, no whitespace, UTF-8)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


CODE_SUFFIXES = ('.py', '.json', '.yaml')
OVERLAY_PREFIX = 'runtime/overlays/'


def tree_digests(files: dict[str, str]) -> tuple[str, str | None]:
    """(code_digest, overlay_defaults_digest) from {relpath: file sha256} of the package tree (INTERFACES §3.2)."""
    code = sorted([k, v] for k, v in files.items() if not k.startswith(OVERLAY_PREFIX))
    overlays = sorted([k, v] for k, v in files.items() if k.startswith(OVERLAY_PREFIX))
    return digest(code), (digest(overlays) if overlays else None)


def overlay_digest(files: dict[str, str] | None) -> str | None:
    """Digest of an L1 overlay set {file name: sha256}: sha256 of canonical_json(sorted [[name, sha256], ...]).
    Same bytes as actions.patch_guard.overlay_digest and runtime Overlays.digest; None without an overlay."""
    return digest([[k, v] for k, v in sorted(files.items())]) if files else None


def code_digest(package_dir: str | Path) -> str:
    """Digest of the infrarsi package tree at package_dir (the directory holding contracts/, runtime/, ...)."""
    root = Path(package_dir)
    files = {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in CODE_SUFFIXES and '__pycache__' not in p.parts}
    return tree_digests(files)[0]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def new_attempt_id() -> str:
    return 'a' + uuid.uuid4().hex[:12]


class InfrastructureError(RuntimeError):
    """A run could not be carried out for a reason outside the candidate: the transport to the hosts failed (copying
    the tree, an ssh connection error). Not a verdict on the candidate: the search stops instead of marking it invalid."""


# ---------------------------------------------------------------- tasks

@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """A frozen UpdateBench task (design/task_specs/<id>.yaml). `raw` is the authoritative content."""
    raw: dict

    @classmethod
    def load(cls, task_id_or_path: str | Path) -> 'TaskSpec':
        p = Path(task_id_or_path)
        if not p.suffix:
            p = DESIGN_DIR / 'task_specs' / f'{task_id_or_path}.yaml'
        raw = yaml.safe_load(p.read_text(encoding='utf-8'))
        if raw.get('schema') != 'updatebench-task-v0':
            raise ValueError(f'{p}: unexpected schema {raw.get("schema")!r}')
        return cls(raw=raw)

    @property
    def id(self) -> str:
        return self.raw['id']

    @property
    def gpus(self) -> int:
        return int(self.raw['hardware']['gpus'])

    @property
    def nodes(self) -> int:
        return int(self.raw['hardware']['nodes'])

    @property
    def seq_len(self) -> int:
        return int(self.raw['workload']['seq_len'])

    @property
    def screen_steps(self) -> int:
        return int(self.raw['workload']['screen_steps'])

    @property
    def window(self) -> tuple[int, int]:
        a, b = self.raw['workload']['scoring_window']
        return int(a), int(b)

    @property
    def digest(self) -> str:
        return digest(self.raw)

    def model_dir(self) -> str:
        return f"{RemotePaths.models}/{self.raw['model']['repo_id'].replace('/', '__')}"


def load_all_tasks() -> dict[str, TaskSpec]:
    return {p.stem: TaskSpec.load(p) for p in sorted((DESIGN_DIR / 'task_specs').glob('*.yaml'))}


# ---------------------------------------------------------------- run request

@dataclasses.dataclass
class RunRequest:
    """Driver -> WINH. Written to runs/<run_id>/<attempt_id>/request.json before torchrun starts.

    config: L0 action keys (validated by infrarsi.actions against the L0 schema) plus 'env' for whitelisted env vars.
    overlay: None, or {'dir': remote dir holding L1 overlay files, 'files': {name: sha256}}.
    """
    run_id: str
    purpose: str
    task: dict
    config: dict
    nodes: list[str]
    gpus_per_node: int
    input_plan: str = 'A'
    steps: int | None = None              # default: task screen_steps (probe: 5)
    window: tuple[int, int] | None = None  # default: task scoring window (probe: None)
    overlay: dict | None = None
    lifecycle: dict | None = None          # purpose == 'lifecycle': {'save_at', 'resume_steps', 'export', 'forward_compare'}
    reference_receipt: str | None = None   # remote path of the numerics reference receipt (FR plan-A screen), if any
    data_manifest: str | None = None       # manifest the driver validated; None -> adapters.data.manifest_for_task
    numerics_role: str = 'candidate'       # 'reference' only for the task's FR (or ED fallback) reference runs
    effective_overlay_digest: str | None = None   # driver-set before upload: overlay_digest(defaults merged with request files)
    code_root: str = ''
    code_digest: str = ''
    seed: int = 42
    wall_seconds_cap: int = 3600
    attempt_id: str = dataclasses.field(default_factory=new_attempt_id)
    created_utc: str = dataclasses.field(default_factory=utc_now)
    schema: str = SCHEMA_REQUEST

    def __post_init__(self):
        if self.purpose not in PURPOSES:
            raise ValueError(f'purpose {self.purpose!r} not in {PURPOSES}')
        if self.input_plan not in INPUT_PLANS:
            raise ValueError(f'input_plan {self.input_plan!r} not in {INPUT_PLANS}')
        if self.numerics_role not in NUMERICS_ROLES:
            raise ValueError(f'numerics_role {self.numerics_role!r} not in {NUMERICS_ROLES}')
        if self.purpose == 'lifecycle' and not self.lifecycle:
            raise ValueError('lifecycle requests need the lifecycle block')
        if self.steps is None:
            self.steps = 5 if self.purpose == 'probe' else int(self.task['workload']['screen_steps'])
        if self.window is None and self.purpose != 'probe':
            a, b = self.task['workload']['scoring_window']
            self.window = (int(a), int(b))

    @property
    def world_size(self) -> int:
        return len(self.nodes) * self.gpus_per_node

    @property
    def out_dir(self) -> str:
        return f'{RemotePaths.runs}/{self.run_id}/{self.attempt_id}'

    @property
    def config_digest(self) -> str:
        return digest(self.config)

    @property
    def request_overlay_digest(self) -> str | None:
        """Digest of the request-supplied overlay files only; the runtime reports the full effective set, which the
        driver pins in effective_overlay_digest (defaults from the code tree merged with these replacements)."""
        return overlay_digest((self.overlay or {}).get('files'))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d['window'] = list(self.window) if self.window else None
        d['world_size'] = self.world_size
        d['out_dir'] = self.out_dir
        d['config_digest'] = self.config_digest
        d['request_overlay_digest'] = self.request_overlay_digest
        d['task_digest'] = digest(self.task)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'RunRequest':
        fields = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in fields}
        if kw.get('window') is not None:
            kw['window'] = tuple(kw['window'])
        return cls(**kw)


# ---------------------------------------------------------------- receipts

_SHA256 = re.compile(r'[0-9a-f]{64}')

RECEIPT_REQUIRED = (
    'schema', 'run_id', 'attempt_id', 'purpose', 'task_id', 'task_digest', 'config_digest', 'code_digest',
    'overlay_digest', 'input_plan', 'nodes', 'world_size',
    'steps_completed', 'steps', 'window', 'step_time', 'per_step', 'invariants', 'memory', 'events', 'phases',
    'env', 'valid', 'invalid_reasons', 'signals_digest',
)


def check_receipt_shape(receipt: dict) -> list[str]:
    """Structural check only; semantic validation (numerics, invariants vs task) lives in harness/reward."""
    problems = [f'missing {k}' for k in RECEIPT_REQUIRED if k not in receipt]
    if receipt.get('schema') not in (SCHEMA_RECEIPT, 'infra' + 'smith-receipt-v1'):
        problems.append(f'schema {receipt.get("schema")!r}')
    if not _SHA256.fullmatch(str(receipt.get('signals_digest') or '')):
        problems.append('signals_digest is not a sha256 hex digest')     # binds the scored signal files
    for row in receipt.get('per_step', []):
        for k in ('step', 'max_rank_whole_s', 'main_ce', 'grad_norm', 'lr'):
            if k not in row:
                problems.append(f'per_step row missing {k}')
                break
    return problems
