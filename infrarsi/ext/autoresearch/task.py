"""ENV-C task contracts (design §4): one EnvTask per arm, same environment, judge, budget and margin.

  arm           action space   track    judge
  C-BASE        patch          infra    trusted judge_bpb (the unmodified baseline; R2 / locked reference)
  C-NAT-free    patch          free     printed val_bpb (published protocol; only prepare.py and data are checked)
  C-NAT-infra   patch          infra    trusted judge_bpb + infra-only checks, run after every experiment
  C-IS          patch          infra    same; the static check is also a pre-launch guard (validate_patch)
  C-RND         knobs          infra    same; knob space of knobs.py, seed 42
Budget per arm: 4 h wall, <= 36 experiments, <= 3M LLM tokens ($40 hard cap), 1 GPU (4 GPU-hours); C-RND has no LLM.
Margin 0.002 bpb (fixed, absolute); tau: y = bpb_base - bpb >= 0.002. Frozen files: prepare.py, data shards, tokenizer,
FA3 kernel sources, and the trusted judge/tool files (assets.json holds the hashes measured on WINH). D-AR-7 (race-free
loader) applies to every arm, the calibration and the locked re-scores (AMENDMENT_EXT_002): arm tasks refuse any other
prepare variant; the upstream loader is reachable only through the named diagnostic task (diagnostic=True, task id
ENV-C-DIAG, never an arm). Anchor = the solo D-AR-7 baseline on the arms' node (assets.json `reference`, WINH0
a21767e9dbd4f: val_bpb 0.996365, 936 steps): bpb_base of tau and AUC_budget, the reference losses of check (4), the steps
of the co-location gate. Arms: WINH0 GPUs 0/2/4/6.
Trust domains (Codex W9 B-02/B-03, O 2026-10-21): every untrusted process of an arm - its agent's shell and all its
candidate trainings - runs under the arm's own numeric uid (ARM_UIDS; no passwd entry, never 65534); the arm's files are
owned by it (0700) inside root:<arm gid> 0750 directories, and after every tool call / attempt every process of the uid is
killed. The calibration's concurrent runs use CALIB_UIDS (one per GPU, so that ending one cannot kill another).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.env_task import SCHEMA_TASK, EnvTask
from . import infra_only_check, judge_files, knobs, source

ASSETS_PATH = Path(__file__).resolve().parent / 'assets.json'
ARMS = {'C-BASE': ('patch', 'infra'), 'C-NAT-free': ('patch', 'free'), 'C-NAT-infra': ('patch', 'infra'),
        'C-IS': ('patch', 'infra'), 'C-RND': ('knobs', 'infra'),
        'C-IS2': ('patch', 'infra')}     # 10-24 phase E: InfraRSI's own flow searched it; only its locked pair runs here
BUDGET = {'screens': 36, 'llm_tokens': 3_000_000, 'wall_seconds': 4 * 3600.0, 'gpu_hours': 4.0, 'llm_usd': 40.0}
MARGIN = {'kind': 'fixed', 'value': 0.002}
TAU = 0.002
MAX_PATCH_LINES = 300
TRAIN_TIMEOUT_S = 600          # program.md: a run over 10 minutes is a failure
NODE = 'WINH0'
ARM_GPUS = {'C-RND': 0, 'C-NAT-free': 2, 'C-NAT-infra': 4, 'C-IS': 6, 'C-BASE': 0, 'C-IS2': 6}   # design §4.3 / §4.5
CALIB_GATE = 0.02              # co-location: C-BASE steps within 2 % of the anchor's
JUDGE_BATCH_SIZE = 128         # fixed eval batch of the trusted judge (the baseline's DEVICE_BATCH_SIZE)
ARM_PREPARE = 'h2d_sync'       # D-AR-7: the only prepare variant an arm task accepts
ARM_UIDS = {'C-BASE': 64000, 'C-NAT-free': 64001, 'C-NAT-infra': 64002, 'C-IS': 64003, 'C-RND': 64004, 'C-IS2': 64008}
CALIB_UIDS = (64000, 64005, 64006, 64007)      # calibration: C-BASE on gpus[0], then one per co-running knob point
DIAG_UID = 64009                               # the named upstream-loader diagnostic (check.py), never an arm
ENVC_UIDS = tuple(sorted(set(ARM_UIDS.values()) | set(CALIB_UIDS)))       # 64000-64007: co-located siblings


def check_arm_variant(prepare_variant: str, *, diagnostic: bool = False) -> str:
    """D-AR-7 is mandatory for arm tasks (AMENDMENT_EXT_002); 'upstream' only for the explicit diagnostic task."""
    if prepare_variant == ARM_PREPARE:
        return prepare_variant
    if diagnostic and prepare_variant in source.PREPARE_VARIANTS:
        return prepare_variant
    raise ValueError(f'ENV-C arms use prepare variant {ARM_PREPARE!r} (D-AR-7) only, not {prepare_variant!r}; the upstream '
                     'loader is reserved for the named diagnostic (check.py --diagnostic-upstream-loader)')


def require_arm_task(task) -> None:
    """Arm entry points: the task must be an arm task with the D-AR-7 loader (never the diagnostic one)."""
    raw = task.raw
    if raw.get('diagnostic') or task.id != 'ENV-C':
        raise ValueError(f'{task.id}: a diagnostic task cannot run an arm')
    check_arm_variant(raw.get('prepare_variant'))
    if raw.get('uid') != ARM_UIDS.get(task.arm):
        raise ValueError(f'{task.arm}: task uid {raw.get("uid")} is not the trust domain of the arm ({ARM_UIDS.get(task.arm)})')


def load_assets(path: str | Path = ASSETS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'{p}: ENV-C assets not frozen yet (run infrarsi.ext.autoresearch.prep on WINH)')
    return json.loads(p.read_text(encoding='utf-8'))


def frozen_files(assets: dict, prepare_variant: str = source.DEFAULT_PREPARE) -> dict:
    ff = dict(judge_files.trusted_sha256(prepare_variant))
    ff.update({f'data/{k}': v for k, v in assets['data'].items()})
    ff.update({f'tokenizer/{k}': v for k, v in assets['tokenizer'].items()})
    ff.update({f'kernel/{k}': v for k, v in assets['kernel'].items()})
    return dict(sorted(ff.items()))


def anchor(assets: dict) -> dict:
    """The ENV-C anchor (assets.json reference); R2 is the fallback before one is recorded."""
    ref = assets.get('reference') or {}
    if ref.get('val_bpb') and ref.get('num_steps') and isinstance(ref.get('first_losses'), list):
        return {k: ref.get(k) for k in ('val_bpb', 'num_steps', 'first_losses', 'attempt_id', 'node', 'gpu', 'prepare_variant')}
    return {'val_bpb': source.R2['val_bpb'], 'num_steps': source.R2['num_steps'], 'first_losses': source.R2['first_losses'],
            'attempt_id': source.R2['run_id'], 'node': source.R2['node'], 'gpu': [0], 'prepare_variant': 'upstream'}


def expected_runtime(assets: dict) -> dict:
    """What the infra-only runtime check compares run_meta.json with (baseline values)."""
    ref = assets.get('reference') or {}
    return {'constants': infra_only_check.expected_constants(source.base_train()), 'time_budget': 300, 'max_seq_len': 2048,
            'config': source.R2['model_config'], 'num_params': source.R2['num_params'],
            'optimizer_groups': ref.get('optimizer_groups')}


def env_c_task(arm: str = 'C-IS', *, assets: dict | None = None, budget: dict | None = None, node: str = NODE,
               prepare_variant: str = ARM_PREPARE, diagnostic: bool = False) -> EnvTask:
    """prepare_variant: 'h2d_sync' = D-AR-7, mandatory for every ENV-C arm; diagnostic=True (C-BASE only) builds the
    named diagnostic task ENV-C-DIAG, which may use 'upstream' and is refused by every arm entry point."""
    if arm not in ARMS:
        raise ValueError(f'unknown ENV-C arm {arm!r}; arms: {sorted(ARMS)}')
    if diagnostic and arm != 'C-BASE':
        raise ValueError('the diagnostic task runs the baseline only (arm C-BASE)')
    check_arm_variant(prepare_variant, diagnostic=diagnostic)
    assets = assets if assets is not None else load_assets()
    action_space, track = ARMS[arm]
    anc = anchor(assets)
    b = dict(budget or BUDGET)
    if arm == 'C-RND':
        b.update(llm_tokens=0, llm_usd=None)
    facts = {'model': {'name': 'autoresearch GPT (nanochat-derived, single GPU)', 'params': source.R2['num_params'],
                       'config': source.R2['model_config'], 'flops_per_token': source.R2['num_flops_per_token']},
             'data': {'name': 'climbmix-400b-shuffle (ModelScope mirror, D-AR-1)', 'train_shards': 10, 'val_shard': 'shard_06542.parquet',
                      'tokenizer': 'rustbpe BPE, vocab 8192', 'seq_len': 2048, 'eval_tokens': 40 * 524288},
             'phases': {'time_budget_s': 300, 'timer_starts_after_step': 10, 'r2_total_seconds': source.R2['total_seconds'],
                        'r2_training_seconds': source.R2['training_seconds']},
             'hardware': {'gpu_type': 'H100 80GB HBM3', 'gpus': 1, 'nodes': 1, 'node': node, 'bf16_peak_flops': 989.5e12},
             'baseline': {'val_bpb': anc['val_bpb'], 'num_steps': anc['num_steps'], 'node': anc['node'],
                          'note': 'solo D-AR-7 baseline on the arms node (the anchor)'},
             'r2': {k: source.R2[k] for k in ('val_bpb', 'num_steps', 'mfu_percent', 'peak_vram_mb', 'total_tokens_M',
                                              'tok_per_sec', 'grad_accum_steps')},
             'published': source.PUBLISHED, 'track': track,
             'prepare_constants': {'MAX_SEQ_LEN': 2048, 'TIME_BUDGET': 300, 'EVAL_TOKENS': 40 * 524288, 'VOCAB_SIZE': 8192},
             'train_timeout_s': TRAIN_TIMEOUT_S, 'judge': 'trusted judge_bpb.py' if track == 'infra' else 'printed val_bpb'}
    spec = {'schema': SCHEMA_TASK, 'id': 'ENV-C-DIAG' if diagnostic else 'ENV-C', 'env': 'autoresearch',
            'split': 'diagnostic' if diagnostic else 'external', 'arm': arm, 'track': track, 'diagnostic': bool(diagnostic),
            'uid': DIAG_UID if diagnostic else ARM_UIDS[arm],
            'metric_name': 'val_bpb', 'action_space': action_space,
            'knob_schema': knobs.KNOB_SCHEMA if action_space == 'knobs' else {},
            **({'knob_space': knobs.frozen_reference()} if action_space == 'knobs' else {}),     # AMENDMENT_EXT_007
            'frozen_files': frozen_files(assets, prepare_variant), 'mutable_globs': ['train.py'], 'max_patch_lines': MAX_PATCH_LINES,
            'prepare_variant': prepare_variant,
            'budget': b, 'margin': MARGIN, 'tau': TAU, 'world_size': 1, 'facts': facts,
            'upstream': source.UPSTREAM, 'base_train_sha256': source.BASE_TRAIN_SHA256,
            'deviations': list(source.DEVIATIONS) + (['D-AR-5'] if track == 'infra' else []) + (['D-AR-6'] if arm.startswith('C-NAT') else [])
                          + (['D-AR-7'] if prepare_variant == 'h2d_sync' else []),
            'anchor': anc, 'reference_losses': anc['first_losses'], 'expected_runtime': expected_runtime(assets),
            'judge_batch_size': JUDGE_BATCH_SIZE, 'assets': {k: assets.get(k) for k in ('measured_utc', 'node', 'reference')}}
    return EnvTask.from_dict(spec)
