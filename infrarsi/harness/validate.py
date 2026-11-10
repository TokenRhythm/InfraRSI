"""Turn a collected attempt into a validated receipt: identity, invariants vs task spec and reference, numerics.

The validated copy replaces step_time/per_step/memory/steps_completed with values re-derived from signals,
keeps the training process's own numbers under harness.reported, and ANDs `valid` with the harness checks.

Fail-closed rules (Codex R1 review, round 6):
  - signals binding: the receipt's signals_digest must be a sha256 under the single contract convention and equal the
    digest of the collected rank files; otherwise the receipt is invalid and no scored timing (step_time, profile) is
    emitted from the unbound files.
  - references: one loader for every use (initial validation, harness.recheck_against, reward) -
    reference_problems(): signed (HMAC verifies, harness/signing.py), valid, same task id and task digest, expected input plan,
    numerics_role 'reference', expected purpose (plan A -> 'screen', plan B -> 'score'), steps_completed == steps,
    finite CE / grad norm / LR on every row, every update 1..N and the scoring window present, complete invariants.
    A run given an unusable reference is invalid; comparisons require every update 1..N in both receipts.
  - invariants: a task-aware mandatory schema for every non-probe reference and candidate (invariant_schema_problems)
    and mandatory equality with the reference for the layout-invariant fields; the layout that ran
    (top-level effective_config) must contain the requested L0 config verbatim.
  - clocks: UTC stamps on every rank and update, strictly positive and ordered; screens and scores tolerate no clock
    outlier inside the scoring window and need the training-span and external clocks; InterStep only from signals.

Numerics (AMENDMENT-001, design 1.7 as amended 2026-10-20; keys from the task spec numerics.tolerance):
  - CE: per-step relative deviation <= main_ce_rel_max_per_step on every compared update; window mean relative
    deviation <= main_ce_rel_mean_over_window.
  - grad norm: update 1 gated (relative deviation <= grad_norm_rel_max_first_update); scoring window gated on the
    signed mean of ln(gn_cand / gn_ref) (|bias| <= grad_norm_log_bias_window_max); steps >= 2 reported only
    (numerics.grad_norm.{first_update_rel, rel_max_all_steps, per_step_rel[], log_bias_window, n_window}).
  - LR exactly equal at every update, aux loss present where the reference has one.
  - E0 = every tracked scalar bitwise equal; E1 = the gates above; otherwise FAIL.
  - self-check without a reference, every run: two adjacent updates with bitwise-equal (main_ce, grad_norm) make the
    run invalid (numerics.self_check.repeated_pairs).
  - lifecycle (numerics.lifecycle_tolerance): receipt lifecycle.state_exact must be true; the first resumed update
    matches the uninterrupted run within first_resumed_update (CE 1e-4, grad norm 5e-3); later resumed updates within
    the task's per-step CE tolerance with LR, tokens and clip calls exactly equal (grad norm reported); export_ok /
    forward_ok and the forward metrics (rel_l2, top-1 agreement) only when the request enables export / forward.
  - invariant: every rank's backward micro-batch count of every update (signals 'microbatches') equals
    packs_per_update / (DP x micro_batch_size), DP from the receipt's effective_config.

Numerics tiers (AMENDMENT-002, execution/design_consults/FABLE_MOE_NONDETERMINISM_20261020.md): the gates follow the
task's model facts, never an agent choice. numerics_tier(task) is dense (not MoE), moe_deterministic (MoE with a
deterministic Megatron-Core path: D3/D4/D5/H1/X2) or moe_nondeterministic (Gated DeltaNet bridges qwen3_next /
qwen4_exp: D6/H5/F1). Each tier's frozen table is design/reward.yaml (tier_tables): the spec's numerics.tolerance and
numerics.lifecycle_tolerance must equal it key for key, otherwise every non-probe run is invalid (numerics.tier), and
the gates are taken from the table. Lifecycle: a 'required' tier needs lifecycle.deterministic true with the
replaced settings in lifecycle.deterministic_overrides; an 'unavailable' tier (no deterministic GDN kernels) needs
lifecycle.deterministic_unavailable; a deterministic lifecycle compares backends_flags with its reference without
deterministic_algorithms / cudnn.deterministic; resumed_bitwise_updates is reported, never gated.

Model-defined fp32 tensors (Codex R3 HIGH): A_log / dt_bias / expert_bias are exempt from the bf16 parameter contract
only as exactly fp32 (dtype.model_defined_fp32); the runtime census reports them under dtype_census.model_fp32.

Trust boundary (Codex R2): the validated receipt is HMAC-signed over everything but its signature field
(harness/signing.py; verify_receipt_digest is that verification). The raw receipt must echo the trusted request's
request_digest, numerics_role, data_manifest and effective_overlay_digest, and its overlay_digest (the runtime's
effective L1 set) must equal the pinned effective_overlay_digest. A lifecycle receipt must carry
lifecycle.signals_resume_digest equal to the collected signals_resume/ files (rank-ordered concatenation) before any
continuity is derived from them.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from infrarsi.contracts import SCHEMA_RECEIPT, RECEIPT_REQUIRED, TaskSpec, check_receipt_shape, digest, utc_now

from . import dataindex as _dataindex
from . import derive as _derive
from . import signing as _signing

TRACKED = ('main_ce', 'aux_loss', 'grad_norm', 'lr')
NUMERICS_ORDER = {'E0': 0, 'E1': 1, 'E2': 2}
OOM_RE = re.compile(r'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|'
                    r'CUDA error: out of memory|NCCL WARN .*Failed to allocate|Cuda failure .*out of memory', re.I)
NCCL_RE = re.compile(r'ncclInternalError|ncclSystemError|ncclUnhandledCudaError|ncclRemoteError|NCCL error|'
                     r'Watchdog caught collective operation timeout|ProcessGroupNCCL.*(?:timed out|Timeout)|'
                     r'NCCL communicator was aborted')
HEX64 = re.compile(r'[0-9a-f]{64}')
DTYPE_ALIASES = {'torch.float32': 'fp32', 'float32': 'fp32', 'fp32': 'fp32', 'float': 'fp32', 'torch.float': 'fp32',
                 'torch.bfloat16': 'bf16', 'bfloat16': 'bf16', 'bf16': 'bf16', 'torch.float16': 'fp16',
                 'float16': 'fp16', 'fp16': 'fp16', 'half': 'fp16'}
# Precision contract of every public task: bf16 params, fp32 master / grads (reduce) / Adam moments.
DTYPE_CONTRACT = {'param': 'bf16', 'grad': 'fp32', 'master': 'fp32', 'exp_avg': 'fp32', 'exp_avg_sq': 'fp32'}
# Invariants that must equal the reference whatever the parallel layout (the reference is always on the same plan).
INV_EQUAL = ('gbs_packs', 'physical_tokens_per_update', 'optimizer_steps', 'lr_schedule_digest', 'trainable_params',
             'trainable_param_digest', 'loss_definition_digest', 'router_topk', 'capacity_factor', 'num_experts',
             'frozen_zone_digest', 'model_config_digest', 'backends_flags')
INV_SAME_PLAN = ('supervised_tokens_total', 'data_order_digest', 'data_consumed_digest')
# A lifecycle attempt runs save_at + resume_steps updates against a plan-B score reference of `steps` updates: its
# step-count totals are checked against its own expected counts instead (optimizer_steps == completed updates,
# lr_schedule_digest == lr_schedule_expected_digest, supervised_tokens_total and data_consumed_digest from INDEX.json).
STEP_COUNT_KEYS = ('optimizer_steps', 'lr_schedule_digest', 'supervised_tokens_total', 'data_consumed_digest')
# Mandatory invariant schema (non-probe runs; real field names of runtime/receipt_writer + adapters trainer).
INV_INTS = ('gbs_packs', 'physical_tokens_per_update', 'optimizer_steps', 'supervised_tokens_total', 'clip_calls_total',
            'trainable_params')
INV_DIGESTS = ('lr_schedule_digest', 'lr_schedule_expected_digest', 'trainable_param_digest', 'loss_definition_digest',
               'model_config_digest', 'frozen_zone_digest', 'data_order_digest')
MOE_BRIDGES = ('qwen3_moe', 'qwen3_next', 'deepseek_v2', 'qwen4_exp')
# AMENDMENT-002 numerics tiers. Gated DeltaNet (packed THD) has no deterministic Megatron-Core path, so its lifecycle
# keeps the default kernels (adapters/megatron_core/config_map.model_facts 'gated_delta_net').
NUMERICS_TIERS = ('dense', 'moe_deterministic', 'moe_nondeterministic')
GDN_BRIDGES = ('qwen3_next', 'qwen4_exp')
DETERMINISTIC_POLICY = 'required_unless_gated_delta_net'      # reward.yaml lifecycle_tolerance.deterministic
DETERMINISTIC_RULES = ('required', 'unavailable')             # what a tier's table (and the spec) says
DETERMINISTIC_FLAG_KEYS = ('deterministic_algorithms', 'cudnn.deterministic')   # what a deterministic lifecycle flips
# Model-defined fp32 tensors (runtime/gates.FP32_PARAM_FRAGMENTS) and their census category.
MODEL_FP32_FRAGMENTS = ('A_log', 'dt_bias', 'expert_bias')
MODEL_FP32_KEY = 'model_fp32'
REF_PURPOSE = {'A': 'screen', 'B': 'score'}           # the numerics reference of a plan-A / plan-B run
# Independent clocks. Per step: derive.CLOCK_REL/ABS; screens and scores allow no outlier inside the scoring window
# and at most CLOCK_MAX_OUTLIERS per rank outside it (other purposes: at most CLOCK_MAX_OUTLIERS per rank). Training
# span: UTC first-start..last-end vs phases.train within 5 % + 5 s. External: node-0 wrapper wall vs phases.total
# after explicit startup/teardown accounting, within max(120 s, 12 %).
LIFECYCLE_TOLERANCE_DEFAULT = {             # AMENDMENT-001 values; the task spec's numerics.lifecycle_tolerance wins
    'state_digest_exact': True, 'first_resumed_update': {'main_ce_rel_max': 1.0e-4, 'grad_norm_rel_max': 5.0e-3},
    'later_updates': {'main_ce': 'task_per_step_tolerance', 'grad_norm': 'report_only', 'lr_tokens_clip_calls': 'exact'},
    'forward': {'rel_l2_max': 0.05, 'top1_agreement_min': 0.90}}
LIFECYCLE_EXACT = ('lr', 'supervised_tokens', 'physical_tokens', 'clip_calls')
STRICT_CLOCK_PURPOSES = ('screen', 'score')
CLOCK_MAX_OUTLIERS = 1
TRAIN_CLOCK_REL, TRAIN_CLOCK_ABS = 0.05, 5.0
EXTERNAL_CLOCK_REL, EXTERNAL_CLOCK_ABS = 0.12, 120.0
REQUIRE_UTC_FIELDS = True        # every runtime/signals.py row carries utc_start_s/utc_end_s (T-train, 2026-10-20)


def _rel(x, y):
    return abs(x - y) / max(abs(y), 1e-12)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def json_safe(obj):
    """Non-finite floats -> None: canonical digests reject NaN/inf, and a run with non-finite scalars is invalid
    anyway (derive records the updates in nan_inf_updates before this)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def norm_dtype(v):
    return DTYPE_ALIASES.get(str(v), str(v))


def dtype_set(v) -> set:
    """A census entry is a dtype or a {dtype: tensor count} map (runtime/gates.py); counts depend on the layout."""
    if isinstance(v, dict):
        return {norm_dtype(k) for k, n in v.items() if n}
    return {norm_dtype(v)}


def census_sets(census: dict | None) -> dict:
    return {k: sorted(dtype_set(v)) for k, v in (census or {}).items()}


def _lim(tol: dict | None, key: str) -> float:
    """A numeric tolerance from the task spec; a missing or non-numeric gate is 0 (fail closed)."""
    v = (tol or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 0.0


def lifecycle_tolerance(task: dict, cfg: dict | None = None) -> dict:
    """The lifecycle gates of the task's tier (its frozen table, AMENDMENT-002); the spec's block only when the table
    cannot be resolved (the run is then invalid through numerics.tier), else the AMENDMENT-001 defaults."""
    table = numerics_tier_check(task, cfg)['table']
    if table is not None:
        return table['lifecycle_tolerance']
    lt = ((task or {}).get('numerics') or {}).get('lifecycle_tolerance')
    return lt if isinstance(lt, dict) else LIFECYCLE_TOLERANCE_DEFAULT


def self_check(per_step: list[dict]) -> dict:
    """Reference-free (AMENDMENT-001): adjacent updates whose (main_ce, grad_norm) are bitwise equal."""
    rows = sorted((p for p in per_step if isinstance(p.get('update'), int)), key=lambda p: p['update'])
    pairs = [[p['update'], q['update']] for p, q in zip(rows, rows[1:])
             if q['update'] == p['update'] + 1 and _derive.finite(p.get('main_ce')) and _derive.finite(p.get('grad_norm'))
             and p.get('main_ce') == q.get('main_ce') and p.get('grad_norm') == q.get('grad_norm')]
    return {'repeated_pairs': pairs}


def sharpness(per_step: list[dict], clip_grad) -> dict:
    """Report-only in-run statistics (AMENDMENT-001 E0 record): share of updates whose grad norm exceeds the clip
    threshold (clipping active) and the median / max |ln(gn_{u+1} / gn_u)| over adjacent updates."""
    gn = [p.get('grad_norm') for p in sorted(per_step, key=lambda p: p.get('update', 0))]
    ok = [x for x in gn if _derive.finite(x)]
    d = sorted(abs(math.log(b / a)) for a, b in zip(gn, gn[1:]) if _derive.finite(a) and _derive.finite(b) and a > 0 and b > 0)
    clip = float(clip_grad) if isinstance(clip_grad, (int, float)) and not isinstance(clip_grad, bool) else None
    return {'clip_grad': clip, 'clip_active_fraction': (sum(x > clip for x in ok) / len(ok)) if ok and clip else None,
            'adjacent_abs_dlog_gn_median': d[len(d) // 2] if d else None, 'adjacent_abs_dlog_gn_max': d[-1] if d else None,
            'n_updates': len(ok)}


def expected_microbatches(packs_per_update: int, world_size: int, effective: dict | None) -> int | None:
    """Backward micro-batches per rank per update: packs_per_update / (DP x micro_batch_size), DP = world / (TP x PP x CP)
    from the layout that ran (with PP this is the pipeline's micro-batch count); None when the layout does not divide."""
    eff = effective if isinstance(effective, dict) else {}
    try:
        tp, pp, cp, mbs = (int(eff.get(k) or 1) for k in ('tensor_model_parallel_size', 'pipeline_model_parallel_size',
                                                          'context_parallel_size', 'micro_batch_size'))
    except (TypeError, ValueError):
        return None
    if min(tp, pp, cp, mbs) < 1 or world_size % (tp * pp * cp):
        return None
    dp = world_size // (tp * pp * cp)
    return int(packs_per_update) // (dp * mbs) if int(packs_per_update) % (dp * mbs) == 0 else None


def task_is_moe(task: dict) -> bool:
    m = (task or {}).get('model') or {}
    return m.get('bridge_type') in MOE_BRIDGES or 'moe' in str(m.get('family', ''))


# ---------------------------------------------------------------- AMENDMENT-002 numerics tiers

def task_is_gdn(task: dict) -> bool:
    return ((task or {}).get('model') or {}).get('bridge_type') in GDN_BRIDGES


def numerics_tier(task: dict) -> str:
    """The task's numerics tier from its model facts (task_is_moe, Gated DeltaNet bridge): never an agent choice."""
    if not task_is_moe(task):
        return 'dense'
    return 'moe_nondeterministic' if task_is_gdn(task) else 'moe_deterministic'


_REWARD_CACHE: dict = {}


def reward_cfg() -> dict:
    """design/reward.yaml through the reward engine's loader, re-read when the file changes (callers must not mutate)."""
    from infrarsi.reward import presets
    p = Path(presets.REWARD_YAML)
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if _REWARD_CACHE.get('key') != key:
        _REWARD_CACHE.update(key=key, cfg=presets.load_reward_cfg(p))
    return _REWARD_CACHE['cfg']


def is_tier_map(value) -> bool:
    """A gate given per tier in the reward config: {tier name | 'moe': value}, keys only from NUMERICS_TIERS and 'moe'."""
    return isinstance(value, dict) and bool(value) and set(value) <= set(NUMERICS_TIERS) | {'moe'}


def tier_value(value, tier: str):
    """A gate's value for `tier`: the value itself, or its tier-map entry ('moe' covers both MoE tiers); KeyError when a
    tier map has no entry for the tier."""
    if not is_tier_map(value):
        return value
    if tier in value:
        return value[tier]
    if str(tier).startswith('moe_') and 'moe' in value:
        return value['moe']
    raise KeyError(f'no {tier!r} entry in {value!r}')


def _resolve(value, tier: str):
    if is_tier_map(value):
        return _resolve(tier_value(value, tier), tier)
    if isinstance(value, dict):
        return {k: _resolve(v, tier) for k, v in value.items()}
    return value


def tier_tables(cfg: dict | None = None) -> dict[str, dict]:
    """{tier: {'tolerance', 'lifecycle_tolerance'}}: the frozen numerics table of every tier, resolved from the reward
    config (default design/reward.yaml). lifecycle_tolerance.deterministic 'required_unless_gated_delta_net' becomes
    'required' / 'unavailable' and moe_nondeterministic takes first_resumed_update_nondeterministic when present; a
    reward config without per-tier entries gives every tier the same table (the design before AMENDMENT-002)."""
    cfg = reward_cfg() if cfg is None else cfg
    tol, life = cfg.get('tolerance'), cfg.get('lifecycle_tolerance')
    if not (isinstance(tol, dict) and isinstance(life, dict)):
        raise ValueError('reward config has no tolerance / lifecycle_tolerance mapping')
    out = {}
    for tier in NUMERICS_TIERS:
        lt = _resolve(life, tier)
        nondet = lt.pop('first_resumed_update_nondeterministic', None)
        if tier == 'moe_nondeterministic' and nondet is not None:
            lt['first_resumed_update'] = nondet
        if lt.get('deterministic') == DETERMINISTIC_POLICY:
            lt['deterministic'] = 'unavailable' if tier == 'moe_nondeterministic' else 'required'
        if 'deterministic' in lt and lt['deterministic'] not in DETERMINISTIC_RULES:
            raise ValueError(f'lifecycle_tolerance.deterministic {lt["deterministic"]!r} for tier {tier}')
        out[tier] = {'tolerance': _resolve(tol, tier), 'lifecycle_tolerance': lt}
    return out


def _canon(value):
    """Comparable form of a YAML value: numbers as floats (0.10 == 0.1, 1 == 1.0), booleans and strings kept apart."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return repr(value)


def numerics_tier_check(task: dict, cfg: dict | None = None) -> dict:
    """{'tier', 'table', 'problems'}: the task's tier, its frozen table (None when the reward config cannot be resolved)
    and every key of the spec's numerics.tolerance / numerics.lifecycle_tolerance that differs from the table. Any
    problem invalidates the run: a spec can neither loosen, tighten nor omit its tier's gates."""
    tier = numerics_tier(task)
    try:
        table = tier_tables(cfg)[tier]
    except Exception as exc:                                       # noqa: BLE001 - fail closed, reported
        return {'tier': tier, 'table': None,
                'problems': [f'frozen numerics table unavailable ({type(exc).__name__}: {exc})']}
    num = (task or {}).get('numerics')
    num = num if isinstance(num, dict) else {}
    problems = []
    for block in ('tolerance', 'lifecycle_tolerance'):
        spec, want = num.get(block), table[block]
        if not isinstance(spec, dict):
            problems.append(f'numerics.{block} missing')
            continue
        for k in sorted(set(spec) | set(want)):
            if k not in spec:
                problems.append(f'numerics.{block}.{k} missing ({tier} table: {_short(want[k])})')
            elif k not in want:
                problems.append(f'numerics.{block}.{k} = {_short(spec[k])} is not in the {tier} table')
            elif _canon(spec[k]) != _canon(want[k]):
                problems.append(f'numerics.{block}.{k} = {_short(spec[k])} != {tier} table {_short(want[k])}')
    return {'tier': tier, 'table': table, 'problems': problems}


def strip_deterministic_flags(flags):
    """backends_flags without the two keys a deterministic lifecycle sets; anything else unchanged."""
    return {k: v for k, v in flags.items() if k not in DETERMINISTIC_FLAG_KEYS} if isinstance(flags, dict) else flags


def deterministic_lifecycle(receipt: dict, purpose: str) -> bool:
    lc = receipt.get('lifecycle') if isinstance(receipt, dict) else None
    return purpose == 'lifecycle' and isinstance(lc, dict) and lc.get('deterministic') is True


def model_fp32_problems(entries, census_entry) -> list[str]:
    """Codex R3 HIGH: a model-defined fp32 tensor (A_log, dt_bias, expert_bias) is exempt from the bf16 parameter
    contract only as exactly fp32. entries: model_defined_fp32_params ('name:dtype' from runtime/gates.dtype_census, or
    bare names); census_entry: dtype_census.model_fp32 (absent in receipts of the pre-fix census)."""
    p = []
    for e in entries if isinstance(entries, list) else []:
        name, sep, dt = str(e).rpartition(':')
        if not sep:
            name, dt = str(e), ''
        if not any(f in name for f in MODEL_FP32_FRAGMENTS):
            p.append(f'{_short(e)} is not a model-defined fp32 tensor ({", ".join(MODEL_FP32_FRAGMENTS)})')
        elif dt and norm_dtype(dt) != 'fp32':
            p.append(f'{_short(e)}: a model-defined fp32 tensor must be exactly fp32')
    if census_entry is not None:
        extra = dtype_set(census_entry) - {'fp32'}
        if extra:
            p.append(f'dtype_census.{MODEL_FP32_KEY} holds {sorted(extra)} (exactly fp32 required)')
    return p


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, name, ok, detail='', severity='invalid'):
        self.items.append({'name': name, 'ok': bool(ok), 'detail': detail, 'severity': severity})
        return ok

    @property
    def invalid(self):
        return [f'{c["name"]}: {c["detail"]}' for c in self.items if not c['ok'] and c['severity'] == 'invalid']

    @property
    def warnings(self):
        return [f'{c["name"]}: {c["detail"]}' for c in self.items if not c['ok'] and c['severity'] == 'warn']


def _short(v):
    s = repr(v)
    return s if len(s) < 80 else s[:77] + '...'


def ref_steps(reference: dict | None) -> list[dict] | None:
    if not reference:
        return None
    rows = reference.get('per_step') or []
    out = []
    for i, p in enumerate(rows, 1):
        q = dict(p)
        q.setdefault('update', i)
        out.append(q)
    return out


# ---------------------------------------------------------------- numerics

def compare_numerics(cand: list[dict], ref: list[dict] | None, tolerance: dict, window, allowed: str = 'E1',
                     role: str = 'candidate', steps: int | None = None, updates=None) -> dict:
    """AMENDMENT-001 numerics classes against a reference (never derived from the candidate): E0 = every tracked scalar
    bitwise equal; E1 = CE per-step and window-mean gates, grad norm gated on the first compared update and on the
    scoring-window mean log ratio (steps >= 2 reported only), LR exactly equal, aux present; else FAIL.
    Without a reference only an explicit numerics_role='reference' run is a reference; a candidate stays unverified.
    The compared updates are `updates`, else 1..steps; every one must be present in both receipts with finite
    main_ce, grad_norm and lr on both sides (a short, gapped or non-finite reference is a FAIL, never a pass)."""
    if ref is None:
        if role == 'reference':
            return {'compared': False, 'class': None, 'is_reference': True, 'violations': []}
        return {'compared': False, 'class': None, 'is_reference': False, 'violations': ['no numerics reference']}
    r, cm = {}, {}
    for p in ref:
        r.setdefault(p['update'], p)
    for p in cand:
        cm.setdefault(p['update'], p)
    if updates is not None:
        want = sorted({int(u) for u in updates})
    elif steps:
        want = list(range(1, int(steps) + 1))
    else:
        want = sorted(set(cm) & set(r))
    res = {'compared': True, 'class': 'FAIL', 'is_reference': False, 'compared_updates': 0, 'reference_updates': len(ref),
           'violations': [], 'allowed': allowed, 'allowed_ok': False}
    v = res['violations']
    if window:
        a, b = int(window[0]), int(window[1])
        if any(u not in want for u in range(a, b + 1)):
            v.append(f'scoring window {[a, b]} outside the compared updates')
    if not want:
        v.append('no updates to compare')
    miss_c = [u for u in want if u not in cm]
    miss_r = [u for u in want if u not in r]
    if miss_c:
        v.append(f'candidate lacks updates {miss_c[:8]}')
    if miss_r:
        v.append(f'reference lacks updates {miss_r[:8]}')
    if v:
        return res
    nf_c = [u for u in want if not all(_derive.finite(cm[u].get(k)) for k in ('main_ce', 'grad_norm', 'lr'))]
    nf_r = [u for u in want if not all(_derive.finite(r[u].get(k)) for k in ('main_ce', 'grad_norm', 'lr'))]
    if nf_c:
        v.append(f'non-finite tracked scalars at updates {nf_c[:8]}')
    if nf_r:
        v.append(f'non-finite reference scalars at updates {nf_r[:8]}')
    if v:
        return res
    res['compared_updates'] = len(want)
    bitwise = all(cm[u].get(k) == r[u].get(k) for u in want for k in TRACKED)
    ce = {u: _rel(cm[u]['main_ce'], r[u]['main_ce']) for u in want}
    gn = {u: _rel(cm[u]['grad_norm'], r[u]['grad_norm']) for u in want}
    a, b = (int(window[0]), int(window[1])) if window else (want[0], want[-1])
    win_u = [u for u in want if a <= u <= b]
    win = [ce[u] for u in win_u]
    first = want[0]
    positive = all(cm[u]['grad_norm'] > 0 and r[u]['grad_norm'] > 0 for u in win_u)
    log_bias = (sum(math.log(cm[u]['grad_norm'] / r[u]['grad_norm']) for u in win_u) / len(win_u)) if win_u and positive else None
    res.update(main_ce_rel_max=max(ce.values()), grad_norm_rel_max=max(gn.values()),
               main_ce_rel_mean_window=sum(win) / len(win) if win else None, bitwise=bitwise,
               grad_norm={'first_update': first, 'first_update_rel': gn[first], 'rel_max_all_steps': max(gn.values()),
                          'per_step_rel': [gn[u] for u in want], 'log_bias_window': log_bias, 'n_window': len(win_u)})
    aux = [(u, cm[u].get('aux_loss'), r[u].get('aux_loss')) for u in want]
    aux_rel = [_rel(c, q) for _, c, q in aux if _derive.finite(c) and _derive.finite(q) and q != 0]
    res['aux_rel_max'] = max(aux_rel) if aux_rel else None
    vanished = [u for u, c, q in aux if _derive.finite(q) and q != 0 and (not _derive.finite(c) or c == 0)]
    if vanished:
        v.append(f'aux loss vanished at updates {vanished[:8]}')
    lr_bad = [u for u in want if cm[u]['lr'] != r[u]['lr']]
    if lr_bad:
        v.append(f'LR differs from reference at updates {lr_bad[:8]}')
    tol = tolerance or {}
    lim = _lim(tol, 'main_ce_rel_max_per_step')
    if res['main_ce_rel_max'] > lim:
        v.append(f'main CE rel dev {res["main_ce_rel_max"]:.3g} > {lim}')
    lim = _lim(tol, 'main_ce_rel_mean_over_window')
    if win and res['main_ce_rel_mean_window'] > lim:
        v.append(f'main CE window mean rel dev {res["main_ce_rel_mean_window"]:.3g} > {lim}')
    lim = _lim(tol, 'grad_norm_rel_max_first_update')
    if gn[first] > lim:
        v.append(f'grad norm rel dev at update {first} {gn[first]:.3g} > {lim}')
    lim = _lim(tol, 'grad_norm_log_bias_window_max')
    if log_bias is None:
        v.append('grad norm window log bias not computable (missing window or non-positive grad norm)')
    elif abs(log_bias) > lim:
        v.append(f'grad norm window log bias {log_bias:+.3g} (|mean ln(gn_cand/gn_ref)| over {len(win_u)} updates) > {lim}')
    if bitwise and not lr_bad:
        res['class'] = 'E0'
    elif not v:
        res['class'] = 'E1'
    else:
        res['class'] = 'FAIL'
    res['allowed_ok'] = res['class'] in ('E0', 'E1', 'E2') and NUMERICS_ORDER[res['class']] <= NUMERICS_ORDER.get(allowed, 1)
    return res


# ---------------------------------------------------------------- invariant schema and the reference loader

def layout_diff(config: dict | None, effective) -> list[str]:
    """Requested L0 keys whose value differs in the layout that actually ran (receipt top-level effective_config:
    the trainer's schema defaults updated with the request config). 'env' and planner-only '_' keys are not layout."""
    if not isinstance(effective, dict):
        return ['effective_config missing']
    out = []
    for k, v in (config or {}).items():
        if k == 'env' or str(k).startswith('_'):
            continue
        if k not in effective or json.dumps(effective[k], sort_keys=True) != json.dumps(v, sort_keys=True):
            out.append(f'{k}: requested {_short(v)}, ran {_short(effective.get(k, "<absent>"))}')
    return out


def invariant_schema_problems(receipt: dict, task: dict, *, purpose: str, plan: str) -> list[str]:
    """Task-aware mandatory invariants of a non-probe run (every reference and every candidate). Field names are the
    training process's (runtime/receipt_writer + adapters/megatron_core/trainer): complete dtype census; loss, LR,
    trainable-parameter, model-config and frozen-zone digests; precision (dtype census + backends_flags +
    model_defined_fp32_params) and layout (top-level effective_config); per-update clipping totals; data identity
    (source manifest, order digest, consumed-pack digests, supervised-token vector); MoE router/capacity fields."""
    inv = receipt.get('invariants') if isinstance(receipt.get('invariants'), dict) else {}
    p = []
    for k in INV_INTS:
        if not (_int(inv.get(k)) and inv[k] >= 0):
            p.append(f'{k} missing or not a non-negative integer')
    if _int(inv.get('trainable_params')) and inv['trainable_params'] <= 0:
        p.append('trainable_params is 0')
    for k in INV_DIGESTS:
        if not HEX64.fullmatch(str(inv.get(k) or '')):
            p.append(f'{k} missing or not a sha256 digest')
    if inv.get('lr_schedule_digest') != inv.get('lr_schedule_expected_digest'):
        p.append('lr_schedule_digest != lr_schedule_expected_digest (LR differs from the task schedule)')
    census = inv.get('dtype_census')
    if not isinstance(census, dict):
        p.append('dtype_census missing')
    else:
        p += [f'dtype_census.{k} missing' for k in DTYPE_CONTRACT if not census.get(k)]
    if not (isinstance(inv.get('backends_flags'), dict) and inv['backends_flags']):
        p.append('backends_flags missing (precision flags)')
    if 'model_defined_fp32_params' not in inv:
        p.append('model_defined_fp32_params missing')
    if not isinstance(receipt.get('effective_config'), dict) or not receipt['effective_config']:
        p.append('effective_config missing (layout that ran)')
    vec = inv.get('supervised_tokens_per_update')
    if not (isinstance(vec, dict) and isinstance(vec.get(plan), list)):
        p.append(f'supervised_tokens_per_update[{plan}] missing')
    if not _dataindex.is_synthetic(task):
        d = inv.get('data') if isinstance(inv.get('data'), dict) else {}
        if not (d.get('source') and d.get('manifest_digest')):
            p.append('data.source / data.manifest_digest missing')
        if purpose != 'lifecycle':
            if not HEX64.fullmatch(str(inv.get('data_consumed_digest') or '')):
                p.append('data_consumed_digest missing')
            if not isinstance(inv.get('data_consumed_per_update'), list):
                p.append('data_consumed_per_update missing')
    if task_is_moe(task):
        if not (_int(inv.get('router_topk')) and inv['router_topk'] >= 1):
            p.append('router_topk missing (MoE)')
        if not (_int(inv.get('num_experts')) and inv['num_experts'] >= 2):
            p.append('num_experts missing (MoE)')
        if 'capacity_factor' not in inv:
            p.append('capacity_factor missing (MoE)')
    return p


def reference_problems(ref, *, task: dict | None = None, task_id: str | None = None, task_digest: str | None = None,
                       plan: str, steps, window, role: str | None = 'reference', purpose: str | None = None) -> list[str]:
    """Why `ref` cannot be the numerics reference (role='reference') or the reward parent (role=None) of a run of the
    task on input plan `plan` with `steps` updates and scoring `window`; [] when it can. The one fail-closed loader used
    by validate(), runner.load_reference / recheck_against and reward.evaluate."""
    if not isinstance(ref, dict):
        return ['reference receipt missing or not a JSON object']
    tid = task_id if task_id is not None else (task or {}).get('id')
    tdig = task_digest if task_digest is not None else (digest(task) if task is not None else None)
    if not verify_receipt_digest(ref):              # nothing in an untrusted receipt is read any further
        return ['harness signature does not verify (not validated by this harness, or altered)']
    p = []
    if ref.get('valid') is not True:
        p.append('receipt is not valid')
    if ref.get('task_id') != tid or tdig is None or ref.get('task_digest') != tdig:
        p.append(f'task {ref.get("task_id")}/{str(ref.get("task_digest"))[:12]} != {tid}/{str(tdig)[:12]}')
    if ref.get('input_plan') != plan:
        p.append(f'input plan {ref.get("input_plan")!r} != {plan!r}')
    if role is not None and ref.get('numerics_role') != role:
        p.append(f'numerics_role {ref.get("numerics_role")!r} != {role!r}')
    want_purpose = purpose or REF_PURPOSE.get(plan)
    if ref.get('purpose') != want_purpose:
        p.append(f'purpose {ref.get("purpose")!r} != {want_purpose!r}')
    if not _int(steps) or ref.get('steps') != steps or ref.get('steps_completed') != ref.get('steps'):
        p.append(f'{ref.get("steps_completed")}/{ref.get("steps")} updates; {steps} required')
    want_w = [int(window[0]), int(window[1])] if window else None
    if want_w is not None and list(ref.get('window') or []) != want_w:
        p.append(f'scoring window {ref.get("window")} != {want_w}')
    rows = ref_steps(ref) or []
    have = [q.get('update') for q in rows]
    if _int(steps):
        gaps = [u for u in range(1, steps + 1) if u not in have]
        if gaps:
            p.append(f'lacks updates {gaps[:8]} of 1..{steps}')
    if len(have) != len(set(have)):
        p.append('duplicate update rows')
    if want_w and any(u not in have for u in range(want_w[0], want_w[1] + 1)):
        p.append(f'does not cover the scoring window {want_w}')
    nonfinite = [q.get('update') for q in rows if not all(_derive.finite(q.get(k)) for k in ('main_ce', 'grad_norm', 'lr'))]
    if nonfinite:
        p.append(f'non-finite main_ce / grad_norm / lr at updates {nonfinite[:8]}')
    if task is not None:
        p += [f'invariants: {x}' for x in invariant_schema_problems(ref, task, purpose=ref.get('purpose'), plan=plan)]
    return p


def reference_invariants(c: Checks, inv: dict, per: list[dict], reference: dict, purpose: str,
                         deterministic: bool = False) -> None:
    """Mandatory equality with a usable (same task, same plan) reference; absent-on-both is not equality for the
    schema fields (the schema check catches it), and lifecycle step-count totals are checked against the lifecycle's
    own expected counts instead (STEP_COUNT_KEYS). A deterministic lifecycle (AMENDMENT-002) compares backends_flags
    without deterministic_algorithms / cudnn.deterministic; every other precision flag must still be equal."""
    rinv = reference.get('invariants') or {}
    skip = STEP_COUNT_KEYS if purpose == 'lifecycle' else ()
    for k in INV_EQUAL + INV_SAME_PLAN:
        if k in skip:
            continue
        mine, ref = inv.get(k), rinv.get(k)
        if k == 'backends_flags' and deterministic:
            mine, ref = strip_deterministic_flags(mine), strip_deterministic_flags(ref)
        c.add(f'ref.{k}', mine == ref, f'{_short(mine)} != reference {_short(ref)}')
    rs = {q['update']: q.get('supervised_tokens') for q in ref_steps(reference) or []}
    bad = [q['update'] for q in per if q['update'] in rs and q.get('supervised_tokens') != rs[q['update']]]
    c.add('ref.supervised_tokens_per_update', not bad, f'updates {bad[:8]} differ from reference')
    rc, mine = census_sets(rinv.get('dtype_census')), census_sets(inv.get('dtype_census'))
    c.add('ref.dtype_census', bool(rc) and mine == rc, f'{mine} != reference {rc}')


# ---------------------------------------------------------------- failure classification and synthesized receipts

def classify_failure(receipt: dict | None, collection: dict, stop: dict | None) -> str:
    logs = '\n'.join((n.get('log_tail') or '') for n in collection.get('nodes', []))
    exits = [n.get('exit') or {} for n in collection.get('nodes', [])]
    ev = (receipt or {}).get('events') or {}
    excs = ' '.join(str(e) for e in (ev.get('exceptions') or []))
    if (ev.get('alloc_ooms') or 0) > 0 or OOM_RE.search(excs) or OOM_RE.search(logs):
        return 'oom'
    if (ev.get('nccl_errors') or 0) > 0 or NCCL_RE.search(excs) or NCCL_RE.search(logs):
        return 'nccl'
    if stop:
        return {'heartbeat': 'heartbeat_lost', 'wall_cap': 'timeout', 'startup': 'startup_hang',
                'node_lost': 'node_lost', 'launch_failed': 'launch_error'}.get(stop.get('kind'), 'stopped')
    if any(e.get('reason') == 'deadline' for e in exits):
        return 'timeout'
    if any(e.get('reason') == 'wrapper_error' for e in exits):
        return 'launch_error'
    if any(e.get('exit_code') not in (0, None) for e in exits) or excs:
        return 'exception'
    return 'ok'


def synth_receipt(request, collection: dict, reason: str) -> dict:
    """Receipt for an attempt that died before rank 0 wrote one (valid=false, all keys present)."""
    r = {k: None for k in RECEIPT_REQUIRED}
    r.update(schema=SCHEMA_RECEIPT, run_id=request.run_id, attempt_id=request.attempt_id, purpose=request.purpose,
             task_id=request.task.get('id'), task_digest=digest(request.task), config_digest=request.config_digest,
             code_digest=request.code_digest, overlay_digest=None, input_plan=request.input_plan,
             world_size=request.world_size, nodes=list(request.nodes), steps=request.steps,
             window=list(request.window) if request.window else None, steps_completed=0, step_time=None, per_step=[],
             invariants={}, memory={}, events={'alloc_retries': 0, 'alloc_ooms': 0, 'nccl_errors': 0, 'nan_inf_steps': 0,
                                               'exceptions': []}, phases={}, env={}, valid=False,
             invalid_reasons=[f'no receipt written by the training process ({reason})'], signals_digest=None,
             synthesized_by_harness=True)
    return r


# ---------------------------------------------------------------- receipt checks

def check_identity(c: Checks, receipt: dict, request) -> None:
    want = {'run_id': request.run_id, 'attempt_id': request.attempt_id, 'purpose': request.purpose,
            'task_id': request.task.get('id'), 'config_digest': request.config_digest, 'code_digest': request.code_digest,
            'world_size': request.world_size, 'steps': request.steps, 'input_plan': request.input_plan,
            'task_digest': digest(request.task)}
    for k, v in want.items():
        got = receipt.get(k)
        c.add(f'identity.{k}', got == v, f'receipt {got!r} != request {v!r}' if got != v else '')
    rw = receipt.get('window')
    want_w = list(request.window) if request.window else None
    c.add('identity.window', (list(rw) if rw else None) == want_w, f'{rw} != {want_w}')
    # Authorization-bearing request fields echoed by the training process (Codex R2 finding 2) ...
    echo = {'request_digest': digest(request.to_dict()), 'numerics_role': getattr(request, 'numerics_role', None),
            'data_manifest': request.data_manifest, 'effective_overlay_digest': getattr(request, 'effective_overlay_digest', None)}
    for k, v in echo.items():
        got = receipt.get(k, '<absent>')
        c.add(f'identity.{k}', k in receipt and got == v, f'receipt {_short(got)} != trusted request {_short(v)}')
    # ... and the effective L1 overlay set the runtime actually loaded (finding 4c)
    eff = getattr(request, 'effective_overlay_digest', None)
    c.add('identity.overlay_digest', bool(eff) and receipt.get('overlay_digest') == eff,
          f'runtime effective overlay digest {_short(receipt.get("overlay_digest"))} != pinned {_short(eff)}')


def check_lifecycle(c: Checks, receipt: dict, derived: dict, collection: dict, request, save_at: int,
                    resume_steps: int, tier: dict | None = None) -> dict:
    """AMENDMENT-001 lifecycle rules with the tier's gates (AMENDMENT-002 frozen table, numerics_tier_check):
    restored state bitwise equal to the saved state (receipt lifecycle.state_exact); deterministic kernels as the tier
    rules ('required': lifecycle.deterministic true with lifecycle.deterministic_overrides; 'unavailable': the receipt
    says lifecycle.deterministic_unavailable); the first resumed update (signals_resume/, step save_at+1) matches the
    uninterrupted run within first_resumed_update; later resumed updates match within the task's per-step CE tolerance
    with LR, supervised / physical tokens and clip calls exactly equal (grad norm reported only, as is the count of
    bitwise-identical resumed updates); export_ok / forward_ok and the forward metrics only when the request enables
    export / forward_compare. Returns the report."""
    task = TaskSpec(raw=request.task).raw
    chk = tier or numerics_tier_check(task)
    table = chk.get('table') or {}
    lt = table.get('lifecycle_tolerance') or lifecycle_tolerance(task)
    tol = table.get('tolerance') or (task.get('numerics') or {}).get('tolerance') or {}
    flags = receipt.get('lifecycle') if isinstance(receipt.get('lifecycle'), dict) else {}
    resumed = _derive.derive(collection.get('signals_resume_text') or {}, world_size=request.world_size,
                             steps=resume_steps, window=None, start_step=save_at + 1)
    for p in resumed['problems']:
        c.add('lifecycle.resume_signals', False, p)
    c.add('lifecycle.resume_complete', resumed['steps_completed'] == resume_steps,
          f'{resumed["steps_completed"]}/{resume_steps} resumed updates in signals_resume')
    if lt.get('state_digest_exact', True):
        c.add('lifecycle.state_exact', flags.get('state_exact') is True,
              f'restored state not bitwise equal to the saved state (lifecycle.state_exact={flags.get("state_exact")!r})')
    rule = lt.get('deterministic')
    det, why, ov = flags.get('deterministic'), flags.get('deterministic_unavailable'), flags.get('deterministic_overrides')
    if rule == 'required':                         # no opting out of deterministic kernels to dodge the tight gates
        c.add('lifecycle.deterministic', det is True and isinstance(ov, dict) and bool(ov),
              f'tier {chk.get("tier")} requires a deterministic lifecycle with its overrides recorded '
              f'(lifecycle.deterministic={det!r}, deterministic_overrides={_short(ov)})')
    elif rule == 'unavailable':
        c.add('lifecycle.deterministic', det is not True and isinstance(why, str) and bool(why.strip()),
              f'tier {chk.get("tier")} has no deterministic kernels: the receipt must state '
              f'lifecycle.deterministic_unavailable (deterministic={det!r}, deterministic_unavailable={_short(why)})')
    elif rule is not None:
        c.add('lifecycle.deterministic', False, f'unknown lifecycle deterministic rule {rule!r}')
    first_tol = lt.get('first_resumed_update') or LIFECYCLE_TOLERANCE_DEFAULT['first_resumed_update']
    lim_ce1, lim_gn1 = _lim(first_tol, 'main_ce_rel_max'), _lim(first_tol, 'grad_norm_rel_max')
    lim_ce = _lim(tol, 'main_ce_rel_max_per_step')
    orig = {p['step']: p for p in derived['per_step'] if p['step'] > save_at}
    back = {p['step']: p for p in resumed['per_step']}
    steps = list(range(save_at + 1, save_at + resume_steps + 1))
    report = {'steps': [], 'first_resumed_update': save_at + 1, 'state_exact': flags.get('state_exact'),
              'training_process_scalars_match': flags.get('resume_scalars_match'), 'tier': chk.get('tier'),
              'first_resumed_tolerance': {'main_ce_rel_max': lim_ce1, 'grad_norm_rel_max': lim_gn1},
              'deterministic': {'rule': rule, 'deterministic': det, 'unavailable': why, 'overrides': ov},
              'resumed_updates': resume_steps, 'resumed_bitwise_updates': None,
              'training_process_bitwise_updates': flags.get('resumed_bitwise_updates')}
    gaps = [s for s in steps if s not in orig or s not in back]
    nonfinite = [s for s in steps if s not in gaps and not all(_derive.finite(x.get(k)) for x in (orig[s], back[s])
                                                              for k in ('main_ce', 'grad_norm', 'lr'))]
    if gaps or nonfinite:
        c.add('lifecycle.continuation', False, f'resumed steps {gaps[:8]} missing and {nonfinite[:8]} non-finite in '
              'signals / signals_resume')
        return report
    first_bad, later_bad, exact_bad, aux_bad = [], [], [], []
    bitwise = 0
    for s in steps:
        p, q = back[s], orig[s]
        ce_rel, gn_rel = _rel(p['main_ce'], q['main_ce']), _rel(p['grad_norm'], q['grad_norm'])
        exact = [k for k in LIFECYCLE_EXACT if p.get(k) != q.get(k)]
        same = all(p.get(k) == q.get(k) for k in TRACKED)
        bitwise += same
        report['steps'].append({'step': s, 'main_ce_rel': ce_rel, 'grad_norm_rel': gn_rel, 'exact_mismatch': exact,
                                'bitwise': same})
        if s == steps[0]:
            if ce_rel > lim_ce1 or gn_rel > lim_gn1:
                first_bad.append(f'step {s}: CE rel {ce_rel:.3g} (<= {lim_ce1}), grad norm rel {gn_rel:.3g} (<= {lim_gn1})')
        elif ce_rel > lim_ce:
            later_bad.append(f'step {s}: CE rel {ce_rel:.3g} > {lim_ce}')
        if exact:
            exact_bad.append(f'step {s}: {exact}')
        if _derive.finite(q.get('aux_loss')) and q['aux_loss'] != 0 and not (_derive.finite(p.get('aux_loss')) and p['aux_loss'] != 0):
            aux_bad.append(s)
    report['grad_norm_rel_max_later'] = max((x['grad_norm_rel'] for x in report['steps'][1:]), default=None)
    report['resumed_bitwise_updates'] = bitwise          # AMENDMENT-002: report only (deterministic tiers expect all)
    c.add('lifecycle.first_resumed_update', not first_bad, '; '.join(first_bad))
    c.add('lifecycle.continuation', not later_bad and not aux_bad,
          '; '.join(later_bad[:6] + ([f'aux loss vanished at steps {aux_bad[:8]}'] if aux_bad else [])))
    c.add('lifecycle.exact_scalars', not exact_bad, 'LR / tokens / clip calls differ from the uninterrupted run: '
          + '; '.join(exact_bad[:6]))
    lc = request.lifecycle or {}
    need = ['resumed_ok'] + (['export_ok'] if lc.get('export', True) else []) + \
        (['forward_ok'] if lc.get('forward_compare', True) else [])
    missing = [k for k in need if flags.get(k) is not True]
    c.add('lifecycle.flags', not missing, f'lifecycle flags not true: {missing} ({_short(flags)})')
    if lc.get('forward_compare', True):
        fwd = flags.get('forward') if isinstance(flags.get('forward'), dict) else {}
        ft = lt.get('forward') or LIFECYCLE_TOLERANCE_DEFAULT['forward']
        rel, top1 = fwd.get('rel_l2'), fwd.get('top1_agreement')
        ok = (_derive.finite(rel) and rel <= _lim(ft, 'rel_l2_max') and _derive.finite(top1)
              and top1 >= float(ft.get('top1_agreement_min', 1.0)))
        report['forward'] = {'rel_l2': rel, 'top1_agreement': top1}
        c.add('lifecycle.forward', ok, f'forward compare rel_l2 {rel!r} (<= {ft.get("rel_l2_max")}), top-1 {top1!r} '
              f'(>= {ft.get("top1_agreement_min")})')
    return report


def check_invariants(c: Checks, receipt: dict, derived: dict, request, reference: dict | None,
                     expected_steps: int | None = None) -> None:
    """Invariants vs the task spec, the mandatory schema, the layout binding and (with a usable reference, already
    authenticated by reference_problems) mandatory equality with the reference."""
    task = TaskSpec(raw=request.task)
    wl = task.raw['workload']
    inv = receipt.get('invariants') if isinstance(receipt.get('invariants'), dict) else {}
    per = derived['per_step']
    probe = request.purpose == 'probe'
    want_steps = expected_steps or request.steps
    c.add('steps.complete', derived['steps_completed'] == want_steps,
          f'{derived["steps_completed"]}/{want_steps} updates in signals')
    c.add('world_size.task', request.world_size == task.gpus, f'world {request.world_size} vs task gpus {task.gpus}')
    if not probe:
        schema = invariant_schema_problems(receipt, task.raw, purpose=request.purpose, plan=request.input_plan)
        c.add('inv.schema', not schema, '; '.join(schema[:12]) + (f' (+{len(schema) - 12} more)' if len(schema) > 12 else ''))
    c.add('inv.gbs_packs', inv.get('gbs_packs') == wl['packs_per_update'], f'{inv.get("gbs_packs")} vs {wl["packs_per_update"]}')
    c.add('inv.physical_tokens_per_update', inv.get('physical_tokens_per_update') == wl['physical_tokens_per_update'],
          f'{inv.get("physical_tokens_per_update")} vs {wl["physical_tokens_per_update"]}')
    bad_phys = [p['update'] for p in per if p.get('physical_tokens') != wl['physical_tokens_per_update']]
    c.add('signals.physical_tokens', not bad_phys, f'updates {bad_phys[:8]} differ from {wl["physical_tokens_per_update"]}')
    if per:
        sup = [p.get('supervised_tokens') for p in per]
        ints = all(_int(x) for x in sup)
        c.add('inv.supervised_tokens_total', ints and sum(sup) == inv.get('supervised_tokens_total'),
              f'signals sum {sum(x for x in sup if _int(x))} vs receipt {inv.get("supervised_tokens_total")!r}')
        clip = [p.get('clip_calls') for p in per]
        clip_ok = all(_int(x) and x >= 1 for x in clip) and len(set(clip)) == 1
        c.add('signals.clip_calls', clip_ok, f'clip calls per update {clip[:8]}: clipping skipped, missing or irregular '
              f'(INTERFACES: a constant count >= 1 per update, 2 with the offloaded hybrid optimizer)')
        tot = inv.get('clip_calls_total')
        c.add('inv.clip_calls_total', clip_ok and tot == sum(clip), f'receipt clip_calls_total {tot!r} vs signals {clip[:4]}...')
    opt = inv.get('optimizer_steps')
    c.add('inv.optimizer_steps', opt == derived['steps_completed'], f'{opt} optimizer steps for {derived["steps_completed"]} updates')
    if per:                     # AMENDMENT-001: no update may skip part of its backward pass on any rank
        mb = derived.get('microbatches') or {}
        want_mb = expected_microbatches(wl['packs_per_update'], request.world_size, receipt.get('effective_config'))
        seen = mb.get('values') or {}
        ok = want_mb is not None and not mb.get('n_missing') and set(seen) == {str(want_mb)}
        c.add('signals.microbatches', ok, f'per-rank backward micro-batches {seen} (missing on {mb.get("n_missing")} '
              f'rank-updates, e.g. {mb.get("missing", [])[:4]}); expected {want_mb} on every rank and update '
              f'(packs_per_update / (DP x micro_batch_size))')
    obj = task.raw.get('objective') if isinstance(task.raw.get('objective'), dict) else {}
    if task_is_moe(task.raw):
        want_cf = (obj.get('moe') or {}).get('capacity_factor')
        c.add('inv.capacity_factor', 'capacity_factor' in inv and inv.get('capacity_factor') == want_cf,
              f'{inv.get("capacity_factor", "<absent>")} vs task {want_cf}')
    raw_census = inv.get('dtype_census') if isinstance(inv.get('dtype_census'), dict) else {}
    model_fp32 = inv.get('model_defined_fp32_params') or []
    # The fixed census (Codex R3 HIGH) reports model-defined fp32 tensors apart (dtype_census.model_fp32): fp32 in the
    # parameter census is then never an exemption. Receipts of the pre-fix census keep the bounded allowance.
    legacy_fp32 = MODEL_FP32_KEY not in raw_census
    for k, want in DTYPE_CONTRACT.items():
        if not raw_census.get(k):
            if not probe:
                c.add(f'dtype.{k}', False, f'dtype census has no {k} entry (incomplete census)')
            continue
        got = dtype_set(raw_census[k])
        allowed = {want} | ({'fp32'} if k == 'param' and model_fp32 and legacy_fp32 else set())
        extra = got - allowed
        if k == 'param' and model_fp32 and legacy_fp32 and isinstance(raw_census[k], dict):
            n32 = sum(n for d, n in raw_census[k].items() if norm_dtype(d) == 'fp32')
            if n32 > (len(model_fp32) if isinstance(model_fp32, list) else int(model_fp32)):
                extra.add(f'fp32 x{n32} > model-defined {model_fp32 if isinstance(model_fp32, int) else len(model_fp32)}')
        c.add(f'dtype.{k}', want in got and not extra, f'{sorted(got)} vs contract {want} (silent precision change)')
    bad_fp32 = model_fp32_problems(model_fp32, raw_census.get(MODEL_FP32_KEY))
    c.add('dtype.model_defined_fp32', not bad_fp32, '; '.join(bad_fp32[:4]))
    eff = receipt.get('effective_config')
    if isinstance(eff, dict) or not probe:
        diff = layout_diff(request.config, eff)
        c.add('inv.layout', not diff, '; '.join(diff[:6]) + ' (the layout that ran is not the requested one)')
    ev = receipt.get('events') or {}
    nan = derived['nan_inf_updates']
    c.add('stability.nan_inf', not nan and not (ev.get('nan_inf_steps') or 0), f'non-finite at updates {nan[:8]}')
    if reference:
        reference_invariants(c, inv, per, reference, request.purpose,
                             deterministic=deterministic_lifecycle(receipt, request.purpose))


def check_receipt_vs_signals(c: Checks, receipt: dict, derived: dict, lenient_counts: bool = False) -> bool:
    """The receipt summary must agree with the signals it claims to summarise (tamper / lost-signal detector).
    Returns whether the receipt binds the collected signal files (signals_digest present, well formed and equal)."""
    st = receipt.get('step_time') or {}
    dst = derived['step_time'] or {}
    for k in ('mean_window_max_rank_s', 'median_window_max_rank_s'):
        if _derive.finite(st.get(k)) and _derive.finite(dst.get(k)):
            c.add(f'receipt.step_time.{k}', _rel(st[k], dst[k]) <= 1e-6, f'receipt {st[k]} vs signals {dst[k]}')
    if _derive.finite(st.get('rsd_window')) and _derive.finite(dst.get('rsd_window')):
        c.add('receipt.step_time.rsd_window', abs(st['rsd_window'] - dst['rsd_window']) <= 1e-6 + 0.02 * dst['rsd_window'],
              f'receipt {st["rsd_window"]} vs signals {dst["rsd_window"]}', severity='warn')
    rows = {p.get('step'): p for p in receipt.get('per_step') or []}
    bad = []
    for p in derived['per_step']:
        q = rows.get(p['step'])
        if q is None:
            continue
        for k in ('max_rank_whole_s', 'main_ce', 'grad_norm', 'lr'):
            if _derive.finite(q.get(k)) and _derive.finite(p.get(k)) and _rel(q[k], p[k]) > 1e-6:
                bad.append((p['step'], k))
    c.add('receipt.per_step', not bad, f'receipt per_step disagrees with signals at {bad[:6]}')
    sd = receipt.get('signals_digest')
    if not (isinstance(sd, str) and HEX64.fullmatch(sd)):
        bound = c.add('receipt.signals_digest', False,
                      f'signals_digest {_short(sd)} is not a sha256 hex digest: nothing binds the scored signal files')
    else:
        bound = c.add('receipt.signals_digest', sd == derived['signals_digest'],
                      'signal files differ from those the training process digested for its receipt (sha256 of the '
                      'rank files concatenated in increasing rank order)')
    if receipt.get('steps_completed') is not None:
        c.add('receipt.steps_completed', receipt['steps_completed'] == derived['steps_completed'],
              f'receipt {receipt["steps_completed"]} vs signals {derived["steps_completed"]}',
              severity='warn' if lenient_counts else 'invalid')
    return bool(bound)


def external_clock(collection: dict, phases: dict) -> dict | None:
    """Node-0 wrapper wall (its own monotonic clock) vs the process's phases.total, with what only the wrapper sees
    accounted explicitly from wrapper UTC anchors and the receipt's mtime (kernel clock): GPU checks before launch,
    interpreter/torchrun startup until the first heartbeat appeared, teardown after the receipt, post-exit checks."""
    nodes = collection.get('nodes') or []
    ex = ((nodes[0] or {}).get('exit') if nodes else None) or {}
    wall, total = ex.get('wall_s'), phases.get('total')
    if not (_derive.finite(wall) and _derive.finite(total)) or ex.get('synthesized_by_driver'):
        return None

    def span(a, b):
        x, y = ex.get(a), ex.get(b)
        return max(0.0, y - x) if _derive.finite(x) and _derive.finite(y) else 0.0
    rm, child_exit = collection.get('receipt_mtime'), ex.get('child_exit_utc_s')
    comp = {'pre_launch_s': span('wrapper_start_utc_s', 'child_start_utc_s'),
            'startup_before_heartbeat_s': span('child_start_utc_s', 'first_heartbeat_utc_s'),
            'teardown_after_receipt_s': max(0.0, child_exit - rm) if _derive.finite(rm) and _derive.finite(child_exit) else 0.0,
            'post_exit_s': span('child_exit_utc_s', 'wrapper_end_utc_s')}
    explained = sum(comp.values())
    unexplained = wall - total - explained
    tol = max(EXTERNAL_CLOCK_ABS, EXTERNAL_CLOCK_REL * wall)
    return {'wall_s': wall, 'process_total_s': total, 'explained_s': explained, 'components': comp,
            'anchors': bool(_derive.finite(child_exit) and _derive.finite(rm)), 'unexplained_s': unexplained,
            'tolerance_s': tol, 'ok': abs(unexplained) <= tol}


def check_clocks(c: Checks, receipt: dict, derived: dict, collection: dict, request) -> tuple[dict, dict]:
    """(a) per-step UTC span vs whole_update_s, (b) UTC training span vs phases.train, (c) wrapper wall vs phases.total.
    Screens and scores fail closed on every missing piece."""
    clk = derived['clock']
    phases = receipt.get('phases') or {}
    strict = request.purpose in STRICT_CLOCK_PURPOSES
    info = {'per_step': {k: clk[k] for k in ('present', 'complete', 'rows', 'rows_with_utc', 'missing', 'disordered',
                                              'outliers', 'max_abs_dev_s')}}
    if not clk['complete']:
        c.add('clock_fields', False, f'utc_start_s/utc_end_s missing on {clk["rows"] - clk["rows_with_utc"]} of '
              f'{clk["rows"]} rank-update rows (ranks {sorted(clk["missing"], key=int)[:8]}): the independent per-step '
              'clock is incomplete', severity='invalid' if REQUIRE_UTC_FIELDS else 'warn')
    dis = clk['disordered']
    c.add('clock_order', not dis, f'UTC stamps not strictly positive and ordered on ranks {sorted(dis, key=int)[:8]} '
          f'(rank {next(iter(sorted(dis, key=int)), None)}: steps {dis.get(next(iter(sorted(dis, key=int)), ""), [])[:6]})')
    win = set()
    if strict and request.window:
        base = derived['step_base']
        win = {base + u - 1 for u in range(int(request.window[0]), int(request.window[1]) + 1)}
    bad = {r: s for r, s in clk['outliers'].items()
           if any(x in win for x in s) or len([x for x in s if x not in win]) > CLOCK_MAX_OUTLIERS}
    first = next(iter(sorted(bad, key=int)), None)
    c.add('clock_mismatch', not bad, f'ranks {sorted(bad, key=int)[:8]}: UTC span and whole_update_s differ by more than '
          f'{derive_pct()} ({"any step inside the scoring window or " if win else ""}more than {CLOCK_MAX_OUTLIERS} '
          f'step(s) per rank; rank {first}: steps {bad.get(first, [])[:6]}; max deviation {clk["max_abs_dev_s"]})')
    train_utc, proc = clk.get('train_utc_s'), phases.get('train')
    wallclock = {'train_utc_s': train_utc, 'process_train_s': proc if _derive.finite(proc) else None}
    if _derive.finite(train_utc) and _derive.finite(proc):
        tol = TRAIN_CLOCK_REL * max(train_utc, proc) + TRAIN_CLOCK_ABS
        info['train'] = {'diff_s': abs(train_utc - proc), 'tolerance_s': tol}
        c.add('train_clock_mismatch', abs(train_utc - proc) <= tol,
              f'UTC training span {train_utc:.1f} s vs phases.train {proc:.1f} s (tolerance {tol:.1f} s)',
              severity='warn' if request.purpose == 'lifecycle' else 'invalid')
    else:
        c.add('train_clock', False, 'UTC training span or phases.train unavailable: training-span clock unchecked',
              severity='invalid' if strict else 'warn')
    ext = external_clock(collection, phases)
    if ext is None:
        c.add('external_clock', False, 'node wrapper wall or phases.total unavailable: external clock unchecked',
              severity='invalid' if strict else 'warn')
    else:
        info['external'] = ext
        c.add('external_clock_mismatch', ext['ok'],
              f'wrapper wall {ext["wall_s"]:.0f} s - phases.total {ext["process_total_s"]:.0f} s - explained '
              f'{ext["explained_s"]:.0f} s = {ext["unexplained_s"]:.0f} s unexplained (tolerance {ext["tolerance_s"]:.0f} s)')
    return wallclock, info


def derive_pct() -> str:
    return f'{_derive.CLOCK_REL:.0%} + {_derive.CLOCK_ABS * 1000:.0f} ms'


def gpu_hours(collection: dict, world_size: int, gpus_per_node: int) -> tuple[float | None, float | None]:
    walls = [((n.get('exit') or {}).get('wall_s')) for n in collection.get('nodes', [])]
    walls = [w for w in walls if isinstance(w, (int, float))]
    if not walls:
        return None, None
    wall = max(walls)
    return wall, wall * world_size / 3600.0


def validate(collection: dict, request, reference: dict | None, *, stop: dict | None = None,
             harness_info: dict | None = None, data: dict | None = None) -> dict:
    receipt = collection.get('receipt')
    if not isinstance(receipt, dict) or '__unreadable__' in (receipt or {}):
        receipt = synth_receipt(request, collection, 'missing or unreadable receipt.json')
    c = Checks()
    shape = check_receipt_shape(receipt)
    c.add('receipt.shape', not shape, '; '.join(shape))
    life = request.purpose == 'lifecycle'
    lc = request.lifecycle or {}
    save_at = int(lc.get('save_at', request.steps)) if life else None
    resume_steps = int(lc.get('resume_steps', request.task['workload'].get('resume_continuation_steps', 10))) if life else 0
    probe = request.purpose == 'probe'
    # lifecycle: phase 1 runs uninterrupted to save_at + resume_steps (saving at save_at), signals/ (runtime trainer)
    derived = _derive.derive(collection.get('signals_text') or {}, world_size=request.world_size,
                             steps=(save_at + resume_steps) if life else request.steps, window=None if life else request.window,
                             profile_window=None if probe else request.window)
    for p in derived['problems']:
        c.add('signals', False, p)
    check_identity(c, receipt, request)
    exits = [n.get('exit') for n in collection.get('nodes', [])]
    c.add('nodes.exit', all(e is not None and e.get('exit_code') == 0 for e in exits),
          f'exit codes {[(e or {}).get("exit_code") for e in exits]} reasons {[(e or {}).get("reason") for e in exits]}')
    c.add('nodes.released', all((e or {}).get('release', {}).get('process_group_gone', False) for e in exits),
          'a node still has processes of this attempt', severity='warn')
    hb = collection.get('heartbeat') or {}
    c.add('heartbeat.done', hb.get('phase') == 'done', f'final phase {hb.get("phase")!r}')
    c.add('stop', stop is None and not collection.get('stop'),
          f'stopped: {(stop or collection.get("stop") or {}).get("reason")}')
    role = getattr(request, 'numerics_role', 'candidate')
    ref_problems = []
    if reference is not None and not probe:
        ref_problems = reference_problems(reference, task=request.task, plan=request.input_plan, steps=request.steps,
                                          window=request.window)
        c.add('ref.usable', not ref_problems, '; '.join(ref_problems[:8]))
    usable = reference if (reference is not None and not probe and not ref_problems) else None
    wallclock = {'train_utc_s': derived['clock']['train_utc_s'], 'process_train_s': None}
    clock_info = {'per_step': {k: derived['clock'][k] for k in ('present', 'complete', 'rows', 'rows_with_utc', 'missing',
                                                                 'disordered', 'outliers', 'max_abs_dev_s')}}
    bound = False
    if not receipt.get('synthesized_by_harness'):
        check_invariants(c, receipt, derived, request, usable, expected_steps=(save_at + resume_steps) if life else None)
        bound = check_receipt_vs_signals(c, receipt, derived, lenient_counts=life)
        _dataindex.check(c, receipt, derived, request, data)
        wallclock, clock_info = check_clocks(c, receipt, derived, collection, request)
    task = TaskSpec(raw=request.task)
    num = task.raw.get('numerics') or {}
    tier = numerics_tier_check(task.raw)            # AMENDMENT-002: the gates of the task's tier, spec equal to it
    if not probe:
        c.add('numerics.tier', not tier['problems'], f'tier {tier["tier"]}: ' + '; '.join(tier['problems'][:6]))
    life_report = None
    if life:                    # Codex R2 finding 3: no continuity is derived from unbound signals_resume/ files
        want_rd = (receipt.get('lifecycle') or {}).get('signals_resume_digest') if isinstance(receipt.get('lifecycle'), dict) else None
        got_rd = _derive.signals_digest(collection.get('signals_resume_text') or {})
        resume_bound = isinstance(want_rd, str) and bool(HEX64.fullmatch(want_rd)) and want_rd == got_rd
        c.add('lifecycle.signals_resume_digest', resume_bound,
              f'lifecycle.signals_resume_digest {_short(want_rd)} does not bind the collected signals_resume files ({got_rd[:16]})')
        if resume_bound:
            life_report = check_lifecycle(c, receipt, derived, collection, request, save_at, resume_steps, tier=tier)
    sc = self_check(derived['per_step'])
    c.add('self_check.repeated_pairs', not sc['repeated_pairs'],
          f'adjacent updates with bitwise-equal (main_ce, grad_norm): {sc["repeated_pairs"][:8]}')
    tol = (tier['table'] or {}).get('tolerance') or num.get('tolerance') or {}
    numerics = {'compared': False, 'class': None, 'skipped': 'probe'} if probe else compare_numerics(
        derived['per_step'], ref_steps(usable), tol, request.window,
        num.get('numerics_class_allowed', 'E1'), role=role, steps=request.steps)
    numerics['tier'], numerics['tolerance'] = tier['tier'], tol         # the gates this receipt was judged with
    if tier['problems']:
        numerics['tier_problems'] = tier['problems'][:8]
    if reference is not None and not probe:
        numerics['reference_attempt_id'] = reference.get('attempt_id') if isinstance(reference, dict) else None
        numerics['reference_plan'] = reference.get('input_plan') if isinstance(reference, dict) else None
        if ref_problems:
            numerics['reference_problems'] = ref_problems[:8]
    if not probe:
        numerics['role'] = role
    numerics['self_check'] = sc
    numerics['sharpness'] = sharpness(derived['per_step'], (task.raw.get('objective') or {}).get('clip_grad')
                                      if isinstance(task.raw.get('objective'), dict) else None)
    if numerics.get('compared'):
        c.add('numerics.class', numerics.get('allowed_ok', False),
              f'{numerics.get("class")} not within {numerics.get("allowed")}: {numerics.get("violations")}')
    elif not probe and not numerics.get('is_reference'):
        c.add('numerics.verified', False, 'candidate run without a comparable numerics reference '
              f'({"; ".join(numerics.get("violations") or []) or "not compared"}); only numerics_role="reference" runs '
              'may lack one')
    failure = 'ok' if receipt.get('valid') and not c.invalid else classify_failure(receipt, collection, stop)
    if failure == 'ok' and (not receipt.get('valid') or c.invalid):
        failure = 'invalid'
    wall, gpuh = gpu_hours(collection, request.world_size, request.gpus_per_node)
    out = dict(receipt)
    reported = {k: receipt.get(k) for k in ('step_time', 'memory', 'steps_completed', 'valid')}
    reported['per_step_digest'] = digest(json_safe(receipt.get('per_step'))) if receipt.get('per_step') is not None else None
    out['step_time'] = derived['step_time'] if bound else None          # nothing scored from unbound signal files
    out['per_step'] = derived['per_step']
    out['steps_completed'] = derived['steps_completed']
    out['profile'] = None if (probe or not bound) else derived['profile']
    out['wallclock'] = wallclock
    out['numerics_role'] = role                     # inside the signed part: a flipped harness flag cannot make a reference
    out['memory'] = derived['memory'] if derived['memory']['per_rank'] else (receipt.get('memory') or {})
    events = dict(receipt.get('events') or {})
    events['alloc_ooms'] = max(events.get('alloc_ooms') or 0, derived['events']['alloc_ooms'])
    events['alloc_retries'] = max(events.get('alloc_retries') or 0, derived['events']['alloc_retries'])
    events['nan_inf_steps'] = max(events.get('nan_inf_steps') or 0, len(derived['nan_inf_updates']))
    out['events'] = events
    out['valid'] = bool(receipt.get('valid')) and not c.invalid
    out['invalid_reasons'] = list(receipt.get('invalid_reasons') or []) + [f'harness: {x}' for x in c.invalid]
    inv = receipt.get('invariants') if isinstance(receipt.get('invariants'), dict) else {}
    eff = receipt.get('effective_config')
    out['harness'] = {
        'schema': 'infrarsi-harness-validation-v1', 'validated_utc': utc_now(), 'failure_kind': failure,
        'reported': reported, 'derived': {k: derived[k] for k in ('ranks_present', 'step_base', 'tail_step_ratio',
                                                                   'window_complete', 'signals_digest', 'events',
                                                                   'nan_inf_updates')},
        'bindings': {'signals_bound': bound, 'layout_digest': digest(eff) if isinstance(eff, dict) else None,
                     'precision_digest': digest({'dtype_census': census_sets(inv.get('dtype_census')),
                                                 'backends_flags': strip_deterministic_flags(inv.get('backends_flags'))
                                                 if deterministic_lifecycle(receipt, request.purpose)
                                                 else inv.get('backends_flags'),
                                                 'model_defined_fp32_params': inv.get('model_defined_fp32_params')})},
        'checks': c.items, 'warnings': c.warnings, 'numerics': numerics, 'stop': stop or collection.get('stop'),
        'exits': exits, 'heartbeat_final': hb, 'wall_s': wall, 'gpu_hours': gpuh, 'clocks': clock_info,
        'lifecycle': life_report, 'microbatches': derived.get('microbatches'),
        'data_index': {k: v for k, v in (data or {}).items() if k not in ('order', 'supervised_tokens_per_update')},
        'request_digest': digest(request.to_dict()), **(harness_info or {})}
    out = json_safe(out)
    out.pop(_signing.SIGNATURE_FIELD, None)                 # a raw receipt never carries a harness signature
    return _signing.sign(out)


def verify_receipt_digest(receipt: dict) -> bool:
    """True only when the receipt's HMAC (harness/signing.py) verifies with this PC's key over the whole receipt."""
    return _signing.verify(receipt)


def gpu_seconds(receipt: dict) -> float:
    h = receipt.get('harness') or {}
    return (h.get('gpu_hours') or 0.0) * 3600.0


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
