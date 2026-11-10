"""The checklist view of the core Megatron-Core harness (reward.evaluate) and of external-environment receipts.

The core harness keeps its own evidence handling (signed receipts, signals re-derivation, invariants, per-tier
numerics gates predeclared by each benchmark task); these functions only regroup its findings into the generic
checklist items so every path returns the same verdict shape and the same feedback to the search.
"""
from __future__ import annotations

from . import checklist as C

# harness check name prefix -> checklist item (first match wins; order matters)
HARNESS_MAP = (
    ('self_check.', 'no_errors'), ('stability.', 'no_errors'),
    ('inv.optimizer_steps', 'optimizer_steps'), ('inv.clip_calls', 'optimizer_steps'), ('signals.clip_calls', 'optimizer_steps'),
    ('ref.lr_schedule', 'optimizer_steps'),
    ('ref.model_config', 'same_model'), ('ref.trainable', 'same_model'), ('ref.loss_definition', 'same_model'),
    ('inv.capacity_factor', 'same_model'),
    ('dtype.', 'precision'), ('ref.dtype', 'precision'), ('ref.backends', 'precision'),
    ('data.order', 'same_data'), ('data.consumed', 'same_data'), ('data.index', 'same_data'), ('data.manifest', 'same_data'),
    ('data.request_manifest', 'same_data'),
    ('inv.', 'same_workload'), ('data.', 'same_workload'), ('signals.physical', 'same_workload'),
    ('signals.microbatches', 'same_workload'), ('ref.supervised', 'same_workload'),
    ('clock', 'timing_consistent'), ('external_clock', 'timing_consistent'), ('train_clock', 'timing_consistent'),
    ('numerics.', 'loss_within_tolerance'),
    ('lifecycle.', 'resume_continues'),
)

# reward.evaluate hard item -> checklist item
HARD_MAP = {
    'receipt_signature': 'completes', 'receipt_valid': 'completes', 'complete_updates': 'completes',
    'step_time_derived': 'completes', 'parent_receipt': 'completes', 'reference_receipt': 'completes',
    'numerics_tier': 'loss_within_tolerance', 'numerics_compared': 'loss_within_tolerance',
    'numerics_class_matches_allowed': 'loss_within_tolerance', 'loss_guard': 'loss_within_tolerance',
    'repeated_adjacent_scalars': 'no_errors', 'alloc_ooms_max': 'no_errors', 'nccl_errors_max': 'no_errors',
    'stability': 'no_errors', 'inter_step_consistency': 'timing_consistent', 'frozen_zone_hash_unchanged': 'frozen_files',
    'peak_reserved_gib_max': 'memory_cap', 'lifecycle_for_promotion': 'resume_continues',
    'second_input_plan_consistency': 'speed_reproduces',
}


def harness_item(name: str) -> str:
    for prefix, cid in HARNESS_MAP:
        if name.startswith(prefix):
            return cid
    return 'completes'


def from_core(hard: list[dict], receipt: dict | None) -> list[dict]:
    """Checklist results from reward.evaluate's hard items plus the harness checks of the receipt (a failed harness
    check lands on its own item instead of the generic 'receipt not valid')."""
    groups: dict[str, list] = {}
    for h in hard:
        if h.get('name') == 'invariants':
            continue                                    # its harness checks are mapped one by one below
        groups.setdefault(HARD_MAP.get(h['name'], 'completes'), []).append((h['ok'], f'{h["name"]}: {h.get("detail") or "ok"}'))
    hc = ((receipt or {}).get('harness') or {}).get('checks') if isinstance((receipt or {}).get('harness'), dict) else None
    named = []
    for c in hc or []:
        if isinstance(c, dict) and not c.get('ok') and c.get('severity', 'invalid') == 'invalid':
            named.append(c)
            groups.setdefault(harness_item(str(c.get('name'))), []).append((False, f'{c.get("name")}: {c.get("detail")}'))
    if named and 'completes' in groups:     # 'receipt not valid' is explained by the mapped harness checks
        groups['completes'] = [(ok, d) for ok, d in groups['completes'] if not d.startswith('receipt_valid:')] or \
            [(True, 'receipt valid apart from the items below')]
    out = []
    for cid in [c.id for c in C.CHECKS]:
        if cid not in groups:
            continue
        rows = groups[cid]
        bad = [d for ok, d in rows if not ok]
        out.append(C.result(cid, C.FAIL if bad else C.PASS, '; '.join(bad or [d for _, d in rows])[:600]))
    return out


def from_env(receipt: dict | None, reasons: list) -> list[dict]:
    """External environments: the environment's own judge decides `completes`/correctness of its metric, InfraRSI's
    guards map to frozen files and run health; everything the environment does not expose stays unchecked."""
    r = receipt if isinstance(receipt, dict) else {}
    out = []
    jv = r.get('judge_valid')
    out.append(C.result('completes', C.PASS if jv is True and not reasons else C.FAIL,
                        '; '.join(map(str, reasons))[:600] if reasons else 'environment judge: valid'))
    for g in r.get('guard_rejections') or []:
        gid = str((g or {}).get('id'))
        cid = 'frozen_files' if 'frozen' in gid else ('timing_consistent' if 'clock' in gid or 'time' in gid else 'no_errors')
        out.append(C.result(cid, C.FAIL, f'guard {gid}: {(g or {}).get("detail")}'))
    return out
