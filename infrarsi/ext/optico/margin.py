"""ENV-B noise margin per workload (AMENDMENT_EXT_002 Q1, decision D11): run-to-run RSD of the judge value over the
protocol-produced default runs, computed mechanically from signed receipts and written to margin_<W>.json.

S_w = judge values T_i (metric.raw.tflops_per_gpu = run_monitor.sh's value) of every protocol run of w's default
configuration (workloads.py defaults = OptiCo's start) that is valid, by run kind only (closed list):
  b_def            the B-DEF run (arm B-DEF, purpose b_def)
  search_start     screen 0 of B-IS and B-RND, and of B-ISW on W1 (purpose screen, candidate c0)
  b_nat_cycle0     the B-NAT cycle-0 row (purpose native, row kind 'default'), or the first valid RETRY of it when
                   OptiCo retried it (one value per loop)
  locked_default   the locked-default half of every locked pair of B-ED, B-NAT, B-IS, B-RND, and B-ISW on W1
                   (purpose locked, candidate locked-default); B-DEF has no locked pair
Builder checks and calibration runs carry other purposes (functional_check, calibration) and never enter S_w.
Valid = receipt valid and judge_valid (Result success, 5 iterations, G-B1/G-B3, output integrity, frozen files), HMAC
verified, parameters equal to the workload defaults, no sanitizer event. No run is added, dropped or replaced because of
its value. Lease held / no concurrent process: by construction (GpuLease refuses a busy node).
Estimator: RSD_w = s_w / mean_w (sample SD, n-1), margin_w = max(0.03, 3 x RSD_w), n_w >= 4. Otherwise the pooled
fallback RSD_pool = sqrt(sum_g sum_i (T_gi / mean_g - 1)^2 / sum_g (n_g - 1)) over every ENV-B workload's S with
n_g >= 2 plus the W3 calibration set (R3 + the G-B smokes, reported beside S_w, never in it) as its own group.
The within-run fields of locked.json are diagnostics only and never feed a label.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ...harness import signing
from . import params as pm
from . import workloads as wl

FLOOR, K, MIN_N = 0.03, 3.0, 4
# AMENDMENT_EXT_002: the W3 calibration group (judge values as listed there; other runner / WINH4, not protocol runs)
CALIBRATION = {'W3': [('r3_20261021a', 228.83), ('x3_gb_smoke_20261021a', 233.67), ('x3_gb_smoke_20261021b_d15', 226.20),
                      ('x3_gb_smoke_20261021c', 223.50)]}
KINDS = ('b_def', 'search_start', 'b_nat_cycle0', 'locked_default')
# The amendment's closed list: search starts of B-IS and B-RND (B-NAT: its cycle-0 row) and locked pairs of B-ED, B-NAT,
# B-IS and B-RND on every workload; B-ISW (start and locked pair) on W1 only. B-DEF has no locked pair.
SEARCH_START_ARMS = ('B-IS', 'B-RND', 'B-IS2', 'B-IS2-noexp')   # + the phase-E arms of 10-24 where they ran
LOCKED_ARMS = ('B-ED', 'B-NAT', 'B-IS', 'B-RND', 'B-IS2', 'B-IS2-noexp')
WEAK_ARM_WORKLOADS = ('W1',)


def _with_weak(arms: tuple, w) -> tuple:
    return arms + (('B-ISW',) if w.key in WEAK_ARM_WORKLOADS else ())


def run_kind(r: dict, workload) -> str | None:
    """The S_w membership kind of a receipt of `workload` (by run kind only, the amendment's closed list), or None."""
    w = wl.get(workload)
    if not isinstance(r, dict) or r.get('env') != 'optico' or r.get('task_id') != w.task_id:
        return None
    arm, purpose, cid = r.get('arm'), r.get('purpose'), r.get('candidate_id')
    native = r.get('native')
    if isinstance(native, dict):
        cycle0 = native.get('kind') == 'default' or (native.get('kind') == 'retry' and native.get('retry_of') == 0)
        return 'b_nat_cycle0' if cycle0 and arm == 'B-NAT' and purpose == 'native' else None
    if arm == 'B-DEF' and purpose == 'b_def':
        return 'b_def'
    if arm in _with_weak(SEARCH_START_ARMS, w) and purpose == 'screen' and cid == 'c0':
        return 'search_start'
    if arm in _with_weak(LOCKED_ARMS, w) and purpose == 'locked' and cid == 'locked-default':
        return 'locked_default'
    return None


def judge_value(r: dict) -> float | None:
    v = ((r.get('metric') or {}).get('raw') or {}).get('tflops_per_gpu')
    return float(v) if type(v) in (int, float) and math.isfinite(v) and v > 0 else None


def ineligible(r: dict, workload) -> str | None:
    """Why a member-kind receipt does not enter S_w (None = it does)."""
    w = wl.get(workload)
    if r.get('task_id') != w.task_id:
        return f'task {r.get("task_id")} is not {w.task_id}'
    if not signing.verify({k: v for k, v in r.items() if k != '_path'}):     # _path: where load_receipts found it
        return 'harness signature missing or invalid'
    if r.get('valid') is not True or r.get('judge_valid') is not True:
        return 'not valid (judge or guards)'
    if judge_value(r) is None:
        return 'no judge value'
    params = r.get('params')
    try:
        same = isinstance(params, dict) and pm.params_key(pm.typed(params, strict=True)) == pm.params_key(w.defaults)
    except ValueError:
        same = False
    if not same:
        return 'parameters are not the workload defaults'
    native = r.get('native') if isinstance(r.get('native'), dict) else {}
    if (r.get('sanitizer') or {}).get('fired') or native.get('sanitizer_events'):
        return 'sanitizer event'
    return None


def load_receipts(paths) -> list[dict]:
    """Receipt JSONs from files and directories (recursive: receipts/*.json, receipt_*.json); deduplicated by attempt."""
    out, seen = [], set()
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(x for x in p.rglob('*.json') if x.parent.name == 'receipts' or x.name.startswith('receipt'))
        elif p.is_file():
            files.append(p)
    for f in files:
        try:
            r = json.loads(f.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and r.get('schema') == 'ext-receipt-v1' and r.get('attempt_id') not in seen:
            seen.add(r.get('attempt_id'))
            out.append({**r, '_path': str(f)})
    return out


def sample(receipts, workload) -> tuple[list[dict], list[dict]]:
    """(members of S_w, excluded member-kind receipts with reasons). B-NAT: per loop, row 0 if valid, else the first
    valid RETRY of row 0 (one value per loop)."""
    w = wl.get(workload)
    members, excluded, nat = [], [], {}
    for r in receipts:
        kind = run_kind(r, w)
        if kind is None:
            continue
        why = ineligible(r, w)
        if kind == 'b_nat_cycle0':
            nat.setdefault(r['native'].get('native_attempt') or r.get('run_id'), []).append((r, why))
            continue
        (excluded if why else members).append({'kind': kind, 'receipt': r, 'why': why})
    for loop, rows in nat.items():
        rows.sort(key=lambda x: x[0]['native'].get('row_index', 0))
        pick = next(((r, why) for r, why in rows if why is None), None)
        if pick:
            members.append({'kind': 'b_nat_cycle0', 'receipt': pick[0], 'why': None})
        for r, why in rows:
            if pick is None or r is not pick[0]:
                excluded.append({'kind': 'b_nat_cycle0', 'receipt': r,
                                 'why': why or 'not the first valid cycle-0 row of its loop'})
    return members, excluded


def stats(values) -> dict:
    n = len(values)
    mean = sum(values) / n if n else None
    s = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n >= 2 else None
    return {'n': n, 'mean': mean, 's': s, 'rsd': (s / mean) if s is not None and mean else None}


def pooled_rsd(groups: dict) -> tuple[float | None, int]:
    """(RSD_pool, degrees of freedom) over groups {name: [T]} with n >= 2."""
    num, df = 0.0, 0
    for vals in groups.values():
        if len(vals) < 2:
            continue
        m = sum(vals) / len(vals)
        num += sum((v / m - 1.0) ** 2 for v in vals)
        df += len(vals) - 1
    return (math.sqrt(num / df) if df else None), df


def compute(workload, receipts, *, calibration: dict | None = None) -> dict:
    w = wl.get(workload)
    calibration = CALIBRATION if calibration is None else calibration
    members, excluded = sample(receipts, w)
    vals = [judge_value(m['receipt']) for m in members]
    st = stats(vals)
    rec = {'schema': 'infrarsi-ext-optico-margin-v1', 'workload': w.key, 'task_id': w.task_id,
           'rule': 'AMENDMENT_EXT_002 Q1 (D11): margin_w = max(0.03, 3 x RSD_w), RSD_w = s/mean over S_w, n >= 4; '
                   'else pooled fallback',
           'S_w': [{'attempt_id': m['receipt'].get('attempt_id'), 'kind': m['kind'], 'arm': m['receipt'].get('arm'),
                    'run_id': m['receipt'].get('run_id'), 'node': m['receipt'].get('node'), 'T': judge_value(m['receipt']),
                    'path': m['receipt'].get('_path')} for m in members],
           'excluded': [{'attempt_id': x['receipt'].get('attempt_id'), 'kind': x['kind'], 'why': x['why'],
                         'path': x['receipt'].get('_path')} for x in excluded],
           'n': st['n'], 'mean': st['mean'], 's': st['s'], 'rsd': st['rsd'],
           'calibration': {g: [{'run': r, 'T': t} for r, t in v] for g, v in calibration.items()}}
    if st['n'] >= MIN_N:
        rec.update(estimator='run_to_run', rsd_used=st['rsd'])
    else:
        groups = {}
        for other in wl.WORKLOADS.values():
            m_other, _ = sample(receipts, other)
            groups[other.key] = [judge_value(m['receipt']) for m in m_other]
        for g, v in calibration.items():
            groups[f'calibration_{g}'] = [t for _, t in v]
        rsd_pool, df = pooled_rsd(groups)
        rec.update(estimator='pooled_fallback', rsd_used=rsd_pool, pooled_df=df,
                   pooled_groups={g: v for g, v in groups.items() if len(v) >= 2})
    rec['margin'] = max(FLOOR, K * rec['rsd_used']) if rec['rsd_used'] is not None else None
    rec['for_trajectory_compare'] = {'rsd': rec['rsd_used'], 'note': 'pass as trajectory.compare(task, ..., rsd=rsd); '
                                     'margin_y = ln(1 + max(0.03, 3 x rsd))'}
    return rec


def write(workload, receipts, out_dir, *, calibration: dict | None = None) -> Path:
    rec = compute(workload, receipts, calibration=calibration)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'margin_{wl.get(workload).key}.json'
    path.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    return path
