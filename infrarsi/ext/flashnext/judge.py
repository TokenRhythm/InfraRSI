"""The F1 judge (AMENDMENT_EXT_003 Q3): the production path's own window metric and the ruled numerics gate.

    evaluate(parsed, reference, gates=None, role='candidate') -> {judge_valid, numerics_class, reasons, metric_s, details}

parsed = parse.collect(local_dir) of the screen (or parse.from_status_cell). reference = what the screen is compared
with: the parsed U+C reference screen (every arm screen, F-HT, both locked halves), the stored U+C A cell
rsi_perf_v1_c1_p03 (the U+C reference screen itself; stored_reference()), a signed ext receipt of a screen judged by
this module (reference_from_receipt) or None (always invalid: no screen is exempt from the comparison). role is
recorded only. gates = numerics_gates.json (frozen; its sha256 goes into every verdict).

Validity = the screen is complete (32 rank signal logs with updates 1..30, 4 node exits 0: the collector's
complete_verified, so the metric exists) and its numerics class against the reference is E0 or E1:
  harness.validate.compare_numerics (pure) on (main_ce, aux_loss = main_aux_weighted, grad_norm, lr) over updates 1..30,
  window 17..30, with the frozen moe_nondeterministic tolerance; plus MTP CE gated exactly like main CE (per update
  0.005, window mean 0.002), MTP aux present, and no repeated adjacent (main_ce, grad_norm) pairs.
  E0 = main CE, MTP CE, grad norm, LR and both aux scalars bitwise equal over 1..30; E1 = within every gate; else FAIL.
The window metric is recorded, never gated (metric_s is the collector's value whenever the screen is complete; the
receipt keeps it only when the screen is valid). Guards (guards.py) are separate and only add vetoes.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

from ...harness import signing
from ...harness.validate import compare_numerics, self_check
from . import parse as P

HERE = Path(__file__).resolve().parent
GATES_PATH = HERE / 'numerics_gates.json'
STORED_UC_A = Path(r'E:\rsi\f54v1\final\waves\rsi_perf_v1_c1_p03\STATUS_AT_FINISH_fec1d463f97845688330438217f62a95.json')
STORED_UC_A_SHA256 = '99765e15608387d7a11221fb873776b1e9609fbb3f2c134c0d412454c7670e45'
STORED_UC_A_METRIC = 12.489992013068072
SCHEMA = 'flashnext-judge-v1'
CLASSES = ('E0', 'E1')
BASE_KEYS = ('main_ce', 'aux_loss', 'grad_norm', 'lr')            # harness.validate.TRACKED
E0_KEYS = ('main_ce', 'mtp_ce', 'aux_loss', 'mtp_aux', 'grad_norm', 'lr')
INTER_STEP = 'not_applicable: the production path has no inter-step timer (D-F1-4)'


def load_gates(path: str | Path = GATES_PATH) -> dict:
    g = json.loads(Path(path).read_text(encoding='utf-8'))
    if g.get('schema') != 'flashnext-numerics-gates-v1':
        raise ValueError(f'{path}: not a flashnext numerics gate file')
    return g


def gates_sha256(path: str | Path = GATES_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(x: float, y: float) -> float:
    return abs(x - y) / max(abs(y), 1e-12)


def _bits(x):
    """IEEE-754 binary64 bits of a number (E0 is bitwise: +0.0 and -0.0 differ); anything else as itself."""
    return struct.pack('<d', float(x)) if type(x) in (int, float) else ('nonnumeric', repr(x))


def _r(v, n: int = 6):
    if isinstance(v, float) and math.isfinite(v):
        return float(f'{v:.{n}g}')
    return v


# ---------------------------------------------------------------- references
def stored_reference(path: str | Path = STORED_UC_A, *, sha256: str | None = STORED_UC_A_SHA256) -> dict:
    """The stored U+C A cell (formal-54 wave rsi_perf_v1_c1_p03, group A) as a parsed reference; the STATUS file must
    be the pinned one (sha256) unless sha256=None."""
    ref = P.from_status_cell(Path(path), 'A')
    if sha256 is not None and ref['source']['sha256'] != sha256:
        raise ValueError(f'{path}: not the pinned stored U+C A cell ({str(ref["source"]["sha256"])[:12]} != {sha256[:12]})')
    return ref


def reference_from_receipt(receipt: dict) -> dict:
    """A signed, valid flashnext screen receipt as a reference (its judge block keeps the per-update scalars); the
    receipt itself is kept under '_receipt' so every use re-verifies it."""
    if not isinstance(receipt, dict) or not signing.verify(receipt):
        raise ValueError('reference receipt: harness signature missing or invalid')
    if receipt.get('env') != 'flashnext' or receipt.get('valid') is not True or receipt.get('judge_valid') is not True:
        raise ValueError('reference receipt: not a valid flashnext screen')
    det = (receipt.get('judge') or {}).get('details') or {}
    steps = det.get('per_step')
    if not isinstance(steps, list) or [x.get('update') if isinstance(x, dict) else None for x in steps] != \
            list(range(1, P.UPDATES + 1)):
        raise ValueError('reference receipt: its judge block does not carry updates 1..30')
    return {'schema': P.SCHEMA, 'complete': True, 'per_step': steps, '_receipt': receipt,
            'steps': [{'update': i + 1, 'max_rank_s': t} for i, t in enumerate(det.get('step_max_rank_s') or [])],
            'metric': {'value': (receipt.get('metric') or {}).get('value')},
            'source': {'kind': 'receipt', 'attempt_id': receipt.get('attempt_id'), 'run_id': receipt.get('run_id'),
                       'purpose': receipt.get('purpose'), 'candidate_id': receipt.get('candidate_id')}}


def _provenance(reference: dict) -> tuple[dict | None, list]:
    """Re-derive a parsed reference from what it claims to come from; anything not re-derivable is refused: a stored
    STATUS file (path + sha256 re-read), a signed receipt (re-verified) or a collected screen directory (re-parsed)."""
    src = reference.get('source') if isinstance(reference.get('source'), dict) else {}
    kind = src.get('kind')
    try:
        if kind == 'stored_status':
            if not src.get('path') or not src.get('sha256'):
                return None, ['stored reference without its STATUS file path and sha256']
            fresh = P.from_status_cell(Path(src['path']), src.get('group') or 'A')
            if fresh['source']['sha256'] != src['sha256']:
                return None, ['stored reference: the STATUS file changed (sha256)']
        elif kind == 'receipt':
            fresh = reference_from_receipt(reference.get('_receipt'))
        elif kind == 'signals':
            fresh = P.collect(src.get('dir') or '')
        else:
            return None, [f'reference of unknown provenance {kind!r}']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, [f'reference provenance: {type(exc).__name__}: {str(exc)[:200]}']
    if fresh.get('per_step') != reference.get('per_step'):
        return None, ['reference per-update scalars differ from their source']
    return fresh, []


def reference_view(reference) -> tuple[list | None, dict, list]:
    """(per-update scalars, reference description, problems). Accepted: a signed ext receipt of a valid screen, or a
    parsed reference whose source re-derives it (_provenance). Anything else is refused."""
    if reference is None:
        return None, {'kind': None}, []
    if isinstance(reference, (str, Path)):
        try:
            reference = json.loads(Path(reference).read_text(encoding='utf-8-sig'))
        except (OSError, ValueError) as exc:
            return None, {'kind': 'file'}, [f'reference unreadable: {exc!r}'[:200]]
    if not isinstance(reference, dict):
        return None, {'kind': type(reference).__name__}, ['reference is not an object']
    if reference.get('schema') == 'ext-receipt-v1':
        try:
            reference = reference_from_receipt(reference)
        except ValueError as exc:
            return None, {'kind': 'receipt'}, [str(exc)]
    if reference.get('schema') != P.SCHEMA:
        return None, {'kind': 'unknown'}, [f'reference schema {reference.get("schema")!r} is not a parsed screen or receipt']
    info = dict(reference.get('source') or {})
    fresh, problems = _provenance(reference)
    if problems:
        return None, info, problems
    if fresh.get('complete') is not True:
        return None, info, ['reference screen is not complete']
    steps = fresh.get('per_step')
    if not isinstance(steps, list) or not steps:
        return None, info, ['reference has no per-update scalars']
    return steps, info, []


# ---------------------------------------------------------------- numerics
def compare(cand: list, ref: list, gates: dict) -> dict:
    """The ruled gate: compare_numerics on the four tracked scalars + MTP CE like main CE + MTP aux present + no repeated
    adjacent scalars. Returns the compare_numerics dict extended with mtp_* fields and the final class."""
    tol, window = gates['tolerance'], gates['window']
    steps = int(gates['compared_updates'][1])
    base = lambda rows: [{'update': p['update'], **{k: p.get(k) for k in BASE_KEYS}} for p in rows]   # noqa: E731
    res = compare_numerics(base(cand), base(ref), tol, window, allowed=gates.get('allowed', 'E1'), steps=steps)
    v = list(res.get('violations') or [])
    out = {**res, 'violations': v}
    if not res.get('compared') or res.get('compared_updates', 0) != steps:
        out.update(**{'class': 'FAIL', 'allowed_ok': False})
        return out
    cm = {p['update']: p for p in cand}
    rm = {p['update']: p for p in ref}
    want = list(range(1, steps + 1))
    bad = [u for u in want if not (P.finite(cm[u].get('mtp_ce')) and P.finite(rm[u].get('mtp_ce')))]
    if bad:
        v.append(f'MTP CE missing or non-finite at updates {bad[:8]}')
    else:
        rel = {u: _rel(cm[u]['mtp_ce'], rm[u]['mtp_ce']) for u in want}
        a, b = int(window[0]), int(window[1])
        win = [rel[u] for u in want if a <= u <= b]
        mt = gates['mtp_ce']
        out.update(mtp_ce_rel_max=max(rel.values()), mtp_ce_rel_mean_window=sum(win) / len(win) if win else None)
        if out['mtp_ce_rel_max'] > float(mt['mtp_ce_rel_max_per_step']):
            v.append(f'MTP CE rel dev {out["mtp_ce_rel_max"]:.3g} > {mt["mtp_ce_rel_max_per_step"]}')
        if win and out['mtp_ce_rel_mean_window'] > float(mt['mtp_ce_rel_mean_over_window']):
            v.append(f'MTP CE window mean rel dev {out["mtp_ce_rel_mean_window"]:.3g} > {mt["mtp_ce_rel_mean_over_window"]}')
    for k, label in (('aux_loss', 'aux loss'), ('mtp_aux', 'MTP aux loss')):          # gate 'aux: present'
        missing = [u for u in want if not (P.finite(cm[u].get(k)) and P.finite(rm[u].get(k)))]
        if missing:
            v.append(f'{label} missing or non-finite at updates {missing[:8]}')
    vanished = [u for u in want if P.finite(rm[u].get('mtp_aux')) and rm[u]['mtp_aux'] != 0
                and not (P.finite(cm[u].get('mtp_aux')) and cm[u]['mtp_aux'] != 0)]
    if vanished:
        v.append(f'MTP aux loss vanished at updates {vanished[:8]}')
    pairs = self_check([{'update': p['update'], 'main_ce': p.get('main_ce'), 'grad_norm': p.get('grad_norm')}
                        for p in cand])['repeated_pairs']
    out['repeated_pairs'] = pairs
    if pairs and gates['tolerance'].get('repeated_adjacent_scalars') == 'invalid':
        v.append(f'repeated adjacent (main_ce, grad_norm) at updates {pairs[:6]}')
    bitwise_all = all(_bits(cm[u].get(k)) == _bits(rm[u].get(k)) for u in want for k in E0_KEYS)
    out['bitwise_all'] = bitwise_all
    if v:
        cls = 'FAIL'
    elif bitwise_all:
        cls = 'E0'
    else:
        cls = 'E1'
    out['class'] = cls
    out['allowed_ok'] = cls in CLASSES
    return out


# ---------------------------------------------------------------- the receipt view (D-F1-5)
def receipt_view(parsed: dict, numerics: dict | None) -> dict:
    """Per-update max-rank times, per-rank spread, numerics deltas and class, every values.* scalar family: the same
    summary for every arm (what read_receipt shows; no analysis tools)."""
    steps = parsed.get('steps') or []
    a, b = P.WINDOW
    win = [s for s in steps if a <= s['update'] <= b]
    spread = {}
    if win and all('spread_s' in s for s in win):
        means = parsed.get('rank_window_mean_s') or []
        counts = {}
        for s in win:
            counts[s['argmax_rank']] = counts.get(s['argmax_rank'], 0) + 1
        ok = [(m, r) for r, m in enumerate(means) if P.finite(m)]
        spread = {'window_mean_spread_s': _r(statistics.fmean(s['spread_s'] for s in win)),
                  'max_spread_s': _r(max(s['spread_s'] for s in steps)),
                  'slowest_rank': max(ok)[1] if ok else None, 'slowest_rank_window_mean_s': _r(max(ok)[0]) if ok else None,
                  'fastest_rank_window_mean_s': _r(min(ok)[0]) if ok else None,
                  'slowest_rank_counts_window': dict(sorted(counts.items(), key=lambda kv: -kv[1])[:4])}
    n = numerics or {}
    gn = n.get('grad_norm') if isinstance(n.get('grad_norm'), dict) else {}
    return {
        'step_time_s': _r(parsed['metric'].get('unchecked_value')),
        'window': list(P.WINDOW),
        'per_step_max_rank_s': [_r(s['max_rank_s'], 5) for s in steps],
        'per_rank_spread': spread,
        'numerics': {'class': n.get('class'), 'main_ce_rel_max': _r(n.get('main_ce_rel_max'), 3),
                     'main_ce_rel_mean_window': _r(n.get('main_ce_rel_mean_window'), 3),
                     'mtp_ce_rel_max': _r(n.get('mtp_ce_rel_max'), 3),
                     'mtp_ce_rel_mean_window': _r(n.get('mtp_ce_rel_mean_window'), 3),
                     'grad_norm_first_update_rel': _r(gn.get('first_update_rel'), 3),
                     'grad_norm_log_bias_window': _r(gn.get('log_bias_window'), 3),
                     'violations': (n.get('violations') or [])[:3]},
        'values': {k: {kk: _r(vv, 5) for kk, vv in v.items()} for k, v in (parsed.get('values') or {}).items()},
    }


# ---------------------------------------------------------------- the verdict
def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)


def evaluate(parsed: dict, reference, gates: dict | None = None, *, role: str = 'candidate') -> dict:
    frozen = load_gates()
    reasons = []
    supplied = None
    if gates is not None and _canon(gates) != _canon(frozen):
        supplied = hashlib.sha256(_canon(gates).encode()).hexdigest()
        reasons.append('numerics gates differ from the frozen numerics_gates.json: the screen is judged with the frozen '
                       'gates and is not valid')
    gates = frozen
    comp = parsed.get('completeness') or {}
    if not parsed.get('complete'):
        why = list(comp.get('problems') or [])
        if comp.get('n_ranks_ok') is not None and comp.get('n_ranks_ok') != P.RANKS:
            why.append(f'{comp.get("n_ranks_ok")}/{P.RANKS} rank signal logs complete')
        if comp.get('nodes_ok') is False:
            why.append('node exits: ' + '; '.join(f'node {x["node"]}: {", ".join(x["problems"])}'
                                                   for x in comp.get('node_problems') or [])[:300])
        reasons.append('incomplete screen (metric null): ' + ('; '.join(why) or 'not complete_verified'))
    metric = parsed['metric'].get('value')
    if parsed.get('complete') and metric is None:
        reasons.append(f'metric null: {parsed["metric"].get("n_window")} window updates')
    ref_steps, ref_info, ref_problems = reference_view(reference)
    if ref_problems:
        reasons += [f'reference: {p}' for p in ref_problems]
    numerics = None
    if ref_steps is None:
        # every F1 screen is compared, the U+C reference screen too (against the stored U+C A cell): no exemption
        reasons.append('no numerics reference: the screen cannot be verified')
        numerics = {'compared': False, 'class': None, 'violations': []}
    elif not parsed.get('per_step'):
        reasons.append('numerics: rank 0 per-update scalars missing')
        numerics = {'compared': False, 'class': 'FAIL', 'violations': ['rank 0 scalars missing']}
    else:
        numerics = compare(parsed['per_step'], ref_steps, gates)
        if numerics['class'] not in CLASSES:
            reasons += [f'numerics FAIL: {x}' for x in (numerics.get('violations') or ['outside the gates'])[:6]]
    judge_valid = not reasons
    ncls = numerics.get('class') if numerics else None
    keep = ('compared', 'class', 'allowed', 'allowed_ok', 'compared_updates', 'reference_updates', 'bitwise', 'bitwise_all',
            'main_ce_rel_max', 'main_ce_rel_mean_window', 'grad_norm_rel_max', 'mtp_ce_rel_max', 'mtp_ce_rel_mean_window',
            'aux_rel_max', 'repeated_pairs', 'grad_norm', 'violations', 'is_reference')
    details = {
        'schema': SCHEMA, 'role': role,
        'gates': {'file': GATES_PATH.name, 'sha256': gates_sha256(), 'tier': gates.get('tier'), 'allowed': gates.get('allowed'),
                  'supplied_differing_sha256': supplied},
        'metric': {k: parsed['metric'].get(k) for k in ('value', 'unchecked_value', 'n_window', 'window', 'stored_value')
                   if k in parsed['metric']},
        'numerics': {k: numerics[k] for k in keep if k in (numerics or {})},
        'reference': ref_info,
        'identity': parsed.get('identity'),
        'per_step': parsed.get('per_step') or [],
        'step_max_rank_s': [s['max_rank_s'] for s in parsed.get('steps') or []],
        'view': receipt_view(parsed, numerics),
        'memory': {'max_peak_reserved_gib': parsed.get('peak_reserved_gib')},
        'completeness': {k: comp.get(k) for k in ('n_ranks_ok', 'nodes_ok', 'rank_problems', 'node_problems', 'problems')
                         if k in comp},
        'scalar_consistency': parsed.get('scalar_consistency'),
        'inter_step_consistency': INTER_STEP,
    }
    return {'judge_valid': judge_valid, 'numerics_class': ncls, 'reasons': reasons,
            'metric_s': metric if P.finite(metric) else None, 'details': details}


def receipt_extra(verdict: dict) -> dict:
    """Receipt fields from a verdict: 'summary' (the D-F1-5 view read by the hooks' digest), 'memory' (read by
    EnvJudge as peak_reserved_gib), 'numerics' (class and deltas), the gate file hash."""
    d = verdict.get('details') or {}
    return {'summary': d.get('view'), 'memory': d.get('memory'),
            'numerics': {'class': verdict.get('numerics_class'), **{k: (d.get('numerics') or {}).get(k) for k in
                         ('main_ce_rel_max', 'main_ce_rel_mean_window', 'mtp_ce_rel_max', 'mtp_ce_rel_mean_window',
                          'bitwise_all', 'violations')}},
            'numerics_gates_sha256': (d.get('gates') or {}).get('sha256'),
            'inter_step_consistency': INTER_STEP}
