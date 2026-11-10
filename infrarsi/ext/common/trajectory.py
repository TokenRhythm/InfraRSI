"""Search trajectories of ext arms -> scoring.score events, CTG(tau), AUC_budget and margins (design §1.4, §0.3).

y is environment specific: ENV-A ln(t_S0/t), ENV-B ln(TFLOPS/TFLOPS_def), ENV-C bpb_base - bpb. scoring.score is used
unchanged by encoding each valid screen as the pseudo step time exp(-y) against t_ref = 1, so log_gain = y. The x axis of
AUC_budget is max(wall, LLM tokens, number of screens) as fractions of the arm's caps (the screen count rides in the
gpuh slot of a shadow event list); GPU-hours, USD and wall stay exact in CTG. Failed and invalid screens cost as usual.
Input: a SearchTrace dict (`screens` with cumulative `cum` snapshots, `used`) or the basic agent's trace (same shape).
"""
from __future__ import annotations

import dataclasses
import math

from ...scoring import score

COST_KEYS = (('gpuh', 'gpu_hours'), ('usd', 'llm_usd'), ('wall_s', 'wall_seconds'), ('tokens', 'llm_tokens'))


def y_value(env: str, value: float | None, ref: float | None, *, raw: dict | None = None, ref_raw: dict | None = None) -> float | None:
    """y of one valid measurement of the minimized metric against the reference (S0 / default / base)."""
    if not (type(value) in (int, float) and math.isfinite(value) and value > 0 and type(ref) in (int, float) and ref > 0):
        return None
    if env == 'autoresearch':
        return float(ref) - float(value)
    if env == 'optico' and raw and ref_raw and (raw.get('tflops') or 0) > 0 and (ref_raw.get('tflops') or 0) > 0:
        return math.log(raw['tflops'] / ref_raw['tflops'])
    return math.log(ref / value)


def pseudo_step_time(y: float) -> float:
    return math.exp(-y)


def margin_y(task, rsd: float | None = None) -> float:
    """The pre-declared margin expressed on y: ENV-C absolute bpb; ENV-A t <= (1 - v) t_ref; ENV-B and generic
    TFLOPS >= (1 + m) ref with m = v or max(floor, k x RSD)."""
    m = task.margin
    v = float(m['value']) if m['kind'] == 'fixed' else max(float(m['floor']), float(m['k']) * float(rsd or 0.0))
    if task.env == 'autoresearch':
        return v
    if task.env == 'rebench_llmf':
        return -math.log(1.0 - v)
    return math.log1p(v)


def events(trace: dict, task, ref: float, *, ref_raw: dict | None = None, values=None) -> list[score.Event]:
    """One event per screen with the cost increments since the previous screen (LLM calls in between included),
    plus a tail event for cost after the last screen. values: optional {row position: (metric value, raw)} overriding
    the trace's step_time (e.g. the trusted judge value) - per row, since a confirmation rerun repeats its candidate's
    id; the trace's verdict validity is kept."""
    out, prev = [], {k: 0.0 for _, k in COST_KEYS}
    for k, s in enumerate(trace.get('screens') or []):
        cum = s.get('cum') or {}
        inc = {ek: max(0.0, float(cum.get(tk) or 0.0) - prev[tk]) for ek, tk in COST_KEYS}
        prev = {tk: float(cum.get(tk) or 0.0) for _, tk in COST_KEYS}
        v, raw = (values or {}).get(k, (s.get('step_time'), None))
        y = y_value(task.env, v, ref, raw=raw, ref_raw=ref_raw) if s.get('valid') else None
        out.append(score.Event(step_time=pseudo_step_time(y) if y is not None else None, kind='screen',
                               note=str(s.get('candidate_id')), **inc))
    used = trace.get('used') or {}
    tail = {ek: max(0.0, float(used.get(tk) or 0.0) - prev[tk]) for ek, tk in COST_KEYS}
    if any(tail.values()):
        out.append(score.Event(kind='tail', note='cost after the last screen', **tail))
    return out


def ctg(evs: list[score.Event], task) -> dict:
    """Cumulative GPUh / USD / wall at the first screen with y >= tau (inf if never)."""
    return score.ctg(evs, 1.0, task.tau)


def auc(evs: list[score.Event], task) -> dict:
    b = task.budget
    shadow = [dataclasses.replace(e, gpuh=1.0 if e.kind == 'screen' else 0.0) for e in evs]
    caps = {'gpuh': b.get('screens'), 'tokens': b.get('llm_tokens'), 'wall_s': b.get('wall_seconds')}
    return score.auc_budget(shadow, 1.0, caps)


def best_search(trace: dict, task, ref: float) -> dict:
    ok = [(s['step_time'], s.get('index', i), s.get('candidate_id')) for i, s in enumerate(trace.get('screens') or [])
          if s.get('valid') and type(s.get('step_time')) in (int, float)]
    if not ok:
        return {'value': None, 'candidate_id': None, 'y': None}
    v, _, cid = min(ok)
    return {'value': v, 'candidate_id': cid, 'y': y_value(task.env, v, ref)}


def accounting(trace: dict) -> tuple[bool, list[str]]:
    """(complete, reasons) of a trace's accounting (X3 #7): complete unless the trace, its `used` block or any screen's
    `cum` says accounting_complete false (backends charge upper bounds for unknown usage; a kept lease, an unmeasured
    clock offset or an ambiguous attribution make CTG / AUC bounds rather than values). Producers' own notes
    (extra.accounting_notes) are passed on."""
    why = []
    if trace.get('accounting_complete') is False:
        why.append('trace')
    if (trace.get('used') or {}).get('accounting_complete') is False:
        why.append('used')
    bad = [s.get('index', i) for i, s in enumerate(trace.get('screens') or []) if (s.get('cum') or {}).get('accounting_complete') is False]
    if bad:
        why.append(f'screens {bad[:8]}{" ..." if len(bad) > 8 else ""}')
    notes = (trace.get('extra') or {}).get('accounting_notes')
    if why and notes:
        why += [str(n) for n in (notes if isinstance(notes, list) else [notes])][:8]
    return not why, why


def arm_row(trace: dict, task, ref: float, *, locked: float | None = None, locked_valid: bool | None = None,
            invalid_accept_locked: int = 0, ref_raw: dict | None = None, values=None) -> dict:
    """Per-arm numbers of the design's tables (search side; the locked value comes from the arm's locked re-score)."""
    evs = events(trace, task, ref, ref_raw=ref_raw, values=values)
    complete, why = accounting(trace)
    screens = trace.get('screens') or []
    best = best_search(trace, task, ref)
    c, a = ctg(evs, task), auc(evs, task)
    return {'task': task.id, 'env': task.env, 'arm': trace.get('arm') or task.arm, 'ref': ref,
            'best_search': best['value'], 'best_search_candidate': best['candidate_id'], 'y_best_search': best['y'],
            'locked': locked, 'locked_valid': locked_valid,
            'y_locked': y_value(task.env, locked, ref) if locked is not None and locked_valid else None,
            'n_screens': len(screens), 'n_valid': sum(1 for s in screens if s.get('valid')),
            'invalid_accept_locked': int(invalid_accept_locked), 'CTG': c, 'AUC_budget': a['auc'], 'AUC_final_y': a['final_y'],
            'used': trace.get('used'), 'stop_reason': trace.get('stop_reason'),
            'accounting_complete': complete, 'accounting_reasons': why}


def compare(task, row_is: dict, row_nat: dict, *, rsd: float | None = None) -> dict:
    """Pre-declared advantage rule (design §0.3), per environment: (1) the InfraRSI final solution is valid in its
    locked re-score, (2) invalid_accept_locked = 0, (3) locked y beyond the native locked y by the margin,
    (4) AUC_budget(IS) >= AUC_budget(native). parity = (3) within the margin both ways. A pair where either row's
    accounting is incomplete is labelled `incomplete` (X3 #7); the label the numbers would give is kept as
    `label_if_complete`."""
    res = _compare(task, row_is, row_nat, rsd=rsd)
    open_rows = [name for name, row in (('IS', row_is), ('native', row_nat)) if row.get('accounting_complete') is False]
    if open_rows:
        res = {**res, 'label_if_complete': res['label'], 'label': 'incomplete', 'incomplete': open_rows,
               'accounting_reasons': {name: row.get('accounting_reasons') for name, row in (('IS', row_is), ('native', row_nat))
                                      if name in open_rows}}
    return res


def _compare(task, row_is: dict, row_nat: dict, *, rsd: float | None = None) -> dict:
    m = margin_y(task, rsd)
    if not row_is.get('locked_valid'):
        return {'label': 'invalid', 'invalid_accept': 1, 'margin_y': m, 'delta_y': None}
    ya, yb = row_is.get('y_locked'), row_nat.get('y_locked') if row_nat.get('locked_valid') else None
    if yb is None:
        yb = 0.0                                         # native final invalid: it falls back to the reference (y = 0)
    d = ya - yb
    eps = 1e-12
    cond = {'valid_locked': True, 'invalid_accept_zero': int(row_is.get('invalid_accept_locked') or 0) == 0,
            'beyond_margin': d >= m - eps, 'auc_not_worse': (row_is.get('AUC_budget') or 0.0) >= (row_nat.get('AUC_budget') or 0.0)}
    if all(cond.values()):
        label = 'advantage'
    elif -d >= m - eps:
        label = 'disadvantage'
    elif abs(d) < m:
        label = 'parity'
    else:
        label = 'no_advantage'                              # beyond the margin but another condition failed
    return {'label': label, 'conditions': cond, 'delta_y': d, 'margin_y': m}
