"""Task scoring (design 1.6): Speedup_FR/ED, margin rule, CTG(tau=log 1.10), AUC_budget, win/feasibility inputs.

A search trajectory is a list of events in time order, each with incremental cost (gpuh, usd, wall_s, tokens) and,
for screens, the verified step time (None when the screen was invalid). y = best verified log(t_FR/t_screen) so far
(screen measurement), floored at y0 (0 = fall back to FR). Costs of failed screens, loads and verification count.

Predeclared experiment task sets (orchestrator decision, Codex R1 finding 13, 2026-10-20). Every scored group is
evaluated on an exact frozen task set chosen by its mode; the denominator is the size of that set and is never
inferred from the rows supplied. A task of the set without a row (or with an ineligible row) counts as a failure
for every criterion; a row for a task outside the set is an error.
  mode 'full' (design section 2):
    E1 = {S1, S2, D1, D2, D3, D4, D5, D6, H1, H2, H3, H4, H5, H6} (14 tasks):
         FeasibilityRate >= 12/14, A0 ValidWinRate vs FR >= 10/14, ED beats FR beyond the margin >= 9/14,
         InvalidAccept = 0.
    E6 held-out = {H1, H2, H3, H4, H5, H6} (rsi_criteria: 5/6 within margin, 3/6 beyond, M arm 4/6).
    R-arm dev sequence = S1, S2, D1, D2, D3, D4, D5, D6.
  mode 'stoploss1' (design section 6 stop-loss 1: MoE path not working at Day 2 12:00):
    E1 = {S1, S2, D1, D2, H2, H3, H4, H6} (8 tasks): FeasibilityRate >= 7/8, A0 ValidWinRate >= 6/8,
         ED beats FR >= 6/8, InvalidAccept = 0.
    E6 held-out = {H2, H3, H4, H6} (rsi_criteria: 4/4 within margin, 2/4 beyond, M arm 3/4).
    R-arm dev sequence = S1, S2, D1, D2.
Rows with eligible=False (a search that exceeded a budget dimension despite worst-case admission, A's
scoring_inputs) are excluded from every aggregate: they never count as a win, a valid candidate or a feasible task.
E1 verdicts carry an explicit InputComplete criterion (one eligible feasibility, A0 and ED entry for every task of the
mode; ineligible rows count as missing) that is ANDed into every pass flag.
"""
from __future__ import annotations

import dataclasses
import math

E1_MODES = {
    'full': {'tasks': ('S1', 'S2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'),
             'feasible': 12, 'a0_wins': 10, 'ed_wins': 9},
    'stoploss1': {'tasks': ('S1', 'S2', 'D1', 'D2', 'H2', 'H3', 'H4', 'H6'), 'feasible': 7, 'a0_wins': 6, 'ed_wins': 6},
}
R_DEV_SEQUENCE = {'full': ('S1', 'S2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6'), 'stoploss1': ('S1', 'S2', 'D1', 'D2')}


def mode_spec(mode: str) -> dict:
    if mode not in E1_MODES:
        raise ValueError(f'unknown scoring mode {mode!r}; predeclared modes: {sorted(E1_MODES)}')
    return E1_MODES[mode]


def eligible(row: dict) -> bool:
    return row.get('eligible', True) is not False

TAU = math.log(1.10)
NOISY_RSD = 0.05


def margin_from_rsd(rsd: float | None) -> float:
    """margin = max(3%, 3 x RSD of max-rank step times in the FR locked scoring window)."""
    return max(0.03, 3.0 * (rsd or 0.0))


def margin_of(fr_scoring_receipt: dict) -> float:
    return margin_from_rsd(((fr_scoring_receipt or {}).get('step_time') or {}).get('rsd_window'))


def is_noisy(rsd: float | None) -> bool:
    return (rsd or 0.0) > NOISY_RSD


def log_gain(t_ref: float, t: float) -> float:
    return math.log(t_ref / t)


def is_win(t_ref: float, t: float, margin: float) -> bool:
    """Win only if log(t_ref/t) > log(1 + margin)."""
    return log_gain(t_ref, t) > math.log1p(margin)


def within_margin(t_ref: float, t: float, margin: float) -> bool:
    return log_gain(t_ref, t) >= -math.log1p(margin)


@dataclasses.dataclass
class Event:
    gpuh: float = 0.0
    usd: float = 0.0
    wall_s: float = 0.0
    tokens: float = 0.0
    step_time: float | None = None      # verified screen step time (valid screens only)
    kind: str = 'screen'
    note: str = ''


def _cum(events):
    g = u = w = k = 0.0
    for e in events:
        g, u, w, k = g + e.gpuh, u + e.usd, w + e.wall_s, k + e.tokens
        yield e, {'gpuh': g, 'usd': u, 'wall_s': w, 'tokens': k}


def ctg(events: list[Event], t_fr: float, tau: float = TAU) -> dict:
    """Cumulative GPUh / USD / wall at the first event whose verified gain reaches tau; inf if never."""
    for e, c in _cum(events):
        if e.step_time and log_gain(t_fr, e.step_time) >= tau:
            return {'gpuh': c['gpuh'], 'usd': c['usd'], 'wall_s': c['wall_s'], 'reached': True}
    return {'gpuh': math.inf, 'usd': math.inf, 'wall_s': math.inf, 'reached': False}


def budget_fraction(c: dict, caps: dict) -> float:
    fr = [c[k] / caps[k] for k in ('gpuh', 'tokens', 'wall_s') if caps.get(k)]
    return min(1.0, max(fr)) if fr else 0.0


def auc_budget(events: list[Event], t_fr: float, caps: dict, y0: float = 0.0) -> dict:
    """Area under y(x), x = max(GPUh, tokens, wall) fraction of each cap in [0, 1]; step function, gains take effect
    when the screen that found them has been paid for; the final best is held to x = 1."""
    x_prev, y, area, points = 0.0, y0, 0.0, [(0.0, y0)]
    for e, c in _cum(events):
        x = budget_fraction(c, caps)
        area += y * max(0.0, x - x_prev)
        x_prev = max(x_prev, x)
        if e.step_time:
            y = max(y, log_gain(t_fr, e.step_time))
        points.append((x_prev, y))
    area += y * max(0.0, 1.0 - x_prev)
    return {'auc': area, 'final_y': y, 'x_end': x_prev, 'points': points}


def t_ref_fallback(t_fr: float | None, t_ed: float | None, t_first_feasible: float | None) -> tuple[float | None, str]:
    """t_ref = FR if feasible, else ED if feasible, else the arm's first feasible config (CTG-to-feasibility only)."""
    if t_fr:
        return t_fr, 'FR'
    if t_ed:
        return t_ed, 'ED'
    if t_first_feasible:
        return t_first_feasible, 'first_feasible'
    return None, 'none'


def score_task(task_id: str, arm: str, *, t_fr_locked: float | None, t_cand_locked: float | None, fr_rsd: float | None,
               t_ed: float | None = None, events: list[Event] | None = None, caps: dict | None = None,
               invalid_accept: int = 0, numerics_class: str | None = None, lifecycle_pass: bool | None = None,
               breakeven: dict | None = None, t_first_feasible: float | None = None, t_fr_screen: float | None = None,
               human_minutes: float = 0.0, eligible: bool = True, ineligible_reason: str | None = None) -> dict:
    """One (task x arm) row. t_*_locked come from the arm's own locked {FR, candidate} pair (plan B).
    eligible=False (over budget): the row is kept for reporting but excluded from every aggregate."""
    m = margin_from_rsd(fr_rsd)
    t_ref, ref_kind = t_ref_fallback(t_fr_locked, t_ed, t_first_feasible)
    row = {'task': task_id, 'arm': arm, 't_FR': t_fr_locked, 't_ED': t_ed, 't_cand': t_cand_locked, 'ref_kind': ref_kind,
           'margin': m, 'fr_rsd': fr_rsd, 'noisy': is_noisy(fr_rsd), 'invalid_accept': invalid_accept,
           'NumericsClass': numerics_class, 'LifecyclePass': lifecycle_pass, 'HumanMinutes': human_minutes,
           'Breakeven': breakeven, 'red': invalid_accept > 0, 'eligible': bool(eligible),
           'ineligible_reason': None if eligible else (ineligible_reason or 'over budget')}
    if not eligible:
        events, t_cand_locked = None, None            # excluded from locked scoring and AUC / CTG
    cand_ok = t_cand_locked is not None and invalid_accept == 0
    if ref_kind in ('FR', 'ED') and cand_ok:
        row['Speedup_FR'] = t_fr_locked / t_cand_locked if t_fr_locked else None
        row['Speedup_ED'] = t_ed / t_cand_locked if t_ed else None
        row['log_gain'] = log_gain(t_ref, t_cand_locked)
        row['win'] = is_win(t_ref, t_cand_locked, m)
    else:
        # no valid promoted candidate (or InvalidAccept > 0): fall back to the reference, zero gain, failures still cost
        row.update(Speedup_FR=1.0 if t_fr_locked else None,
                   Speedup_ED=(t_ed / t_fr_locked if (t_ed and t_fr_locked) else None), log_gain=0.0, win=False)
    row['valid_candidate'] = cand_ok
    row['feasible_ref'] = ref_kind != 'none'
    if events is not None and t_ref:
        t_scr = t_fr_screen or t_ref        # screen-measured FR time for the screen-metric curves
        for unit, v in ctg(events, t_scr).items():
            row[f'CTG_{unit}'] = v
        if caps:
            a = auc_budget(events, t_scr, caps)
            row['AUC_budget'] = a['auc']
            row['AUC_final_y'] = a['final_y']
        row['spent'] = dict(list(_cum(events))[-1][1]) if events else {'gpuh': 0.0, 'usd': 0.0, 'wall_s': 0.0, 'tokens': 0.0}
    return row


def valid_win_rate(rows: list[dict], n: int | None = None) -> dict:
    """Wins over eligible rows; `n` is the predeclared denominator (default: the number of eligible rows)."""
    kept = [r for r in rows if eligible(r)]
    wins = sum(1 for r in kept if r.get('win') and r.get('valid_candidate'))
    n = len(kept) if n is None else n
    return {'wins': wins, 'n': n, 'rate': wins / n if n else None}


def feasibility_rate(rows: list[dict], n: int | None = None) -> dict:
    kept = [r for r in rows if eligible(r)]
    ok = sum(1 for r in kept if r.get('feasible_ref') or r.get('valid_candidate'))
    n = len(kept) if n is None else n
    return {'feasible': ok, 'n': n, 'rate': ok / n if n else None}


def invalid_accept_total(rows: list[dict]) -> int:
    return sum(int(r.get('invalid_accept') or 0) for r in rows if eligible(r))


def _by_task(rows: list[dict], tasks: tuple, name: str) -> dict:
    out = {}
    for r in rows:
        t = r.get('task')
        if t not in tasks:
            raise ValueError(f'{name}: task {t!r} is not in the predeclared set {list(tasks)}')
        if t in out:
            raise ValueError(f'{name}: duplicate row for task {t}')
        out[t] = r
    return out


def e1_verdicts(rows_a0: list[dict], rows_ed: list[dict], feasible: dict, mode: str = 'full') -> dict:
    """E1 pre-declared criteria on the exact task set of `mode` (module docstring): FeasibilityRate, A0 ValidWinRate
    vs FR, ED beats FR beyond the margin, InvalidAccept 0. feasible: {task: bool} (FR or ED feasible). A task of the
    set that is missing from any input, or whose row is ineligible, fails the criteria it would feed."""
    spec = mode_spec(mode)
    tasks = spec['tasks']
    if not isinstance(feasible, dict):
        raise TypeError('feasible must map task id -> bool for the predeclared task set')
    unknown = sorted(set(feasible) - set(tasks))
    if unknown:
        raise ValueError(f'feasible: tasks {unknown} are not in the predeclared set of mode {mode!r}')
    a0, ed = _by_task(rows_a0, tasks, 'rows_a0'), _by_task(rows_ed, tasks, 'rows_ed')
    win = lambda r: bool(r and eligible(r) and r.get('win') and r.get('valid_candidate'))   # noqa: E731
    n = len(tasks)
    n_feasible = sum(1 for t in tasks if feasible.get(t) is True)
    a0_wins = sum(1 for t in tasks if win(a0.get(t)))
    ed_wins = sum(1 for t in tasks if win(ed.get(t)))
    missing = {'feasible': [t for t in tasks if t not in feasible], 'rows_a0': [t for t in tasks if t not in a0],
               'rows_ed': [t for t in tasks if t not in ed],
               'ineligible': sorted(t for t in tasks for r in (a0.get(t), ed.get(t)) if r and not eligible(r))}
    res = {'mode': mode, 'tasks': list(tasks), 'n': n, 'feasibility': n_feasible, 'a0_wins': a0_wins, 'ed_wins': ed_wins,
           'invalid_accept': invalid_accept_total(list(a0.values())) + invalid_accept_total(list(ed.values())),
           'missing': missing, 'need': {k: spec[k] for k in ('feasible', 'a0_wins', 'ed_wins')}}
    # Codex R2 finding 5: no E1 verdict passes unless every task of the mode has an eligible feasibility, A0 and ED
    # entry (ineligible rows count as missing); InputComplete is ANDed into every criterion.
    complete = not any(missing.values())
    res['pass'] = {'InputComplete': complete,
                   'FeasibilityRate': complete and n_feasible >= spec['feasible'],
                   'A0_ValidWinRate': complete and a0_wins >= spec['a0_wins'],
                   'ED_vs_FR': complete and ed_wins >= spec['ed_wins'],
                   'InvalidAccept': complete and res['invalid_accept'] == 0}
    res['passed'] = all(res['pass'].values())
    return res
