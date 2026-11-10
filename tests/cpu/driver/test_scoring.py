"""score.py numerics on synthetic trajectories: margin rule, wins, CTG at log 1.10, AUC_budget, fallbacks, E1 verdicts."""
import math

from infrarsi.scoring import (TAU, Event, auc_budget, ctg, e1_verdicts, is_noisy, is_win, margin_from_rsd, score_task,
                                t_ref_fallback, tables, valid_win_rate, within_margin)


def test_margin_rule():
    assert margin_from_rsd(0.002) == 0.03
    assert math.isclose(margin_from_rsd(0.02), 0.06)
    assert is_noisy(0.051) and not is_noisy(0.05)
    assert is_win(10.0, 9.6, 0.03) and not is_win(10.0, 9.75, 0.03)        # 10/9.75 = 1.0256 < 1.03
    assert within_margin(10.0, 10.25, 0.03) and not within_margin(10.0, 10.4, 0.03)


def test_ctg_first_crossing():
    ev = [Event(gpuh=1, usd=2, wall_s=100, step_time=None), Event(gpuh=1, usd=2, wall_s=100, step_time=9.5),
          Event(gpuh=1, usd=2, wall_s=100, step_time=9.0), Event(gpuh=1, usd=2, wall_s=100, step_time=8.0)]
    c = ctg(ev, 10.0)
    assert math.log(10 / 9.0) >= TAU and c == {'gpuh': 3, 'usd': 6, 'wall_s': 300, 'reached': True}
    assert ctg(ev[:2], 10.0)['gpuh'] == math.inf


def test_auc_budget_step_function():
    caps = {'gpuh': 10.0, 'tokens': 1e6, 'wall_s': 1000.0}
    ev = [Event(gpuh=2, tokens=1e5, wall_s=100, step_time=8.0), Event(gpuh=3, tokens=1e5, wall_s=100, step_time=5.0)]
    a = auc_budget(ev, 10.0, caps)
    # x: 0.2 after event 1 (gpuh dominates), 0.5 after event 2; y: 0 -> log(10/8) -> log 2
    want = 0.0 * 0.2 + math.log(10 / 8) * 0.3 + math.log(2) * 0.5
    assert math.isclose(a['auc'], want) and math.isclose(a['final_y'], math.log(2))
    worse = auc_budget([Event(gpuh=2, step_time=12.0)], 10.0, caps)
    assert worse['auc'] == 0.0                                             # floored at y0 = 0 (fall back to FR)


def test_auc_budget_clips_overspend():
    a = auc_budget([Event(gpuh=20, step_time=5.0)], 10.0, {'gpuh': 10.0})
    assert a['x_end'] == 1.0 and a['auc'] == 0.0


def test_score_task_rows_and_invalid_accept():
    row = score_task('D2', 'A0', t_fr_locked=10.0, t_cand_locked=8.0, fr_rsd=0.01, t_ed=9.0,
                     events=[Event(gpuh=5, step_time=8.2)], caps={'gpuh': 20.0})
    assert math.isclose(row['Speedup_FR'], 1.25) and math.isclose(row['Speedup_ED'], 9 / 8) and row['win']
    assert row['CTG_gpuh'] == 5 and row['AUC_budget'] > 0
    red = score_task('D2', 'A0', t_fr_locked=10.0, t_cand_locked=8.0, fr_rsd=0.01, invalid_accept=1)
    assert red['red'] and red['log_gain'] == 0.0 and not red['win'] and red['Speedup_FR'] == 1.0
    assert valid_win_rate([row, red]) == {'wins': 1, 'n': 2, 'rate': 0.5}


def test_reference_fallback():
    assert t_ref_fallback(None, 9.0, None) == (9.0, 'ED')
    assert t_ref_fallback(None, None, 7.0) == (7.0, 'first_feasible')
    row = score_task('H6', 'A0', t_fr_locked=None, t_cand_locked=None, fr_rsd=None, t_first_feasible=7.0)
    assert row['ref_kind'] == 'first_feasible' and row['log_gain'] == 0.0 and row['feasible_ref']


FULL = ['S1', 'S2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6']
STOPLOSS1 = ['S1', 'S2', 'D1', 'D2', 'H2', 'H3', 'H4', 'H6']


def _rows(tasks, wins):
    return [{'task': t, 'win': i < wins, 'valid_candidate': True, 'invalid_accept': 0} for i, t in enumerate(tasks)]


def test_e1_verdicts_thresholds():
    v = e1_verdicts(_rows(FULL, 10), _rows(FULL, 9), {t: i < 12 for i, t in enumerate(FULL)})
    assert all(v['pass'].values()) and v['n'] == 14 and v['need'] == {'feasible': 12, 'a0_wins': 10, 'ed_wins': 9}
    v = e1_verdicts(_rows(FULL, 9), _rows(FULL, 9), {t: i < 11 for i, t in enumerate(FULL)})
    assert not v['pass']['FeasibilityRate'] and not v['pass']['A0_ValidWinRate']


def test_e1_denominator_is_the_predeclared_set():
    """Codex R1 13A: one feasible winning task does not pass E1; missing tasks count as failures."""
    v = e1_verdicts(_rows(['S1'], 1), _rows(['S1'], 1), {'S1': True})
    assert v['n'] == 14 and not any(v['pass'][k] for k in ('FeasibilityRate', 'A0_ValidWinRate', 'ED_vs_FR'))
    assert len(v['missing']['rows_a0']) == 13 and len(v['missing']['feasible']) == 13
    try:
        e1_verdicts(_rows(['S1', 'X1'], 1), [], {})
        raise AssertionError('a task outside the set must be rejected')
    except ValueError:
        pass
    try:
        e1_verdicts([], [], [True] * 14)                                   # a bare list cannot name the tasks
        raise AssertionError('feasible must be keyed by task')
    except TypeError:
        pass


def test_e1_stoploss_mode_is_exact():
    v = e1_verdicts(_rows(STOPLOSS1, 6), _rows(STOPLOSS1, 6), {t: i < 7 for i, t in enumerate(STOPLOSS1)}, mode='stoploss1')
    assert all(v['pass'].values()) and v['n'] == 8 and v['need'] == {'feasible': 7, 'a0_wins': 6, 'ed_wins': 6}
    v = e1_verdicts(_rows(STOPLOSS1[:7], 6), _rows(STOPLOSS1, 6), {t: True for t in STOPLOSS1}, mode='stoploss1')
    assert v['missing']['rows_a0'] == ['H6'] and v['a0_wins'] == 6                 # the count alone would pass ...
    assert not v['pass']['InputComplete'] and not v['pass']['A0_ValidWinRate'] and not v['passed']   # ... the gate does not
    v = e1_verdicts(_rows(STOPLOSS1[:5], 5), _rows(STOPLOSS1, 6), {t: True for t in STOPLOSS1}, mode='stoploss1')
    assert not v['pass']['A0_ValidWinRate']                                 # 5 of 8, the missing three count as losses
    try:
        e1_verdicts(_rows(['D3'], 1), [], {}, mode='stoploss1')            # D3 is not in the stop-loss set
        raise AssertionError('D3 must be rejected in stoploss1')
    except ValueError:
        pass
    try:
        e1_verdicts([], [], {}, mode='half')
        raise AssertionError('only predeclared modes')
    except ValueError:
        pass


def test_ineligible_rows_are_excluded_from_every_aggregate():
    """A item 8: an over-budget (eligible=False) search row never counts, not even as feasible or valid."""
    row = score_task('D2', 'A0', t_fr_locked=10.0, t_cand_locked=8.0, fr_rsd=0.01, eligible=False,
                     events=[Event(gpuh=5, step_time=8.2)], caps={'gpuh': 20.0})
    assert row['eligible'] is False and not row['win'] and not row['valid_candidate'] and 'AUC_budget' not in row
    assert row['ineligible_reason'] == 'over budget'
    ok = score_task('D1', 'A0', t_fr_locked=10.0, t_cand_locked=8.0, fr_rsd=0.01)
    assert valid_win_rate([row, ok]) == {'wins': 1, 'n': 1, 'rate': 1.0}
    forged = dict(row, win=True, valid_candidate=True, invalid_accept=1)
    assert valid_win_rate([forged]) == {'wins': 0, 'n': 0, 'rate': None}
    from infrarsi.scoring import feasibility_rate, invalid_accept_total
    assert feasibility_rate([forged])['feasible'] == 0 and invalid_accept_total([forged]) == 0
    rows = _rows(FULL, 14)
    rows[0] = dict(rows[0], eligible=False)
    v = e1_verdicts(rows, _rows(FULL, 9), {t: True for t in FULL})
    assert v['a0_wins'] == 13 and v['missing']['ineligible'] == ['S1']


def test_tables_render():
    row = score_task('D2', 'A0', t_fr_locked=10.0, t_cand_locked=8.0, fr_rsd=0.01)
    md = tables.e1_table([row])
    assert md.splitlines()[0].startswith('| task | arm |') and '| D2 | A0 |' in md
    assert tables.to_csv([row], tables.E1_COLUMNS).splitlines()[1].startswith('D2,A0,10,')


def test_e1_requires_complete_inputs():
    """Codex R2 finding 5: exactly the threshold counts (12 feasible, 10 A0 wins, 9 ED wins) with the other tasks
    omitted, or an ineligible row, never pass."""
    feasible = {t: True for t in FULL[:12]}
    v = e1_verdicts(_rows(FULL[:10], 10), _rows(FULL[:9], 9), feasible)
    assert (v['feasibility'], v['a0_wins'], v['ed_wins']) == (12, 10, 9)
    assert v['pass'] == {'InputComplete': False, 'FeasibilityRate': False, 'A0_ValidWinRate': False, 'ED_vs_FR': False,
                         'InvalidAccept': False} and not v['passed']
    full = e1_verdicts(_rows(FULL, 10), _rows(FULL, 9), {t: i < 12 for i, t in enumerate(FULL)})
    assert full['passed'] and full['pass']['InputComplete']
    rows = _rows(FULL, 14)
    rows[3] = dict(rows[3], eligible=False)                       # over budget: counts as missing
    v = e1_verdicts(rows, _rows(FULL, 9), {t: True for t in FULL})
    assert v['a0_wins'] == 13 and not v['pass']['InputComplete'] and not v['passed']
