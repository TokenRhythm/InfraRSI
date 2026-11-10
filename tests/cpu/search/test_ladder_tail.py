"""The rung ladder (search/ladder.py) and the tail as a set (search/tail.py): shared components of the candidates
(possible_search_algorithm/20 sections 4 and 5)."""
import math
from types import SimpleNamespace

import pytest

from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.ladder import NOTE, RungLadder, account, lowfi_spent
from infrarsi.search.tail import TailScreen
from test_search import TOGGLE, setup


class FakeEngine:
    """What the ladder reads of an engine: the session's rows and budget, the costs, the noise, the incumbent."""

    def __init__(self, screens=10, full=1.0, probe=0.35, short=0.5, rows=(), t0=2.0, t_inc=None, inc='c0'):
        self.s = SimpleNamespace(screens=list(rows), budget=SimpleNamespace(screens=screens))
        self._costs = {'full': full, 'probe': probe, 'short': short}
        self.warmup, self.short_steps = 3, 10
        self._t = {'c0': t0, inc: t_inc or t0}
        self._inc = inc

    def costs(self):
        return dict(self._costs)

    def sigma_run(self):
        return 0.015

    def sigma_step(self):
        return 0.03

    def sigma_pair(self):
        return 0.01

    def t(self, cid):
        return self._t.get(cid)

    def incumbent(self):
        return self._inc


def test_the_ladder_plans_the_rungs_by_feasibility_warmup_and_allowance():
    lad = RungLadder()
    eng = FakeEngine()
    assert lad.plan(eng, 0.5) == ('probe', 'short') and lad.plan(eng, 0.9) == ('short',)
    assert lad.plan(eng, 0.9, warmup=6) == ('full',) and lad.plan(eng, 0.5, warmup=6) == ('probe', 'full')
    assert RungLadder(short=False).plan(eng, 0.5) == ('probe', 'full')
    assert lad.plan(eng, 0.9, probe=True) == ('probe', 'short')
    spent = FakeEngine(rows=[{'fidelity': 'probe', 'gpu_hours': 2.4}, {'fidelity': 'short', 'gpu_hours': 2.4},
                             {'fidelity': 'full', 'gpu_hours': 3.0}])
    assert lowfi_spent(spent) == pytest.approx(4.8)
    assert lad.allowance(spent) == pytest.approx(5.0)                # half the screens' GPU-hours
    assert lad.affordable(spent, 'probe') is False and lad.plan(spent, 0.5) == ('full',)
    assert lad.counts['lowfi_capped'] >= 1


def test_a_short_run_is_promoted_only_when_it_may_beat_the_incumbent():
    lad = RungLadder()
    eng = FakeEngine(t0=2.0, t_inc=1.6, inc='c5')                   # the incumbent: log(2 / 1.6) = 22.3 %
    s = lad.sigma_short(eng)
    assert s == pytest.approx(math.sqrt(0.015 ** 2 + 0.03 ** 2 / 7))
    near = Outcome('c7', 'short', True, 1.62, math.log(2.0 / 1.62), 'short', 0.1, [], [])
    far = Outcome('c8', 'short', True, 1.9, math.log(2.0 / 1.9), 'short', 0.1, [], [])
    failed = Outcome('c9', 'invalid', False, None, None, 'short', 0.1, ['out of memory'], [])
    assert lad.promote(eng, near) and not lad.promote(eng, far) and not lad.promote(eng, failed)
    assert lad.counts['promoted'] == 1 and lad.counts['not_promoted'] == 2
    assert RungLadder().promote(FakeEngine(), Outcome('c1', 'short', True, 2.0, -0.005, 'short', 0.1, [], []))
    a = account(FakeEngine(rows=[{'fidelity': 'probe', 'gpu_hours': 0.3}, {'fidelity': 'full', 'gpu_hours': 1.0}]), 0.5)
    assert a['by_fidelity'] == {'full': {'runs': 1, 'gpu_hours': 1.0}, 'probe': {'runs': 1, 'gpu_hours': 0.3}}
    assert a['lowfi_share_of_gpu_hours'] == pytest.approx(0.231, abs=1e-3) and a['note'] == NOTE.format(share=0.5)
    assert 'not counted as screens' in a['note'] and 'counted in the search' in a['note']


def test_a_short_run_and_its_full_run_are_two_engine_keys_on_one_candidate():
    _, s, eng = setup(TOGGLE, tokens=0)
    cfg = {**s.candidates['c0'].config, **{k: v for k, v in [('fused_adam', True)]}}
    short = Proposal(change=dict(cfg), absolute=True, fidelity=('short',), origin='test')
    out = eng.evaluate(short)
    assert out.status == 'short' and out.gain is not None and s.ledger.used['screens'] == 0
    assert s.ledger.used.get('trials') == 1 and s.ledger.used['gpu_hours'] > 0   # GPU-hours, not a screen
    full = Proposal(change=dict(cfg), absolute=True, fidelity=('full',), origin='test')
    assert eng.key(full) != eng.key(short) and eng._fresh(full)
    out2 = eng.evaluate(full)
    assert out2.status == 'screened' and out2.cid == out.cid and s.ledger.used['screens'] == 1


def test_the_tail_accepts_regresses_into_a_bounded_bisection_or_stops():
    eng = FakeEngine()
    vals = [('a', 1, 0.010, 'input'), ('b', True, 0.008, 'input'), ('c', 2, 0.006, 'comm'), ('d', 'x', 0.004, 'comm'),
            ('e', False, -0.01, 'other'), ('a', 2, 0.002, 'input')]
    t = TailScreen()
    first = t.start(vals)
    assert [m[0] for m in first] == ['a', 'b', 'c', 'd'] and first[0][1] == 1   # positive only, one value per knob
    assert t.threshold(eng) == pytest.approx(max(math.log1p(0.02), 2 * eng.sigma_pair()))
    assert t.observe(eng, 0.03) is None and t.done and [m[0] for m in t.accepted] == ['a', 'b', 'c', 'd']
    t = TailScreen()
    t.start(vals)
    nxt = t.observe(eng, -0.05)                                         # a regression: by family
    assert sorted(m[3] for m in nxt) in (['input', 'input'], ['comm', 'comm'])
    third = t.observe(eng, -0.04)                                       # the other family next
    assert third is not None and {m[3] for m in third} != {m[3] for m in nxt}
    assert t.observe(eng, 0.03) is None and t.diag == 2 and t.accepted == third
    t = TailScreen()
    t.start(vals)
    assert t.observe(eng, 0.005) is None and t.done and not t.accepted  # within the threshold: nothing more
    assert TailScreen().start([('x', 1, -0.01, None)]) is None
    t = TailScreen()
    t.start([('p', 1, 0.01, None), ('q', 1, 0.009, None), ('r', 1, 0.008, None)])
    halves = t.observe(eng, None)                                       # failed: a regression; halves without families
    assert [m[0] for m in halves] == ['p', 'q'] and len(t.describe()['history']) == 1


def test_the_next_runs_cost_is_at_least_what_passed_runs_cost_and_a_rung_can_be_skipped():
    rows = [{'fidelity': 'probe', 'gpu_hours': 0.05, 'valid': False}, {'fidelity': 'probe', 'gpu_hours': 0.8, 'valid': True}]
    eng = FakeEngine(rows=rows, probe=0.425)                          # (the engine's estimate: the mean of both)
    lad = RungLadder()
    assert lad.next_cost(eng, 'probe') == pytest.approx(0.8) and lad.next_cost(eng, 'short') == pytest.approx(0.5)
    assert lad.plan(FakeEngine(), 0.5, short=False) == ('probe', 'full') and lad.plan(FakeEngine(), 0.9, short=False) == ('full',)
    assert RungLadder(short=False).plan(FakeEngine(), 0.9, short=True) == ('full',)   # (never more rungs than the ladder has)


def test_the_allowance_follows_the_budget_that_binds():
    from infrarsi.search.ladder import allowance_of
    eng = FakeEngine(screens=10, full=1.0)
    assert allowance_of(eng, 0.5) == pytest.approx(5.0)                  # the screens bind: half of 10 full runs
    eng.s.budget.gpu_hours = 1000.0                                     # generous GPU-hours: the same
    assert allowance_of(eng, 0.5) == pytest.approx(5.0) == pytest.approx(RungLadder().allowance(eng))
    eng.s.budget.screens, eng.s.budget.gpu_hours = 64, 16.0             # the GPU-hours bind first (16 full runs, 64 screens)
    assert allowance_of(eng, 0.5) == pytest.approx(0.5 * 16.0 / 1.5)    # the full runs keep two thirds of them
    assert account(eng, 0.5)['allowance_gpu_hours'] == pytest.approx(round(0.5 * 16.0 / 1.5, 4))
