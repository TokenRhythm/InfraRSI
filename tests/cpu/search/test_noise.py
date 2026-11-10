"""Measured between-run noise (Fable H1 decision 3): the split halves of every run's scoring window and the pairs of runs
of one configuration replace the fixed 2 % floor; confirmation re-runs add pairs; a screening design combines only the
factors above its resolution and hands the smaller positives to HLS (decision 5.2)."""
import math
from types import SimpleNamespace

import pytest

from infrarsi.project.signals import split_half
from infrarsi.search.engine import Outcome, Proposal, pooled_sigma_pair, white_floor
from infrarsi.search.strategies import DOE, HLS, pb_design
from infrarsi.search.toytrain import ToySim, ToyTask
from test_search import TOGGLE, setup

QUIET = ToyTask('7b', 'h100', 'toggle', 'none', 'low', gain_scale=0.4, run_sd=0.0007, drift=True, cv=0.0015)


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def test_split_half_on_the_clock_and_on_reported_times():
    rows = [{'step': s, 't': 1.0 * s if s <= 20 else 20 + 1.02 * (s - 20)} for s in range(1, 31)]
    assert split_half(rows, [11, 30]) == pytest.approx(math.log(1.02))
    reported = [{'step': s, 'step_time': 1.0 if s <= 20 else 1.02} for s in range(1, 31)]
    assert split_half(reported, [11, 30]) == pytest.approx(math.log(1.02))
    assert split_half(rows, [30, 30]) is None


def test_measured_noise_pools_pairs_and_split_halves():
    sim, s, eng = setup(TOGGLE)                                   # simulator engines keep the fixed floor
    assert eng.noise()['mode'] == 'fixed'
    assert eng.sigma_pair() == pytest.approx(math.sqrt(2) * max(0.015, eng.sigma_step()))
    sim, s, eng = setup(TOGGLE, run_noise=None, noise_pairs=[0.002, -0.001])
    halves = eng.split_halves()
    floor = white_floor(s.candidates['c0'].receipt['evidence'])
    assert len(halves) == 1 and eng.noise()['mode'] == 'measured'
    assert eng.sigma_pair() == pytest.approx(max(pooled_sigma_pair([0.002, -0.001], halves), floor))
    assert eng.sigma_run() == pytest.approx(eng.sigma_pair() / math.sqrt(2))
    eng.pairs, s.candidates['c0'].receipt['evidence']['split_half'] = [], None
    assert eng.sigma_pair() == pytest.approx(math.sqrt(2) * max(0.02, eng.sigma_step()))   # nothing measured


def test_split_halves_track_the_between_run_noise_when_runs_drift():
    task = ToyTask('7b', 'h100', 'toggle', 'none', 'high', run_sd=0.04, drift=True)
    sim, s, eng = setup(task)
    times, halves = [], []
    for _ in range(60):                                           # one configuration, 60 runs (a test of the estimator)
        ev = s.runner.run(s.task, sim.reference(), None, 'rescore', 't', candidate_id='c0', steps=30)['evidence']
        times.append(ev['step_time'])
        halves.append(ev['split_half'])
    pairs = [math.log(b / a) for a, b in zip(times[0::2], times[1::2])]
    target = math.sqrt(2) * 0.04                                  # sigma_pair of the simulator
    assert abs(rms(halves) / target - 1) < 0.3 and abs(rms(pairs) / target - 1) < 0.3
    assert abs(pooled_sigma_pair(pairs, halves) / target - 1) < 0.3


def test_a_small_lever_moves_the_incumbent_alone_on_a_quiet_node():
    sim = ToySim(QUIET)
    ref = sim.reference()
    change = {'pin_memory': not ref.get('pin_memory')}
    t0, t1 = sim.truth(ref, {})[0], sim.truth(dict(ref, **change), {})[0]
    assert 0.003 < math.log(t0 / t1) < 0.008                      # a 0.5 % lever
    for run_noise, moves in ((None, True), (0.015, False)):       # measured noise vs the old fixed floor
        _, s, eng = setup(QUIET, run_noise=run_noise)
        out = eng.evaluate(Proposal(change=change, mu=0.005, abortable=False))
        assert out.status == 'screened' and (eng.incumbent() == out.cid) is moves


def test_confirmation_reruns_add_pairs(monkeypatch):
    sim, s, eng = setup(TOGGLE, run_noise=None)
    out = eng.evaluate(Proposal(change={'fused_adam': True}, mu=0.02, abortable=False))
    t_inc, t_c = eng.t_screen('c0'), eng.t_screen(out.cid)
    monkeypatch.setattr(s, 'rescore', lambda c: {'valid': True, 'step_time': (t_inc if c == 'c0' else t_c) * 1.001})
    eng._inc = 'c0'
    eng.confirm(out.cid, 0.01)
    assert eng.pairs == pytest.approx([math.log(1.001)] * 2)


class FakeEngine:
    def __init__(self, sigma_pair=0.0014):
        self.s = SimpleNamespace(candidates={'c0': SimpleNamespace(config={f'f{j}': False for j in range(8)}, overlay={})},
                                 hooks=None)
        self._sp = sigma_pair

    def incumbent(self):
        return 'c0'

    def sigma_pair(self):
        return self._sp

    def sigma_run(self):
        return self._sp / math.sqrt(2)


def test_design_combines_resolved_factors_and_hands_on_the_small_ones():
    beta = [-0.043, -0.020, -0.005, -0.006, -0.005, -0.007, -0.004, 0.041]   # log step time per factor (+ level)
    doe = DOE(None)
    doe.factors = [{'name': f'f{j}', 'plus': {f'f{j}': True}} for j in range(8)]
    doe.design, doe.base, doe.state = pb_design(8), 'c0', 'combine'
    doe.y = {i: math.exp(sum(b * x / 2 for b, x in zip(beta, row))) for i, row in enumerate(doe.design)}
    eng = FakeEngine()
    combo = doe.propose(eng, 16)[0]
    assert combo.change == {'f0': True, 'f1': True}               # 4/12 of the largest effect is the resolution
    assert doe.resolution == pytest.approx(0.043 / 3)
    assert [h['mechanism'] for h in doe.handoff] == ['doe:f2', 'doe:f3', 'doe:f4', 'doe:f5', 'doe:f6']
    assert doe.describe()['below_resolution'] == ['f2', 'f3', 'f4', 'f5', 'f6']
    hls = HLS(SimpleNamespace(describe=dict, hypotheses=lambda *a, **k: []), explore=False)
    hls.receive(eng, doe.handoff)
    assert {h['status'] for h in hls.ledger.values()} == {'unresolved'}
    bundle = next(p for p in hls.propose(eng, 16) if p.origin == 'hls-bundle')
    assert sorted(bundle.change) == ['f2', 'f3', 'f4', 'f5', 'f6'] and bundle.confirm
    hls.observe(eng, bundle, Outcome('c1', 'screened', True, 1.0, -0.001, 'full', 0.1, [], []))
    assert len(hls.pending) == 2                                  # a failed bundle of five is split


# ---------------------------------------------------------------- review R7 (gpt-6.1-sol)
def test_rescore_split_halves_join_the_pool():
    sim, s, eng = setup(TOGGLE, run_noise=None)
    n = len(eng.split_halves())
    r = s.rescore('c0')                                           # a valid full re-run of the reference
    assert r['valid'] and type(r['split_half']) is float and eng.split_halves() == eng.split_halves()[:n] + [r['split_half']]


def test_zero_measured_noise_keeps_the_white_noise_floor():
    assert pooled_sigma_pair([], [0.0, 0.0]) == 0.0 and pooled_sigma_pair([], []) is None
    sim, s, eng = setup(TOGGLE, run_noise=None)
    s.candidates['c0'].receipt['evidence']['split_half'] = 0.0
    assert eng.sigma_pair() == pytest.approx(white_floor(s.candidates['c0'].receipt['evidence']))


def test_white_noise_floor_counts_the_scoring_window():
    assert white_floor({'rsd_window': 0.02, 'window': [11, 30]}) == pytest.approx(math.sqrt(2) * 0.02 / math.sqrt(20))
    assert white_floor({'rsd_window': None, 'window': [11, 30]}) == 0.0
