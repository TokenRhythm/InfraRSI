"""hls2 on the recorded Q9U-code-env-opus run and on ToyTrain (FABLE_DESIGN_SEARCH_FIX T9, T10).

T10: the replay of the recorded run (data/q9u_code_env_opus.json: the model's hypotheses per incumbent, the noise and
the run costs of each decision, the measured effects) reproduces, with hls, the ten recorded decisions (best value/cost
to 1e-3, the same choices, the same number of proposals) - the engine's arithmetic and hls's priors as they ran; with
hls2 the Liger knob, never tested in the recorded ten screens, is screened, and every move the transfer judgment calls
'+' is measured. T9: a planted job (a +8 % knob the model rates 2 %, five +0.1 % patches it rates 3 %) - hls2 tests the
knob within four screens, hls does not; and ToyTrain's 36 tasks at 16 runs: hls2 within 0.02 of hls in paired frac,
with no larger share of GPU-hours on failed runs."""
import math
import statistics

import pytest

from hls2_helpers import BASE_TEXT, FIXTURE, PATCH_FILE, SimEngine, SimSession, q9u_replay, snippet
from infrarsi.search import toybench
from infrarsi.search.strategies import HLS, HLS2
from infrarsi.search.toytrain import tasks
from test_composition import FakeProposer


def screened(eng, knob, value) -> list[int]:
    """The screens (1-based, in order) whose change set knob = value."""
    return [i + 1 for i, (_, change, status, _, _) in enumerate(eng.order)
            if change.get(knob) == value and status in ('screened', 'aborted', 'invalid', 'infeasible')]


# ---------------------------------------------------------------- T10
def test_replica_regression_hls_reproduces_the_recorded_run():
    eng, hls, trace = q9u_replay(lambda prop, **kw: HLS(prop, **kw))
    rec = FIXTURE['decisions']
    assert len(eng.decisions) == len(rec) == 10 and trace.stop_reason == 'budget:screens'
    for i, (d, r) in enumerate(zip(eng.decisions, rec)):
        assert d['best_value_per_cost'] == pytest.approx(r['best_value_per_cost'], abs=1e-3), i
        assert d['best_mu'] == pytest.approx(r['best_mu'], abs=1e-4), i
        assert d['best_mechanism'][:60] == r['best_mechanism'] and d['n_open'] == r['n_open'], i
    moved = [o[4] for o in eng.order]
    assert sorted(set(moved), key=moved.index) == ['c1', 'c2', 'c3', 'c8'] and trace.selected == 'c8'
    assert hls.kappa() == pytest.approx(FIXTURE['kappa'], abs=1e-3)
    assert screened(eng, 'use_liger_kernel', True) == []          # the finding: Liger never got its turn


def test_replica_hls2_screens_the_liger_knob_and_every_judged_move():
    old, _, _ = q9u_replay(lambda prop, **kw: HLS(prop, **kw))
    eng, hls2, trace = q9u_replay(lambda prop, **kw: HLS2(prop, **kw))
    assert len(eng.order) == 10 and trace.stop_reason == 'budget:screens'
    first = screened(eng, 'use_liger_kernel', True)
    assert first and first[0] <= 10
    for a in FIXTURE['judgment']:                                   # R1: every move judged '+' measured (or failed)
        if a['sign'] == '+':
            (k, v), = a['change'].items()
            assert screened(eng, k, v), a['knob']
    t0 = FIXTURE['c0']['step_time']
    assert eng.t(eng.incumbent()) < old.t(old.incumbent())
    print(f'\nhls x{t0 / old.t(old.incumbent()):.3f}, hls2 x{t0 / eng.t(eng.incumbent()):.3f}; hls2 screens: '
          + '; '.join(f'{i + 1} {m[:40]} ({st})' for i, (m, _, st, _, _) in enumerate(eng.order)))
    # the design's replay F/J (section 4.4): Liger 3, the conv patch 4, no-reshard 5, fused AdamW 6, compile 7
    assert first[0] == 3 and screened(eng, 'fsdp_config', 'fsdp_v2_noreshard.json')[0] == 5
    assert eng.t(eng.incumbent()) == pytest.approx(0.922, abs=0.002)


# ---------------------------------------------------------------- T9
PLANTED_TRUE = {'fast': 0.077, **{f'p{i}': 0.001 for i in range(5)}}


def planted_moves(cfg, overlay) -> set:
    text = (overlay or {}).get(PATCH_FILE, '')
    return ({'fast'} if cfg.get('fast') else set()) | {m for m in PLANTED_TRUE if f'def _m_{m}(' in text}


def planted(cls, screens: int = 6) -> SimEngine:
    """A cheap knob worth 8 % that the model rates 2 %, and five patches worth 0.1 % that it rates 3 %."""
    s = SimSession({'fast': False}, knobs={'fast': [False, True]}, screens=screens)

    def effect(base, cfg, overlay):
        new = planted_moves(cfg, overlay) - planted_moves(base.config, base.overlay)
        return sum(PLANTED_TRUE[m] for m in new), None
    eng = SimEngine(s, effect, sigma_runs=[0.005], full_costs=[0.3])
    hyps = [{'mechanism': 'fast knob', 'change': {'fast': True}, 'patch': None, 'mu': math.log1p(0.02), 'sigma': 0.02,
             'p_feasible': 0.9}] + \
        [{'mechanism': f'patch {i}', 'change': {}, 'patch': {PATCH_FILE: BASE_TEXT + snippet(f'p{i}')},
          'mu': math.log1p(0.03), 'sigma': 0.02, 'p_feasible': 0.9} for i in range(5)]
    eng.run(cls(FakeProposer(hyps)))
    return eng


def test_planted_cheap_knob():
    new, old = planted(HLS2), planted(HLS)
    assert screened(new, 'fast', True)[0] <= 4
    assert not [i for i in screened(old, 'fast', True) if i <= 4]   # hls: the over-rated patches (and their bundle) first


def test_toybench_hls2_keeps_up_with_hls():
    rows = {(t.name, arm): toybench.run_one(t, arm, 16) for t in tasks(36) for arm in ('hls', 'hls2')}
    names = sorted({t for t, _ in rows})
    d = [rows[(t, 'hls2')]['frac'] - rows[(t, 'hls')]['frac'] for t in names]
    inv = {arm: statistics.mean(rows[(t, arm)]['invalid_share'] for t in names) for arm in ('hls', 'hls2')}
    print(f'\npaired frac hls2 - hls {statistics.mean(d):+.4f} (wins {sum(x > 1e-6 for x in d)}, losses '
          f'{sum(x < -1e-6 for x in d)}); invalid share hls {inv["hls"]:.3f}, hls2 {inv["hls2"]:.3f}')
    assert statistics.mean(d) >= -0.02 and inv['hls2'] <= inv['hls'] + 1e-9
