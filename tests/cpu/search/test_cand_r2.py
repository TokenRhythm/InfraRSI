"""Regressions of the Codex review CAND_r2 (reports/infrarsi_newmodels_20261028/reviews/CAND_r2_out.md): R2-1 the two
low-fidelity pools strand a probe the whole allowance could fund; R2-2 a full-table draw after many observations redoes
the measured-dominance scan for every configuration left."""
import math
import random
import time

import pytest

from hls2_helpers import SimEngine, SimSession
from infrarsi.search.ladder import RungLadder
from infrarsi.search.rpp import RPP, weighted_draw
from test_ladder_tail import FakeEngine


# ---------------------------------------------------------------- R2-1
def test_r2_1_a_probe_the_two_pools_strand_is_funded_by_one_transfer():
    def probe(cid):
        return {'fidelity': 'probe', 'gpu_hours': 0.35, 'valid': True, 'candidate_id': cid}
    rows = [probe('k1'), probe('k2')] + [probe(f'n{i}') for i in range(11)]       # code 0.70, knob side 3.85
    eng = FakeEngine(screens=10, full=1.0, probe=0.35, short=0.5, rows=rows)
    lad = RungLadder(code_reserve=0.2)
    lad.code_cids |= {'k1', 'k2'}
    assert lad.allowance(eng, code=True) == pytest.approx(1.0) and lad.allowance(eng) == pytest.approx(4.0)
    assert lad.spent(eng, code=True) == pytest.approx(0.70) and lad.spent(eng) == pytest.approx(3.85)
    assert lad.affordable(eng, 'probe', code=True)                    # before: False - 0.30 + 0.15 stranded
    assert lad.allowance(eng, code=True) + lad.allowance(eng) == pytest.approx(5.0)   # moved, not created
    eng.s.screens.append(probe('k3'))                                 # the code proof ran: 0.10 left in all
    lad.code_cids.add('k3')
    assert not lad.affordable(eng, 'probe', code=True) and lad.plan(eng, 0.5) == ('full',)   # charged once
    knob = RungLadder(code_reserve=0.2)                               # the knob side, the same balances
    knob.code_cids |= {'k1', 'k2'}
    eng2 = FakeEngine(screens=10, full=1.0, probe=0.35, short=0.5, rows=rows)
    assert knob.plan(eng2, 0.5) == ('probe', 'full')                 # before: ('full',)
    assert knob.allowance(eng2, code=True) + knob.allowance(eng2) == pytest.approx(5.0)
    keep = RungLadder(code_reserve=0.2)                               # a pool that can still fund its own probe keeps it
    eng3 = FakeEngine(screens=10, full=1.0, probe=0.35, short=0.5, rows=[probe(f'n{i}') for i in range(11)])
    assert not keep.affordable(eng3, 'probe') and keep.affordable(eng3, 'probe', code=True)


# ---------------------------------------------------------------- R2-2
class BigPrior:
    calls = 0

    def __init__(self, n):
        self.n = n

    def knob_prior(self, engine, base):
        return {'levels': [{'knob': f'b{i}', 'value': True, 'gain_pct': i - 4, 'sd_pct': 3,
                            'memory': {'direction': 'spends' if i % 3 == 0 else 'frees', 'gib': 5 + i}}
                           for i in range(self.n)], 'interactions': [], 'configs': [], 'notes': ''}

    def describe(self):
        return {}


def big_table(n_bool):
    knobs = {f'b{i}': [False, True] for i in range(n_bool)}
    knobs['c2'] = ['x', 'y']
    knobs['c3'] = [1, 2, 3]
    start = {k: v[0] for k, v in knobs.items()}
    eng = SimEngine(SimSession(start, knobs=knobs, screens=10), lambda base, cfg, ov: (0.0, None), sigma_runs=[0.01])
    r = RPP(BigPrior(n_bool), facts={'cap_gib': 80.0}, seed=1)
    r._init(eng)
    return r, eng, start


def observe_24(r, start):
    """12 runs that fitted, each one knob from the reference (a measured move each), and 12 out of memory."""
    rng = random.Random(7)
    keys = [k for k in r.knobs if k.startswith('b')]
    for i in range(12):
        cfg = {**start, keys[i % len(keys)]: True}
        r.mem.record(f'ok{i}', r._m_of(cfg), 'ok', peak=50.0 + i)
        r.mem.relate(r._m_of(start), r._m_of(cfg), ('spends', 'neutral', 'frees')[i % 3])
        r.mledger.observe_run(cfg, 'ok', peak=50.0 + i, source=f'ok{i}')
    for i in range(12):
        cfg = {**start, **{k: rng.random() < 0.5 for k in keys}, 'c3': rng.choice([1, 2, 3])}
        r.mem.record(f'oom{i}', r._m_of(cfg), 'oom')
        r.mledger.observe_run(cfg, 'oom', oom={'need_gib': 90.0}, source=f'oom{i}')
    r._forget()


def test_r2_2_the_first_draw_after_many_observations_stays_near_the_first_draw():
    r, eng, start = big_table(10)                                     # 6,144 configurations
    t = time.perf_counter()
    assert weighted_draw(r, eng, 'c0', a=1.0, T=math.inf, salt=0) is not None
    first = time.perf_counter() - t
    observe_24(r, start)
    t = time.perf_counter()
    got = weighted_draw(r, eng, 'c0', a=1.0, T=math.inf, salt=1)
    after = time.perf_counter() - t
    assert got is not None and got[4]['pool'] == len(r._candidates(eng, 'c0'))   # (the whole table still weighed)
    assert after <= 3.0 * first + 0.5, (first, after)                # before: about 12x the first draw
    # the same verdicts as wcs's own measured-only dominance (MemoryModel.predict, hard = the measured moves)
    rows = [c for c, _ in r._candidates(eng, 'c0')]
    for cfg in random.Random(3).sample(rows, 400):
        m = r._m_of(cfg)
        verdict = r.mem.predict(m, r._known(), r.runs['c0']['cfg'], hard=r.mem.measured)[0]
        assert (r._measured_out(m) is not None) == (verdict == 'infeasible'), cfg
