"""Regressions of the Codex review CAND_r1 (reports/infrarsi_newmodels_20261028/reviews/CAND_r1_out.md), one per defect:
C1 esls's lattice loses an invariant-coupled spender; C2 rpp's pre-sampled pool skews the draw and ends the arm on a
false-empty pool; C3 the ledger's evidence crosses code overlays; C4 the ledger loses wcs's final class (the jitter
cliff) and esls's cliff; C5 the ledger bypasses wcs's measured dominance; C6 a malformed prior crashes esls; C7 the
hard prune rests on one overwritten enabler value; C8 a plan's probe and short run pass the allowance together and code
proofs bypass it. (C9, the control's experience store, is in tests/cpu/project/test_candidates_project.py.)"""
import json
import math

import pytest

from cand_helpers import toy
from infrarsi.search.engine import Proposal
from infrarsi.search.esls import ESLS
from infrarsi.search.ladder import RungLadder
from infrarsi.search.rpp import RPP, weighted_draw
from infrarsi.search.toytrain import mem_tasks
from infrarsi.search.wcs import vk
from test_ladder_tail import FakeEngine
from test_wcs import START, PriorProposer, answer, engine
from test_wcs_r1 import START4, coupled

TIGHT = 'mem-small-c80-tight-free-low'


def js(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


def flat(seed=3, **kw):
    eng = engine()
    r = RPP(PriorProposer(None), facts={'cap_gib': 100.0}, seed=seed, **kw)
    r._init(eng)
    return r, eng


class RawPrior:
    """A proposer whose prior answer is kept as the model gave it (`last_prior`), however malformed."""

    def __init__(self, answer_):
        self.answer, self.calls, self.last_prior = answer_, 0, None

    def knob_prior(self, engine_, base):
        self.calls += 1
        self.last_prior = self.answer
        return self.answer

    def describe(self):
        return {'kind': 'RawPrior'}


# ---------------------------------------------------------------- C1
def test_c1_a_coupled_spender_is_completed_from_the_legal_table_not_dropped():
    eng = coupled()                                                   # mb x ga == 2; the reference mb 1, ga 2
    e = ESLS(PriorProposer(answer()), facts={'cap_gib': 100.0, 'coupled': [['ga', 'mb']]}, seed=3)
    e._init(eng)
    c0 = eng.s.candidates['c0'].config
    assert all(eng.s.check_config({**c0, **c}, None)[0] is not None for c in e.legal_levels)
    mb2 = [c for c in e.legal_levels if c['mb'] == 2]
    assert mb2 and all(c['ga'] == 1 for c in mb2)                     # before: no level held micro-batch 2
    assert any(r['cfg']['mb'] == 2 for r in e.lattice)                # ... and the chain reaches it
    assert all({k for k in e.knobs if r['cfg'][k] != START4[k]} <= set(e.spender_knobs) | {'ga'} for r in e.lattice)


# ---------------------------------------------------------------- C2
def test_c2_the_draw_weighs_every_remaining_configuration_and_never_ends_on_a_partial_pool(monkeypatch):
    import infrarsi.search.rpp as rpp_mod
    monkeypatch.setattr(rpp_mod, 'POOL', 10, raising=False)          # before: a uniform 10 of the 143 left were scored
    r, eng = flat()

    def key(cfg):
        return js({k: cfg[k] for k in r.knobs})
    only = {**START, 'rc': 'half', 'mb': 2, 'tf32': True, 'wk': 4}
    r._pf = lambda cfg, ids=(): ((0.9 if key(cfg) == key(only) else 0.01), 'planted')
    draws = [weighted_draw(r, eng, 'c0', a=1.0, T=math.inf, salt=i) for i in range(12)]
    assert all(d is not None and key(d[0]) == key(only) for d in draws)   # before: None (a false-empty pool) mostly

    class Stub:                                                       # the annealed weight: one dominant configuration
        @staticmethod
        def value(feats, mean):
            return 0.3 if feats == r._feats(only) else 0.0
    monkeypatch.setattr(rpp_mod, 'Surrogate', Stub)
    r._pf = lambda cfg, ids=(): (1.0, 'planted')
    hot = [weighted_draw(r, eng, 'c0', a=1.0, T=0.005, salt=i)[0] for i in range(12)]
    assert all(key(c) == key(only) for c in hot)                     # before: only when it entered the 10
    r._pf = lambda cfg, ids=(): (0.01, 'planted')                     # nothing clears P_MIN: a draw still, not the end
    assert weighted_draw(r, eng, 'c0', a=1.0, T=math.inf, salt=0) is not None


# ---------------------------------------------------------------- C3
def test_c3_ledger_evidence_does_not_cross_code_overlays():
    r, eng = flat()
    cfg = {**START, 'mb': 2}
    r.mledger.observe_run(cfg, 'oom', oom={'need_gib': 95.0}, source='m1')     # out of memory without any patch
    assert r._pf(cfg)[0] == 0.0
    r.patches['p1'] = {'status': 'proven', 'mem_dir': 'frees', 'mem_gib': 10.0}
    assert r._pf(cfg, ('p1',))[0] > 0.05                              # before: 0.0 - the unpatched run's status
    ok = {**START, 'tf32': True}
    r.mledger.observe_run(ok, 'ok', peak=60.0, source='m2')           # fitted without any patch
    assert r._pf(ok)[0] == 1.0
    r.patches['p2'] = {'status': 'proven', 'mem_dir': 'spends', 'mem_gib': 50.0}
    assert r._pf(ok, ('p2',))[0] < 0.5                                # before: 1.0 - the unpatched run's status


# ---------------------------------------------------------------- C4
def test_c4_the_ledger_takes_wcs_s_final_class_and_esls_s_cliff(monkeypatch):
    r, eng = flat()
    cfg = {**START, 'rc': 'half'}
    monkeypatch.setattr(r, '_jitter', lambda *a, **k: 'thrash')     # wcs's last word on the run: the cliff (F23)
    p = Proposal(change=cfg, absolute=True, hypothesis='j1', origin='rpp')
    r.ideas['j1'] = frozenset()
    out = eng.evaluate(p)
    r.observe(eng, p, out)
    assert out.status == 'screened' and r.runs[out.cid]['kind'] == 'thrash'
    assert r.mledger.ran_status(cfg) == 'thrash' and r._pf(cfg)[0] == 0.0     # before: 'ok', 1.0
    # esls's boundary probe that its cliff rule rejects: an interval (thrash), not an exact fit
    for t in mem_tasks():                                             # (a task with a spend level that fits)
        eng2, sim, sp, facts = toy(t)
        e = ESLS(sp(), facts=facts, seed=1, off=['esls_enabler_probe'])
        e._init(eng2)
        fit = [(i, r_['cfg']) for i, r_ in enumerate(e.lattice) if i and sim.truth(r_['cfg'])[1] is None]
        if fit:
            break
    i, level = fit[0]
    e.cur = {'set': (), 'lo': 0, 'hi': len(e.lattice), 'probes': 1, 'ledge': False, 'mid': i}
    q = e._probe(eng2, 'c0', level, 'bound', 'the boundary')
    out2 = eng2.evaluate(q)
    monkeypatch.setattr(e, '_cliff', lambda engine_, out_: True)
    e.observe(eng2, q, out2)
    assert out2.status == 'probed' and e.cur['hi'] == i
    assert e.mledger.ran_status(level) == 'thrash' and e._pf(level)[0] == 0.0  # before: 'ok', 1.0
    assert any(o[1] == 'interval' for o in e.mledger.obs)


# ---------------------------------------------------------------- C5
def test_c5_wcs_s_measured_dominance_rules_out_before_the_ledger():
    r, eng = flat()
    oom = {**START, 'rc': 'none', 'mb': 2}
    r.mem.record('m1', r._m_of(oom), 'oom')                          # a measured out-of-memory run
    r.mledger.observe_run(oom, 'oom', oom={'need_gib': 101.0}, source='m1')
    r.mem.relate(r._m_of(START), r._m_of({**START, 'tf32': True}), 'neutral')  # a move the runs measured neutral
    r._forget()
    worse = {**oom, 'tf32': True}                                    # the out-of-memory run plus that neutral move
    assert r._p_feasible(worse)[0] == 0.0                             # wcs: infeasible
    assert r._pf(worse)[0] == 0.0                                     # before: the ledger's 0.24
    fitted = {**START, 'rc': 'none', 'mb': 2, 'tf32': True}           # a run of it that fitted: its own result first
    r.mledger.observe_run(fitted, 'ok', peak=90.0, source='m3')
    assert r._pf(fitted)[0] == 1.0


# ---------------------------------------------------------------- C6
@pytest.mark.parametrize('bad', [{'levels': 5}, {'levels': [{'knob': ['rc'], 'value': 'half', 'family': 'activations'}]},
                                 {'levels': [5, None, {'knob': 'rc', 'value': [1], 'family': 3}]}, ['not', 'a', 'dict']])
def test_c6_a_malformed_prior_degrades_esls_as_it_degrades_wcs(bad):
    eng = engine()
    e = ESLS(RawPrior(bad), facts={'cap_gib': 100.0}, seed=3)
    e._init(eng)                                                      # before: TypeError
    assert e.prior is not None and e.propose(eng, 10)


# ---------------------------------------------------------------- C7
def test_c7_the_hard_prune_needs_every_allowed_enabler_assignment():
    eng = engine()
    e = ESLS(PriorProposer(answer()), facts={'cap_gib': 100.0}, seed=3, off=['esls_enabler_probe'])
    e._init(eng)
    strong, weak = 0, 4                                               # two enabler values of one knob, the weaker last
    e.enablers = {('wk', vk(v)): {'knob': 'wk', 'value': v, 'status': 'candidate', 'peak': None, 'frees': None,
                                  'declared': None} for v in (strong, weak)}

    def lb(cfg, *args, **kwargs):
        top = cfg.get('rc') == 'none' and cfg.get('mb') == 2
        return (101.0 - (31.9 if cfg.get('wk') == strong else 0.1 if cfg.get('wk') == weak else 0.0)) if top else 10.0
    e.order_ledger.lower_bound = lb
    e._build_chain()
    top = js({**e.ref, 'rc': 'none', 'mb': 2})
    assert not any(p['lower_bound_gib'] > 100 for p in e.pruned)       # before: pruned at 100.9 (the weak value)
    assert any(js(r['cfg']) == top for r in e.lattice)


# ---------------------------------------------------------------- C8
def test_c8_the_allowance_holds_for_a_whole_plan_and_for_code_proofs(monkeypatch):
    rows = [{'fidelity': 'probe', 'gpu_hours': 1.1, 'valid': False}] * 4      # 4.4 of the 5.0 spent
    eng = FakeEngine(screens=10, full=1.0, probe=0.35, short=0.5, rows=rows)
    lad = RungLadder()
    assert lad.allowance(eng) == pytest.approx(5.0)
    assert lad.plan(eng, 0.5) == ('probe', 'full')                   # before: ('probe', 'short') - 5.25 of 5.0
    assert lad.plan(eng, 0.9) == ('short',)                           # (a short run alone still fits: 4.9)
    held = RungLadder(code_reserve=0.2)                               # a code track: a fifth is the code proofs' pool
    assert held.allowance(eng) == pytest.approx(4.0) and held.allowance(eng, code=True) == pytest.approx(1.0)
    assert held.plan(eng, 0.5) == ('full',)                           # the knob side's pool (4.0) is spent
    assert held.affordable(eng, 'probe', code=True) and not held.affordable(eng, 'probe')
    held.code_cids.add('c9')                                          # a code proof's probe of 0.8: its own pool's
    eng.s.screens.append({'fidelity': 'probe', 'gpu_hours': 0.8, 'valid': True, 'candidate_id': 'c9'})
    assert held.spent(eng, code=True) == pytest.approx(0.8) and held.spent(eng) == pytest.approx(4.4)
    assert not held.affordable(eng, 'probe', code=True)               # 0.8 + 0.8 (the passed probes' mean) > 1.0
    eng2, sim, sp, facts = toy(mem_tasks()[0])
    for make in (lambda: RPP(sp(), facts=facts, seed=1), lambda: ESLS(sp(), facts=facts, seed=1)):
        st = make()
        st._init(eng2)
        cfg = next(c for c, _ in st._candidates(eng2, 'c0'))
        q = st._knob_q(eng2, 'c0', cfg, kind='patch', plan=('probe', 'full'))   # a code proof's probe-first plan
        monkeypatch.setattr(st, '_code', lambda engine_, inc, left, _q=q: [_q])
        monkeypatch.setattr(st.ladder, 'affordable', lambda engine_, f, **kw: f != 'probe')   # no probe left in it
        got = st.propose(eng2, 16) if isinstance(st, RPP) else [st._step_code(eng2, 'c0', 10)]
        assert got[0] is not q                                        # before: the proof's probe past the allowance
