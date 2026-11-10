"""wcs (search/wcs.py; FABLE_ARCH_REVIEW_20261031 section 4): the prior's parsing, the surrogate's posterior and draws,
the feasibility model (dominance, the pair table, the cliff), the acquisition (fresh, diverse, probe-first, the partner
probe, the coverage screen), the code track (patches only, the self-proof, a proven patch as a pseudo-knob), every
component off, the simulated model's prior, the LLMProposer's prior call and patch-only mode, the coupling task family,
and the toybench end to end (OOM, thrash, probes) on the memory-coupled job."""
import json
import math
import random
from types import SimpleNamespace

import pytest

from hls2_helpers import SimEngine, SimSession
from infrarsi.search import ABLATIONS, LLM_STRATEGIES, STRATEGIES, WCS_COMPONENTS, resolve
from infrarsi.search.engine import Proposal
from infrarsi.search.proposers import PATCH_ONLY_RULE, PRIOR_SCHEMA, PRIOR_SYSTEM, LLMProposer, SimProposer
from infrarsi.search.toybench import run_one
from infrarsi.search.toytrain import CoupleSim, CoupleTask, MemTask, ToyTask, couple_tasks, make_sim, mem_tasks
from infrarsi.search.wcs import WCS, Prior, Surrogate, vk

KNOBS = {'rc': ['full', 'half', 'none'], 'mb': [1, 2], 'liger': [False, True], 'comp': [False, True], 'tf32': [False, True],
         'wk': [0, 2, 4]}
START = {'rc': 'full', 'mb': 1, 'liger': False, 'comp': False, 'tf32': False, 'wk': 2}
# a landscape like G4U's: recompute off (+22 %) and micro-batch 2 (+8 %) spend memory; the declared enabler (the fused
# loss, +5 %, frees 10) makes half recompute + micro-batch 2 fit (+21 %), only the undeclared one (the compiler, -5 %
# alone, frees 30) makes recompute off fit (+25 % with micro-batch 2); the compiler and the fused loss crash together;
# tf32 and the loader do nothing
GAIN = {('rc', 'half'): 0.08, ('rc', 'none'): 0.22, ('mb', 2): 0.08, ('liger', True): 0.05, ('comp', True): -0.05,
        ('tf32', True): 0.0, ('wk', 0): -0.005, ('wk', 4): 0.002}
CAP = 40


def mem_of(c):
    return (0 if c['rc'] == 'full' else 20 if c['rc'] == 'half' else 55) + (10 if c['mb'] == 2 else 0) \
        - (10 if c['liger'] else 0) - (30 if c['comp'] else 0)


def landscape(base, cfg, overlay):
    """(log gain vs the base, failure) of the Engine's SimEngine: the landscape above."""
    mem = mem_of
    if cfg['comp'] and cfg['liger']:
        return None, 'timeout'
    if mem(cfg) > CAP:
        return None, 'oom'
    g = sum(GAIN.get((k, v), 0.0) for k, v in cfg.items() if v != START[k])
    gb = sum(GAIN.get((k, v), 0.0) for k, v in base.config.items() if v != START[k])
    if base.config.get('comp') and base.config.get('liger'):
        gb = 0.0
    d = g - gb
    return d, 'aborted' if d < -0.02 else None


def answer(memory=True):
    """A model's prior over KNOBS in PRIOR_SYSTEM's format (strings for numbers, a value no knob has, unknowns)."""
    lv = [{'knob': 'rc', 'value': 'half', 'gain_pct': 8, 'sd_pct': 4, 'memory': {'direction': 'spends', 'gib': 20}},
          {'knob': 'rc', 'value': 'none', 'gain_pct': '22', 'sd_pct': 8, 'memory': {'direction': 'spends', 'gib': 60}},
          {'knob': 'mb', 'value': '2', 'gain_pct': 8, 'sd_pct': 3, 'memory': {'direction': 'spends', 'gib': 10}},
          {'knob': 'liger', 'value': 'true', 'gain_pct': 5, 'sd_pct': 3, 'memory': {'direction': 'frees', 'gib': 10}},
          {'knob': 'comp', 'value': True, 'gain_pct': -5, 'sd_pct': 10, 'memory': {'direction': 'unknown', 'gib': None}},
          {'knob': 'tf32', 'value': True, 'gain_pct': None, 'memory': None},
          {'knob': 'wk', 'value': 7, 'gain_pct': 1}, {'knob': 'nope', 'value': 1, 'gain_pct': 1}, 'junk']
    if not memory:
        for r in lv:
            if isinstance(r, dict):
                r.pop('memory', None)
    return {'levels': lv, 'interactions': [{'members': [{'knob': 'comp', 'value': True}, {'knob': 'liger', 'value': True}],
                                            'kind': 'conflicts'}, {'members': [{'knob': 'x', 'value': 1}], 'kind': 'enables'},
                                           'junk'],
            'configs': [{'name': 'aggressive', 'change': {'rc': 'none', 'mb': 2, 'liger': True, 'bogus': 1}}, 'junk'],
            'notes': 'a test prior'}


class PriorProposer:
    """A proposer with a scripted prior and no patches (the knob side alone)."""

    def __init__(self, prior=None):
        self.answer, self.calls = prior, 0

    def knob_prior(self, engine, base):
        self.calls += 1
        return self.answer

    def describe(self):
        return {'kind': 'PriorProposer'}


def engine(knobs=KNOBS, start=START, *, screens=10, sp=0.01, effect=landscape, validate=None):
    s = SimSession(start, knobs=knobs, screens=screens)
    return SimEngine(s, effect, sigma_runs=[sp / math.sqrt(2)], validate=validate)


def test_registration_ablations_and_resolve():
    assert 'wcs' in STRATEGIES and 'wcs' in LLM_STRATEGIES and resolve('wcs', has_llm=False) == 'wcs'
    assert set(WCS_COMPONENTS) <= set(ABLATIONS) and all(ABLATIONS[k].startswith('wcs: ') for k in WCS_COMPONENTS)
    assert 'wcs_prior' in WCS_COMPONENTS and 'wcs_code' in WCS_COMPONENTS and len(WCS_COMPONENTS) == 9


# ---------------------------------------------------------------- the prior
def test_prior_parsing_is_robust_and_flat_without_a_model():
    p = Prior.from_answer(answer(), KNOBS, START)
    assert p.source == 'model' and p.dropped == 3                       # wk=7, knob 'nope', 'junk'
    rc = p.levels[('rc', vk('none'))]
    assert rc['mu'] == pytest.approx(math.log1p(0.22)) and rc['mem_dir'] == 'spends' and rc['mem_gib'] == 60
    assert p.levels[('mb', vk(2))]['mu'] == pytest.approx(math.log1p(0.08))      # '2' matched to 2
    assert p.levels[('liger', vk(True))]['mem_dir'] == 'frees'                    # 'true' matched to True
    assert p.levels[('comp', vk(True))]['mem_dir'] is None and p.levels[('tf32', vk(True))]['mu'] is None
    assert p.pairs == [{'members': [('comp', vk(True)), ('liger', vk(True))], 'kind': 'conflicts', 'why': '', 'mu': None, 'sd': None}]
    assert p.configs == [{'name': 'aggressive', 'change': {'rc': 'none', 'mb': 2, 'liger': True}, 'why': '', 'mu': None, 'sd': None}]
    assert p.declared(START) == {frozenset({('rc', vk('full'), vk('half'))}): 'spends',
                                 frozenset({('rc', vk('full'), vk('none'))}): 'spends',
                                 frozenset({('mb', vk(1), vk(2))}): 'spends', frozenset({('liger', vk(False), vk(True))}): 'frees'}
    assert p.describe()['known'] == 5 and p.describe()['top'][0]['value'] == 'rc="none"'
    flat = Prior.flat(KNOBS, START)
    assert flat.source == 'flat' and all(e['mu'] is None for e in flat.levels.values()) and len(flat.levels) == 8
    assert Prior.from_answer(None, KNOBS, START).source == 'model' and Prior.from_answer('x', KNOBS, START).levels
    assert p.absorb([{'change': {'tf32': True}, 'gain': 0.012, 'level': 'covered'}, {'change': {'a': 1, 'b': 2}, 'gain': 1,
                                                                                    'level': 'covered'}], KNOBS, START) == 1
    assert p.levels[('tf32', vk(True))]['mu'] == pytest.approx(0.012)


# ---------------------------------------------------------------- the surrogate
def test_surrogate_posterior_follows_the_data_and_draws_shrink():
    s = Surrogate()
    for f in ('a', 'b', 'c'):
        s.ensure(f, 0.0, 0.05)
    truth = {'a': 0.2, 'b': 0.08, 'c': 0.0}
    rng = random.Random(1)
    sd0 = s.sd()
    for _ in range(12):
        feats = {f: 1 for f in truth if rng.random() < 0.5}
        s.observe(feats, sum(truth[f] for f in feats) + rng.gauss(0, 0.005), 0.007)
    mean, sd = s.mean(), s.sd()
    assert abs(mean['a'] - 0.2) < 0.02 and abs(mean['b'] - 0.08) < 0.02 and abs(mean['c']) < 0.02
    assert all(sd[f] < sd0[f] for f in truth)
    draws = [s.sample(rng)['a'] for _ in range(50)]
    assert max(draws) - min(draws) < 0.1 and abs(sum(draws) / 50 - 0.2) < 0.03
    assert 0 < s.var({'a': 1, 'b': 1}) <= (sd['a'] + sd['b']) ** 2 + 1e-9            # (correlated: within the bound)
    assert Surrogate.value({'a': 1, 'c': 1}, mean) == pytest.approx(mean['a'] + mean['c'])


# ---------------------------------------------------------------- feasibility: dominance, the pair table, the cliff
def test_feasibility_from_the_runs_and_the_pair_table():
    eng = engine()
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0})
    w._init(eng)
    assert w.prior.source == 'model' and len(w.table) == 144           # 3 x 2 x 2 x 2 x 2 x 3 (no invariant in the toy)
    assert w.pairs[frozenset({('comp', vk(True)), ('liger', vk(True))})]['p'] == 0.3     # declared conflict
    p0, _ = w._p_feasible({**START, 'rc': 'none'})
    assert 0 < p0 < 1                                                   # declared spends against the floor: unsure, not out
    out = eng.evaluate(Proposal(change={**START, 'rc': 'none'}, absolute=True, hypothesis='x', origin='wcs'))
    w.ideas['x'] = frozenset()
    assert out.status == 'invalid' and 'OutOfMemory' in out.reasons[0]
    w.observe(eng, Proposal(change={**START, 'rc': 'none'}, absolute=True, hypothesis='x'), out)
    assert w.mem.runs[out.cid]['status'] == 'oom' and w.failures[-1]['class'] == 'oom'
    p1, why = w._p_feasible({**START, 'rc': 'none', 'mb': 2})          # by the declared move it needs at least what the
    assert 0 < p1 <= 0.4 and 'declared' in why                          # out-of-memory run needed: a probe, not ruled out
    p1b, why = w._p_feasible({**START, 'rc': 'none', 'mb': 2}, ())
    assert p1b == p1
    p2, _ = w._p_feasible({**START, 'rc': 'none', 'liger': True})       # a declared enabler added: unsure, a probe
    assert 0 < p2 < 0.7
    p3, why3 = w._p_feasible({**START, 'rc': 'none', 'tf32': True})     # an unknown-direction move added: a probe
    assert p3 <= 0.3 and 'unknown memory direction' in why3
    # a run that fitted one move away from the out-of-memory one: the move's direction is measured
    out2 = eng.evaluate(Proposal(change={**START, 'rc': 'none', 'comp': True}, absolute=True, hypothesis='y'))
    w.ideas['y'] = frozenset()
    w.observe(eng, Proposal(change={**START, 'rc': 'none', 'comp': True}, absolute=True, hypothesis='y'), out2)
    assert out2.status == 'screened' and w.mem.runs[out2.cid]['status'] == 'ok'
    assert frozenset({('comp', vk(True), vk(False))}) in w.mem.measured   # going back spends: the compiler frees
    assert w._p_feasible({**START, 'rc': 'none', 'comp': True})[0] == 1.0
    assert 0 < w._p_feasible({**START, 'rc': 'none', 'comp': True, 'tf32': True})[0] < 1.0   # unknown knob: a probe
    # two failed runs one move apart say nothing about that move (WCS_r1 D2): both stay failure bounds, the class
    # with another unknown move is still a probe, not ruled out
    out3 = eng.evaluate(Proposal(change={**START, 'rc': 'none', 'wk': 4}, absolute=True, hypothesis='z'))
    w.ideas['z'] = frozenset()
    w.observe(eng, Proposal(change={**START, 'rc': 'none', 'wk': 4}, absolute=True, hypothesis='z'), out3)
    assert out3.status == 'invalid' and w.mem.measured.get(frozenset({('wk', vk(2), vk(4))})) is None
    assert 0 < w._p_feasible({**START, 'rc': 'none', 'wk': 4, 'tf32': True})[0] < 0.7
    # the pair table: a start failure with two new values marks the pair; a valid run with both clears it
    out4 = eng.evaluate(Proposal(change={**START, 'comp': True, 'liger': True}, absolute=True, hypothesis='q'))
    w.ideas['q'] = frozenset()
    w.observe(eng, Proposal(change={**START, 'comp': True, 'liger': True}, absolute=True, hypothesis='q'), out4)
    assert out4.status == 'invalid' and w.pairs[frozenset({('comp', vk(True)), ('liger', vk(True))})]['p'] == 0.05
    assert w._p_feasible({**START, 'comp': True, 'liger': True})[0] <= 0.05
    w._record_pairs({**START, 'comp': True, 'liger': True}, 'c0', 'ok', SimpleNamespace(cid='c9'))
    assert w.pairs[frozenset({('comp', vk(True)), ('liger', vk(True))})]['p'] == 1.0
    assert w.describe()['wcs']['memory']['runs'][out.cid]['status'] == 'oom'


def test_the_cliff_is_read_from_scattered_updates():
    eng = engine()
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0})
    w._init(eng)
    cfg = {**START, 'mb': 2}

    def per_step(sd, n=24):
        rng, t, rows = random.Random(3), 0.0, []
        for s in range(1, n + 1):
            t += 1.0 * math.exp(rng.gauss(0, sd))
            rows.append({'step': s, 't': t})
        return rows
    eng.s.candidates['c0'].receipt = {'cost': {'gpu_seconds': 1000}, 'evidence': {'per_step': per_step(0.002)}}
    eng.s.candidates['c5'] = SimpleNamespace(id='c5', parent='c0', config=cfg, overlay={}, status='screened', results={},
                                             verdict={'valid': True, 'step_time': 1.1},
                                             receipt={'evidence': {'per_step': per_step(0.12)}})
    assert w._scatter(eng, 'c5') > 0.02 > w._scatter(eng, 'c0')
    out = SimpleNamespace(cid='c5', status='screened', fidelity='full', valid=True, reasons=[], failed_checks=[])
    assert w._jitter(eng, Proposal(change=cfg, absolute=True), out, w._m_of(cfg), 'c0', 'ok') == 'thrash'
    eng.s.candidates['c5'].receipt = {'evidence': {'per_step': per_step(0.003)}}
    assert w._jitter(eng, Proposal(change=cfg, absolute=True), out, w._m_of(cfg), 'c0', 'ok') == 'ok'
    off = WCS(PriorProposer(answer()), off=['wcs_feasibility'])
    off._init(eng)
    eng.s.candidates['c5'].receipt = {'evidence': {'per_step': per_step(0.12)}}
    assert off._jitter(eng, Proposal(change=cfg, absolute=True), out, off._m_of(cfg), 'c0', 'ok') == 'ok'


# ---------------------------------------------------------------- the acquisition
def test_an_aborted_run_without_a_partial_time_is_a_bound_on_its_belief():
    """The replay's and the toy engines (and a run aborted inside the warm-up) give no partial time: the abort rule is
    the observation - the incumbent's gain less its margin, never above the belief - so the stack is not drawn again."""
    a = answer()
    for e in a['levels']:
        if isinstance(e, dict) and e.get('knob') == 'comp':
            e['gain_pct'], e['sd_pct'] = 10, 4                    # the prior likes the compiler; the landscape does not
    eng = engine()
    w = WCS(PriorProposer(a), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0})
    w._init(eng)
    f = ('comp', vk(True))
    assert 0.08 < w.surr.mean()[f] < 0.12
    p = Proposal(change={**START, 'comp': True}, absolute=True, hypothesis='a', origin='wcs')
    w.ideas['a'] = frozenset()
    out = eng.evaluate(p)
    assert out.status == 'aborted' and eng.t_partial(out.cid) is None
    w.observe(eng, p, out)
    assert w.runs[out.cid]['y'] is not None and w.runs[out.cid]['y'] <= -0.02 + 1e-9
    assert w.surr.mean()[f] < 0.06                                 # one abort takes about half the belief away
    # a bound never raises a belief already below it
    e2 = engine()
    w2 = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0})
    w2._init(e2)
    m0 = w2.surr.mean()[f]
    assert -0.06 < m0 < -0.04
    w2.ideas['a'] = frozenset()
    out2 = e2.evaluate(p)
    w2.observe(e2, p, out2)
    assert w2.surr.mean()[f] <= m0 + 1e-9


def test_the_search_reaches_the_coupled_optimum_and_stays_fresh():
    eng = engine(screens=10)
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, seed=7)
    trace = eng.run(w)
    runs = [r for r in eng.order]
    assert trace.stop_reason.startswith('budget') and eng.s.ledger.used['screens'] == 10
    cfgs = [json.dumps(c, sort_keys=True) for _, c, st, _, _ in runs if st != 'duplicate']
    assert len(cfgs) == len(set(cfgs))                                  # never the same configuration twice
    best = eng.s.candidates[eng.incumbent()].config
    assert best['rc'] == 'none' and best['mb'] == 2 and best['comp'] and not best['liger']   # the undeclared enabler's class
    assert sum(GAIN.get((k, v), 0.0) for k, v in best.items() if v != START[k]) >= 0.24
    d = w.describe()['wcs']
    assert d['prior']['source'] == 'model' and d['counts']['thompson'] >= 3 and d['table']['legal'] == 144
    assert all(p['kind'] in ('prior', 'thompson', 'explore', 'coverage', 'partner') for p in d['picks'])
    assert d['picks'][0]['kind'] == 'prior' and d['counts']['prior_config'] == 1   # the prior's 'aggressive' ran first
    first = [r for r in runs if r[2] in ('screened', 'invalid', 'aborted')][:3]
    assert len({json.dumps(c, sort_keys=True) for _, c, _, _, _ in first}) == len(first)


def test_probe_first_below_the_feasibility_threshold_and_full_above():
    eng = engine()
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 2.0}, seed=1)
    w._init(eng)
    low = w._proposal(eng, 'c0', {**START, 'rc': 'none'}, (), mu=0.2, sigma=0.05, p=0.3, why='', probe=True, origin='wcs',
                      kind='thompson')
    assert low.fidelity == ('probe', 'full') and low.absolute and low.keep_probe and low.gate is not None
    high = w._proposal(eng, 'c0', {**START, 'liger': True}, (), mu=0.05, sigma=0.02, p=0.95, why='', probe=False,
                       origin='wcs', kind='thompson')
    assert high.fidelity == ('full',) and high.gate is None and w.counts['probe_first'] == 1
    off = WCS(PriorProposer(answer()), off=['wcs_probe_first'])
    off._init(eng)
    q = off._proposal(eng, 'c0', {**START, 'rc': 'none'}, (), mu=0.2, sigma=0.05, p=0.3, why='', probe=True, origin='wcs',
                      kind='thompson')
    assert q.fidelity == ('full',)


def test_a_blocked_spender_gets_a_partner_probe_with_an_undeclared_enabler():
    eng = engine()
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, seed=3)
    w._init(eng)
    # recompute off ran out of memory alone and with the declared enabler: the runs block it
    for cfg in ({**START, 'rc': 'none'}, {**START, 'rc': 'none', 'mb': 2, 'liger': True}):
        p = Proposal(change=cfg, absolute=True, hypothesis=f'h{len(w.ideas)}', origin='wcs')
        w.ideas[p.hypothesis] = frozenset()
        w.observe(eng, p, eng.evaluate(p))
    assert sum(1 for r in w.mem.runs.values() if r['status'] == 'oom') == 2
    w.surr.observe(w._feats({**START, 'rc': 'none'}), 0.3, 0.01)        # the posterior wants recompute off
    q = w._partner(eng, 'c0')
    assert q is not None and q.origin == 'wcs-partner' and q.change['comp'] and q.change['rc'] == 'none'
    assert not q.change['liger'] and q.change['mb'] == 1          # the incumbent + the spender + the partner, nothing else
    assert q.fidelity == ('probe', 'full') and q.keep_probe and w.counts['partner_probes'] == 1
    out = eng.evaluate(q)
    w.observe(eng, q, out)
    assert out.status == 'screened' and w.mem.runs[out.cid]['status'] == 'ok'   # the compiler was the undeclared enabler
    assert w._p_feasible({**START, 'rc': 'none', 'comp': True})[0] == 1.0
    assert 0 < w._p_feasible({**START, 'rc': 'none', 'mb': 2, 'comp': True})[0] < 1.0   # a declared spend on top: a probe
    again = w._partner(eng, 'c0')
    assert again is None or again.change != q.change                      # once per (run, knob)
    off = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, off=['wcs_partner_probe'])
    off._init(eng)
    assert not [p for p in off.propose(eng, 16) if p.origin == 'wcs-partner']


def test_the_last_screen_is_a_coverage_bundle_of_the_posteriors_positives():
    eng = engine(screens=2)
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, seed=5, off=['wcs_thompson'])
    w._init(eng)
    p = Proposal(change={**START, 'liger': True}, absolute=True, hypothesis='h1', origin='wcs')
    w.ideas['h1'] = frozenset()
    w.observe(eng, p, eng.evaluate(p))
    assert eng.full_screens() == 1
    w.surr.observe(w._feats({**START, 'tf32': True}), 0.03, 0.005)
    w.surr.observe(w._feats({**START, 'wk': 4}), 0.03, 0.005)
    props = w.propose(eng, 16)
    cov = [q for q in props if q.origin == 'wcs-coverage']
    assert cov and cov[0].change['tf32'] and cov[0].change['wk'] == 4 and cov[0].change['liger']
    off = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, off=['wcs_coverage'])
    off._init(eng)
    assert not [q for q in off.propose(eng, 16) if q.origin == 'wcs-coverage']


def test_the_greedy_and_diversity_switches():
    eng = engine()
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, off=['wcs_thompson', 'wcs_diversity'])
    w._init(eng)
    q = w._acquire(eng, 'c0', 10)
    assert q is not None and w.counts['greedy'] == 1 and w.counts['thompson'] == 0
    assert w.describe()['wcs']['off'] == ['wcs_diversity', 'wcs_thompson']


# ---------------------------------------------------------------- the code track on ToyTrain (patches, the self-proof)
def test_the_code_track_proves_patches_and_adopts_them_as_pseudo_knobs():
    from test_search import setup
    task = ToyTask('7b', 'h100', 'both', 'hostsync', 'low')
    sim, s, eng = setup(task, budget=12, job_gpu_hours=None)
    prop = SimProposer(sim, q=1.0, b=1.0, rho=0.0)
    w = WCS(prop, seed=2, off=['wcs_diversity'], table_cap=4000)   # (the toy's 49k-row table sampled: minutes, not 25)
    assert prop.ask_proof is True
    trace = eng.run(w).to_dict()
    d = w.describe()['wcs']
    assert d['counts']['patches'] >= 1 and d['llm_calls'] >= 2
    statuses = {p['status'] for p in d['patches'].values()}
    assert statuses & {'proven', 'inert at its probe', 'queued', 'oom', 'crash', 'slower'}
    assert any(q.origin == 'wcs-patch' for q in w.queue) or d['patches']
    if d['chain']:                                   # an adopted patch: the selected candidate carries the chain's file
        c = s.candidates[trace['selected']]
        assert all(d['patches'][pid]['adopted'] for pid in d['chain'])
        assert c.overlay.get('patches.txt') and any(line for line in c.overlay['patches.txt'].splitlines()
                                                    if line and not line.startswith('proof:'))


def test_every_component_off_runs_on_the_memory_toy():
    task = mem_tasks(36)[0]
    for arm in ['wcs', 'wcs-noprior', 'wcs-nofeas'] + [f'wcs-no-{c[4:]}' for c in WCS_COMPONENTS]:
        r = run_one(task, arm, 6, job_factor=None, proposer={'memory': True}, bind='screens')
        assert r['arm'] == arm and r['screens'] == 6 and r['stop'].startswith('budget'), (arm, r)


def test_the_memory_toy_end_to_end_shows_oom_thrash_probes_and_a_gain():
    rows = {}
    for task in mem_tasks(36)[:4]:
        for arm in ('wcs', 'random'):
            rows[(task.name, arm)] = run_one(task, arm, 10, job_factor=None, proposer={'memory': True}, bind='screens')
    w = [r for (t, a), r in rows.items() if a == 'wcs']
    assert all(r['screens'] == 10 for r in w) and sum(r['probes'] for r in w) >= 1
    assert sum(r['frac'] for r in w) > sum(r['frac'] for (t, a), r in rows.items() if a == 'random')
    assert sum(r['oom_screens'] for r in w) <= sum(r['oom_screens'] for (t, a), r in rows.items() if a == 'random')


# ---------------------------------------------------------------- the simulated and the real model's prior
def test_the_simulated_prior_is_parsed_like_the_models():
    task = mem_tasks(36)[3]
    sim = make_sim(task)
    from test_search import setup as _setup  # noqa: F401 (the toy session helper)
    from infrarsi.agent.budget import VirtualClock
    from infrarsi.agent.search import SearchSession
    from infrarsi.ext.common.env_task import EnvTask
    from infrarsi.ext.common.session_hooks import SessionHooks
    from infrarsi.search.engine import Engine
    from infrarsi.search.toytrain import ToyRunner, toy_task_spec
    from infrarsi.verify import Verifier
    clock = VirtualClock()
    runner = ToyRunner(sim, clock)
    ref_cfg = sim.reference()
    t = EnvTask.from_dict(toy_task_spec(sim, {'screens': 8, 'llm_tokens': 0, 'wall_seconds': 1e12, 'gpu_hours': 1e9}))
    ref = runner.run(t, ref_cfg, None, 'reference', 't', candidate_id='c0', steps=30)
    s = SearchSession(task=t, budget=t.search_budget(), runner=runner, judge=Verifier(0.03, memory_cap_gib=sim.hw['mem']),
                      run_id='t', arm='t', start_config=ref_cfg, start_receipt=ref, reference_receipt=ref,
                      hooks=SessionHooks(t, base_files={'patches.txt': ''}), clock=clock, stop_after_same_failures=10 ** 6)
    eng = Engine(s, full_steps=30, short_steps=10, probe_steps=2)
    prop = SimProposer(sim, q=0.7, memory=True)
    a = prop.knob_prior(eng, 'c0')
    assert {'levels', 'interactions', 'configs'} <= set(a) and a['configs'][0]['name'] == 'aggressive'
    from infrarsi.agent.jsonschema_lite import validate
    assert validate(a, PRIOR_SCHEMA) == []
    p = Prior.from_answer(a, sim.knobs(), ref_cfg)
    assert p.known if hasattr(p, 'known') else p.describe()['known'] >= 1
    assert any(e['mem_dir'] for e in p.levels.values())
    hidden = set(sim.hidden_memory)
    assert all(e['mem_dir'] is None for (k, _), e in p.levels.items() if k in hidden)   # a hidden enabler stays unknown


def test_the_llm_proposer_prior_call_and_patch_only_mode():
    from infrarsi.agent.backends.fake import ScriptedBackend
    from test_search import TOGGLE, setup
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    seen = []

    def script(role, system, prompt):
        seen.append((role, system, json.loads(prompt)))
        if role == 'prior':
            return {'levels': [{'knob': 'fused_adam', 'value': True, 'gain_pct': 3, 'sd_pct': 2,
                                'memory': {'direction': 'neutral', 'gib': 0}}], 'interactions': [], 'configs': [], 'notes': ''}
        if role == 'code_review':
            return {'verdict': 'patch', 'findings': []}
        return {'hypotheses': [{'mechanism': 'knob', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3},
                               {'mechanism': 'host sync', 'change': {}, 'patch': {'patches.txt': 'remove_host_sync\n'},
                                'predicted_gain_pct': 6, 'site': 'loop'}]}
    drv = ScriptedBackend(script)
    prop = LLMProposer(drv, s.hooks, search_calibrates=True, hls3={'sites': True, 'self_proof': True})
    a = prop.knob_prior(eng, 'c0')
    assert a['levels'][0]['knob'] == 'fused_adam' and prop.prior_calls == 1 and s.llm_calls[-1]['role'] == 'prior'
    role, system, payload = seen[0]
    assert system.startswith(PRIOR_SYSTEM) and 'knob_table' in payload and payload['knob_table']['fused_adam']['reference'] is False
    prop.patch_only = True
    out = prop.hypotheses(eng, 'c0', ledger=[])
    hyp = [x for x in seen if x[0] == 'hypotheses'][-1]
    assert PATCH_ONLY_RULE in hyp[1] and hyp[2]['max_hypotheses'] == 3
    assert len(out) == 1 and out[0]['patch'] and not out[0]['change'] and out[0]['site'] == 'loop'
    assert prop.describe()['patch_only'] and prop.describe()['prior_calls'] == 1
    plain = LLMProposer(ScriptedBackend(script), s.hooks, search_calibrates=True)
    assert 'patch_only' not in plain.describe() and not plain.patch_only


# ---------------------------------------------------------------- the coupling task family
def test_the_coupling_family_adds_noise_knobs_and_a_pair_that_only_pays_together():
    ts = couple_tasks(4)
    assert len(ts) == 4 and all(isinstance(t, CoupleTask) and isinstance(t, MemTask) for t in ts)
    assert ts[0].name.startswith('couple-') and len(couple_tasks()) == 36
    sim = make_sim(ts[0])
    assert isinstance(sim, CoupleSim) and len(sim.knobs()) == 14 and sim.reference()['collective_channels'] == 'default'
    ref = sim.reference()
    t0, _ = sim.truth(ref)
    t_ch, _ = sim.truth({**ref, 'collective_channels': '32'})
    t_both, _ = sim.truth({**ref, 'collective_channels': '32', 'fused_loss': True})
    t_fl, _ = sim.truth({**ref, 'fused_loss': True})
    assert t_ch == pytest.approx(t0) and math.log(t_fl / t_both) > 0.035          # the pair pays only together
    assert sim.truth({**ref, 'deterministic': True})[0] > t0                       # the harmful noise knob
    assert abs(math.log(t0 / sim.truth({**ref, 'prefetch_factor': 4})[0])) < 0.006
    t_opt, c_opt, _ = sim.optimum()
    assert t_opt < t0 and c_opt['collective_channels'] == '32' and c_opt['fused_loss']
    r = run_one(ts[0], 'wcs', 6, job_factor=None, proposer={'memory': True}, bind='screens')
    assert r['screens'] == 6 and r['frac'] >= 0 and r['stop'].startswith('budget')   # (runs end to end; no gate on frac)
