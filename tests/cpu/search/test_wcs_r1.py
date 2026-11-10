"""Regressions of the Codex review WCS_r1 (reports/infrarsi_newmodels_20261028/reviews/WCS_r1_out.md), one per defect:
D1 a malformed prior member kills the arm; D2 two out-of-memory runs measure a false neutral move; D3 a queued patch
is not merged onto the adopted overlay; D4 partner probes and the coverage bundle bypass the legal table; D5 the jitter
cliff rule labels a fast, low-peak run thrash; D6 the model-call allowance is not enforced and under-reported; D7 the
acquisition prices a probe with a fixed denominator; D8 the last screen's coverage bundle only competes."""
import json
import math
import random
from types import SimpleNamespace

import pytest

from hls2_helpers import SimEngine, SimSession
from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.proposers import LLMProposer
from infrarsi.search.wcs import WCS, Prior, Surrogate, vk
from test_wcs import CAP, GAIN, KNOBS, START, PriorProposer, answer, engine, landscape, mem_of


def run_of(w, eng, cfg, hid=None):
    p = Proposal(change=cfg, absolute=True, hypothesis=hid or f'h{len(w.ideas)}', origin='wcs')
    w.ideas[p.hypothesis] = frozenset()
    out = eng.evaluate(p)
    w.observe(eng, p, out)
    return out


# ---------------------------------------------------------------- D1
def test_d1_a_malformed_prior_member_gives_a_usable_or_flat_prior_not_a_dead_arm(monkeypatch):
    a = answer()
    a['interactions'].insert(0, {'members': [{'knob': ['mb', 'rc'], 'value': True}, {'knob': 'liger', 'value': True}],
                                 'kind': 'conflicts'})                         # schema-valid, a list for a knob name
    a['levels'].insert(0, {'knob': ['mb'], 'value': 2, 'gain_pct': 5})
    eng = engine()
    w = WCS(PriorProposer(a), facts={'cap_gib': 100.0})
    w._init(eng)                                                               # before: TypeError, the arm dies
    assert w.prior.source == 'model' and len(w.pairs) == 1                     # the junk dropped, the real pair kept
    # an answer the parser cannot use at all: a flat prior, the search goes on

    def boom(*a, **k):
        raise ValueError('unusable')
    monkeypatch.setattr(Prior, 'from_answer', boom)
    w2 = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0})
    w2._init(engine())
    assert w2.prior.source == 'flat' and w2.propose(engine(), 16)


# ---------------------------------------------------------------- D2
CAP2 = 100


def mem2(c):
    """GiB: recompute off needs 130, half 110; the fused loss (declared to free) frees 15."""
    return (0 if c['rc'] == 'full' else 110 if c['rc'] == 'half' else 130) - (15 if c['liger'] else 0)


def landscape2(base, cfg, overlay):
    if mem2(cfg) > CAP2:
        return None, 'oom'
    g = sum(GAIN.get((k, v), 0.0) for k, v in cfg.items() if v != START[k])
    gb = sum(GAIN.get((k, v), 0.0) for k, v in base.config.items() if v != START[k])
    d = g - gb
    return d, 'aborted' if d < -0.02 else None


def test_d2_two_out_of_memory_runs_do_not_measure_a_neutral_move():
    eng = engine(effect=landscape2)
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0})
    w._init(eng)
    for cfg in ({**START, 'rc': 'none'}, {**START, 'rc': 'none', 'liger': True}, {**START, 'rc': 'half'}):
        assert run_of(w, eng, cfg).status == 'invalid'                        # 130, 115 and 110 GiB on a 100 GiB card
    assert 'neutral' not in set(w.mem.measured.values())                      # before: the fused loss 'neutral', hard
    p, why = w._p_feasible({**START, 'rc': 'half', 'liger': True})            # 95 GiB: it fits
    assert p > 0.0, why                                                       # before: 0.0 (ruled out)
    assert eng.evaluate(Proposal(change={**START, 'rc': 'half', 'liger': True}, absolute=True, hypothesis='z')).status == 'screened'


# ---------------------------------------------------------------- D3
BASE_PY = 'import x\n\n\ndef a():\n    return 0\n\n\ndef b():\n    return 0\n'
FILES = {'f.py': BASE_PY}
P_A = BASE_PY.replace('def a():\n    return 0', 'def a():\n    return 1')        # the first patch: a() -> 1
P_B = BASE_PY.replace('def b():\n    return 0', 'def b():\n    return 1')        # the second: b() -> 1
P_AB = P_A.replace('def b():\n    return 0', 'def b():\n    return 1')           # both


def crafted(eng, cid, overlay, gain):
    eng.s.candidates[cid] = SimpleNamespace(id=cid, parent='c0', config=dict(START), overlay=dict(overlay), status='screened',
                                            results={}, verdict={'valid': True, 'step_time': math.exp(-gain)}, receipt={})
    return Outcome(cid, 'screened', True, math.exp(-gain), gain, 'full', 0.3, [], [])


def test_d3_a_queued_patch_is_merged_onto_the_adopted_overlay_and_carries_its_ids():
    s = SimSession(START, knobs=KNOBS, base_files=FILES)
    eng = SimEngine(s, landscape, sigma_runs=[0.01 / math.sqrt(2)])          # sigma_pair 0.01: the engine moves at 0.02
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0})
    w._init(eng)
    h = {'mechanism': 'p', 'mu': 0.02, 'p_feasible': 0.9, 'mem_dir': 'neutral'}
    q1 = w._patch_proposal(eng, 'c0', {**h, 'patch': {'f.py': P_A}})
    q2 = w._patch_proposal(eng, 'c0', {**h, 'patch': {'f.py': P_B}})
    w.queue += [q1, q2]
    pid1, pid2 = q1.hypothesis.split('|', 1)[1], q2.hypothesis.split('|', 1)[1]
    w.pids['c1'] = frozenset({pid1})
    w.observe(eng, q1, crafted(eng, 'c1', q1.patch, 0.015))                   # adopted (> sigma_pair), the incumbent stays
    assert w.chain == [pid1] and w.overlay_best == {'f.py': P_A} and eng.incumbent() == 'c0'
    props = w._code(eng, 'c0', 5)                                             # the queued second patch, rebased
    assert len(props) == 1 and props[0] is q2
    assert q2.patch == {'f.py': P_AB}, q2.patch                               # before: b() alone (the first patch lost)
    assert w.ideas[q2.hypothesis] == frozenset({pid1, pid2}) and set(w.ledger[q2.hypothesis]['patches']) == {pid1, pid2}
    w.pids['c2'] = frozenset({pid1, pid2})
    w.observe(eng, q2, crafted(eng, 'c2', q2.patch, 0.045))
    assert w.chain == [pid1, pid2] and w.overlay_best == {'f.py': P_AB}
    # a new hypothesis from the model (written against the incumbent's files) is merged onto the chain at once
    q3 = w._patch_proposal(eng, 'c0', {**h, 'patch': {'f.py': BASE_PY.replace('import x', 'import x, y')}})
    assert q3 is not None and q3.patch['f.py'].startswith('import x, y') and 'return 1' in q3.patch['f.py']


# ---------------------------------------------------------------- D4
KNOBS4 = {**KNOBS, 'ga': [1, 2]}
START4 = {**START, 'ga': 2}


def mem4(c):
    return mem_of(c) + (45 if c['mb'] == 2 else 0)                            # micro-batch 2 with the fused loss: 45 of 40


def landscape4(base, cfg, overlay):
    if cfg['comp'] and cfg['liger']:
        return None, 'timeout'
    if mem4(cfg) > CAP:
        return None, 'oom'
    g = sum(GAIN.get((k, v), 0.0) for k, v in cfg.items() if v != START4[k])
    gb = sum(GAIN.get((k, v), 0.0) for k, v in base.config.items() if v != START4[k])
    d = g - gb
    return d, 'aborted' if d < -0.02 else None


def coupled():
    s = SimSession(START4, knobs=KNOBS4)
    s.check_config = lambda cfg, c0: ((cfg if cfg['mb'] * cfg['ga'] == 2 else None), {}, [], [], None)   # the global batch
    return SimEngine(s, landscape4, sigma_runs=[0.01 / math.sqrt(2)])


def test_d4_partner_probes_and_the_coverage_bundle_keep_the_coupled_invariant():
    eng = coupled()
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, seed=3)
    w._init(eng)
    assert all(c['mb'] * c['ga'] == 2 for c in w.table)
    blocked = {**START4, 'mb': 2, 'ga': 1, 'liger': True}
    assert run_of(w, eng, blocked).status == 'invalid'                         # the legal spender runs out of memory
    w.surr.observe(w._feats(blocked), 0.3, 0.01)                               # the posterior wants it
    q = w._partner(eng, 'c0')
    assert q is not None and q.change['comp'] and q.change['mb'] == 2
    assert q.change['mb'] * q.change['ga'] == 2, q.change                      # before: mb 2 x ga 2 (the engine refuses it)
    assert eng.s.check_config(q.change, None)[0] is not None
    # the coverage bundle: micro-batch 2 rated positive alone, the incumbent's accumulation 2 - completed from the table
    w2 = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, seed=3, off=['wcs_thompson'])
    w2._init(eng)
    for _ in range(3):
        w2.surr.observe(w2._feats({**START4, 'mb': 2, 'ga': 1}), 0.05, 0.005)
        w2.surr.observe(w2._feats({**START4, 'ga': 1}), 0.0, 0.005)
    cov = w2._coverage(eng, 'c0')
    assert cov is not None and cov.change['mb'] == 2
    assert cov.change['mb'] * cov.change['ga'] == 2, cov.change               # before: mb 2 x ga 2


# ---------------------------------------------------------------- D5
def per_step(sd, n=24, seed=3):
    rng, t, rows = random.Random(seed), 0.0, []
    for s in range(1, n + 1):
        t += 1.0 * math.exp(rng.gauss(0, sd))
        rows.append({'step': s, 't': t})
    return rows


def scattered(eng, cid, step_time, peak=None):
    ev = {'per_step': per_step(0.12)}
    if peak is not None:
        ev['memory_peak_measured_gib'] = peak
    eng.s.candidates[cid] = SimpleNamespace(id=cid, parent='c0', config={**START, 'mb': 2}, overlay={}, status='screened',
                                            results={}, verdict={'valid': True, 'step_time': step_time},
                                            receipt={'evidence': ev})
    return Outcome(cid, 'screened', True, step_time, -math.log(step_time), 'full', 0.3, [], [])


def test_d5_the_jitter_cliff_needs_a_slowdown_and_yields_to_a_measured_peak():
    eng = engine()
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0})
    w._init(eng)
    eng.s.candidates['c0'].receipt = {'cost': {'gpu_seconds': 1000}, 'evidence': {'per_step': per_step(0.002)}}
    cfg = {**START, 'mb': 2}
    for cid, t, peak, kind in (('c5', 0.7, 45.0, 'ok'),       # faster, 45 of 100 GiB measured: not at the memory edge
                               ('c6', 0.7, None, 'ok'),       # faster: not thrash either (F23 wants a slowdown)
                               ('c7', 1.1, 45.0, 'ok'),       # slower, but the measured peak says headroom
                               ('c8', 1.1, None, 'thrash')):  # slower, scattered, no peak, spends memory: F23's cliff
        p = Proposal(change=cfg, absolute=True, hypothesis=cid, origin='wcs')
        w.ideas[cid] = frozenset()
        w.observe(eng, p, scattered(eng, cid, t, peak))
        assert w.runs[cid]['kind'] == kind, (cid, w.runs[cid]['kind'])
        assert w.mem.runs[cid]['status'] == ('thrash' if kind == 'thrash' else 'ok')


# ---------------------------------------------------------------- D6
def test_d6_one_shared_allowance_over_every_model_call_and_the_true_count():
    from infrarsi.agent.backends.fake import ScriptedBackend
    from test_search import TOGGLE, setup
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    seen = []

    def script(role, system, prompt):
        seen.append(role)
        if role == 'prior':
            return {'levels': [{'knob': 'fused_adam', 'value': True, 'gain_pct': 3, 'sd_pct': 2,
                                'memory': {'direction': 'neutral', 'gib': 0}}], 'interactions': [], 'configs': [], 'notes': ''}
        if role == 'code_review':
            return {'verdict': 'patch', 'findings': []}
        if 'previous_reply_errors' in prompt:                                   # the F14 re-ask: a usable patch
            return {'hypotheses': [{'mechanism': 'host sync', 'change': {}, 'patch': {'patches.txt': 'remove_host_sync\n'},
                                    'predicted_gain_pct': 6, 'site': 'loop'}]}
        return {'hypotheses': [{'mechanism': 123, 'change': [], 'predicted_gain_pct': 'five'}]}   # unusable: a re-ask
    prop = LLMProposer(ScriptedBackend(script), s.hooks, search_calibrates=True, features=WCS.proposer_features())
    w = WCS(prop, facts={'cap_gib': 80.0})
    w._init(eng)
    assert seen == ['prior']
    knobs = iter(sorted(w.knobs))
    for i in range(3):
        k = next(knobs)                                                         # a screen between the asks
        v = next(x for x in w.knobs[k] if vk(x) != vk(w.ref.get(k)))
        p = Proposal(change={**w.ref, k: v}, absolute=True, hypothesis=f'k{i}', origin='wcs')
        w.ideas[p.hypothesis] = frozenset()
        w.observe(eng, p, eng.evaluate(p))
        w._code(eng, eng.incumbent(), 5)
        w.queue.clear()                                                         # (as if the patches had run)
    assert len(seen) <= 6, seen                                                 # before: 8 (prior, review, 3 x ask + re-ask)
    assert w.describe()['wcs']['llm_calls'] == len(seen)                        # before: 4 reported
    assert w.propose(eng, 16)                                                   # past the allowance: knobs go on


# ---------------------------------------------------------------- D7
def test_d7_the_acquisition_prices_a_probe_first_candidate_by_its_expected_cost():
    eng = engine()                                                              # a probe costs 0.35 of a full run
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0})
    w._init(eng)
    a, b = {**START, 'tf32': True}, {**START, 'wk': 4}
    pf = {w._ckey(a, ()): 0.9, w._ckey(b, ()): 0.5}
    w._p_feasible = lambda cfg, ids=(): (pf.get(w._ckey(cfg, ids), 0.0), '')
    scored = [(0.17, b, (), w._feats(b)), (0.10, a, (), w._feats(a))]
    best = w._best(scored, None, True)
    assert best is not None and best[0] == b and best[5]                        # 0.17 x 0.5 / (0.35 + 0.5) = 0.10 > 0.09
    assert best[3] == 0.5                                                       # before: 0.17 x 0.5 / 1.35 = 0.063, a wins


# ---------------------------------------------------------------- D8
def test_d8_the_last_screen_goes_to_the_coverage_bundle_alone():
    eng = engine(screens=2)
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, seed=5, off=['wcs_thompson'])
    w._init(eng)
    run_of(w, eng, {**START, 'liger': True}, 'h1')
    assert eng.full_screens() == 1
    w.surr.observe(w._feats({**START, 'tf32': True}), 0.03, 0.005)
    w.surr.observe(w._feats({**START, 'wk': 4}), 0.03, 0.005)
    props = w.propose(eng, 16)
    assert [q.origin for q in props] == ['wcs-coverage'], [q.origin for q in props]   # before: plus a greedy pick
    assert props[0].change['tf32'] and props[0].change['wk'] == 4 and props[0].change['liger']


# ---------------------------------------------------------------- the Q9H smoke (the real prior's suggestion got lost)
def smoke_prior(pair_gain=None, config_gain=None):
    """A prior like the real one of the Q9H smoke: small single gains, the spender's benefit in an unquantified
    enabling pair, the winning class only as a whole 'aggressive' configuration."""
    lv = [{'knob': 'liger', 'value': True, 'gain_pct': 1.5, 'sd_pct': 1, 'memory': {'direction': 'frees', 'gib': 10}},
          {'knob': 'tf32', 'value': True, 'gain_pct': 1, 'sd_pct': 1, 'memory': {'direction': 'neutral', 'gib': 0}},
          {'knob': 'mb', 'value': 2, 'gain_pct': None, 'sd_pct': None, 'memory': {'direction': 'spends', 'gib': 10}},
          {'knob': 'rc', 'value': 'none', 'gain_pct': None, 'memory': {'direction': 'spends', 'gib': 60}},
          {'knob': 'comp', 'value': True, 'gain_pct': -5, 'sd_pct': 10, 'memory': {'direction': 'unknown', 'gib': None}}]
    pair = {'members': [{'knob': 'mb', 'value': 2}, {'knob': 'liger', 'value': True}], 'kind': 'enables'}
    if pair_gain is not None:
        pair['gain_pct'] = pair_gain
    cfgs = [{'name': 'safe', 'change': {'tf32': True}},
            {'name': 'aggressive', 'change': {'mb': 2, 'liger': True, 'tf32': True}}]
    if config_gain is not None:
        cfgs[1]['gain_pct'] = config_gain
    return {'levels': lv, 'interactions': [pair], 'configs': cfgs, 'notes': ''}


def test_s1_the_priors_best_configuration_runs_first_and_an_enabling_pair_carries_a_gain():
    eng = engine(screens=10)
    w = WCS(PriorProposer(smoke_prior()), facts={'cap_gib': 100.0}, seed=3)
    w._init(eng)
    pair = ('pair', frozenset({('mb', vk(2)), ('liger', vk(True))}))
    assert w.surr.mean()[pair] == pytest.approx(0.03, abs=1e-6)              # an enabling pair without a number: +3 %
    assert {'mb', 'liger'} <= w._big_knobs()                                  # its members count as big knobs
    props = w.propose(eng, 16)                                                # before: a Thompson pick of small knobs
    assert len(props) == 1 and props[0].origin == 'wcs-prior' and props[0].hypothesis.startswith('prior|')
    assert props[0].change['mb'] == 2 and props[0].change['liger'] and props[0].change['tf32']   # the 'aggressive' one
    out = eng.evaluate(props[0])
    w.observe(eng, props[0], out)
    assert out.status == 'screened' and eng.incumbent() == out.cid and w.counts['prior_config'] == 1
    assert not [q for q in w.propose(eng, 16) if q.origin == 'wcs-prior']    # once
    # the numbers the model gives are read: the pair's own gain, a configuration's gain as a wide observation
    w2 = WCS(PriorProposer(smoke_prior(pair_gain=8, config_gain=12)), facts={'cap_gib': 100.0}, seed=3)
    w2._init(engine(screens=10))
    assert w2.surr.mean()[pair] > math.log1p(0.08) - 0.02                     # (the configuration's gain lifts it a little)
    assert w2.prior.pairs[0]['mu'] == pytest.approx(math.log1p(0.08)) and w2.prior.configs[1]['mu'] == pytest.approx(math.log1p(0.12))
    cfg = {**START, 'mb': 2, 'liger': True, 'tf32': True}
    assert w2.surr.mean()[('mb', vk(2))] + w2.surr.mean()[('liger', vk(True))] + w2.surr.mean()[('tf32', vk(True))] \
        + w2.surr.mean()[pair] > 0.09                                        # the whole configuration believed at ~+12 %
    assert w2.describe()['wcs']['prior']['configs'][1]['gain_pct'] == 12.0
    off = WCS(PriorProposer(smoke_prior()), facts={'cap_gib': 100.0}, off=['wcs_prior'])
    off._init(engine(screens=10))
    assert not [q for q in off.propose(engine(screens=10), 16) if q.origin == 'wcs-prior']


def test_s2_the_prior_configuration_is_probed_first_when_doubtful_and_skipped_when_ruled_out():
    eng = engine(screens=10)
    a = smoke_prior()
    a['configs'][1]['change'] = {'rc': 'none', 'mb': 2, 'liger': True}        # 55 + 10 - 10 of 40: out of memory
    facts = {'cap_gib': 100.0, 'world': 8, 'params_b': 20.0}                  # (the floor + 60 GiB declared: doubtful)
    w = WCS(PriorProposer(a), facts=facts, seed=3)
    w._init(eng)
    props = w.propose(eng, 16)
    assert len(props) == 1 and props[0].origin == 'wcs-prior' and props[0].fidelity == ('probe', 'full')   # doubtful: probed
    assert props[0].p_feasible < 0.7
    w.observe(eng, props[0], eng.evaluate(props[0]))
    assert w.mem.runs[[c for c in eng.s.candidates if c != 'c0'][0]]['status'] == 'oom'
    # a configuration the runs rule out is not proposed: the next one by rank is
    w3 = WCS(PriorProposer(a), facts=facts, seed=3)
    w3._init(eng)
    w3.mem.runs = dict(w.mem.runs)
    w3.tried = set(w.tried)
    q = w3._prior_pick(eng, 'c0')
    assert q is not None and q.change['rc'] == 'full' and q.change['tf32']   # 'safe', not the ruled-out 'aggressive'


# ---------------------------------------------------------------- the G4U / M35H2 smokes (the memory edge)
def test_e1_a_run_at_the_memory_edge_gets_a_rescue_probe_with_its_enabler():
    eng = engine()
    w = WCS(PriorProposer(smoke_prior()), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0}, seed=3)
    w._init(eng)
    eng.s.candidates['c0'].receipt = {'cost': {'gpu_seconds': 1000}, 'evidence': {'per_step': per_step(0.002)}}
    p = Proposal(change={**START, 'mb': 2}, absolute=True, hypothesis='t1', origin='wcs')
    w.ideas['t1'] = frozenset()
    w.observe(eng, p, scattered(eng, 't1', 1.1))                               # slower, scattered, no peak: the cliff
    assert w.mem.runs['t1']['status'] == 'thrash'
    w.counts['prior_config'] = 1                                               # (the prior's own pick made)
    props = w.propose(eng, 16)
    assert [x.origin for x in props] == ['wcs-rescue']                        # proposed alone
    q = props[0]
    assert q.fidelity == ('probe', 'full') and q.keep_probe and q.p_feasible >= 0.4
    assert q.change['mb'] == 2 and q.change['liger'] and 'names it the enabler' in w.ledger[q.hypothesis]['why_feasible']
    assert w.counts['rescue_probes'] == 1
    assert w._rescue(eng, 'c0') is None                                        # once per (run, value)
    # out of memory by a sliver: the largest declared freeing value it lacks; a conflicting value swapped back
    w2 = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0}, seed=3)
    w2._init(eng)
    out = run_of(w2, eng, {**START, 'rc': 'none', 'comp': True, 'tf32': True})   # 55 - 30 = 25: fits in the landscape,
    assert out.status == 'screened'                                              # but say it ran out by 0.3 GiB
    w2.mem.runs[out.cid]['status'] = 'oom'
    w2.mem.runs[out.cid]['oom'] = {'short_gib': 0.3}
    q2 = w2._rescue(eng, 'c0')
    assert q2 is not None and q2.change['liger'] and q2.change['rc'] == 'none' and q2.change['tf32']
    assert not q2.change['comp']                                              # the compiler conflicts with the fused loss
    assert w2._rescue(eng, 'c0') is None or w2._rescue(eng, 'c0').change != q2.change   # once per (run, value)
    off = WCS(PriorProposer(smoke_prior()), facts={'cap_gib': 100.0}, off=['wcs_partner_probe'])
    off._init(eng)
    assert not [x for x in off.propose(eng, 16) if x.origin == 'wcs-rescue']


def test_e2_the_arithmetic_reads_a_measured_reference_peak_against_the_card_not_the_cliff_line():
    eng = engine()
    a = answer()
    a['levels'].append({'knob': 'wk', 'value': 4, 'gain_pct': 0.5, 'sd_pct': 1, 'memory': {'direction': 'neutral', 'gib': 0}})
    w = WCS(PriorProposer(a), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0})
    w._init(eng)
    w.mem.runs['c0']['peak'] = 90.0                                             # the reference reserves 90 % of the card
    w._forget()
    p_mb, why = w._p_feasible({**START, 'mb': 2})                              # +10 GiB declared: 100 of 100, +-7
    assert 0.2 < p_mb < 0.7 and 'reference peak 90' in why                     # before: 0.15 (the 90 % line was crossed)
    w.mem.runs['c0']['peak'] = 95.0
    w._forget()
    p_rc, why = w._p_feasible({**START, 'rc': 'none'})                         # +60 declared: 155 of 100
    assert p_rc == pytest.approx(0.15), why                                    # the floor: a probe, not ruled out
    assert w._p_feasible({**START, 'wk': 4})[0] >= 0.7                         # a declared neutral move: fits
    assert w._p_feasible({**START, 'tf32': True})[0] < 0.7                     # an unknown-direction move at 95 %: a probe


def test_e3_a_failure_the_arithmetic_did_not_expect_scales_the_declared_spends():
    eng = engine()
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0})
    w._init(eng)
    cfg = {**START, 'rc': 'none'}
    p0 = w._p_feasible(cfg)[0]
    q = w._proposal(eng, 'c0', cfg, (), mu=0.2, sigma=0.05, p=0.6, why='about 95 GiB by the declared memory of 100',
                    probe=False, origin='wcs', kind='thompson')
    out = eng.evaluate(q)
    assert out.status == 'invalid'
    w.observe(eng, q, out)
    assert w.spend_scale == 1.25 and w.counts['calibrations'] == 1
    assert w._p_feasible({**START, 'rc': 'half'})[0] <= w._p_feasible({**START, 'rc': 'half'})[0]
    assert 'x1.25' in w._p_arith(w._m_of({**START, 'rc': 'half'}))[1]
    w.observe(eng, q, out)                                                     # (a second surprise: one more step)
    assert w.spend_scale == 1.5


def test_e4_the_exploration_fallback_takes_no_doubtful_bet_with_a_negative_mean():
    a = smoke_prior()
    a['levels'] = [{'knob': 'liger', 'value': True, 'gain_pct': -2, 'sd_pct': 1, 'memory': {'direction': 'frees', 'gib': 10}},
                   {'knob': 'comp', 'value': True, 'gain_pct': -5, 'sd_pct': 12, 'memory': {'direction': 'unknown', 'gib': None}},
                   {'knob': 'rc', 'value': 'half', 'gain_pct': -3, 'sd_pct': 1, 'memory': {'direction': 'spends', 'gib': 20}},
                   {'knob': 'rc', 'value': 'none', 'gain_pct': -3, 'sd_pct': 1, 'memory': {'direction': 'spends', 'gib': 60}},
                   {'knob': 'mb', 'value': 2, 'gain_pct': -1, 'sd_pct': 1, 'memory': {'direction': 'spends', 'gib': 10}},
                   {'knob': 'wk', 'value': 0, 'gain_pct': -1, 'sd_pct': 1}, {'knob': 'wk', 'value': 4, 'gain_pct': -1, 'sd_pct': 1}]
    a['interactions'], a['configs'] = [], []
    eng = engine()
    w = WCS(PriorProposer(a), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0}, off=['wcs_thompson'], seed=3)
    w._init(eng)
    q = w._acquire(eng, 'c0', 5)                                               # nothing improves by the mean: the fallback
    assert q is not None and w.ledger[q.hypothesis]['kind'] == 'explore'
    assert not q.change['comp'], q.change                                      # before: the compiler (mean -5 %, spread 12 %)
    assert q.change['tf32'] and q.p_feasible >= 0.7                            # the unknown with a non-negative mean


# ---------------------------------------------------------------- WCS_r2 (Codex round 2) residuals
def test_r2_1_a_queued_patch_on_another_file_keeps_the_adopted_file():
    files = {'a.py': 'A=0\n', 'b.py': 'B=0\n'}
    s = SimSession(START, knobs=KNOBS, base_files=files)
    eng = SimEngine(s, landscape, sigma_runs=[0.01 / math.sqrt(2)])
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0})
    w._init(eng)
    h = {'mechanism': 'p', 'mu': 0.02, 'p_feasible': 0.9, 'mem_dir': 'neutral'}
    q1 = w._patch_proposal(eng, 'c0', {**h, 'patch': {'a.py': 'A=1\n'}})
    q2 = w._patch_proposal(eng, 'c0', {**h, 'patch': {'b.py': 'B=1\n'}})
    w.queue += [q1, q2]
    pid1, pid2 = q1.hypothesis.split('|', 1)[1], q2.hypothesis.split('|', 1)[1]
    w.pids['c1'] = frozenset({pid1})
    w.observe(eng, q1, crafted(eng, 'c1', q1.patch, 0.015))                   # adopted, the incumbent stays c0
    assert w.chain == [pid1] and w.overlay_best == {'a.py': 'A=1\n'}
    props = w._code(eng, 'c0', 5)
    assert len(props) == 1 and props[0] is q2
    assert q2.patch == {'a.py': 'A=1\n', 'b.py': 'B=1\n'}, q2.patch            # before: b.py alone, ids {p1, p2}
    q3 = w._patch_proposal(eng, 'c0', {**h, 'patch': {'b.py': 'B=2\n'}})       # a new hypothesis on the other file
    assert q3 is not None and q3.patch == {'a.py': 'A=1\n', 'b.py': 'B=2\n'}


def test_r2_2_an_impossible_first_partner_does_not_consume_the_opportunity():
    s = SimSession(START4, knobs=KNOBS4)
    s.check_config = lambda cfg, c0: ((cfg if cfg['mb'] * cfg['ga'] == 2 and not (cfg['comp'] and cfg['mb'] == 2) else None),
                                      {}, [], [], None)                       # the compiler cannot take micro-batch 2
    eng = SimEngine(s, landscape4, sigma_runs=[0.01 / math.sqrt(2)])
    w = WCS(PriorProposer(answer()), facts={'cap_gib': 100.0}, seed=3)
    w._init(eng)
    blocked = {**START4, 'mb': 2, 'ga': 1, 'liger': True}
    assert run_of(w, eng, blocked).status == 'invalid'
    w.surr.observe(w._feats(blocked), 0.3, 0.01)
    q = w._partner(eng, 'c0')                                                  # the most uncertain partner (the compiler)
    assert q is not None, 'before: None - the impossible partner consumed the chance'   # has no legal form: the next one
    assert q.change['mb'] == 2 and q.change['ga'] == 1 and not q.change['comp'] and q.origin == 'wcs-partner'
    assert eng.s.check_config(q.change, None)[0] is not None and w.counts['partner_probes'] == 1


def test_r2_3_the_price_of_a_probe_includes_its_declared_warm_up():
    eng = engine()                                                             # nominal probe: 2 updates of a 30-update run
    a = answer()
    for e in a['levels']:
        if isinstance(e, dict) and e.get('knob') == 'comp':
            e['warmup_updates'] = 6                                            # a seven-update probe: 5 more updates
    w = WCS(PriorProposer(a), facts={'cap_gib': 100.0})
    w._init(eng)
    cand_a, cand_b = {**START, 'tf32': True}, {**START, 'comp': True}
    pf = {w._ckey(cand_a, ()): 0.9, w._ckey(cand_b, ()): 0.5}
    w._p_feasible = lambda cfg, ids=(): (pf.get(w._ckey(cfg, ids), 0.0), '')
    w._price_context(eng, 'c0')
    scored = [(0.17, cand_b, (), w._feats(cand_b)), (0.10, cand_a, (), w._feats(cand_a))]
    best = w._best(scored, None, True)
    assert best is not None and best[0] == cand_a                             # 0.17 x 0.5 / (0.35 + 5/30 + 0.5) = 0.084 < 0.09
    assert w._price(cand_b, True) == pytest.approx(0.35 + 5 / 30 + 0.5, abs=1e-6)
    assert w._price(cand_a, False) == 1.0


def test_r3_the_price_follows_the_engines_probe_removal_test_when_screens_are_not_scarce():
    """Codex WCS_r3: Engine.plan drops a probe that is not kept only when its NOMINAL cost reaches half a full run; a
    longer declared warm-up does not drop it, it adds updates to the probe the plan keeps."""
    eng = engine()                                                             # nominal probe: 0.35 of a 30-update run
    a = answer()
    for e in a['levels']:
        if isinstance(e, dict) and e.get('knob') == 'comp':
            e['warmup_updates'] = 6                                            # a seven-update probe: 5 more updates
    w = WCS(PriorProposer(a), facts={'cap_gib': 100.0})
    w._init(eng)
    w._screens_scarce = lambda *args, **kw: False                              # GPU-hours bind: keep_probe False
    cand_a, cand_b = {**START, 'tf32': True}, {**START, 'comp': True}
    pf = {w._ckey(cand_a, ()): 0.9, w._ckey(cand_b, ()): 0.4}
    w._p_feasible = lambda cfg, ids=(): (pf.get(w._ckey(cfg, ids), 0.0), '')
    w._price_context(eng, 'c0')
    assert w._price(cand_b, True) == pytest.approx(0.35 + 5 / 30 + 0.4, abs=1e-6)   # the probe stays (before: 1.0)
    scored = [(0.22, cand_b, (), w._feats(cand_b)), (0.10, cand_a, (), w._feats(cand_a))]
    best = w._best(scored, None, True)
    assert best is not None and best[0] == cand_b                             # 0.22 x 0.4 / 0.9167 = 0.096 > 0.09


def test_r2_4_config_suggestions_yield_to_a_measurement_and_are_deduplicated():
    a = smoke_prior()
    a['interactions'] = []
    a['configs'] = [{'name': 'c1', 'change': {'tf32': True, 'wk': 4}, 'gain_pct': 20, 'sd_pct': 1},
                    {'name': 'c1 again', 'change': {'wk': 4, 'tf32': True}, 'gain_pct': 20, 'sd_pct': 1},
                    {'name': 'c2', 'change': {'tf32': True, 'wk': 0}, 'gain_pct': 20, 'sd_pct': 1},
                    {'name': 'c3', 'change': {'tf32': True, 'liger': True}, 'gain_pct': 20, 'sd_pct': 1}]
    eng = engine(sp=0.03 * math.sqrt(2))                                       # run noise 3 % (the review's reproduction)
    w = WCS(PriorProposer(a), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0}, seed=3)
    w._init(eng)
    assert len(w.surr.obs) == 3                                                # the duplicate suggestion once (before: 4)
    c1, c3 = {**START, 'tf32': True, 'wk': 4}, {**START, 'tf32': True, 'liger': True}
    p = Proposal(change=c1, absolute=True, hypothesis='m1', origin='wcs')
    w.ideas['m1'] = frozenset()
    eng.s.candidates['m1'] = SimpleNamespace(id='m1', parent='c0', config=dict(c1), overlay={}, status='screened', results={},
                                             verdict={'valid': True, 'step_time': math.exp(0.05)}, receipt={})
    w.observe(eng, p, Outcome('m1', 'screened', True, math.exp(0.05), -0.05, 'full', 0.3, [], []))   # measured -5 %
    mean = w.surr.mean()
    assert Surrogate.value(w._feats(c1), mean) < 0.0                           # the measurement, not the suggestion
    assert Surrogate.value(w._feats(c3), mean) < 0.10                          # before: the sibling kept at +12.8 %


def test_r2_5_the_fallback_exhausts_non_negative_candidates_before_a_negative_mean():
    a = smoke_prior()
    a['levels'] = [{'knob': 'liger', 'value': True, 'gain_pct': -2, 'sd_pct': 1, 'memory': {'direction': 'frees', 'gib': 10}},
                   {'knob': 'comp', 'value': True, 'gain_pct': -5, 'sd_pct': 12, 'memory': {'direction': 'unknown', 'gib': None}},
                   {'knob': 'rc', 'value': 'half', 'gain_pct': -3, 'sd_pct': 1, 'memory': {'direction': 'spends', 'gib': 20}},
                   {'knob': 'rc', 'value': 'none', 'gain_pct': -3, 'sd_pct': 1, 'memory': {'direction': 'spends', 'gib': 60}},
                   {'knob': 'mb', 'value': 2, 'gain_pct': -1, 'sd_pct': 1, 'memory': {'direction': 'spends', 'gib': 10}},
                   {'knob': 'wk', 'value': 0, 'gain_pct': -1, 'sd_pct': 1}, {'knob': 'wk', 'value': 4, 'gain_pct': -1, 'sd_pct': 1}]
    a['interactions'], a['configs'] = [], []
    eng = engine()
    w = WCS(PriorProposer(a), facts={'cap_gib': 100.0, 'world': 8, 'params_b': 20.0}, off=['wcs_thompson'], seed=3)
    w._init(eng)
    w._probes_affordable = lambda engine: False                                # the probe allowance is spent
    base_pf = w._p_feasible
    w._p_feasible = lambda cfg, ids=(): ((0.5, 'doubtful') if cfg.get('tf32') and not cfg.get('comp') else
                                        (0.9, '') if cfg.get('comp') else base_pf(cfg, ids))
    q = w._acquire(eng, 'c0', 5)
    assert q is not None and w.ledger[q.hypothesis]['kind'] == 'explore'
    assert not q.change['comp'] and q.change['tf32'], q.change                # before: the compiler (mean -5 %) at 0.9
    assert q.fidelity == ('full',)                                             # the doubtful one in full (its permitted plan)
    assert w.ledger[q.hypothesis]['predicted_gain_pct'] >= -0.01


def test_e5_the_stores_warnings_lower_the_feasibility_by_what_they_say():
    lvl = WCS._warn_level
    assert lvl(['memory (estimate): about 145 GiB of 80 (closed-form accounting, +-25 %)']) == 0.03
    assert lvl(['memory (estimate): about 85 GiB of 80 (closed-form accounting, +-25 %)']) == 0.2
    assert lvl(['memory (estimate): about 70 GiB of 80 (closed-form accounting, +-25 %)']) == 0.4
    assert lvl(['memory: run-x/c3 peaks at 79.5 GiB of 80']) == 0.1
    assert lvl(['memory: run-x/c3 ran out of memory']) == 0.1
    assert lvl(['c-12 (oom): some constraint']) == 0.4
