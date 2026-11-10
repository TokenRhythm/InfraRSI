"""esls (search/esls.py; possible_search_algorithm/20 section 1): the chain of spend levels (legal, on the Pareto front,
ordered by the ledger), the enabler probes (what each frees, within their share of the low-fidelity allowance), the
probabilistic bisection on a planted chain, the ledge after a failed boundary, the hedge, the undeclared enablers that
unlock the largest spend on the memory toy, every component off, the extended prior call (wcs's unchanged), and the wcs
methods the candidates build on."""
import inspect
import json
import math

import pytest

from cand_helpers import run_kept, toy
from infrarsi.search import ESLS_COMPONENTS
from infrarsi.search.esls import ENABLER_SHARE, PRIOR_RULE, ESLS
from infrarsi.search.ladder import lowfi_spent
from infrarsi.search.proposers import PRIOR_SCHEMA, PRIOR_SYSTEM, LLMProposer
from infrarsi.search.toytrain import enabler_tasks, mem_tasks
from infrarsi.search.wcs import WCS, Surrogate

TIGHT = 'mem-small-c80-tight-free-low'          # the reference fits; the largest spend fits only with three enablers,
#                                                 two of them undeclared by the simulated model


def task(name):
    return next(t for t in mem_tasks() + enabler_tasks() if t.name == name)


def js(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


def started(name=TIGHT, **kw):
    eng, sim, sp, facts = toy(task(name))
    e = ESLS(sp(), facts=facts, seed=1, **kw)
    e._init(eng)
    return e, eng, sim


def test_the_chain_is_legal_on_the_pareto_front_and_ordered_by_the_ledger():
    e, eng, _ = started()
    lat = e.lattice
    assert lat[0]['cfg'] == e.ref and len(lat) >= 2 and e.spender_knobs
    mems, gains = [r['mem'] for r in lat], [r['gain'] for r in lat]
    assert mems == sorted(mems) and all(b > a for a, b in zip(gains, gains[1:]))   # more memory only for more gain
    assert {js(r['cfg']) for r in lat} <= {js(c) for c in e.legal_levels} and all(e._legal(eng, r['cfg']) for r in lat)
    assert all({k for k in e.knobs if r['cfg'][k] != e.ref[k]} <= set(e.spender_knobs) for r in lat)
    assert e.enablers and not set(k for k, _ in e.enablers) & set(e.spender_knobs)
    before = [js(r['cfg']) for r in lat]
    e.mledger.observe_lower(lat[-1]['cfg'], 10 * e.mem.cap)          # the ledger learns: the chain is rebuilt by it
    e._build_chain()
    assert [js(r['cfg']) for r in e.lattice] != before or len(before) == 1 or e.pruned


def test_enabler_probes_measure_what_each_frees_within_their_share():
    e, eng, sim = started()
    seen = []
    while True:
        q = e.propose(eng, 16)[0]
        if q.hypothesis.split('|', 1)[0] != 'enabler':
            break
        assert tuple(q.fidelity) == ('probe',) and q.keep_probe
        out = eng.evaluate(q)
        e.observe(eng, q, out)
        seen.append(out.status)
    assert seen and set(seen) == {'probed'} and eng.s.ledger.used['screens'] == 0   # probes are no screens
    ref = sim.reference()
    measured = [(k, en) for (k, _), en in e.enablers.items() if en['status'] == 'measured']
    assert measured and all(en['frees'] == pytest.approx(sim.memory(ref) - sim.memory({**ref, k: en['value']}), abs=0.05)
                            for k, en in measured)
    assert all(e.mledger.ran_status({**e.ref, k: en['value']}) == 'ok' for k, en in measured)   # its own result now
    biggest = max(r['gpu_hours'] for r in eng.s.screens)
    assert lowfi_spent(eng) <= ENABLER_SHARE * e.ladder.allowance(eng) + biggest
    assert len(measured) < len(e.enablers) or lowfi_spent(eng) + biggest <= ENABLER_SHARE * e.ladder.allowance(eng) + biggest


def test_the_bisection_probes_the_median_level_and_screens_the_boundary():
    e, eng, _ = started(off=['esls_enabler_probe'])
    chain = [e.ref] + [c for c in e.legal_levels if c != e.ref][:8]
    assert len(chain) == 9
    p_of = dict(zip(map(js, chain), (1.0, 0.98, 0.9, 0.7, 0.55, 0.4, 0.2, 0.05, 0.005)))
    e.lattice = [{'cfg': c, 'mem': float(i), 'gain': 0.01 * i} for i, c in enumerate(chain)]
    e._build_chain = lambda: None                                     # (the planted chain stays)
    e._sets = lambda: [()]
    e._gain = lambda cfg: 0.01 * [js(c) for c in chain].index(js(cfg))
    e._pf = lambda cfg, ids=(): (p_of[js(cfg)], 'planted')
    probes = []
    for _ in range(9):
        q = e._step_boundary(eng, 'c0', 10)
        if q.hypothesis.split('|', 1)[0] != 'bound':
            break
        assert tuple(q.fidelity) == ('probe',)
        mid = e.cur['mid']
        probes.append(mid)
        if mid <= 5:                                                  # the planted boundary: level 5 fits, 6 does not
            e.cur['lo'], e.cur['probed_at'] = mid, mid
        else:
            e.cur['hi'] = mid
    assert probes == [4, 5, 6]                                        # nearest one half first: 0.55, then 0.4, then 0.2
    assert q.hypothesis.startswith('boundary|') and e.cur['level'] == 5 and tuple(q.fidelity) == ('full',)
    assert js({k: q.change[k] for k in e.knobs}) == js(chain[5]) and e.lattice_used == 1


def test_a_failed_boundary_gets_the_ledge_one_level_down():
    e, eng, sim = started(off=['esls_enabler_probe'])
    fits = [c for c in e.legal_levels if c != e.ref and sim.truth(c)[1] is None]
    ooms = [c for c in e.legal_levels if sim.truth(c)[1] == 'oom']
    assert fits and ooms
    e.lattice = [{'cfg': e.ref, 'mem': 0.0, 'gain': 0.0}, {'cfg': fits[0], 'mem': 1.0, 'gain': 0.02},
                 {'cfg': ooms[0], 'mem': 2.0, 'gain': 0.05}]
    e.cur = {'set': (), 'lo': 2, 'hi': 3, 'probes': 0, 'ledge': False}   # the bisection believed the top level fits
    q = e._screen_boundary(eng, 'c0')
    assert q.hypothesis.startswith('boundary|') and q.fidelity[0] == 'probe'
    out = eng.evaluate(q)
    e.observe(eng, q, out)
    assert out.status == 'infeasible' and e.cur is None and e.boundaries[-1]['outcome'] in ('oom', 'infeasible')
    ledge = e.pending[-1]
    assert ledge.hypothesis.startswith('ledge|') and tuple(ledge.fidelity) == ('full',) and e.lattice_used == 2
    assert js({k: ledge.change[k] for k in e.knobs}) == js(fits[0])
    assert e.propose(eng, 16)[0] is ledge                             # pending first


def test_the_hedge_is_one_draw_when_the_boundaries_tie():
    e, eng, _ = started(off=['esls_enabler_probe'])
    assert e._step_hedge(eng, 'c0', 10) is None                       # no boundary yet
    e.boundaries = [{'set': (), 'level': 1, 'cfg': e.ref, 'gain': 0.050}, {'set': (), 'level': 2, 'cfg': e.ref, 'gain': 0.051}]
    q = e._step_hedge(eng, 'c0', 10)
    assert q is not None and q.hypothesis.startswith('hedge|') and e._step_hedge(eng, 'c0', 10) is None
    off = ESLS(e.proposer, facts=e.facts, off=['esls_hedge'])
    assert off._step_hedge(eng, 'c0', 10) is None


def test_undeclared_enablers_enter_by_their_measured_gib_and_unlock_the_largest_spend(monkeypatch):
    row, e = run_kept(monkeypatch, task(TIGHT), lambda sp, facts, sim: ESLS(sp, facts=facts, seed=sim.task.seed % 2 ** 31))
    eng = e.engine
    best = eng.s.candidates[eng.incumbent()].config
    undeclared = [(k, en['value']) for (k, _), en in e.enablers.items()
                  if en['declared'] is None and en['status'] == 'measured' and en['frees'] >= e._frees_min()]
    assert undeclared and sum(best.get(k) == v for k, v in undeclared) >= 1
    assert any(best.get(k) != e.ref.get(k) for k in e.spender_knobs)   # ... with a spend the enablers made fit
    d = e.describe()['esls']
    assert row['frac'] >= 0.85 and row['oom_screens'] == 0 and d['probes_bisection']
    assert any(b.get('gain_pct') is not None and b['set'] for b in d['boundaries'])
    assert d['ladder']['account']['note'].startswith('Probes and short runs are not counted as screens')


@pytest.mark.parametrize('comp', sorted(ESLS_COMPONENTS))
def test_every_component_off_runs(monkeypatch, comp):
    row, e = run_kept(monkeypatch, mem_tasks()[3], lambda sp, facts, sim: ESLS(sp, facts=facts, off=[comp], seed=5))
    assert row['screens'] == 10 and e.describe()['esls']['off'] == [comp]
    if comp == 'esls_ledger':
        assert e.mledger is None and e.order_ledger is not None       # (wcs's feasibility; the chain by its own ledger)


def test_the_extended_prior_call_asks_for_terms_and_families_and_wcs_is_unchanged():
    from infrarsi.agent.backends.fake import ScriptedBackend
    from test_search import TOGGLE, setup
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    seen = []
    level = {'knob': 'fused_adam', 'value': True, 'gain_pct': 3, 'sd_pct': 2, 'memory': {'direction': 'neutral', 'gib': 0}}

    def extended(role, system, prompt):
        seen.append(system)
        return {'levels': [{**level, 'family': 'optimizer'}], 'interactions': [], 'configs': [], 'notes': '',
                'mem_terms': [{'name': 'optimizer state', 'gib_expr': '2 * 3', 'when': None}]}
    prop = LLMProposer(ScriptedBackend(extended), s.hooks, search_calibrates=True)
    ESLS(prop)                                                        # asks for more in the one prior call
    a = prop.knob_prior(eng, 'c0')
    assert a is not None and PRIOR_RULE.strip() in seen[-1] and seen[-1].startswith(PRIOR_SYSTEM)
    assert prop.last_prior['mem_terms'][0]['gib_expr'] == '2 * 3' and prop.last_prior['levels'][0]['family'] == 'optimizer'
    assert 'mem_terms' not in PRIOR_SCHEMA['properties'] and 'family' not in json.dumps(PRIOR_SCHEMA)   # (a copy)

    def plain_script(role, system, prompt):
        seen.append(system)
        return {'levels': [level], 'interactions': [], 'configs': [], 'notes': ''}
    plain = LLMProposer(ScriptedBackend(plain_script), s.hooks, search_calibrates=True)
    assert plain.knob_prior(eng, 'c0') is not None and PRIOR_RULE.strip() not in seen[-1]   # wcs's call: unchanged
    assert plain.prior_extra is None and plain.last_prior['levels'][0]['knob'] == 'fused_adam'


def test_the_wcs_methods_the_candidates_build_on_exist():
    for name in ('_init', '_table', '_ckey', '_feats', '_words', '_m_of', '_p_feasible', '_proposal', '_speed_gate', '_code',
                 'observe', 'describe', '_candidates', '_incumbent_feats', 'classify', '_mcfg', '_patch_ideas', '_warned',
                 '_pair_words', '_levels', '_legal', '_warn_level', '_llm_used', '_allowance'):   # (the final wcs's too)
        assert callable(getattr(WCS, name, None)), name
    assert {'mu', 'sigma', 'p', 'why', 'probe', 'origin', 'kind'} <= set(inspect.signature(WCS._proposal).parameters)
    assert 'tag' in inspect.signature(Surrogate.observe).parameters and callable(Surrogate.retire)
    assert callable(Surrogate.value) and math.isfinite(ENABLER_SHARE)


def test_the_enablers_are_probed_most_promising_first():
    e, eng, _ = started()
    led = e.order_ledger
    base = led.predict(e.ref)[0]

    def potential(key):                                               # what it may free: its estimate plus its spread
        mu, sd = led.predict({**e.ref, key[0]: e.enablers[key]['value']})
        return max(0.0, base - mu) + sd
    pots = [potential(key) for key in e.enablers]
    assert len(pots) >= 2 and pots == sorted(pots, reverse=True)
    q = e.propose(eng, 16)[0]
    first = next(iter(e.enablers.values()))
    assert q.hypothesis.startswith('enabler|') and q.change[first['knob']] == first['value'] and first['status'] == 'probing'


def test_a_set_with_no_spend_level_beyond_a_reached_boundary_takes_no_probe():
    e, eng, _ = started(off=['esls_enabler_probe'])
    sk = e.spender_knobs
    chain = [e.ref] + [c for c in e.legal_levels if c != e.ref][:4]
    assert len(chain) == 5
    keys = [js({k: c[k] for k in sk}) for c in chain]
    e.lattice = [{'cfg': c, 'mem': float(i), 'gain': 0.01 * i} for i, c in enumerate(chain)]
    e._build_chain = lambda: None                                     # (the planted chain stays)
    e._gain = lambda cfg: 0.01 * keys.index(js({k: cfg[k] for k in sk}))
    e._pf = lambda cfg, ids=(): (0.9, 'planted')
    a, b = list(e.enablers)[:2]
    e._sets = lambda: [(), (a,), (b,)]
    e.boundaries = [{'set': (a,), 'level': 3, 'cfg': e._with((a,), chain[3]), 'gain': 0.03, 'outcome': 'ok'}]   # level 3 reached
    for s in [(), (a,)]:                                              # their one level beyond it ran in full already
        e.full_keys.add(e._ckey(e._with(s, chain[4]), tuple(e.chain)))
    q = e._step_boundary(eng, 'c0', 10)
    assert e.cur['set'] == (b,) and q.hypothesis.startswith('bound|') and e.cur['mid'] == 4
    assert js({k: q.change[k] for k in e.knobs}) == js(e._with((b,), chain[4]))
    skipped = [d for d in e.sets_done if d.get('level') is None]
    assert [d['set'] for d in skipped] == [(), (a,)] and all('beyond those reached' in d['why'] for d in skipped)


def test_esls_keeps_to_wcs_s_one_model_call_allowance():
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.search.engine import Proposal
    from infrarsi.search.wcs import LLM_CALLS_MAX, vk
    from test_search import TOGGLE, setup
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    seen = []

    def script(role, system, prompt):
        seen.append(role)
        if role == 'prior':
            return {'levels': [{'knob': 'fused_adam', 'value': True, 'gain_pct': 3, 'sd_pct': 2, 'family': 'optimizer',
                                'memory': {'direction': 'neutral', 'gib': 0}}], 'interactions': [], 'configs': [],
                    'notes': '', 'mem_terms': []}
        if role == 'code_review':
            return {'verdict': 'patch', 'findings': []}
        return {'hypotheses': [{'mechanism': 'host sync', 'change': {}, 'patch': {'patches.txt': 'remove_host_sync\n'},
                                'predicted_gain_pct': 6, 'site': 'loop'}]}
    prop = LLMProposer(ScriptedBackend(script), s.hooks, search_calibrates=True, features=ESLS.proposer_features())
    e = ESLS(prop, facts={'cap_gib': 80.0})
    e._init(eng)
    assert seen == ['prior'] and prop.calls_left == LLM_CALLS_MAX - 1   # the one allowance set before the call
    d = e.describe()['esls']
    assert d['llm_calls'] == 1 and d['llm_calls_max'] == LLM_CALLS_MAX
    k = sorted(e.knobs)[0]                                            # a screen, then the code track asks
    p = Proposal(change={**e.ref, k: next(x for x in e.knobs[k] if vk(x) != vk(e.ref.get(k)))}, absolute=True,
                 hypothesis='k0', origin='esls')
    e.ideas['k0'] = frozenset()
    e.observe(eng, p, eng.evaluate(p))
    e._code(eng, eng.incumbent(), 5)
    assert len(seen) > 1 and e._llm_used() == len(seen) <= LLM_CALLS_MAX
    e.queue.clear()
    prop.driver_calls = LLM_CALLS_MAX                                 # the allowance spent
    n = len(seen)
    assert e._step_code(eng, eng.incumbent(), 5) is None and len(seen) == n   # no more model calls; the knobs go on
    spent = LLMProposer(ScriptedBackend(script), s.hooks, search_calibrates=True, features=ESLS.proposer_features())
    spent.driver_calls = LLM_CALLS_MAX
    e2 = ESLS(spent, facts={'cap_gib': 80.0})
    e2._init(eng)
    assert len(seen) == n and e2.prior.source == 'flat' and e2.propose(eng, 16)   # no prior call; the search still runs


def test_without_a_spender_no_enabler_is_probed():
    from infrarsi.search.toytrain import tasks
    eng, sim, sp, facts = toy(tasks(36)[0], memory=False)               # no card, no declared spend
    e = ESLS(sp(), facts=facts, seed=1)
    e._init(eng)
    assert not e.spender_knobs and len(e.lattice) == 1 and e.enablers
    q = e.propose(eng, 16)[0]
    assert not q.hypothesis.startswith('enabler|')                     # nothing to unlock: no probe measures an enabler
    assert {en['status'] for en in e.enablers.values()} == {'unprobed'}
    assert all(en['why'] == 'no spend level to unlock' for en in e.enablers.values())
