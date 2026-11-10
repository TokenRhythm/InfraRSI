"""xfer (search/xfer.py; FABLE_PICK_C6_20261031 section 2): the registration and the switches, the source environments
(their numbers, the similarity weights, the winners mapped by knob name - a value the table lacks, a value above the card
alone, the completion -, the effects), the RGPE update (agreements, disagreements, its floors, the pairs it reads), the
screens (the top source first, probed below 0.7, the next one after a failure, at most two source full runs, the prior's
configuration by screen 2's rule or first without a source, the tail before the last screen, a source winner within
2 sigma or the draw on the last), the port ask (an additive field of the first patch call only, best weighted gain first,
one per site, the inert list, at most three patches through the unchanged self-proof, by their entry's weighted gain), no
brief or a malformed one (ordinary asks), the home as the only source, an engine-level run of ten screens with its
accounting (screens, probes, short runs, model calls, dollars), every component off, the toybench arms, and no knob,
model or library name in the module."""
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cand_helpers import run_kept
from infrarsi.search import ABLATIONS, LLM_STRATEGIES, STRATEGIES, XFER_COMPONENTS, resolve
from infrarsi.search.llmopt import CALLS_MAX, USD_MAX
from infrarsi.search.toytrain import enabler_tasks, mem_tasks
from infrarsi.search.wcs import CODE_ASKS as WCS_CODE_ASKS
from infrarsi.search.xfer import (CODE_ASKS, F_MIN, PORT_RULE, SOURCE_SCREENS, W_MIN, XFer, brief_entries, env_numbers,
                                    same_env, similarity, sources_from_packs)
from test_wcs import START, PriorProposer, answer, engine

SRC = Path(__file__).resolve().parents[3] / 'infrarsi' / 'search'
TARGET = {'card_gib': 141, 'nodes': 1, 'params_b': 27.0, 'framework': 'hf_trainer'}
NEAR = (141, 1, 25.81)          # card x2, nodes x1.5, size x1.5: 4.5
SAME_CARD = (141, 1, 8.95)      # card, nodes: 3.0
FAR = (77.3, 2, 35.0)           # size only: 1.5


def xid(k):
    return 'x-' + k.replace('_', '-')


def state(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def pack(pid, env, rows, *, noise=1.0, framework='hf_trainer'):
    """A run pack as runs.write_back writes one: the environment's numbers, an observation per row (id, the knob values off
    START, its step time, status; `patch`: it carried one)."""
    lines = [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': pid, 'title': pid, 'license': 'x',
              'tier': 'verified'},
             {'kind': 'env', 'id': 'env', 'gpu': 'g', 'gpus_per_node': 8, 'nodes': env[1], 'memory_gib': env[0],
              'framework': framework, 'model': {'params_b': env[2]}, 'workload': {'seq_len': 4096}, 'noise_pct': noise}]
    for row in rows:
        oid, cfg, t = row[:3]
        status = row[3] if len(row) > 3 else 'ok'
        o = {'kind': 'observation', 'id': oid, 'env': 'env', 'method': 'infrarsi', 'status': status,
             'config': {xid(k): state(v) for k, v in {**START, **cfg}.items()}}
        if 'patch' in row[4:]:
            o['config']['x-patch'] = 'abcdef012345'
        if status == 'ok':
            o['step_time_s'] = t
        lines.append(o)
    return lines


def brief(proven=(), inert=(), framework='hf_trainer'):
    """A transfer brief: entries (id, site, gain or reason, env) in the builder's format."""
    def src(env):
        return {'card_gib': env[0], 'nodes': env[1], 'params_b': env[2], 'framework': framework}
    return {'schema': 'infrarsi-transfer-brief/1', 'builder': 'test',
            'proven': [{'id': i, 'status': 'proven', 'mechanism': f'mechanism {i}', 'site': site, 'gain_pct': g,
                        'source': src(env), 'src': 0} for i, site, g, env in proven],
            'inert': [{'id': i, 'status': 'inert', 'mechanism': f'mechanism {i}', 'site': site, 'reason': r,
                       'source': src(env), 'src': 0} for i, site, r, env in inert],
            'sources': [{'arm': 'an arm'}]}


class PortProposer(PriorProposer):
    """A scripted model: the prior (test_wcs.answer), and patch calls that read the strategy's extra fields
    (`context`) and answer with `answers(ctx, call)` - patches of train_patches.py."""
    context = None

    def __init__(self, prior=None, answers=None):
        super().__init__(prior)
        self.answers, self.seen = answers, []

    def hypotheses(self, engine, base, ledger=None):
        self.calls += 1
        ctx = self.context(engine, base) if self.context is not None else {}
        self.seen.append(json.loads(json.dumps(ctx, default=str)))
        return self.answers(ctx, len(self.seen)) if self.answers else []


def patch_h(name, mech=None, mu=0.03):
    from hls2_helpers import BASE_TEXT, PATCH_FILE, snippet
    return {'mechanism': mech or f'patch {name}', 'change': {}, 'patch': {PATCH_FILE: BASE_TEXT + snippet(name)}, 'mu': mu,
            'sigma': 0.02, 'p_feasible': 0.9, 'site': name}


def xfer(prior=None, *, packs=None, brief_=None, answers=None, env=TARGET, screens=10, effect=None, **kw):
    eng = engine(screens=screens, **({'effect': effect} if effect else {}))
    st = XFer(PortProposer(prior, answers), facts={'cap_gib': 100.0}, seed=3, packs=packs, brief=brief_, env=env, **kw)
    return st, eng


# ---------------------------------------------------------------- registration and switches
def test_registration_switches_and_its_own_asks():
    assert 'xfer' in STRATEGIES and 'xfer' in LLM_STRATEGIES and resolve('xfer', has_llm=True) == 'xfer'
    assert len(XFER_COMPONENTS) == 8 and all(ABLATIONS[k].startswith('xfer: ') for k in XFER_COMPONENTS)
    assert XFer.proposer_features()['self_proof'] is True and XFer.proposer_features(['xfer_code'])['self_proof'] is False
    st = XFer(PriorProposer(None))
    assert st.code_asks == CODE_ASKS == 6 and WCS_CODE_ASKS == 3      # six asks on this instance; wcs keeps three
    assert st.on['wcs_prior'] and st.on['wcs_code'] and st.use_ledger and st.ladder.short is False
    assert (st.calls_max, st.usd_max, st.patches_per_ask) == (CALLS_MAX, USD_MAX, 3)
    off = XFer(PriorProposer(None), off=sorted(XFER_COMPONENTS))
    assert not off.on['wcs_code'] and not off.use_ledger and off.ladder.probe_below == 0.0 and off.on['wcs_prior']
    assert XFer(PriorProposer(None), off=['xfer_probe']).ladder.probe_below == 0.0


# ---------------------------------------------------------------- the environments and the sources
def test_similarity_weighs_card_nodes_and_size_and_requires_the_framework():
    t = env_numbers(TARGET)
    e = lambda c, n, p, fw='hf_trainer': env_numbers({'card_gib': c, 'nodes': n, 'params_b': p, 'framework': fw})  # noqa: E731
    assert similarity(e(*NEAR), t) == pytest.approx(2.0 * 1.5 * 1.5)
    assert similarity(e(*SAME_CARD), t) == pytest.approx(3.0)            # |log(8.95 / 27)| = 1.10 > 0.7
    assert similarity(e(*FAR), t) == pytest.approx(1.5)                  # |log(35 / 27)| = 0.26
    assert similarity(e(77.3, 2, 8.95), t) == pytest.approx(1.0)
    assert similarity(e(141, 1, 27.0, 'megatron'), t) == 0.0             # another framework: not a source
    pk = env_numbers({'memory_gib': 141, 'nodes': 1, 'gpus_per_node': 8, 'model': {'params_b': 27}, 'framework': 'hf_trainer'})
    assert pk['card_gib'] == 141.0 and pk['params_b'] == 27.0 and same_env(pk, t)
    assert not same_env(e(141, 1, 27.5), t) and same_env(e(141.3, 1, 27.1), t)
    assert env_numbers(None) is None and env_numbers({})['framework'] == 'other'


def test_sources_group_by_numbers_keep_knob_only_winners_and_readable_effects():
    a = pack('run-a', NEAR, [('c0', {}, 1.0), ('c1', {'rc': 'half'}, 0.9), ('c2', {'mb': 2}, 0.95),
                             ('c3', {'rc': 'half', 'liger': True}, 0.8), ('c4', {'rc': 'none', 'mb': 2}, 0.5, 'ok', 'patch'),
                             ('c5', {'rc': 'none'}, None, 'oom')])
    a2 = pack('run-a2', NEAR, [('c0', {}, 2.0), ('c1', {'mb': 2, 'liger': True}, 1.5)])   # the same numbers: one source
    noisy = pack('run-b', SAME_CARD, [('c0', {}, 1.0), ('c1', {'mb': 2}, 0.95), ('c2', {'rc': 'half'}, 0.88)], noise=5.0)
    slow = pack('run-c', FAR, [('c0', {}, 1.0), ('c1', {'wk': 0}, 1.1)])
    own = pack('run-own', (141, 1, 27.0), [('c0', {}, 1.0), ('c1', {'rc': 'none'}, 0.5)])
    other = pack('run-mg', NEAR, [('c0', {}, 1.0), ('c1', {'mb': 2}, 0.5)], framework='megatron')
    srcs, notes = sources_from_packs([a, a2, noisy, slow, own, other], env_numbers(TARGET))
    by = {(s['env']['card_gib'], s['env']['params_b']): s for s in srcs}
    assert set(by) == {(141.0, 25.81), (141.0, 8.95), (77.3, 35.0)}
    assert any('own numbers' in n for n in notes) and any('another framework' in n for n in notes)
    s = by[(141.0, 25.81)]
    assert s['w0'] == s['w'] == pytest.approx(4.5) and s['packs'] == ['run-a', 'run-a2']
    assert s['winner']['obs'] == 'c1' and s['winner']['pack'] == 'run-a2'    # x1.333 beats c3's x1.25; patched c4 left out
    assert s['winner']['gain'] == pytest.approx(math.log(2.0 / 1.5))
    assert s['effects'][('x-rc', 'full', 'half')] == pytest.approx(math.log(1 / 0.9))
    assert s['effects'][('x-rc', 'half', 'full')] == pytest.approx(-math.log(1 / 0.9))   # both directions
    assert s['effects'][('x-mb', '1', '2')] == pytest.approx(math.log(1 / 0.95))
    b = by[(141.0, 8.95)]
    assert ('x-mb', '1', '2') not in b['effects'] and ('x-rc', 'full', 'half') in b['effects']   # below 2 x 5 %: dropped
    assert by[(77.3, 35.0)]['winner'] is None                                # nothing faster than its reference


def test_winners_map_by_knob_name_drop_values_above_the_card_and_complete(monkeypatch):
    from infrarsi.search.ledger import MemoryLedger
    rows = [('c0', {}, 1.0), ('c1', {'rc': 'none', 'mb': 2, 'liger': True}, 0.6)]
    p1 = pack('run-near', NEAR, rows)
    for o in p1[2:]:
        o['config']['x-wk'] = '7'                                            # a value this table lacks
    p2 = pack('run-same', SAME_CARD, [('c0', {}, 1.0), ('c1', {'mb': 2}, 0.9)])
    p3 = pack('run-far', FAR, [('c0', {}, 1.0), ('c1', {'mb': 2}, 0.8)])     # the same configuration as p2's: left out
    monkeypatch.setattr(MemoryLedger, 'lower_bound', lambda self, cfg, z=2.0, ids=(): 500.0 if cfg.get('rc') == 'none' else 0.0)
    st, eng = xfer(answer(), packs=[p1, p2, p3])
    eng.s.check_config = lambda cfg, c0: (None,) if cfg['mb'] == 2 and not cfg['tf32'] else (cfg,)   # mb 2 needs tf32
    st._init(eng)
    by = {s['env']['params_b']: s for s in st.sources}
    near = by[25.81]
    assert near['cfg'] == {**START, 'mb': 2, 'liger': True, 'tf32': True}   # rc dropped (above the card alone), wk kept
    assert any('wk: no value of the table' in n for n in near['map_notes'])  # (the reference's), tf32 completed
    assert any('rc=none' in n and 'above the card' in n for n in near['map_notes'])
    assert by[8.95]['cfg'] == {**START, 'mb': 2, 'tf32': True} and by[8.95]['state'] == 'open'
    assert by[35.0]['cfg'] is None and by[35.0]['state'] == 'unusable'
    assert any('heavier source' in n for n in by[35.0]['map_notes'])
    assert [s['index'] for s in st.sources] == [0, 1, 2]                     # by weight: 4.5, 3.0, 1.5


# ---------------------------------------------------------------- the RGPE update
def test_rgpe_counts_sign_agreements_and_floors_the_weight():
    good = pack('run-good', NEAR, [('c0', {}, 1.0), ('c1', {'rc': 'half'}, 0.9), ('c2', {'mb': 2}, 0.92)])
    bad = pack('run-bad', SAME_CARD, [('c0', {}, 1.0), ('c1', {'rc': 'half'}, 1.1), ('c2', {'mb': 2}, 1.08)])
    mute = pack('run-mute', FAR, [('c0', {}, 1.0), ('c1', {'wk': 0}, 0.9)])   # measured nothing this run changes
    st, eng = xfer(answer(), packs=[good, bad, mute])
    st._init(eng)
    g, b, m = (next(s for s in st.sources if s['env']['params_b'] == x) for x in (25.81, 8.95, 35.0))
    st.runs['c1'] = {'cfg': {**START, 'rc': 'half'}, 'kind': 'ok', 'y': 0.08, 'status': 'screened', 'fidelity': 'full'}
    st._rgpe(eng, 'c1')                                                      # the pair (c1, c0): +8 % > 2 sigma_pair
    assert g['w'] == pytest.approx(4.5 * 2) and b['w'] == pytest.approx(3.0 * F_MIN) and m['w'] == pytest.approx(1.5)
    rec = st.weights[-1]
    assert rec['pairs'] == 1 and [x['agree'] for x in rec['sources']] == [1, 0, 0]
    st.runs['c2'] = {'cfg': {**START, 'rc': 'half', 'mb': 2}, 'kind': 'ok', 'y': 0.15, 'status': 'screened', 'fidelity': 'full'}
    st._rgpe(eng, 'c2')                                                      # pairs with c0 and c1: both agree for `good`
    assert g['w'] == pytest.approx(9.0 * 3) and b['w'] == pytest.approx(max(W_MIN, 3.0 * F_MIN * F_MIN))
    for _ in range(12):                                                      # never 0, never negative
        st._rgpe(eng, 'c2')
    assert b['w'] == pytest.approx(W_MIN) and all(s['w'] >= W_MIN for s in st.sources)
    n = len(st.weights)
    st.runs['c3'] = {'cfg': {**START, 'mb': 2}, 'kind': 'ok', 'y': 0.155, 'status': 'screened', 'fidelity': 'full'}
    st.runs['c4'] = {'cfg': {**START, 'mb': 2, 'patch:p1': True}, 'kind': 'ok', 'y': 0.3, 'status': 'screened',
                     'fidelity': 'full'}
    w = [s['w'] for s in st.sources]
    st._rgpe(eng, 'c4')                                                      # other patches than every other run: no pair
    assert st.weights[-1]['pairs'] == 0 and [s['w'] for s in st.sources] == w
    st.runs['c5'] = {'cfg': {**START, 'rc': 'half', 'mb': 2, 'wk': 4}, 'kind': 'ok', 'y': 0.16, 'status': 'screened',
                     'fidelity': 'full'}
    st._rgpe(eng, 'c5')                                                      # c2, c3 within 2 sigma, c4 other patches:
    assert st.weights[-1]['pairs'] == 2                                      # only c0 and c1 are read
    st.runs['c6'] = {'cfg': {**START, 'rc': 'none'}, 'kind': 'ok', 'y': 0.1, 'status': 'aborted', 'fidelity': 'full',
                     'bound': True}
    k = len(st.weights)
    st._rgpe(eng, 'c6')                                                      # a bound is not a measurement: no update
    assert len(st.weights) == k and n < k
    off, eng2 = xfer(answer(), packs=two_sources(), off=['xfer_rgpe'])
    eng2.run(off)
    assert eng2.full_screens() == 10 and off.weights == [] and all(s['w'] == s['w0'] for s in off.sources)


# ---------------------------------------------------------------- screens 1 and 2
def two_sources():
    """A source near this job whose winner runs out of memory (recompute off with micro-batch 2, no enabler) and one
    farther whose winner fits (+21 %)."""
    oom = pack('run-oom', NEAR, [('c0', {}, 1.0), ('c1', {'rc': 'none', 'mb': 2}, 0.7)])
    fit = pack('run-fit', SAME_CARD, [('c0', {}, 1.0), ('c1', {'rc': 'half', 'mb': 2, 'liger': True}, 0.8)])
    return [oom, fit]


def test_screen_one_probes_the_top_source_and_moves_to_the_next_on_failure(monkeypatch):
    from infrarsi.search.llmopt import LLMOptBase
    st, eng = xfer(answer(), packs=two_sources())
    st._init(eng)
    p = {'none': 0.3, 'half': 0.9}
    monkeypatch.setattr(LLMOptBase, '_pf', lambda self, cfg, ids=(): (p.get(cfg['rc'], 1.0), 'planted'))
    q1 = st.propose(eng, 1)[0]
    assert q1.hypothesis.startswith('source|') and tuple(q1.fidelity) == ('probe', 'full')   # below 0.7: probed first
    assert q1.change['rc'] == 'none' and q1.change['mb'] == 2
    o1 = eng.evaluate(q1)
    st.observe(eng, q1, o1)
    assert o1.status == 'infeasible' and st.sources[0]['state'] == 'probe failed' and st.src_fulls == 0
    assert [b for b, _, _ in st._blamed()] == [frozenset({('rc', '"none"')})]   # the ledger blames recompute off, the
    q2 = st.propose(eng, 1)[0]                                               # failed probe's large spender; the next
    assert q2.hypothesis.startswith('source|') and q2.change['rc'] == 'half'  # source, no screen spent, is not of that
    assert tuple(q2.fidelity) == ('full',)                                   # family: at 0.9, a full run
    o2 = eng.evaluate(q2)
    st.observe(eng, q2, o2)
    assert o2.status == 'screened' and eng.incumbent() == o2.cid and st.src_ok and st.src_fulls == 1
    assert eng.full_screens() == 1 and st.sources[1]['state'] == 'screened'
    q3 = st.propose(eng, 1)[0]                                               # the phase is over: screen 2's rule
    assert not q3.hypothesis.startswith('source|')


def test_at_most_two_source_full_runs_count_every_full_run():
    srcs = [pack(f'run-{i}', env, [('c0', {}, 1.0), ('c1', cfg, 0.8)])
            for i, (env, cfg) in enumerate([(NEAR, {'tf32': True}), (SAME_CARD, {'wk': 4}), (FAR, {'wk': 0})])]
    st, eng = xfer(None, packs=srcs, off=['xfer_prior'])                     # three winners that move nothing
    st._init(eng)
    st._pf = lambda cfg, ids=(): (1.0, 'planted')
    phases = []
    for _ in range(3):
        q = st.propose(eng, 1)[0]
        phases.append((st.by_src.get(q.hypothesis) or {}).get('phase'))
        st.observe(eng, q, eng.evaluate(q))
    assert phases == ['source', 'source', 'late']                            # the third: no source screen, a fill (code
    assert st.src_fulls == SOURCE_SCREENS == 2 and not st.src_ok and eng.full_screens() == 3   # had nothing to run)
    assert st.steps[-1]['pick'] == 'fill: source'


def test_screen_two_runs_the_prior_only_when_two_big_knobs_differ_and_the_model_rates_it_above():
    a = answer()
    st, eng = xfer(a, packs=two_sources())
    st._init(eng)
    st.src_tried = True                                                      # (after the source screens)
    q = st._prior_step(eng, 'c0')                                            # aggressive: rc, mb, liger off the incumbent
    assert q is not None and q.hypothesis.startswith('prior|') and q.change['rc'] == 'none'
    one = dict(a, configs=[{'name': 'one', 'change': {'liger': True}, 'gain_pct': 5}])
    st1, eng1 = xfer(one, packs=two_sources())
    st1._init(eng1)
    st1.src_tried = True
    assert st1._prior_step(eng1, 'c0') is None                               # one big knob: code instead
    assert st1.steps[-1]['pick'] == 'prior skipped' and st1.steps[-1]['big_knobs_differ'] == 1
    neg = dict(a, configs=[{'name': 'neg', 'change': {'comp': True, 'liger': True}}])
    st2, eng2 = xfer(neg, packs=two_sources())
    st2._init(eng2)
    st2.src_tried = True
    assert st2._prior_step(eng2, 'c0') is None                               # rated at or below the incumbent
    st3, eng3 = xfer(one, packs=None, brief_=None)                           # no source: the prior first, as it is
    st3._init(eng3)
    q3 = st3.propose(eng3, 1)[0]
    assert q3.hypothesis.startswith('prior|') and q3.change['liger'] is True
    assert any('no usable source configuration' in n for n in st3.notes)
    st4, eng4 = xfer(a, packs=two_sources(), off=['xfer_prior'])
    st4._init(eng4)
    st4.src_tried = True
    assert st4._prior_step(eng4, 'c0') is None


# ---------------------------------------------------------------- out-of-memory families
PAIR = {'rc': 'half', 'mb': 2}


def pair_oom(base, cfg, overlay):
    """Every configuration holding the pair (rc half, micro-batch 2) runs out of memory at its first updates; the rest
    fit (+1 % per value off the reference)."""
    if cfg['rc'] == 'half' and cfg['mb'] == 2:
        return None, 'oom'
    return 0.01 * sum(1 for k, v in cfg.items() if v != START[k]), None


def two_probe_ooms(monkeypatch, effect=None, pin=True, **kw):
    """xfer after two probes out of memory that share the pair and nothing else off the reference; `pin`: the ledger's
    own feasibility pinned at 0.95 (the runs and the ledger do not catch the family: only the family rule can), else
    xfer's own (the ledger's, from the two failures)."""
    from infrarsi.search.llmopt import LLMOptBase
    if pin:
        monkeypatch.setattr(LLMOptBase, '_pf', lambda self, cfg, ids=(): (0.95, 'planted'))
    st, eng = xfer(answer(), packs=None, effect=effect or pair_oom, **kw)
    st._init(eng)
    st.phase = 'code'
    cids = []
    for extra in ({'liger': True}, {'tf32': True, 'wk': 0}):
        q = st._knob_q(eng, 'c0', {**START, **PAIR, **extra}, kind='source', plan=('probe', 'full'), p=0.5)
        o = eng.evaluate(q)
        st.observe(eng, q, o)
        assert o.status == 'infeasible' and o.fidelity == 'probe' and st.runs[o.cid]['kind'] == 'oom'
        cids.append(o.cid)
    return st, eng, cids


@pytest.mark.parametrize('off', [[], ['xfer_ledger']])
def test_two_probe_ooms_sharing_a_pair_send_its_next_carrier_to_a_probe_first(monkeypatch, off):
    st, eng, cids = two_probe_ooms(monkeypatch, off=off)
    blamed = st._blamed()
    pair = {('rc', '"half"'), ('mb', '2')}
    assert blamed and all(cid in cids for _, cid, _ in blamed) and all(w == () for *_, w in blamed)   # nothing fitted
    assert any(b <= pair for b, _, _ in blamed)                                 # what is blamed lies in the shared pair
    if off:                                                                  # (no ledger: the core the two failures share)
        assert [b for b, _, _ in blamed] == [frozenset(pair)]
    nxt = {**START, **PAIR, 'comp': True}                                    # the pair with a value neither failure had
    p, why = st._pf(nxt)
    assert p <= 0.5 < st.ladder.probe_below and 'ran out of memory' in why
    assert st._pf({**START, 'comp': True})[0] == 0.95                       # outside the family: as it was
    assert st._family(dict(START)) is None
    src = {'index': 0, 'cfg': nxt, 'w': 1.0, 'w0': 1.0, 'state': 'open', 'map_notes': [], 'winner': {'gain': 0.1}}
    st.sources = [src]                                                       # a source winner mapped from elsewhere
    q = st._source_q(eng, 'c0', src, phase='late')
    assert q is not None and tuple(q.fidelity) == ('probe', 'full') and st.oom_gate['probed'] >= 1
    o = eng.evaluate(q)
    assert o.status == 'infeasible' and o.fidelity == 'probe'                # caught at its probe, no full screen spent
    assert eng.full_screens() == 0


def test_with_no_probe_left_a_family_member_is_dropped_never_run_in_full(monkeypatch):
    import infrarsi.search.llmopt as llmopt
    st, eng, cids = two_probe_ooms(monkeypatch)
    st.ladder._fund = lambda engine, cost, code=False: False                 # the low-fidelity allowance is spent
    nxt = {**START, **PAIR, 'comp': True}
    src = {'index': 0, 'cfg': nxt, 'w': 1.0, 'w0': 1.0, 'state': 'open', 'map_notes': [], 'winner': {'gain': 0.1}}
    assert st._source_q(eng, 'c0', src, phase='late') is None and src['state'] == 'oom family, no probe'
    assert st.oom_gate['dropped'] == 1
    other = {**START, 'comp': True, 'wk': 4}
    drawn = []

    def draw(s, engine, inc, **kw):                                          # the draw prefers the family member
        cfg = nxt if s._pf(nxt)[0] > 0 else other
        drawn.append(cfg is nxt)
        return cfg, (), s._pf(cfg)[0], 'drawn', {}
    monkeypatch.setattr(llmopt, 'weighted_draw', draw)
    q = st._fallback(eng, 'c0')
    assert drawn == [True, False] and q is not None and tuple(q.fidelity) == ('full',)
    assert {k: q.change.get(k) for k in START} == other                      # drawn again outside the family
    st.tail_base, st.tail.done = None, False
    st.surr.mean = lambda: {('rc', '"half"'): 0.05, ('mb', '2'): 0.05}       # a tail on the pair: refused, the tail ends
    st._knob_cfg = lambda engine, inc: dict(START)
    st.mem.split = lambda base, cfg, known: {'spends': [], 'frees': [], 'unknown': []}   # (no known spender)
    n = st.oom_gate['dropped']
    assert st._tail_step(eng, 'c0', 5) is None and st.tail.done and st.oom_gate['dropped'] == n + 1


def comp_frees(base, cfg, overlay):
    """The pair runs out of memory unless the compiler frees what it needs - and with no loader workers even then; the
    rest fit (+1 % per value off the reference)."""
    if cfg['rc'] == 'half' and cfg['mb'] == 2 and (not cfg['comp'] or cfg['wk'] == 0):
        return None, 'oom'
    return 0.01 * sum(1 for k, v in cfg.items() if v != START[k]), None


def run_cfg(st, eng, cfg, plan=('probe', 'full')):
    q = st._knob_q(eng, 'c0', cfg, kind='source', plan=plan, p=0.5)
    assert q is not None
    o = eng.evaluate(q)
    st.observe(eng, q, o)
    return o


def test_a_fitted_run_holding_a_family_waives_it_only_with_its_compensating_values(monkeypatch):
    st, eng, cids = two_probe_ooms(monkeypatch, effect=comp_frees, off=['xfer_ledger'])   # (the core: the pair)
    bare, held = {**START, **PAIR, 'wk': 4}, {**START, **PAIR, 'comp': True, 'wk': 4}
    o = run_cfg(st, eng, {**START, 'mb': 2, 'wk': 4}, plan=('full',))       # one value of the pair: not of the family
    assert o.status == 'screened' and st._family(bare) is not None and st._family(held) is not None
    o = run_cfg(st, eng, {**START, **PAIR, 'comp': True})                   # the pair in a run that fits, with the
    assert o.status == 'screened' and st.runs[o.cid]['kind'] == 'ok'        # compiler: never cleared (XFER_r1 1) -
    [(b, cid, waive)] = st._blamed()                                         # waived with the compiler only
    assert b == frozenset({('rc', '"half"'), ('mb', '2')}) and cid == cids[0] and waive == (frozenset({('comp', 'true')}),)
    assert st._family(bare) == (cids[0], b) and st._pf(bare)[0] <= 0.5     # without it: a member, probed first
    assert st._family(held) is None and st._pf(held)[0] == 0.95             # with it: what the runs and the ledger say
    assert st._family(dict(START)) is None                                   # the reference: never of a family
    assert 'waived with comp=true' in st.describe()['xfer']['oom_families']['blamed'][0]


def test_codex_r1_a_compensated_fit_does_not_clear_the_family_for_the_uncompensated(monkeypatch):
    """XFER_r1 1, Codex's scenario: the ledger on, xfer's own feasibility; two probes sharing the pair out of memory, a
    third with the pair and the memory-saving compiler fits; a later carrier of the pair with an unrelated value and no
    compiler, the probe allowance spent: dropped, never a full screen (it ran out of memory there as one). The waiver
    holds after another uncompensated member fails, and ends when a run holding the pair and the compiler fails."""
    st, eng, cids = two_probe_ooms(monkeypatch, effect=comp_frees, pin=False)
    assert st.mledger is not None
    o = run_cfg(st, eng, {**START, **PAIR, 'comp': True})
    assert o.status == 'screened' and st.runs[o.cid]['kind'] == 'ok'
    blamed = st._blamed()
    assert blamed and all(b <= {('rc', '"half"'), ('mb', '2')} for b, _, _ in blamed)   # the ledger blames the pair's
    assert all(w == (frozenset({('comp', 'true')}),) for *_, w in blamed)                # (the compiler: the context)
    bare, held = {**START, **PAIR, 'wk': 4}, {**START, **PAIR, 'comp': True, 'wk': 4}
    assert st._family(bare) is not None and st._family(held) is None
    p, why = st._pf(bare)
    assert p <= 0.5 and 'ran out of memory' in why
    st.ladder._fund = lambda engine, cost, code=False: False                 # the low-fidelity allowance is spent
    src = {'index': 0, 'cfg': bare, 'w': 1.0, 'w0': 1.0, 'state': 'open', 'map_notes': [], 'winner': {'gain': 0.1}}
    st.sources = [src]
    n = eng.full_screens()
    assert st._source_q(eng, 'c0', src, phase='late') is None and src['state'] == 'oom family, no probe'
    assert eng.full_screens() == n and st.oom_gate['dropped'] >= 1
    del st.ladder._fund                                                      # (probes again)
    o = run_cfg(st, eng, bare)                                               # another uncompensated member fails at its
    assert o.status == 'infeasible' and o.fidelity == 'probe'                # probe: the waiver stands
    assert st._family(held) is None and st._family(bare) is not None
    o = run_cfg(st, eng, {**START, **PAIR, 'comp': True, 'wk': 0})          # the pair with the compiler fails too: the
    assert o.status == 'infeasible' and st.runs[o.cid]['kind'] == 'oom'      # compensation did not hold, the waiver ends
    assert all(w == () for *_, w in st._blamed())
    assert st._family(held) is not None and st._pf(held)[0] <= 0.5


def test_the_family_rule_keeps_the_pair_off_full_screens_over_a_whole_run(monkeypatch):
    from infrarsi.search.llmopt import LLMOptBase
    monkeypatch.setattr(LLMOptBase, '_pf', lambda self, cfg, ids=(): (0.95, 'planted'))
    pk = [pack('run-pair', NEAR, [('c0', {}, 1.0), ('c1', {**PAIR, 'liger': True}, 0.7)]),
          pack('run-pair2', SAME_CARD, [('c0', {}, 1.0), ('c1', {**PAIR, 'tf32': True}, 0.75)]),
          pack('run-pair3', FAR, [('c0', {}, 1.0), ('c1', {**PAIR, 'wk': 0}, 0.8)])]
    st, eng = xfer(answer(), packs=pk, effect=pair_oom)
    log = []
    ev = eng.evaluate

    def evaluate(p):
        cfg = {k: p.change.get(k) for k in START} if p.absolute else None
        o = ev(p)
        log.append((tuple(p.fidelity), cfg, o.status, o.fidelity))
        return o
    eng.evaluate = evaluate
    eng.run(st)
    first = next(i for i, (_, cfg, st_, fid) in enumerate(log) if cfg and cfg['rc'] == 'half' and cfg['mb'] == 2)
    later = [x for x in log[first + 1:] if x[1] and x[1]['rc'] == 'half' and x[1]['mb'] == 2]
    assert later and all(fid[0] == 'probe' for fid, *_ in later), later     # after the first: a probe before any full run
    assert not [x for x in later if x[3] == 'full']                          # (each caught at its probe)
    assert st.oom_gate['probed'] >= len(later) and eng.full_screens() == 10


# ---------------------------------------------------------------- the last screens
def test_the_tail_comes_before_the_last_screen_and_a_source_within_two_sigma_on_it():
    st, eng = xfer(answer(), packs=two_sources(), screens=3)
    st._init(eng)
    st.phase = 'code'
    st._pf = lambda cfg, ids=(): (1.0, 'planted')
    eng.s.ledger.used['screens'] = 1                                         # two screens left
    q = st.propose(eng, 1)[0]
    assert q.hypothesis.startswith('tail|')
    eng.s.ledger.used['screens'] = 2                                         # the last screen
    st._mu = lambda engine, inc, cfg, ids=(): (-0.01, 0.02)                  # within 2 sigma
    q = st.propose(eng, 1)[0]
    assert q.hypothesis.startswith('source|') and st.steps[-1]['pick'] == 'last: source'
    assert st.by_src[q.hypothesis]['phase'] == 'late'
    st2, eng2 = xfer(answer(), packs=two_sources(), screens=3)
    st2._init(eng2)
    st2.phase = 'code'
    st2._pf = lambda cfg, ids=(): (1.0, 'planted')
    eng2.s.ledger.used['screens'] = 2
    st2._mu = lambda engine, inc, cfg, ids=(): (-0.10, 0.02)                 # below the incumbent by more: the draw
    q2 = st2.propose(eng2, 1)[0]
    assert q2.hypothesis.startswith('draw|') and st2.steps[-1]['pick'] == 'last: draw'


# ---------------------------------------------------------------- the port ask
BRIEF = brief(proven=[('p001', 'Norm.forward', 5.0, NEAR), ('p002', 'conv_fn', 12.0, FAR), ('p003', 'conv_fn', 4.0, NEAR),
                      ('p004', None, 3.0, SAME_CARD), ('p005', 'own_fn', 40.0, (141, 1, 27.0))],
              inert=[('i001', 'conv_fn', 'self-proof failed: the note is still reported', FAR),
                     ('i002', 'other_fn', 'slower: -1.00 % at its full run', NEAR),
                     ('i003', 'conv_fn', 'self-proof failed: again', NEAR)])


def port_answers(ctx, call):
    if 'transfer' in ctx:
        return [patch_h('b', '[port p002] conv routing here'), patch_h('a', '[port p001] the norm here'),
                patch_h('c', 'an idea of its own'), patch_h('d', '[port p003] not shown: counted as unmapped'),
                patch_h('e', '[Port P004] the other one')]
    return [patch_h(f'n{call}', f'ordinary {call}')]


def test_the_port_ask_is_the_first_asks_extra_field_best_weighted_gain_first_one_per_site():
    st, eng = xfer(answer(), packs=two_sources(), brief_=BRIEF, answers=port_answers)
    st._init(eng)
    assert [e['id'] for e in st.brief_proven] == ['p001', 'p002', 'p003', 'p004'] and st.brief_used   # p005: this job's numbers
    assert any("this job's own numbers" in n for n in st.notes)
    view = st._port_view()
    assert view['rule'] == PORT_RULE
    assert [r['id'] for r in view['proven']] == ['p001', 'p002', 'p004']     # 4.5 x 5 > 1.5 x 12 = 4.5 x 4 > 3 x 3
    assert [r['runs'] for r in view['proven']] == [1, 2, 1]                  # conv_fn twice: p002 shown, p003 counted
    assert view['proven'][0]['weight'] == pytest.approx(4.5) and view['proven'][0]['source'] == {
        'card_gib': 141.0, 'nodes': 1, 'params_b': 25.81}
    assert [x['site'] for x in view['do_not_repeat']] == ['conv_fn', 'other_fn']   # a shown site's first; one per reason
    st.phase = 'code'
    eng.s.ledger.used['screens'] = 1                                         # a full screen ran
    q = st.propose(eng, 1)[0]
    assert len(st.proposer.seen) == 1 and st.proposer.seen[0]['transfer']['proven'][0]['id'] == 'p001'
    assert st.port_done and st.asks == 1 and st.port['answers'] == 5 and len(st.port['queued']) == 3
    assert [x['entry'] for x in st.port['queued']] == ['p001', 'p002', 'p004']   # by their entry's weighted gain; an id
    assert set(st.ported.values()) == {'p001', 'p002', 'p004'}                    # not shown (p003) counts as none
    assert q.hypothesis == 'patch|' + st.port['queued'][0]['patch'] and q.origin == 'wcs-patch'
    assert tuple(q.fidelity) == ('probe', 'full') and q.gate is not None     # the unchanged self-proof at its probe
    assert st.context(eng, 'c0').get('transfer') is None                     # only during the port ask
    eng.s.ledger.used['screens'] = 2
    st.queue = []                                                            # (nothing pending: the next ask)
    st.propose(eng, 1)
    assert len(st.proposer.seen) == 2 and 'transfer' not in st.proposer.seen[1] and st.asks == 2


def test_no_brief_or_a_malformed_one_runs_ordinary_asks():
    for b, why in ((None, 'no transfer brief'), ({'schema': 'nope'}, 'unusable'), ({**BRIEF, 'proven': 'x'}, 'unusable'),
                   (brief(proven=[('p1', 'f', -2.0, NEAR)]), 'no proven patch')):
        st, eng = xfer(answer(), packs=two_sources(), brief_=b, answers=port_answers)
        st._init(eng)
        assert any(why in n for n in st.notes), (b, st.notes)
        st.phase = 'code'
        eng.s.ledger.used['screens'] = 1
        st.propose(eng, 1)
        assert st.proposer.seen and 'transfer' not in st.proposer.seen[0] and not st.port_done
    proven, inert, bad = brief_entries({**BRIEF, 'proven': BRIEF['proven'] + [{'id': 'x'}, 'junk', {'mechanism': 'm'}]})
    assert bad is None and len(proven) == 5 and len(inert) == 3              # malformed entries dropped one by one
    st, eng = xfer(answer(), packs=two_sources(), brief_=BRIEF, answers=port_answers, off=['xfer_port'])
    st._init(eng)
    st.phase = 'code'
    eng.s.ledger.used['screens'] = 1
    st.propose(eng, 1)
    assert 'transfer' not in st.proposer.seen[0] and not st.port_done


def test_the_port_asks_patches_are_held_to_three_and_a_failed_call_degrades():
    def many(ctx, call):
        return [patch_h(x, f'[port p00{i % 3 + 1}] {x}') for i, x in enumerate('abcdef')] if 'transfer' in ctx else []
    st, eng = xfer(answer(), packs=two_sources(), brief_=BRIEF, answers=many)
    st._init(eng)
    st.phase, eng.s.ledger.used['screens'] = 'code', 1
    st.propose(eng, 1)
    assert len(st.port['queued']) == 3 and len(st.queue) == 3
    def boom(ctx, call):
        raise RuntimeError('the backend fell over')
    st2, eng2 = xfer(answer(), packs=two_sources(), brief_=BRIEF, answers=boom)
    st2._init(eng2)
    st2.phase, eng2.s.ledger.used['screens'] = 'code', 1
    q = st2.propose(eng2, 1)
    assert st2.queue == [] and q and not q[0].hypothesis.startswith('patch|')   # a fill, not a crash; tried once,
    assert st2.port_tried and not st2.port_done and st2.port['sent'] is False and st2.port['shown'] == []   # not done
    assert st2.steps[-2]['pick'] == 'port ask not sent' and st2.asks == 1


def test_asks_stay_within_six_and_wait_for_a_full_screen():
    st, eng = xfer(answer(), packs=None, brief_=BRIEF, answers=port_answers)
    st._init(eng)
    st.phase = 'code'
    for n in range(1, 10):
        eng.s.ledger.used['screens'] = n
        for _ in range(2):                                                   # twice per screen: one ask at most
            st.queue = []
            st._code(eng, 'c0', 10 - n)
    assert st.asks == 6 and len(st.proposer.seen) == 6
    assert 'transfer' in st.proposer.seen[0] and all('transfer' not in x for x in st.proposer.seen[1:])


def test_the_briefs_entries_never_mix_with_wcs_own_inert_patches():
    """The brief's lists are xfer's own (`brief_proven`, `brief_inert`): wcs's `inert` - this run's inert patches, shown
    to the model as `inert_patches` on every patch call - starts empty and only this run's rows enter it; the brief's
    inert entries reach the model only as the port ask's `do_not_repeat`, and the brief's count stays its own."""
    st, eng = xfer(answer(), packs=two_sources(), brief_=BRIEF, answers=port_answers)
    st._init(eng)
    assert len(st.brief_inert) == 3 and st.inert == []
    st.phase = 'code'
    eng.s.ledger.used['screens'] = 1
    st.propose(eng, 1)                                                       # the port ask
    port = st.proposer.seen[0]
    assert 'inert_patches' not in port and [x['site'] for x in port['transfer']['do_not_repeat']] == ['conv_fn', 'other_fn']
    st.inert.append({'candidate': 'c9', 'mechanism': 'a patch of this run', 'site': 'fn', 'at': 'probe'})   # (wcs's row)
    ctx = st.context(eng, 'c0')
    assert [r['mechanism'] for r in ctx['inert_patches']] == ['a patch of this run'] and 'transfer' not in ctx
    assert st.describe()['xfer']['brief'] == {'proven': 4, 'inert': 3, 'used': True}


# ---------------------------------------------------------------- the home is the only source
def test_the_home_is_read_once_and_only_with_the_experience_plan(tmp_path, monkeypatch):
    from infrarsi.experience.pack import write_lines
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path))
    runs = tmp_path / 'experience' / 'runs'
    runs.mkdir(parents=True)
    for pk in two_sources() + [pack('run-skip', FAR, [('c0', {}, 1.0), ('c1', {'wk': 4}, 0.9)])]:
        write_lines(runs / f'{pk[0]["pack_id"]}.jsonl', pk)
    (tmp_path / 'experience' / 'transfer_brief.json').write_text(json.dumps(BRIEF), encoding='utf-8')
    plan = {'fingerprint': {'memory_gib': 141, 'nodes': 1, 'gpus_per_node': 8, 'model': {'params_b': 27.0},
                            'framework': 'hf_trainer'}, 'packs': ['run-oom', 'run-fit']}
    st = XFer(PortProposer(answer()), facts={'cap_gib': 100.0}, seed=3, experience=plan)
    eng = engine()
    st._init(eng)
    assert st.target['card_gib'] == 141.0 and st.target['params_b'] == 27.0
    assert sorted(s['packs'][0] for s in st.sources) == ['run-fit', 'run-oom'] and len(st.brief_proven) == 4
    none = XFer(PortProposer(answer()), facts={'cap_gib': 100.0}, seed=3)    # no plan: nothing read
    none._init(engine())
    assert none.sources == [] and none.brief_proven == [] and any('no home read' in n for n in none.notes)
    (tmp_path / 'experience' / 'transfer_brief.json').write_text('{not json', encoding='utf-8')
    bad = XFer(PortProposer(answer()), facts={'cap_gib': 100.0}, seed=3, experience=plan)
    bad._init(engine())
    assert bad.brief_proven == [] and any('does not parse' in n for n in bad.notes) and len(bad.sources) == 2


# ---------------------------------------------------------------- an engine-level run and its accounting
def toy_llm_run(verdict='patch', calls_max=CALLS_MAX, findings=()):
    """xfer with an LLMProposer over a priced scripted backend (its code review answers `verdict` with `findings`) on the
    memory toy, with two sibling tasks' packs and a brief, run for ten full screens: (the strategy, the session, the
    backend, the calls seen as (role, system, payload), the trace, the proposer)."""
    import dataclasses
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.agent.budget import VirtualClock
    from infrarsi.search import toybench as tb
    from infrarsi.search.proposers import LLMProposer

    task = mem_tasks()[1]
    sim = tb.make_sim(task)
    clock = VirtualClock()
    runner = tb.ToyRunner(sim, clock)
    ref_cfg = sim.reference()
    probe_task = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                             'gpu_hours': 1e9}))
    ref = runner.run(probe_task, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=tb.FULL)
    c_full = ref['cost']['gpu_seconds'] / 3600
    env = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 10, 'llm_tokens': 10 ** 7, 'wall_seconds': 1e12,
                                                      'gpu_hours': 40 * c_full}))
    hooks = tb.MemoryHooks(env, sim.memory_facts, base_files={'patches.txt': ''})
    s = tb.SearchSession(task=env, budget=env.search_budget(), runner=runner,
                         judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']), run_id='toy-x', arm='x', start_config=ref_cfg,
                         start_receipt=ref, reference_receipt=ref, hooks=hooks, clock=clock, stop_after_same_failures=10 ** 6)
    eng = tb.Engine(s, full_steps=tb.FULL, short_steps=tb.SHORT, probe_steps=tb.PROBE, job_gpu_hours=None,
                    multi_fidelity=True)
    seen = []

    def script(role, system, prompt):
        payload = json.loads(prompt)
        seen.append((role, system, payload))
        if role == 'prior':
            return {'levels': [{'knob': 'fused_loss', 'value': True, 'gain_pct': 2, 'sd_pct': 2,
                                'memory': {'direction': 'frees', 'gib': 3}}],
                    'interactions': [], 'configs': [{'name': 'a', 'change': {'fused_loss': True}, 'gain_pct': 2}], 'notes': ''}
        if role == 'code_review':
            return {'verdict': verdict, 'summary': 'x', 'findings': list(findings)}
        n = sum(1 for r, *_ in seen if r == 'hypotheses')
        ids = [r['id'] for r in (payload.get('transfer') or {}).get('proven') or []]
        names = [f'x{n}{c}' for c in 'abcd']
        return {'hypotheses': [{'mechanism': (f'[port {ids[i]}] ' if i < len(ids) else '') + f'patch {nm}', 'change': {},
                                'patch': {'patches.txt': f'{nm}\nproof:{nm}\n'}, 'predicted_gain_pct': 3, 'site': nm}
                               for i, nm in enumerate(names)]}

    class Priced(ScriptedBackend):
        def __init__(self):
            super().__init__(script, tokens_per_call=2000)
            self.max_retries, self.kw, self.attempts_made = 0, [], 0

        def complete(self, system, prompt, **kw):
            self.kw.append(dict(kw))
            res = super().complete(system, prompt, **{k: v for k, v in kw.items() if k in ('schema', 'call_dir', 'role',
                                                                                           'expect_json')})
            self.attempts_made += 1
            cap = kw.get('max_usd')
            return dataclasses.replace(res, llm_usd=1.0 if cap is None else min(1.0, float(cap)), attempts=1)
    be = Priced()
    prop = LLMProposer(be, hooks, search_calibrates=True, features=XFer.proposer_features())
    packs = tb.toy_packs(task, 2)
    st = XFer(prop, facts=sim.memory_facts, seed=1, packs=packs, env=tb.toy_env(sim), calls_max=calls_max,
              brief=brief(proven=[('p001', 'fn_a', 6.0, (sim.cap, 1, sim.memory_facts['params_b'] * 1.1)),
                                  ('p002', 'fn_b', 9.0, (sim.cap * 2, 2, 999.0))], framework='other'))
    trace = eng.run(st)
    return SimpleNamespace(st=st, s=s, be=be, seen=seen, trace=trace, prop=prop)


def test_ten_screens_on_a_toy_table_and_their_accounting():
    """LLMProposer over a priced scripted backend on the memory toy, with two sibling tasks' packs and a brief: ten
    full screens, probes (not screens), no short run, every backend attempt within the ten calls, every dollar within
    the eight, at most six asks of at most three patches, the port ask first."""
    from infrarsi.search.proposers import PRIOR_SYSTEM
    r = toy_llm_run()
    st, s, be, seen = r.st, r.s, r.be, r.seen
    d = st.describe()['xfer']
    fid = [x.get('fidelity') for x in s.screens]
    assert r.trace.stop_reason.startswith('budget') and s.ledger.used['screens'] == 10 == fid.count('full')
    assert fid.count('short') == 0 and fid.count('probe') >= 1 and s.ledger.used.get('trials', 0) >= fid.count('probe')
    assert st._llm_used() == be.attempts_made == d['llm_calls'] <= CALLS_MAX
    assert s.ledger.used['llm_usd'] <= USD_MAX + 1e-9
    assert all(k.get('max_usd') is not None and k['max_usd'] <= USD_MAX + 1e-9 for k in be.kw)
    assert d['code_asks'] <= CODE_ASKS and len(st.port['queued']) <= 3 and st.port['shown'] == ['p001', 'p002']
    assert st.port_done and st.port['sent'] and d['port']['done']
    hyp = [p for x, _, p in seen if x == 'hypotheses']
    assert 'transfer' in hyp[0] and all('transfer' not in p for p in hyp[1:])
    assert hyp[0]['code_review']['verdict'] == 'patch'                       # (a patch review: shown as it answered)
    assert seen[0][0] == 'prior' and seen[0][1] == PRIOR_SYSTEM and 'transfer' not in seen[0][2]   # wcs's prior call as it is
    assert d['source_full_runs'] <= SOURCE_SCREENS and d['sources']
    assert sum(1 for p in st.patches.values()) <= 3 * d['code_asks']


def test_a_no_patch_review_still_sends_the_port_request_with_the_code():
    """XFER_r1 2: the code review (the proposer's first patch call, without the transfer field) answers no_patch; the
    port request still goes out, with the code and the review's verdict as it answered (and why the code is shown); the
    port is done and its entries shown only then; the review is put back as it answered for every later ask (wcs's
    after a no_patch review: no code, no call)."""
    from infrarsi.search.xfer import PORT_VERDICT
    r = toy_llm_run('no_patch')
    st, s, be, seen, prop = r.st, r.s, r.be, r.seen, r.prop
    roles = [x for x, *_ in seen]
    assert roles[:2] == ['prior', 'code_review'] and roles.count('code_review') == 1
    hyp = [p for x, _, p in seen if x == 'hypotheses']
    assert len(hyp) == 1 and 'transfer' in hyp[0] and 'code' in hyp[0]      # the port request, with the code
    assert hyp[0]['code_review']['verdict'] == PORT_VERDICT and PORT_VERDICT.startswith('no_patch ')
    assert [e['id'] for e in hyp[0]['transfer']['proven']] == ['p001', 'p002']
    assert st.port_tried and st.port_done and st.port['sent'] and st.port['shown'] == ['p001', 'p002']
    assert [x['entry'] for x in st.port['queued']] == ['p001', 'p002', None]   # the ports first, its own idea to fill
    assert prop.review['verdict'] == 'no_patch' and '_review' not in prop.__dict__   # the review as it answered
    assert 'complete' not in st._bdriver.__dict__                            # (the dispatch spy undone)
    assert st._llm_used() == be.attempts_made == len(roles) == 3             # the prior, the review, the port
    assert s.ledger.used['screens'] == 10 and s.ledger.used['llm_usd'] <= USD_MAX + 1e-9
    # (reviewed before the port request: the verdict is lifted for it and put back after it)
    held = SimpleNamespace(review={'verdict': 'no_patch', 'summary': 's', 'findings': []})
    undo = st._lift_review_veto(held)
    assert held.review['verdict'] == PORT_VERDICT and held.review['summary'] == 's'
    undo()
    assert held.review == {'verdict': 'no_patch', 'summary': 's', 'findings': []}
    kept = SimpleNamespace(review={'verdict': 'patch', 'summary': 's', 'findings': []})
    st._lift_review_veto(kept)()
    assert kept.review['verdict'] == 'patch'


def test_a_port_request_that_does_not_go_out_shows_nothing():
    """XFER_r1 2: the model budget lets the code review through but not the port's patch request (two calls: the prior,
    the review): the port is tried once, not done, shows no entry, and the steps and the notes say so; a patch the review
    found is still queued (as the ordinary ask queues it), with no entry."""
    r = toy_llm_run(calls_max=2)
    st, seen = r.st, r.seen
    assert [x for x, *_ in seen] == ['prior', 'code_review']                 # no hypotheses request went out
    assert st.port_tried and not st.port_done and st.port['sent'] is False and st.port['shown'] == []
    assert any(x['pick'] == 'port ask not sent' for x in st.steps)
    assert any('the port request did not go out' in n for n in st.notes)
    d = st.describe()['xfer']['port']
    assert d['tried'] and not d['done'] and d['shown'] == [] and st.ported == {}
    assert r.s.ledger.used['screens'] == 10 and st._llm_used() == 2
    from infrarsi.project.report import _xfer
    assert 'its request did not go out (no entry shown)' in '\n'.join(_xfer(st.describe()['xfer']))
    found = {'file': 'patches.txt', 'issue': 'a slow loop', 'knob': None, 'patch': {'patches.txt': 'rv\nproof:rv\n'},
             'predicted_gain_pct': 3, 'site': 'rv'}
    r2 = toy_llm_run(calls_max=2, findings=[found])
    st2 = r2.st
    assert [x for x, *_ in r2.seen] == ['prior', 'code_review'] and not st2.port_done and st2.port['shown'] == []
    assert [(x['entry'], x['mechanism'].startswith('review: ')) for x in st2.port['queued']] == [(None, True)]
    assert st2.ported == {} and any(x['pick'] == 'port ask not sent' and x['queued'] == 1 for x in st2.steps)


# ---------------------------------------------------------------- every component off, the toybench, the names
@pytest.mark.parametrize('off', [[c] for c in XFER_COMPONENTS] + [sorted(XFER_COMPONENTS)])
def test_every_component_off_runs(monkeypatch, off):
    from infrarsi.search.toybench import toy_env, toy_packs
    task = mem_tasks()[2]
    row, st = run_kept(monkeypatch, task, lambda sp, facts, sim: XFer(sp, facts=facts, off=off, seed=5,
                                                                         packs=toy_packs(sim.task), env=toy_env(sim)), budget=6)
    d = st.describe()['xfer']
    assert row['screens'] == 6 and d['off'] == sorted(off) and d['llm_calls'] <= CALLS_MAX and row['shorts'] == 0
    assert d['source_full_runs'] <= SOURCE_SCREENS


def test_the_report_and_the_result_carry_its_account(monkeypatch):
    from infrarsi.project.flow import _cand
    from infrarsi.project.report import _xfer
    from infrarsi.search.toybench import toy_env, toy_packs
    row, st = run_kept(monkeypatch, mem_tasks()[4], lambda sp, facts, sim: XFer(sp, facts=facts, seed=2, packs=toy_packs(sim.task),
                                                                                env=toy_env(sim)), budget=5)
    d = st.describe()
    assert _cand({'extra': {'strategy': d}}, 'xfer') is d['xfer']
    text = '\n'.join(_xfer(d['xfer']))
    assert text.startswith('\n## xfer: transfer-first') and '| source | card GiB / nodes / params |' in text
    assert 'Note: no transfer brief' in text and 'Transfer brief: 0 proven' in text
    assert _xfer(None) == [] and _xfer({}) == []


def test_the_toybench_runs_the_xfer_arms_and_keeps_the_old_ones():
    from infrarsi.search.toybench import arms, run_one, toy_packs
    sim = SimpleNamespace(task=SimpleNamespace(seed=1), memory_facts=None)
    names = set(arms(sim))
    assert {'xfer', 'xfer-src', 'opro', 'ltune', 'random', 'wcs', 'esls', 'rpp', 'rpp-ctl'} <= names
    t = enabler_tasks()[1]
    packs = toy_packs(t)
    assert len(packs) == 5 and all(p[0]['pack_id'] != f'run-toy-{t.name}' for p in packs)   # never the task itself
    for arm in ('xfer', 'xfer-src'):
        r = run_one(t, arm, 6, job_factor=None, proposer={'memory': True}, bind='screens')
        assert r['arm'] == arm and r['screens'] == 6 and r['stop'].startswith('budget') and r['shorts'] == 0


def test_the_module_names_no_knob_model_or_library():
    src = (SRC / 'xfer.py').read_text(encoding='utf-8')
    code = '\n'.join(ln for ln in src.splitlines() if not ln.strip().startswith('#'))
    for word in ('recompute', 'micro_batch', 'liger', 'compile', 'fsdp', 'deepspeed', 'megatron', 'torch', 'qwen', 'gemma',
                 'llama', 'flash', 'expandable', 'checkpointing', 'nccl', 'cudnn', 'triton', 'sdpa', 'hf_trainer'):
        assert not re.search(rf'\b{word}\b', code, re.I), word


def test_the_shared_strategies_are_unchanged():
    """The files xfer must not edit are the pristine tree's (qa_trials/_pc_code_c6_base), byte for byte."""
    base = SRC.parents[2] / '_pc_code_c6_base' / 'infrarsi' / 'search'
    if not base.is_dir():
        pytest.skip('the pristine tree is not next to this one')
    v41 = SRC.parents[1].name in ('pc_code_o4b', 'pc_code_o2b')    # (v4.1 trees: the search-state checkpoints and the
    changed = {'wcs.py', 'llmopt.py', 'engine.py'} if v41 else set()  # bounded asks changed these - test_v41.py and the
    for name in ('wcs.py', 'llmopt.py', 'opro.py', 'ltune.py', 'cand_base.py', 'esls.py', 'rpp.py',    # Codex review
                 'hls3.py', 'strategies.py', 'engine.py', 'ledger.py', 'ladder.py', 'tail.py', 'proof.py', 'toytrain.py'):
        if name in changed:                                                                              # cover them)
            continue
        a, b = (p / name for p in (SRC, base))
        assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest(), name
    if v41:
        return
    import difflib                                       # (v4: proposers.py gained the patch request's cap - lines only
    old, new = ((p / 'proposers.py').read_text(encoding='utf-8').splitlines() for p in (base, SRC))   # inserted)
    assert all(t in ('equal', 'insert') for t, *_ in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes())
