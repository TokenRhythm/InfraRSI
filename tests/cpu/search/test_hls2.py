"""hls2, the HLS search fix (FABLE_DESIGN_SEARCH_FIX, 2026-10-29; Q9U-code-env-opus never tested the Liger knob in ten
screens): T1-T8 of the design's section 6 - never-tested knob entries follow the incumbent (C1), the model's ledger is
what the strategy can still test (C2), honest priors for untested moves judged '+' (C3a), the neighbourhood on every
incumbent without reverts (C3b, C8), the coverage backstop (C3c), kappa per origin (C4), a value that fails to start
twice is unrunnable (C5), same-anchor appends merge (C7); C6 off by default; registration."""
import json
import math
from types import SimpleNamespace

import pytest

from hls2_helpers import SimEngine, SimSession
from infrarsi.agent.search import START_FAILED
from infrarsi.search import LLM_STRATEGIES, STRATEGIES, resolve
from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.strategies import HLS, HLS2, merge3, merge_patches
from test_composition import FakeProposer, hyp
from test_hls_reask import Script


def ns(cid, parent, **cfg):
    return SimpleNamespace(id=cid, parent=parent, config=cfg, overlay={})


class Lineage:
    """A FakeEngine whose incumbents form a lineage: c0 (m=1) -> c1 (m=2) -> c2 (m=2, c=1)."""

    def __init__(self, task=None, sp=0.02):
        cands = {'c0': ns('c0', None, a=0, b=0, c=0, m=1), 'c1': ns('c1', 'c0', a=0, b=0, c=0, m=2),
                 'c2': ns('c2', 'c1', a=0, b=0, c=1, m=2)}
        self.s = SimpleNamespace(candidates=cands, hooks=None, task=task)
        self._inc, self.sp = 'c1', sp

    def incumbent(self):
        return self._inc

    def sigma_pair(self):
        return self.sp

    def sigma_run(self):
        return self.sp / math.sqrt(2)


def h(name, change, mu=0.03, p=0.8, patch=None):
    return {'mechanism': name, 'change': change, 'patch': patch, 'mu': mu, 'sigma': 0.02, 'p_feasible': p}


C1_HYPS = [h('b', {'b': 1}, p=0.95), h('b, m back', {'b': 1, 'm': 1}), h('c on', {'c': 1}), h('m4', {'m': 4}),
           h('a and c', {'a': 1, 'c': 1}), h('code', {}, patch={'f.py': 'x = 1\n'})]


# ---------------------------------------------------------------- T1 (C1)
def test_carry_open_knobs():
    eng = Lineage()
    hls = HLS2(Script([C1_HYPS, []]), explore=False)
    hls.propose(eng, 16)                                            # the model's ideas on c1
    eng._inc = 'c2'                                                 # c=1 moved the incumbent before any of them ran
    props = hls.propose(eng, 16)
    carried = hls.ledger['c2|b']
    assert carried['status'] == 'open' and carried['carried_from'] == 'c1' and carried['p_feasible'] == 0.9
    assert carried['mu'] == 0.03 and carried['rebased']
    assert [p.hypothesis for p in props] == ['c2|b']               # the only open entry of the new incumbent
    why = {k[3:]: v.get('why') for k, v in hls.ledger.items() if k.startswith('c1|')}
    assert {v['status'] for k, v in hls.ledger.items() if k.startswith('c1|')} == {'superseded'}
    assert why['b'] == 'carried to c2' and 'already in the incumbent' in why['c on']
    assert 'undo' in why['b, m back'] and 'undo' in why['m4']      # m was moved by the lineage (1 -> 2)
    assert 'changed on c2' in why['a and c'] and 'patch' in why['code']
    assert hls.residual(eng)['n_open'] == 1
    d = hls.describe()
    assert d['carried'] == 1 and d['superseded'] == 6 and d['fix']['carry_open']


def test_hls_orphans_them_as_before():
    eng = Lineage()
    hls = HLS(Script([C1_HYPS, []]), explore=False)
    hls.propose(eng, 16)
    eng._inc = 'c2'
    assert hls.propose(eng, 16) == []                               # M1: nothing reaches c2
    assert {v['status'] for v in hls.ledger.values()} == {'open'} and 'fix' not in hls.describe()


def test_a_carried_entry_takes_the_prediction_made_for_the_new_incumbent():
    eng = Lineage()
    hls = HLS2(Script([[h('b', {'b': 1})], [h('b again', {'b': 1}, mu=0.05)]]), explore=False)
    hls.propose(eng, 16)
    eng._inc = 'c2'
    props = hls.propose(eng, 16)
    assert [p.hypothesis for p in props] == ['c2|b'] and hls.ledger['c2|b']['mu'] == 0.05 and hls.ledger['c2|b']['rerated']


def test_pending_bundles_are_rebuilt_on_the_new_incumbent():
    rows = [{'knob': k, 'mechanism': k, 'change': {k: 1}, 'level': 'partial', 'gain': 0.0, 'sign': '+', 'p_feasible': 0.9,
             'evidence_ids': [], 'warnings': []} for k in ('a', 'b', 'x')]
    comp = {'members': ['a', 'b', 'x'], 'change': {'a': 1, 'b': 1, 'x': 1}, 'gain': None, 'warnings': []}
    eng = Lineage()
    hls = HLS2(Script([[], []]), explore=False, experience={'seeds': rows, 'composite': comp})
    first = {p.hypothesis: p for p in hls.propose(eng, 16)}
    assert first['bundle|exp'].base == 'c1'
    eng._inc = 'c2'
    again = {p.hypothesis: p for p in hls.propose(eng, 16)}
    assert again['bundle|exp'].base == 'c2' and again['bundle|exp'].change == {'a': 1, 'b': 1, 'x': 1}
    assert hls.members['bundle|exp'] == ['c2|experience:a', 'c2|experience:b', 'c2|experience:x']


# ---------------------------------------------------------------- T2 (C2)
def test_model_view_consistency():
    seeds = [{'knob': 'x', 'mechanism': 'x', 'change': {'x': 1}, 'level': 'partial', 'gain': 0.0, 'sign': '+',
              'p_feasible': 0.9, 'evidence_ids': [], 'warnings': []}]
    eng = Lineage()
    prop = Script([C1_HYPS, [h('d', {'d': 1})], []])
    seen = []
    ask = prop.hypotheses
    prop.hypotheses = lambda engine, base, ledger=None: seen.append((base, ledger)) or ask(engine, base, ledger)
    hls = HLS2(prop, explore=False, experience={'seeds': seeds})
    hls.propose(eng, 16)
    eng._inc = 'c2'
    props = hls.propose(eng, 16)
    view = seen[-1][1]                                              # what the model got on c2
    assert all(r['base'] for r in view) and 'open' not in {r['status'] for r in view}
    queued = {(r['mechanism'], json.dumps(r['change'])) for r in view if r['status'] == 'queued'}
    assert queued and queued <= {(p.mechanism, json.dumps(p.change)) for p in props}
    assert [r['change'] for r in view].count({'x': 1}) == 1         # the re-seeded seed: one row
    sup = {r['mechanism']: r for r in view if r['status'] == 'superseded'}
    assert set(sup) == {'b, m back', 'c on', 'm4', 'a and c', 'code'} and all(r['why'] for r in sup.values())
    assert 'predicted_gain_pct' in sup['m4']                        # never measured: the model's prediction stays
    assert hls.model_view() == hls._states_view('c2')


def test_the_recorded_view_keeps_the_first_incumbent_and_one_row_per_move():
    from hls2_helpers import q9u_replay
    _, old, _ = q9u_replay(lambda prop, **kw: HLS(prop, **kw))
    _, new, _ = q9u_replay(lambda prop, **kw: HLS2(prop, **kw))
    last_old, last_new = old.proposer.views[-1][1], new.proposer.views[-1][1]
    assert len(last_old) > 40                                       # hls: the proposer cut it to the last 40 rows
    moves = [json.dumps([r['change'], r['mechanism'] if not r['change'] else None]) for r in last_new]
    assert len(last_new) <= 40 and len(moves) == len(set(moves))
    assert any(r['base'] == 'c0' for r in last_new)                 # the first incumbent's rows are still there
    assert sum(r['mechanism'] == 'experience:torch_compile' for r in last_old) >= 4
    assert sum(r['mechanism'] == 'experience:torch_compile' for r in last_new) == 1


# ---------------------------------------------------------------- T3 (C3a)
def c3_engine(**kw):
    """The engine as it stood at the recorded decision 4 (incumbent c3): sigma_run 0.0037, three valid full runs, the
    probe measured (dropped from then on)."""
    s = SimSession({'a': 0}, knobs={'a': [0, 1]}, c0_gpu_hours=0.3085)
    s.screens = [{'fidelity': 'full', 'gpu_hours': g, 'valid': True} for g in (0.27588, 0.30637, 0.24979)] + \
        [{'fidelity': 'probe', 'gpu_hours': 0.2355, 'valid': True}]
    return SimEngine(s, None, sigma_runs=[0.0037], **kw)


def vc(eng, p):
    return eng.value(p) / eng.expected_cost(p)


def test_seed_prior():
    rows = [{'knob': k, 'mechanism': k, 'change': {k: 1}, 'level': 'partial', 'gain': 0.0, 'sign': sg, 'p_feasible': 0.9,
             'evidence_ids': [], 'warnings': []} for k, sg in (('plus', '+'), ('zero', '0'), ('minus', '-'))]
    liger = h('liger', {'liger': 1}, mu=math.log1p(0.04), p=0.6)
    liger['sigma'] = 0.04 * 1.1
    nccl = h('nccl patch', {}, mu=math.log1p(0.003), p=0.95, patch={'f.py': 'y\n'})
    nccl['sigma'] = 0.008 * 1.1
    eng = c3_engine()
    hls = HLS(FakeProposer([liger, nccl]), explore=False, kappa0=0.655, probe_below=0.77, prior_sigma=0.03,
              experience={'seeds': rows})
    hls.pairs = [(math.log1p(0.06), 0.0772), (math.log1p(0.20), 0.2133), (math.log1p(0.05), 0.0724)]   # as at c3
    props = {p.mechanism: p for p in hls.propose(eng, 16)}
    sp = eng.sigma_pair()
    assert props['experience:plus'].sigma == pytest.approx(max(2 * sp, 0.03))
    assert props['experience:zero'].sigma == pytest.approx(2 * sp) and 'experience:minus' not in props
    r = {m: vc(eng, p) for m, p in props.items()}
    # the design's c3 numbers (section 4.1): Liger carried 0.0903, the seed as recorded 0.0144, the vague seed 0.0374
    # with its sigma scaled by kappa 0.955 in the replica (0.0391 here: an experience prior is never scaled by kappa)
    assert r['experience:plus'] == pytest.approx(0.0391, abs=1e-3)
    assert r['experience:zero'] == pytest.approx(0.0144, abs=1e-3)
    assert r['liger'] == pytest.approx(0.0903, abs=1e-3)
    assert r['liger'] > r['experience:plus'] > r['nccl patch'] > r['experience:zero']
    off = HLS(FakeProposer([]), explore=False, experience={'seeds': rows})       # hls: zero prior, '-' seeded
    sig = {p.mechanism: p.sigma for p in off.propose(c3_engine(), 16)}
    assert sig['experience:plus'] == pytest.approx(2 * sp) and 'experience:minus' in sig


# ---------------------------------------------------------------- T4 (C3b, C8)
def knob_task(**knobs):
    return SimpleNamespace(knob_schema={'properties': {k: {'enum': v} for k, v in knobs.items()}})


def judged(*rows):
    return {'judgment': {'assessment': [{'knob': k, 'change': {k: v}, 'sign': s, 'p_feasible': p} for k, v, s, p in rows]}}


def test_neighbourhood_always():
    eng = Lineage(task=knob_task(a=[0, 1], b=[0, 1], c=[0, 1], m=[1, 2, 4], d=[0, 1]), sp=0.005)
    exp = judged(('a', 1, '+', 0.72), ('b', 1, '0', 0.9), ('d', 1, '-', 0.9))
    hls = HLS2(Script([[h('c on', {'c': 1})], []]), experience=exp)
    props = {p.hypothesis: p for p in hls.propose(eng, 16)}          # on c1 (m=2): the model gave a proposal
    assert {'c1|c on', 'c1|explore:a=1', 'c1|explore:b=1', 'c1|explore:m=4'} <= set(props)
    assert 'c1|explore:m=1' not in props                            # back to c0's micro-batch: the retry's job (C8)
    assert 'c1|explore:d=1' not in props                            # judged '-' (C8)
    assert 'c1|explore:c=1' not in hls.ledger                       # the model's entry makes that move already
    sp = eng.sigma_pair()
    assert props['c1|explore:a=1'].sigma == pytest.approx(0.03) and props['c1|explore:a=1'].p_feasible == 0.72
    assert props['c1|explore:b=1'].sigma == pytest.approx(2 * sp) and props['c1|explore:m=4'].sigma == pytest.approx(2 * sp)
    assert hls.ledger['c1|explore:a=1']['judged'] == '+' and props['c1|explore:m=4'].p_feasible == 0.6
    assert props['c1|explore:a=1'].mu == 0.0 and props['c1|explore:a=1'].origin == 'hls-explore'
    plain = HLS(Script([[h('c on', {'c': 1})], []]), experience=exp)
    assert [p.hypothesis for p in plain.propose(eng, 16)] == ['c1|c on']   # hls: only once the model is out of ideas


def test_a_model_rating_replaces_a_neighbours_prior():
    eng = Lineage(task=knob_task(a=[0, 1]))
    hls = HLS2(Script([[], [h('a on', {'a': 1}, mu=0.04)]]), reask=1)
    first = [p.hypothesis for p in hls.propose(eng, 16)]
    assert first == ['c1|explore:a=1']
    hls._reask(eng, 'c1')
    assert 'c1|explore:a=1' not in hls.ledger and hls.ledger['c1|a on']['status'] == 'open'


# ---------------------------------------------------------------- T5 (C3c)
def test_backstop():
    patch = h('small patch', {}, mu=0.004, p=0.95, patch={'f.py': 'y\n'})
    rows = [{'knob': 'k', 'mechanism': 'k', 'change': {'k': 1}, 'level': 'partial', 'gain': 0.0, 'sign': '+',
             'p_feasible': 0.8, 'evidence_ids': [], 'warnings': []}]
    hyps = [patch, h('predicted loss', {'z': 1}, mu=-0.01, p=0.9),
            h('big idea', {}, mu=0.02, p=0.2, patch={'g.py': 'z\n'})]          # resolvable, but unlikely to run
    exp = {**judged(('k', 1, '+', 0.8)), 'seeds': rows}
    eng = c3_engine()
    plain = HLS(FakeProposer(hyps), explore=False, experience=exp)
    plain.pairs = [(0.03, 0.03)] * 3
    every = plain.propose(eng, 16)
    best = max(every, key=lambda p: vc(eng, p))
    assert best.mechanism == 'small patch' and best.mu < 2 * eng.sigma_pair()      # an idea one run cannot resolve
    hls = HLS(FakeProposer(hyps), explore=False, backstop=True, experience=exp)
    hls.pairs = [(0.03, 0.03)] * 3
    props = hls.propose(c3_engine(), 16)
    assert [p.mechanism for p in props] == ['experience:k', 'big idea']   # the '+' knob first; >= 2 sigma_pair stays
    assert hls.describe()['backstop'] == 1                          # (the model's predicted loss is not protected)
    none = HLS(FakeProposer([patch]), explore=False, backstop=True)  # no untested knob move: not triggered
    assert [p.mechanism for p in none.propose(c3_engine(), 16)] == ['small patch'] and none.describe()['backstop'] == 0


def test_the_backstop_defers_but_never_ends_the_search():
    """Review NM25 (P2): with the value/cost stop on (a declared job size, the floor met after 4 of 10 runs), narrowing
    to the judged knob (value/cost 0.72 < lambda) stopped the search while the deferred patch (1.52) cleared lambda;
    now decide() gets the full set whenever its stop rule would stop on the narrowed one."""
    from hls2_helpers import BASE_TEXT, PATCH_FILE, snippet
    rows = [{'knob': 'k', 'mechanism': 'k', 'change': {'k': 1}, 'level': 'partial', 'gain': 0.0, 'sign': '+',
             'p_feasible': 0.9, 'evidence_ids': [], 'warnings': []}]
    exp = {**judged(('k', 1, '+', 0.9)), 'seeds': rows}
    wide = {'mechanism': 'wide patch', 'change': {}, 'patch': {PATCH_FILE: BASE_TEXT + snippet('w')}, 'mu': 0.02,
            'sigma': 0.1, 'p_feasible': 0.9}

    def run(job_gpu_hours):
        s = SimSession({'k': 0}, knobs={'k': [0, 1]}, screens=10)
        s.ledger.used['screens'] = 4
        s.screens = [{'fidelity': 'full', 'gpu_hours': 0.3, 'valid': True} for _ in range(4)]
        eng = SimEngine(s, lambda base, cfg, overlay: (0.005, None), sigma_runs=[0.004], full_costs=[0.3],
                        job_gpu_hours=job_gpu_hours)
        hls = HLS2(FakeProposer([wide]), experience=exp)
        return eng, hls, eng.run(hls)
    eng, hls, trace = run(20.0)
    first = eng.decisions[0]
    assert first['n_open'] == 2 and first['best_mechanism'] == 'wide patch' and first['best_value_per_cost'] > 1.5
    assert eng.order[0][0] == 'wide patch' and eng.full_screens() == 5 and hls.backstops == 0
    knob = eng.decisions[1]                                         # the knob alone: below lambda, a stop of decide()'s own
    assert trace.stop_reason == 'stop_rule' and knob['best_mechanism'] == 'experience:k'
    assert knob['best_value_per_cost'] == pytest.approx(0.724, abs=1e-3)
    eng, hls, trace = run(None)                                     # no value/cost stop: the backstop still defers
    assert eng.order[0][1] == {'k': 1} and eng.order[1][0] == 'wide patch' and hls.backstops == 1


# ---------------------------------------------------------------- T6 (C4)
def test_class_kappa():
    hls = HLS2(FakeProposer([]), explore=False)
    knob, patch, review = {'change': {'a': 1}}, {'change': {}, 'patch': {'f': 'x'}}, \
        {'change': {}, 'patch': {'f': 'x'}, 'mechanism': 'review: f: slow'}
    for _ in range(3):
        hls._pair(knob, 0.04, 0.08)                                 # the model under-predicts knobs (Q9U: 1.18) ...
        hls._pair(patch, 0.03, 0.001)                               # ... and over-predicts its patches (0.19)
    hls._pair(review, 0.05, 0.07)
    by = hls.describe()['kappa_by_class']
    assert by['knob']['kappa'] > 1.0 and by['patch']['kappa'] < 0.4 and by['review']['n'] == 1
    assert hls.kappa('knob') > hls.kappa() > hls.kappa('patch')     # the pooled kappa (hls's) sits between
    assert len(hls.pairs) == 7 and hls.residual(SimpleNamespace(incumbent=lambda: 'c0'))['kappa_by_class'] == by
    eng = Lineage()
    two = HLS2(FakeProposer([h('k', {'a': 1}, mu=0.04), h('p', {}, mu=0.04, patch={'f.py': 'z\n'})]), explore=False)
    two.class_pairs = {'knob': [(0.04, 0.05)] * 3, 'patch': [(0.03, 0.001)] * 3}
    two.pairs = two.class_pairs['knob'] + two.class_pairs['patch']
    mus = {q.mechanism: q.mu for q in two.propose(eng, 16)}
    assert mus['k'] == pytest.approx(two.kappa('knob') * 0.04) and mus['p'] == pytest.approx(two.kappa('patch') * 0.04)
    one = HLS(FakeProposer([h('k', {'a': 1}, mu=0.04)]), explore=False)
    one.pairs = list(two.pairs)
    assert one.propose(Lineage(), 16)[0].mu == pytest.approx(one.kappa() * 0.04) and 'kappa_by_class' not in one.describe()


# ---------------------------------------------------------------- T7 (C5)
def _session():
    from test_search import TOGGLE, setup
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    return s, eng


def _receipt(timed_out=True, oom=0, error=''):
    return {'valid': False, 'evidence': {'steps_completed': 0, 'timed_out': timed_out,
                                         'events': {'oom': str(oom), 'traceback': '0', **({'error': error} if error else {})}}}


def test_unrunnable_after_two_start_failures():
    s, eng = _session()
    s.start_failure_limit = 2
    a = s.propose_config(changes={'compile': True}, base='c0')['candidate_id']
    b0 = s.propose_config(changes={'fused_adam': True}, base='c0')['candidate_id']
    s.candidates[b0].verdict = {'valid': True, 'step_time': 1.0}    # another measured base
    b = s.propose_config(changes={'compile': True}, base=b0)['candidate_id']
    oom = s.propose_config(changes={'cuda_graphs': True}, base='c0')['candidate_id']
    killed = SimpleNamespace(reasons=['completes: killed at the timeout after 0/30 updates'])
    s._note_unrunnable(s.candidates[a], _receipt(), killed)
    assert s.value_unrunnable('compile', True) is None              # once may be the combination
    for _ in range(2):
        s._note_unrunnable(s.candidates[oom], _receipt(False, oom=1), SimpleNamespace(reasons=['completes: out of memory']))
    s._note_unrunnable(s.candidates[b], _receipt(), killed)
    assert 'launch timeout' in s.value_unrunnable('compile', True) and s.value_unrunnable('cuda_graphs', True) is None
    c = s.propose_config(changes={'compile': True, 'pin_memory': True}, base=b0)['candidate_id']
    assert s.start_failed_hit(c).startswith('compile=True: killed at the launch timeout')
    out = eng.evaluate(Proposal(change={'compile': True, 'pin_memory': True}, base=b0, mechanism='third'))
    assert out.status == 'rejected' and out.gpu_hours == 0.0 and out.reasons[0].startswith(START_FAILED)
    assert s.invalid.get('start_failed') == 1 and not s.screens   # refused before any run
    hls = HLS2(FakeProposer([]), explore=False)
    entry = {'mechanism': 'third', 'change': {'compile': True}, 'patch': None, 'mu': 0.1, 'sigma': 0.02,
             'p_feasible': 0.9, 'base': 'c0', 'status': 'open', 'key': 'c0|third'}
    hls.ledger['c0|third'] = entry
    hls.observe(eng, Proposal(hypothesis='c0|third'), out)
    assert entry['status'] == 'invalid' and entry['why'].startswith('unrunnable: compile=True')
    assert hls.describe()['unrunnable'] == {'compile=True': 'compile=True: killed at the launch timeout before its first '
                                                            'update (2 runs that set it)'}


AUTOTUNE = 'torch.OutOfMemoryError: out of resource: shared memory, Required: 131072, Hardware limit: 101376.'


def test_an_autotune_out_of_resource_line_is_not_an_oom():
    """Review NM25 (P2): Inductor's autotune "OutOfMemoryError: out of resource" is a config it skips, not an OOM (the
    log scanner's reading, signals.OOM_RE): two zero-update compile timeouts whose log carries it count and make
    compile=True unrunnable; real CUDA out-of-memory failures still do not count."""
    from infrarsi.project.signals import scan_events
    assert scan_events([(0, AUTOTUNE)])['oom'] == 0
    s, _ = _session()
    s.start_failure_limit = 2
    killed = SimpleNamespace(reasons=['completes: killed at the timeout after 0/30 updates'])
    tuned = {'valid': False, 'evidence': {'steps_completed': 0, 'timed_out': True,
                                          'events': {'oom': '0', 'traceback': '1', 'error': AUTOTUNE}}}
    b0 = s.propose_config(changes={'fused_adam': True}, base='c0')['candidate_id']
    s.candidates[b0].verdict = {'valid': True, 'step_time': 1.0}
    for base in ('c0', b0):
        cid = s.propose_config(changes={'compile': True}, base=base)['candidate_id']
        s._note_unrunnable(s.candidates[cid], tuned, killed)
    assert s.start_failures == {json.dumps(['compile', True]): 2}
    assert s.value_unrunnable('compile', True).startswith('killed at the launch timeout')
    real = 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 3 has a total capacity of 139.81 GiB'
    crash = SimpleNamespace(reasons=['completes: exit code 1, 0/30 updates; the log: ' + real])
    for base, events in (('c0', {'oom': '0', 'traceback': '1', 'error': real}), (b0, {'oom': '8', 'traceback': '9'})):
        cid = s.propose_config(changes={'cuda_graphs': True}, base=base)['candidate_id']
        s._note_unrunnable(s.candidates[cid], {'valid': False, 'evidence': {'steps_completed': 0, 'timed_out': False,
                                                                           'events': events}}, crash)
    assert json.dumps(['cuda_graphs', True]) not in s.start_failures and s.value_unrunnable('cuda_graphs', True) is None


def test_hls2_skips_an_unrunnable_value_and_hls_records_nothing():
    s, eng = _session()
    rows = [{'knob': 'compile', 'mechanism': 'compile', 'change': {'compile': True}, 'level': 'partial', 'gain': 0.0,
             'sign': '+', 'p_feasible': 0.9, 'evidence_ids': [], 'warnings': []}]
    hls = HLS2(FakeProposer([dict(hyp('compile'), change={'compile': True, 'pin_memory': True}, mechanism='model compile')]),
               experience={'seeds': rows})
    hls.propose(eng, 16)
    assert s.start_failure_limit == 2                               # hls2 asked the session to count
    s.unrunnable_values[json.dumps(['compile', True])] = 'killed at the launch timeout before its first update'
    props = hls.propose(eng, 16)
    assert not [p for p in props if p.change.get('compile')]
    assert hls.ledger['c0|model compile']['status'] == 'invalid' and 'unrunnable' in hls.ledger['c0|model compile']['why']
    assert hls.ledger['c0|experience:compile']['status'] == 'invalid'
    s2, eng2 = _session()
    plain = HLS(FakeProposer([]))
    plain.propose(eng2, 16)
    assert s2.start_failure_limit is None                           # hls (and random, tpe): nothing counted
    c = s2.propose_config(changes={'compile': True}, base='c0')['candidate_id']
    s2._note_unrunnable(s2.candidates[c], _receipt(), SimpleNamespace(reasons=['completes: killed']))
    s2._note_unrunnable(s2.candidates[c], _receipt(), SimpleNamespace(reasons=['completes: killed']))
    assert s2.start_failures == {} and s2.unrunnable_values == {}


def test_toy_crash_twice_is_refused_at_no_cost():
    from test_search import BOTH, setup
    sim, s, eng = setup(BOTH, tokens=10 ** 6)
    s.start_failure_limit = 2
    bad = {'tp_overlap': True, 'seq_parallel': False}               # crashes before its first update on these bases
    assert eng.evaluate(Proposal(change=bad, mu=0.01, fidelity=('full',), abortable=False)).status == 'invalid'
    other = eng.evaluate(Proposal(change={'pin_memory': True}, mu=0.01, fidelity=('full',), abortable=False))
    assert other.status in ('screened', 'aborted')
    assert eng.evaluate(Proposal(change=bad, base=other.cid, mu=0.01, fidelity=('full',), abortable=False)).status == 'invalid'
    n, gpu = len(s.screens), s.ledger.used['gpu_hours']
    third = eng.evaluate(Proposal(change={**bad, 'fused_adam': True}, mu=0.01, fidelity=('full',), abortable=False))
    assert third.status == 'rejected' and START_FAILED in third.reasons[0] and 'tp_overlap=True' in third.reasons[0]
    assert len(s.screens) == n and s.ledger.used['gpu_hours'] == gpu


# ---------------------------------------------------------------- T8 (C7)
TAIL = 'import torch\n\n\ndef _install():\n    pass\n\n\n_install()\n'
F1 = TAIL + '\n\ndef _lm_head():\n    pass\n\n\n_lm_head()\n'
F2 = TAIL + '\n\ndef _clip():\n    pass\n\n\n_clip()\n'


def test_merge_append_same_anchor():
    assert merge3(TAIL, F1, F2) is None                             # hls: two appends after the last call conflict
    assert merge3(TAIL, F1, F2, append=True) == F1 + F2[len(TAIL):]
    assert merge3(TAIL, F1, F1 + 'x = 1\n', append=True) == F1 + 'x = 1\n'   # one extends the other: the longer
    edit1, edit2 = TAIL.replace('pass', 'return 1'), TAIL.replace('pass', 'return 2')
    assert merge3(TAIL, edit1, edit2, append=True) is None          # a changed base line still conflicts
    assert merge_patches({'f.py': TAIL}, [{'f.py': F1}, {'f.py': F2}], append=True) == {'f.py': F1 + F2[len(TAIL):]}
    assert merge_patches({'f.py': TAIL}, [{'f.py': F1}, {'f.py': F2}]) is None


def test_loose_bundle_of_appended_patches():
    r1, r2 = TAIL.replace('pass', 'return 1'), TAIL.replace('pass', 'return 2')    # one base line, two edits

    def run(cls):
        eng = Lineage()
        eng.s.hooks = SimpleNamespace(candidate_files=lambda overlay: {'f.py': TAIL, **(overlay or {})})
        hls = cls(FakeProposer([h('lm', {}, patch={'f.py': F1}), h('clip', {}, patch={'f.py': F2}),
                                h('ret1', {}, patch={'f.py': r1}), h('ret2', {}, patch={'f.py': r2})]), explore=False)
        for i, (n, g) in enumerate((('lm', 0.0013), ('clip', 0.0007), ('ret1', 0.0006), ('ret2', 0.0005))):
            p = next(p for p in hls.propose(eng, 16) if p.mechanism == n)
            hls.observe(eng, p, Outcome(f'c{7 + i}', 'screened', True, 1.0, g, 'full', 0.1, [], []))
        b = next(p for p in hls.propose(eng, 16) if p.origin == 'hls-bundle')
        return hls, b
    old, b = run(HLS)
    assert old.members[b.hypothesis] == ['c1|lm', 'c1|ret1']      # M7: the second append left out, silently
    assert not any('why' in old.ledger[k] for k in ('c1|clip', 'c1|ret2'))
    new, b = run(HLS2)
    assert new.members[b.hypothesis] == ['c1|lm', 'c1|clip', 'c1|ret1']
    assert b.patch == {'f.py': (F1 + F2[len(TAIL):]).replace('def _install():\n    pass', 'def _install():\n    return 1')}
    assert new.ledger['c1|ret2']['why'].startswith('patch conflict') and 'why' not in new.ledger['c1|clip']


# ---------------------------------------------------------------- C6 (off in hls2)
def test_cover_bundle_only_when_cheap_moves_outnumber_the_screens():
    exp = judged(*((k, 1, '+', 0.9) for k in 'abcde'))
    s = SimSession({k: 0 for k in 'abcde'}, knobs={k: [0, 1] for k in 'abcde'}, screens=3)
    eng = SimEngine(s, None, sigma_runs=[0.005])
    hls = HLS2(FakeProposer([]), cover_bundle=True, experience=exp)
    props = {p.hypothesis: p for p in hls.propose(eng, 16)}
    cover = props['bundle|cover|c0']
    assert len(cover.change) == 4 and cover.fidelity == ('probe', 'full') and cover.origin == 'hls-cover'
    assert cover.p_feasible == pytest.approx(0.9)
    s.budget.screens = 10                                           # enough runs left to test them one by one
    again = HLS2(FakeProposer([]), cover_bundle=True, experience=exp)
    assert not [p for p in again.propose(SimEngine(s, None, sigma_runs=[0.005]), 16) if p.origin == 'hls-cover']
    assert not [p for p in HLS2(FakeProposer([]), experience=exp).propose(eng, 16) if p.origin == 'hls-cover']


# ---------------------------------------------------------------- registration
def test_hls2_is_a_strategy_that_calls_the_model():
    assert 'hls2' in STRATEGIES and 'hls2' in LLM_STRATEGIES and resolve('hls2', has_llm=True) == 'hls2'
    assert resolve('auto', has_llm=True, knobs={'a': {'type': 'int', 'choices': [1, 2, 4]}}) == 'hls'   # auto unchanged
    s = HLS2(FakeProposer([]))
    assert s.name == 'hls2' and isinstance(s, HLS) and s.cover_bundle is False and s.prior_sigma == 0.03
    assert HLS(FakeProposer([]))._switches() == {}                  # hls: every switch off


def test_cli_help_names_hls2():
    from infrarsi.cli.main import build
    run = next(a for a in build()._subparsers._group_actions[0].choices['run']._actions if '--strategy' in a.option_strings)
    assert 'hls2' in run.help


def test_engine_strategy_builds_hls2_with_the_states_prompt(monkeypatch):
    import infrarsi.search as search
    from infrarsi.search import engine as engine_mod
    from infrarsi.search import strategies
    built = {}

    class Stop(Exception):
        pass

    def fake_run(self, strategy, **kw):
        built['strategy'] = strategy
        raise Stop
    monkeypatch.setattr(engine_mod.Engine, 'run', fake_run)
    from test_search import TOGGLE, setup
    sim, s, _ = setup(TOGGLE)
    for name, cls, states in (('hls2', strategies.HLS2, True), ('hls', strategies.HLS, False)):
        ctx = search.Context(task=s.task, budget=s.budget, start_config=s.candidates['c0'].config, runner=s.runner,
                             judge=s.judge, hooks=s.hooks, run_id='t', out_dir=None, start_receipt={'valid': True},
                             options={'full_steps': 30, 'job_gpu_hours': None, 'capability': 0.9})
        with pytest.raises(Stop):
            search.run(name, ctx)
        st = built['strategy']
        assert type(st) is cls and st.proposer.ledger_states is states and st.proposer.search_calibrates
