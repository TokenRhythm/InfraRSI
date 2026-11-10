"""bcomp v2 (search/bcomp.py; V2_SEARCH_DESIGN_20261101 M1, M2, M4 through search/explore.py's ExploreMixin): the new
switches and their mapping onto the shared module's parts; M2's rules on a W2-like history (a value the guide rates +18 %
at p_runs 0.25 runs out of memory twice - it then needs a saver covering its measured shortfall: the third candidate
without one is never proposed, a completion with too small a saver is dropped from the re-ranked queue, the whole table's
rows holding it are ruled out; three memory failures with a saver held exclude it for the rest of the arm; a pair that
crashed together is never composed again while each value alone stays allowed) and end to end in knob-only mode; the
knob-only schedule of screens 6-10 (completion, contrarian, escape, the ledger's best, then the tail and the draw) and the
code track's path left as v1 had it; M1 in the challenge's slot (a conflict-resolved move, two per arm; v1's challenge
with ex_contrarian off); M4's admissibility while stalled (a near-duplicate refused, the table's rows filtered, the
uniform start exempt, the draw never ending the arm); M2's revise call (after the second failure or one failure against a
declared p_runs >= 0.7; the revised levels merged into the prior's levels, the time model's prior and the ledger's
declared GiB; any failure of the call leaves every estimate unchanged; two per arm, inside the model budget; through an
LLMProposer schema-checked and charged as role 'revise'); the report (which mechanism proposed each screen and why).
No live model call: scripted proposers and the priced scripted backend only."""
import copy
import dataclasses
import json
import math
from types import SimpleNamespace

import pytest

from finalist_helpers import CodeProposer
from infrarsi.search import ABLATIONS, BCOMP_COMPONENTS
from infrarsi.search import bcomp as B
from infrarsi.search.bcomp import BComp
from infrarsi.search.explore import ExploreMixin
from infrarsi.search.finalist_base import CODE_CALLS, FinalistBase, vk
from infrarsi.search.llmopt import CALLS_MAX
from test_bcomp import START, PeakEngine, bc, knobs_of, ph, prior, run

V1 = {'bcomp_ledger', 'bcomp_complete', 'bcomp_challenge', 'bcomp_isolated_blame', 'bcomp_peak_thrash'}
V2 = {'bcomp_feedback', 'bcomp_revise', 'ex_contrarian', 'ex_escape'}
RC_NONE = ('rc', vk('none'))


def mem_w(c):
    """W2 (a Q27U-like job) on a 100 GiB card: recompute none needs 115 GiB (short by 15, a gap of 17 with the card's
    reserve) - no saver of the table frees that much (the fused loss frees 6.5)."""
    return 60.0 + (55.0 if c['rc'] == 'none' else 12.0 if c['rc'] == 'half' else 0.0) + (8.0 if c['mb'] == 2 else 0.0) \
        - (6.5 if c['liger'] else 0.0)


def w2_guide(configs=(), saver_gib=None, **kw):
    """test_bcomp's guide with recompute none rated +18 % but p_runs 0.25 (a nearby environment ran out of memory);
    `saver_gib`: tf32 declared to free that many GiB (it frees nothing)."""
    g = prior(configs=configs, **kw)
    for r in g['levels']:
        if r['knob'] == 'rc' and r['value'] == 'none':
            r.update(gain_pct=18, p_runs=0.25)
    if saver_gib:
        g['levels'] = [r for r in g['levels'] if r['knob'] != 'tf32'] + [
            {'knob': 'tf32', 'value': True, 'gain_pct': 0, 'sd_pct': 1, 'memory': {'direction': 'frees', 'gib': saver_gib}}]
    return g


def knob_only(prop):
    """The knob-only study's proposer: the project's open layers exclude patches (code.mutable empty)."""
    prop.hooks = SimpleNamespace(layers=('L0',))
    return prop


class ReviseProposer(CodeProposer):
    """test_bcomp's scripted model with a revise call: `revise(payload, n)` answers it (or raises); each call counted
    as a model call and its payload kept."""

    def __init__(self, prior=None, answers=None, revise=None):
        super().__init__(prior, answers)
        self.revise, self.revise_seen = revise, []

    def knob_revise(self, engine, base, payload):
        self.calls += 1
        self.revise_seen.append(json.loads(json.dumps(payload, default=str)))
        return self.revise(payload, len(self.revise_seen)) if self.revise else None


def force(st, eng, cfg, plan=('full',)):
    """A knob configuration run as the test says (START plus `cfg`; S6 is not under test)."""
    st.fb_oom_streak = 0
    q = st._knob_q(eng, eng.incumbent(), {**START, **cfg}, kind='t', plan=plan, p=0.9)
    assert q is not None
    return run(st, eng, q)


def two_rc_failures(st, eng):
    """Two memory failures holding recompute none (and no fitted run holds it): W2's history."""
    o1 = force(st, eng, {'rc': 'none', 'wk': 4})
    o2 = force(st, eng, {'rc': 'none', 'mb': 2})
    assert st.runs[o1.cid]['kind'] == st.runs[o2.cid]['kind'] == 'oom'
    return o1.cid, o2.cid


# ---------------------------------------------------------------- the switches, the shared module's parts
def test_registration_the_v2_switches_and_their_shared_parts():
    assert set(BCOMP_COMPONENTS) == V1 | V2 and all(ABLATIONS[k].startswith('bcomp: ') for k in BCOMP_COMPONENTS)
    assert BComp.__mro__[1] is ExploreMixin and issubclass(BComp, FinalistBase)
    assert B.BCOMP_VERSION == 'v2' and (BComp.EX_CONTRA_MAX, BComp.EX_REVISE_MAX) == (2, 2)
    assert (B.REVISE_EVERY, B.REVISE_P_RUNS, B.KNOB_ONLY_ORDER) == (2, 0.7, ('complete', 'contrarian', 'escape', 'ledger'))
    st = BComp(CodeProposer())
    assert all(st._ex_on(x) for x in ('contrarian', 'escape', 'repair', 'attribution')) and all(st.bc_on.values())
    a = BComp(CodeProposer(), off=['ex_contrarian'])
    assert not a._ex_on('contrarian') and a._ex_on('escape') and a._ex_on('attribution')
    b = BComp(CodeProposer(), off=['ex_escape'])
    assert not b._ex_on('escape') and not b._ex_on('repair') and b._ex_on('contrarian')
    c = BComp(CodeProposer(), off=['bcomp_feedback'])
    assert not c._ex_on('attribution') and c._ex_on('escape')


# ---------------------------------------------------------------- M2: W2's rules
def test_w2_two_memory_failures_make_the_value_need_a_saver_the_third_without_one_is_never_proposed():
    """The fused loss declared to free 12 GiB (it frees 6.5): v1 completes the failure with it at P 0.3 - a third
    out-of-memory run -; v2 drops that completion and every other candidate holding recompute none."""
    guide = w2_guide(liger=('frees', 12))
    st, eng = bc(PeakEngine(mem_w), guide)
    st._init(eng)
    c1, c2 = two_rc_failures(st, eng)
    att = st._ex_attribution()
    assert att['needs_saver'][RC_NONE]['failures'] == [c1, c2] and att['needs_saver'][RC_NONE]['gap_gib'] == pytest.approx(17.0)
    p, why = st._pf({**START, 'rc': 'none', 'wk': 0})                       # the third, without a saver: ruled out
    assert p == 0.0 and why.startswith(B.M2_TAG) and 'needs a saver covering 17.0 GiB' in why
    assert st._pf({**START, 'rc': 'none', 'liger': True})[0] == 0.0         # (12 declared GiB do not cover 17)
    assert st._pf({**START, 'mb': 2, 'wk': 4})[0] > 0.0                     # (each other value of the failures: allowed)
    assert all(st._pf(c)[0] == 0.0 for c in st._table(eng) if c['rc'] == 'none')   # no row of the table holds it
    assert st.complete(eng, eng.incumbent()) is None                       # F + the fused loss: dropped from the queue,
    assert {r['run'] for r in st.bc_dropped} == {c1, c2}                    # with the rule that dropped it
    assert all(r['partner'] == 'liger=True' and 'needs a saver' in r['rule'] for r in st.bc_dropped)
    view = st.bcomp_view()['feedback']
    assert view['needs_saver'][0]['value'].startswith('rc=') and view['dropped'] and view['queue'] == []
    v1, e1 = bc(PeakEngine(mem_w), guide, off=['bcomp_feedback'])           # v1 (feedback off): the third is composed
    v1._init(e1)
    two_rc_failures(v1, e1)
    assert v1._pf({**START, 'rc': 'none', 'wk': 0})[0] > 0.0
    q = v1.complete(e1, e1.incumbent())
    assert q is not None and knobs_of(v1, q)['rc'] == 'none' and knobs_of(v1, q)['liger'] is True
    assert run(v1, e1, q).status == 'infeasible'                            # (116.5 GiB on the card: out of memory again)


def test_w2_a_covering_saver_is_allowed_then_three_failures_with_it_exclude_the_value():
    st, eng = bc(PeakEngine(mem_w), w2_guide(saver_gib=20))                # tf32 declared to free 20 GiB (it frees 0)
    st._init(eng)
    two_rc_failures(st, eng)
    assert st._pf({**START, 'rc': 'none', 'wk': 0})[0] == 0.0
    assert st._pf({**START, 'rc': 'none', 'tf32': True})[0] > 0.0           # 20 declared GiB cover the 17: composed
    for extra in ({}, {'wk': 0}, {'wk': 4}):                                # it runs out of memory with the saver, 3x
        force(st, eng, {'rc': 'none', 'tf32': True, **extra})
    att = st._ex_attribution()
    assert RC_NONE in att['excluded'] and len(att['excluded'][RC_NONE]['with_saver']) == 3
    p, why = st._pf({**START, 'rc': 'none', 'tf32': True, 'liger': True})   # excluded for the rest of the arm: even with
    assert p == 0.0 and 'excluded' in why                                   # every saver
    assert st.bcomp_view()['feedback']['excluded'] == ['rc="none"']


def test_a_crash_pair_is_never_composed_again_and_each_value_alone_stays_allowed():
    st, eng = bc(PeakEngine(cap=200.0), prior(configs=()))                  # (the compiler and the fused loss time out
    st._init(eng)                                                           #  together; everything else fits)
    o = force(st, eng, {'comp': True, 'liger': True})
    assert st.runs[o.cid]['kind'] in ('start_cap', 'hang')
    conflicts = st._ex_attribution()['conflicts']
    assert [sorted(c['members']) for c in conflicts] == [[('comp', 'true'), ('liger', 'true')]]
    p, why = st._pf({**START, 'comp': True, 'liger': True, 'wk': 4})
    assert p == 0.0 and why.startswith(B.M2_TAG) and 'failed together' in why
    assert st._pf({**START, 'comp': True, 'wk': 4})[0] > 0.0 and st._pf({**START, 'liger': True, 'tf32': True})[0] > 0.0
    prop = knob_only(CodeProposer(prior(configs=(('both', {'comp': True, 'liger': True, 'mb': 2}, 20),))))
    eng2 = PeakEngine(cap=200.0)
    st2 = BComp(prop, facts={'cap_gib': 200.0}, seed=3)
    st2._init(eng2)
    first = force(st2, eng2, {'comp': True, 'liger': True})                 # the pair crashes on the first screen ...
    eng2.run(st2)                                                           # ... and the arm runs to its end: never
    both = [cid for cid, r in st2.runs.items() if (r.get('cfg') or {}).get('comp') and (r.get('cfg') or {}).get('liger')]
    assert eng2.full_screens() == 10 and both == [first.cid]                 # composed again (not the guide's start either)


def test_w2_end_to_end_in_knob_only_mode_the_value_stops_after_two_memory_failures():
    """A ten-screen knob-only arm: the guide's two starts hold recompute none and run out of memory at their probes; no
    later candidate holds it (no saver covers its gap); screens 6-10 are bcomp's own mechanisms (or the draw), never a
    code ask; every screen says which mechanism proposed it and why."""
    g = w2_guide(configs=(('aggressive', {'rc': 'none', 'mb': 2, 'liger': True}, 30), ('deep', {'rc': 'none', 'tf32': True}, 22)))
    eng = PeakEngine(mem_w)
    st = BComp(knob_only(CodeProposer(g)), facts={'cap_gib': 100.0}, seed=3)
    trace = eng.run(st)
    assert trace.stop_reason.startswith('budget') and eng.full_screens() == 10 and not st._can_code()
    held = [(cid, r['kind']) for cid, r in st.runs.items() if (r.get('cfg') or {}).get('rc') == 'none']
    assert len(held) == 2 and all(k == 'oom' for _, k in held)
    assert st.asks == 0 and not st.fb_asks and all(s['pick'] != 'code ask' for s in st.bc_steps)
    v = st.describe()['bcomp']['bcomp']
    assert v['version'] == 'v2' and v['schedule'].startswith('knob-only')
    assert len(v['screens']) == 10 and all(s.get('by') and s.get('why') for s in v['screens'])
    assert {s['by'] for s in st.bc_screens[5:]} <= {'complete', 'contrarian', 'escape', 'ledger', 'tail', 'draw'}
    assert v['explore']['version'] and json.dumps(st.describe(), default=str)


def test_w2_the_uniform_start_skips_draws_that_break_an_m2_rule():
    """V2B_r1 P1-1, the same arm with seed 0: the two starts run out of memory holding recompute none at their probes; the
    uniform start's first draws hold it without a saver - skipped (recorded on the slot) for the next legal draw, which
    still runs in full, unprobed (S6's exemption kept apart). With bcomp_feedback off the uniform start is that first
    draw and runs out of memory in full (the review's c3)."""
    g = w2_guide(configs=(('aggressive', {'rc': 'none', 'mb': 2, 'liger': True}, 30), ('deep', {'rc': 'none', 'tf32': True}, 22)))
    eng = PeakEngine(mem_w)
    st = BComp(knob_only(CodeProposer(g)), facts={'cap_gib': 100.0}, seed=0)
    eng.run(st)
    held = [(cid, r['kind'], r['fidelity']) for cid, r in st.runs.items() if (r.get('cfg') or {}).get('rc') == 'none']
    assert held == [('c1', 'oom', 'probe'), ('c2', 'oom', 'probe')] and eng.full_screens() == 10
    slot = st.bc_slots['uniform']
    assert slot['skipped'] >= 1 and 'needs a saver covering 17.0 GiB' in slot['skipped_why']
    u = st.bc_by_hid[slot['id']]
    assert u['plan'] == ['full'] and u['candidate'] == 'c3' and u['kind'] == 'ok' and u['change'].get('rc') != 'none'
    assert 'skipped by M2' in u['why']
    eng1 = PeakEngine(mem_w)
    v1 = BComp(knob_only(CodeProposer(g)), facts={'cap_gib': 100.0}, seed=0, off=['bcomp_feedback'])
    eng1.run(v1)
    u1 = v1.bc_by_hid[v1.bc_slots['uniform']['id']]
    assert u1['candidate'] == 'c3' and u1['change'].get('rc') == 'none' and u1['kind'] == 'oom' and u1['plan'] == ['full']


# ---------------------------------------------------------------- the knob-only schedule, the code track's path
def _scripted(st, ret: dict, calls: list):
    """bcomp's mechanisms replaced by recorders returning `ret[name]` (None by default)."""
    def rec(name):
        def f(*a, **k):
            calls.append(name)
            return ret.get(name)
        return f
    st.complete = rec('complete')
    st._bc_contrarian, st._bc_escape, st._bc_ledger_best = rec('contrarian'), rec('escape'), rec('ledger')
    st._tail_step, st._fallback, st._bc_code = rec('tail'), rec('draw'), rec('code')


@pytest.mark.parametrize('first', ['complete', 'contrarian', 'escape', 'ledger', 'tail', 'draw'])
def test_knob_only_screens_6_to_10_take_completion_contrarian_escape_ledger_then_tail_and_draw(first):
    st = BComp(knob_only(CodeProposer(prior())), facts={'cap_gib': 200.0}, seed=3)
    eng = PeakEngine(cap=200.0)
    st._init(eng)
    eng.s.ledger.used['screens'] = 5                                         # (screens 6-10)
    order = ['complete', 'contrarian', 'escape', 'ledger', 'tail', 'draw']
    calls, mark = [], SimpleNamespace(hypothesis=f'{first}|x')
    _scripted(st, {first: mark}, calls)
    assert st._bc_step(eng, 'c0', 5) is mark
    assert calls == order[:order.index(first) + 1] and 'code' not in calls


def test_the_code_track_path_on_screens_6_to_10_is_v1s():
    """With the code track open the code screens ask the code track first and fall back to the tail and the draw, as v1
    did - bcomp's knob mechanisms are not asked there; a real arm's screens 6-10 are code screens."""
    st, eng = bc(PeakEngine(cap=200.0), prior())
    st._init(eng)
    eng.s.ledger.used['screens'] = 5
    calls = []
    _scripted(st, {}, calls)
    st._bc_step(eng, 'c0', 5)
    assert st._can_code() and calls == ['code', 'tail', 'draw']
    calls.clear()
    patch = SimpleNamespace(hypothesis='patch|p1')
    _scripted(st, {'code': patch}, calls)
    assert st._bc_step(eng, 'c0', 5) is patch and calls == ['code']
    st2, eng2 = bc(PeakEngine(cap=200.0), prior(), answers=lambda ctx, n: [ph(f'k{n}')])
    eng2.run(st2)
    late = [s for s in st2.bc_steps if s.get('screens_done', 0) >= 5]
    assert late and {s['pick'] for s in late} <= {'code', 'code ask', 'tail', 'draw'}
    assert any(s['pick'] == 'code' for s in late) and st2.bcomp_view()['schedule'].startswith('code track open')


# ---------------------------------------------------------------- M1 in the challenge's slot
def _contrarian_case(**kw):
    """The compiler declared -1 % with a 10 % spread (its sign in doubt) and to conflict with the fused loss, which the
    incumbent holds (a Gemma-like W1: the move must send the fused loss back)."""
    g = prior(configs=(), conflicts=[(('comp', True), ('liger', True))])
    for r in g['levels']:
        if r['knob'] == 'comp':
            r.update(gain_pct=-1, sd_pct=10)
    st, eng = bc(PeakEngine(cap=200.0), g, **kw)
    st._init(eng)
    force(st, eng, {'liger': True})                                         # +5 %: the incumbent holds the fused loss
    assert eng.s.candidates[eng.incumbent()].config['liger'] is True
    return st, eng


def test_m1_the_contrarian_move_takes_the_challenges_slot_conflict_resolved_two_per_arm():
    st, eng = _contrarian_case()
    st.bc_config.append('complete')                                         # (screen 5: the other kind after a completion)
    q = st._bc_config(eng, eng.incumbent())
    assert q.hypothesis.startswith('contrarian|') and getattr(q, 'why', None) == 'contrarian'
    cfg = knobs_of(st, q)
    assert cfg['comp'] is True and cfg['liger'] is False                     # the conflict resolved: the fused loss back
    rec = st.bc_contras[-1]
    assert rec['value'].startswith('comp=') and rec['flipped'] and 'contrarian' in rec['why']
    assert not st.bc_challenges and st.bc_by_hid[q.hypothesis]['what'] == 'contrarian'
    run(st, eng, q)
    n = 1
    while n < 3:
        q = st._bc_contrarian(eng, eng.incumbent())
        if q is None:
            break
        run(st, eng, q)
        n += 1
    assert len(st._bc_done(st.bc_contras)) <= BComp.EX_CONTRA_MAX and st._bc_contrarian(eng, eng.incumbent()) is None
    v1, e1 = _contrarian_case(off=['ex_contrarian'])                         # off: v1's challenge in the slot
    v1.bc_config.append('complete')
    q = v1._bc_config(e1, e1.incumbent())
    assert q.hypothesis.startswith('challenge|') and not v1.bc_contras


# ---------------------------------------------------------------- M4 while stalled
def _stalled(guide=None):
    """Stalled: one full screen that improved on the reference (the fused loss, +5 %: the plateau counts from the first
    improvement, the shared module's rule), then two full screens without an improvement - radius 2."""
    st, eng = bc(PeakEngine(cap=200.0), guide or prior(configs=()))
    st._init(eng)
    best = force(st, eng, {'liger': True})
    force(st, eng, {'liger': True, 'wk': 0})
    force(st, eng, {'liger': True, 'wk': 0, 'tf32': True})
    assert eng.incumbent() == best.cid and st._ex_stalled() and st._bc_stalled() and st._ex_radius() == 2
    return st, eng


def test_m4_while_stalled_every_knob_candidate_is_admissible_the_uniform_start_exempt():
    st, eng = _stalled()
    inc = eng.incumbent()
    near = {**START, 'liger': True, 'wk': 4}                                 # a near-duplicate of the incumbent (small knob)
    q = st._bc_gated(st._bc_whole, eng, inc, near, kind='guide', why='a test start')
    assert q is None and st.bc_refusals == 1 and st.bc_refused[-1]['kind'] == 'guide'
    assert st.bc_refused[-1]['why'].startswith(('trust region', 'tabu'))
    assert st.ex_counts['refused_trust'] + st.ex_counts['refused_tabu'] >= 1   # (counted in the shared module's view too)
    assert st._bc_whole(eng, inc, near, kind='guide', why='not stalled-gated') is not None   # (outside the gate: as v1)
    rows = st._bc_gated(st._candidates, eng, inc)
    ctx = st._ex_context(st._knob_cfg(eng, inc), eng)
    assert rows and all(st._ex_adm(c, ctx)[0] for c, _ in rows) and len(rows) < len(st._candidates(eng, inc))
    u = st._bc_gated(st.uniform_start_q, eng, inc, {**START, 'liger': True, 'tf32': True, 'wk': 4})
    assert u is not None and u.hypothesis.startswith('uniform|')            # the control start: never held to it
    st.bc_on['ex_escape'] = False                                            # off: no gate at all
    assert not st._bc_stalled() and st._bc_gated(st._bc_whole, eng, inc, {**START, 'liger': True, 'wk': 0, 'comp': True},
                                                 kind='guide', why='x') is not None


def test_m4_a_start_the_stall_gate_refuses_stays_open_and_runs_once_the_stall_ends():
    st, eng = _stalled(prior(configs=(('deep', {'rc': 'none', 'liger': True}, 22),)))   # (one big knob off the incumbent:
    st.bc_slots['uniform']['state'] = 'none'                                             #  radius 2 refuses it)
    q = st._bc_gated(st._bc_knob, eng, eng.incumbent())
    assert q is None or not q.hypothesis.startswith('experience|')
    assert st.bc_slots['experience']['state'] == 'open' and st.bc_refused[0]['kind'] == 'experience'
    assert 'trust region' in st.bc_refused[0]['why']
    st.ex_bad = 0                                                            # (an improvement: the stall is over)
    q = st._bc_gated(st._bc_knob, eng, eng.incumbent())
    assert q.hypothesis.startswith('experience|') and knobs_of(st, q)['rc'] == 'none'


def test_m4_no_admissible_row_never_ends_the_arm_the_draw_runs_as_v1():
    st, eng = _stalled()
    st._bc_admissible = lambda engine, inc, cfg, row=False: (False, 'test: nothing admissible')
    q = st._bc_rest(eng, eng.incumbent(), 8)
    assert q is not None and q.hypothesis.startswith('draw|')
    assert 'no row of the table admissible' in st.bc_steps[-1]['why'] and 'M4 relaxed' in st.bc_steps[-1]['why']


def test_m4_the_stalled_draw_never_runs_a_doubtful_admissible_row_in_full_while_a_safe_row_remains():
    """V2B_r1 P1-2: stalled, the low-fidelity allowance spent (no probe left: the draw's floor is the ladder's
    probe_below), every admissible row at P 0.15 while rows M4 refuses still fit at 0.99 - the draw (which keeps rows
    below its floor only when none reaches it) must not run an admissible P 0.15 row in full: M4 is relaxed and a row at
    the floor or above runs. With one admissible row at the floor, the draw takes that one, M4 kept."""
    st, eng = _stalled()
    inc = eng.incumbent()
    st.ladder.lowfi_share = 0.0
    st.tail.done = True                                                      # (the draw is under test, not the tail)
    assert not st.ladder.affordable(eng, 'probe') and st._bc_draw_floor(eng) == st.ladder.probe_below >= 0.5
    real = st._pf
    admissible = lambda cfg: st._bc_admissible(eng, inc, cfg, row=True)[0]   # noqa: E731
    st._pf = lambda cfg, ids=(): (0.15, 'planted') if admissible(cfg) else real(cfg, ids)
    assert any(admissible(c) for c, _ in st._candidates(eng, inc))          # (48 such rows in the review's case)
    q = st._bc_rest(eng, inc, 7)
    assert q.hypothesis.startswith('draw|') and tuple(q.fidelity) == ('full',)
    assert q.p_feasible >= st.ladder.probe_below and not admissible(st._cfg_of(q))   # a safe row, M4 relaxed
    assert 'M4 relaxed' in st.bc_steps[-1]['why']
    target = next(c for c, _ in st._candidates(eng, inc) if admissible(c))
    key = st._ckey(target, ())
    st._pf = lambda cfg, ids=(): (0.99, 'planted') if st._ckey(cfg, ()) == key else         (0.15, 'planted') if admissible(cfg) else real(cfg, ids)
    q = st._bc_rest(eng, inc, 7)
    assert q.hypothesis.startswith('draw|') and st._ckey(st._cfg_of(q), ()) == key and tuple(q.fidelity) == ('full',)
    assert 'over the admissible rows' in st.bc_steps[-1]['why']


# ---------------------------------------------------------------- M2: the revise call
def _revise_answer(payload, n):
    return {'levels': [{'knob': 'rc', 'value': 'none', 'gain_pct': 6, 'sd_pct': 4, 'p_runs': 0.1,
                        'memory': {'direction': 'spends', 'gib': 45}, 'why': 'two runs out of memory'},
                       {'knob': 'tf32', 'value': True, 'gain_pct': 50, 'why': 'not asked: ignored'},
                       {'knob': 'nope', 'value': 1}, 'junk'], 'notes': ''}


def _snapshot(st):
    led = {b['name']: (b['mu'], b['sd'], b.get('declared')) for b in st.mledger.basis}
    return copy.deepcopy(st.prior.levels), list(st.surr.mu0), list(st.surr.sd0), led, dict(st.declared)


def test_revise_after_the_second_failure_merges_the_revised_levels_into_the_guide():
    prop = ReviseProposer(w2_guide(), revise=_revise_answer)
    eng = PeakEngine(mem_w)
    st = BComp(prop, facts={'cap_gib': 100.0}, seed=3)
    st._init(eng)
    force(st, eng, {'rc': 'none', 'wk': 4})
    assert st.bc_revise_due is None and st.bc_fails == 1
    force(st, eng, {'rc': 'none', 'mb': 2})
    assert st.bc_revise_due and 'since the last revise call' in st.bc_revise_due
    calls = prop.calls
    st._bc_revise(eng)
    assert prop.calls == calls + 1 and st.bc_revise_calls == 1 and st.bc_revise_due is None
    sent = prop.revise_seen[0]
    assert sent['revise'] == [{'knob': 'rc', 'value': 'none'}, {'knob': 'mb', 'value': 2}, {'knob': 'wk', 'value': 4}]
    assert {r['knob'] for r in sent['guide']} == {'rc', 'mb', 'wk'} and set(sent['knob_table']) == {'rc', 'mb', 'wk'}
    assert sum('out of memory' in r['outcome'] for r in sent['measured']) == 2
    e = st.prior.levels[RC_NONE]
    assert e['mu'] == pytest.approx(math.log1p(0.06)) and e['sd'] == pytest.approx(math.log1p(0.04))
    assert (e['p_runs'], e['mem_dir'], e['mem_gib']) == (0.1, 'spends', 45) and e['why'].endswith('[revised]')
    i = st.surr.names.index(RC_NONE)                                         # the time model's prior (wcs's rule)
    assert st.surr.mu0[i] == pytest.approx(math.log1p(0.06)) and st.surr.sd0[i] == pytest.approx(math.log1p(0.04))
    b = next(b for b in st.mledger.basis if b['name'] == f'rc={vk("none")}')    # the ledger's declared GiB
    assert (b['mu'], b['sd'], b['declared']) == (45.0, 22.5, 'spends')
    assert st.prior.levels[('tf32', 'true')]['mu'] == pytest.approx(0.0)    # (not asked: unchanged)
    rec = st.bc_revisions[-1]
    assert rec['outcome'] == '1 value(s) revised' and rec['changes'][0]['before']['gain_pct'] == pytest.approx(18.0)
    assert rec['changes'][0]['after']['gain_pct'] == pytest.approx(6.0) and rec['calls'] == [calls, calls + 1]
    assert st.bcomp_view()['revisions']['calls'] == 1


def test_revise_after_one_failure_against_a_declared_p_runs_of_07():
    g = w2_guide()
    for r in g['levels']:
        if r['knob'] == 'mb':
            r['p_runs'] = 0.9
    st, eng = bc(PeakEngine(lambda c: 60.0 + (45.0 if c['mb'] == 2 else 0.0)), g)
    st._init(eng)
    force(st, eng, {'mb': 2})
    assert st.bc_fails == 1 and 'p_runs >= 0.7' in st.bc_revise_due and ('mb', '2') in st.bc_contradicted


@pytest.mark.parametrize('answer', ['raise', None, {'levels': 'x'}, {'levels': [{'knob': 'rc', 'value': 'off', 'gain_pct': 1}]}])
def test_a_failed_revise_call_leaves_every_estimate_unchanged(answer):
    def revise(payload, n):
        if answer == 'raise':
            raise RuntimeError('the backend fell over')
        return answer
    prop = ReviseProposer(w2_guide(), revise=revise)
    eng = PeakEngine(mem_w)
    st = BComp(prop, facts={'cap_gib': 100.0}, seed=3)
    st._init(eng)
    two_rc_failures(st, eng)
    before = _snapshot(st)
    st._bc_revise(eng)
    assert _snapshot(st) == before and st.bc_revise_calls == 1 and len(prop.revise_seen) == 1
    out = st.bc_revisions[-1]['outcome']
    assert out.startswith('failed') if answer == 'raise' else ('unchanged' in out or 'none of the values' in out)


def test_a_merge_that_fails_half_way_puts_every_estimate_back():
    def revise(payload, n):
        return {'levels': [{'knob': 'rc', 'value': 'none', 'gain_pct': 6, 'memory': {'direction': 'spends', 'gib': 45}},
                           {'knob': 'mb', 'value': 2, 'gain_pct': 1, 'memory': {'direction': 'spends', 'gib': 30}}]}
    prop = ReviseProposer(w2_guide(), revise=revise)
    eng = PeakEngine(mem_w)
    st = BComp(prop, facts={'cap_gib': 100.0}, seed=3)
    st._init(eng)
    two_rc_failures(st, eng)
    before = _snapshot(st)
    real, n = st._bc_ledger_prior, []

    def second_fails(k, key, e):                                             # the first value merged, the second's
        n.append(k)                                                          # ledger update raises
        if len(n) == 2:
            raise ValueError('planted')
        real(k, key, e)
    st._bc_ledger_prior = second_fails
    st._bc_revise(eng)
    assert len(n) == 2 and _snapshot(st) == before
    assert 'could not be merged' in st.bc_revisions[-1]['outcome'] and st.bc_revise_calls == 1


def test_at_most_two_revise_calls_per_arm_inside_the_model_budget():
    prop = ReviseProposer(w2_guide(saver_gib=20), revise=_revise_answer)
    eng = PeakEngine(mem_w)
    st = BComp(prop, facts={'cap_gib': 100.0}, seed=3)
    st._init(eng)
    for extra in ({'wk': 4}, {'mb': 2}, {'tf32': True}, {'tf32': True, 'wk': 0}, {'tf32': True, 'wk': 4}, {'mb': 2, 'wk': 0}):
        force(st, eng, {'rc': 'none', **extra})
        st._bc_revise(eng)
    assert st.bc_fails == 6 and len(prop.revise_seen) == 2 == st.bc_revise_calls == BComp.EX_REVISE_MAX
    st2 = BComp(ReviseProposer(w2_guide(), revise=_revise_answer), facts={'cap_gib': 100.0}, seed=3)
    eng2 = PeakEngine(mem_w)
    st2._init(eng2)
    two_rc_failures(st2, eng2)
    st2.calls_max = st2._llm_used()                                          # the budget's calls spent
    st2._bc_revise(eng2)
    assert not st2.proposer.revise_seen and st2.bc_revise_calls == 0 and st2.bc_revisions[-1]['outcome'].startswith('not made')
    st3 = BComp(ReviseProposer(w2_guide(), revise=_revise_answer), facts={'cap_gib': 100.0}, seed=3)
    eng3 = PeakEngine(mem_w)                                                 # the code track open and only its CODE_CALLS
    st3._init(eng3)                                                          # left: S1 keeps them for it
    two_rc_failures(st3, eng3)
    st3.calls_max = st3._llm_used() + CODE_CALLS
    assert st3._fb_code_can_use_calls(eng3)
    st3._bc_revise(eng3)
    assert not st3.proposer.revise_seen and st3.bc_revisions[-1]['outcome'].startswith('not made')


def _bcomp_toy(script):
    """bcomp with an LLMProposer over a priced scripted backend (test_bcomp's bcomp_llm_toy, not run): (strategy,
    session, backend, calls seen as (role, payload), engine)."""
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.agent.budget import VirtualClock
    from infrarsi.search import toybench as tb
    from infrarsi.search.proposers import LLMProposer
    from infrarsi.search.toytrain import mem_tasks
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
    s = tb.SearchSession(task=env, budget=env.search_budget(), runner=runner, judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']),
                         run_id='toy-v2', arm='b', start_config=ref_cfg, start_receipt=ref, reference_receipt=ref,
                         hooks=hooks, clock=clock, stop_after_same_failures=10 ** 6)
    eng = tb.Engine(s, full_steps=tb.FULL, short_steps=tb.SHORT, probe_steps=tb.PROBE, job_gpu_hours=None, multi_fidelity=True)
    seen = []

    def wrapped(role, system, prompt):
        payload = json.loads(prompt)
        seen.append((role, payload))
        return script(role, system, payload)

    class Priced(ScriptedBackend):
        def __init__(self):
            super().__init__(wrapped, tokens_per_call=2000)
            self.max_retries, self.kw = 0, []

        def complete(self, system, prompt, **kwargs):
            self.kw.append(dict(kwargs))
            res = super().complete(system, prompt, **{k: v for k, v in kwargs.items() if k in ('schema', 'call_dir', 'role',
                                                                                               'expect_json')})
            return dataclasses.replace(res, llm_usd=0.5, attempts=1)
    be = Priced()
    prop = LLMProposer(be, hooks, search_calibrates=True, features=BComp.proposer_features(()))
    st = BComp(prop, facts=sim.memory_facts, seed=1, env=tb.toy_env(sim))
    return st, s, be, seen, eng


def test_the_revise_call_through_an_llm_proposer_is_schema_checked_charged_and_merged():
    def script(role, system, payload):
        if role == 'prior':
            return {'levels': [{'knob': 'micro_batch', 'value': 2, 'gain_pct': 8, 'sd_pct': 3, 'p_runs': 0.9,
                                'memory': {'direction': 'spends', 'gib': 10}}], 'interactions': [], 'configs': [], 'notes': ''}
        if role == 'revise':
            return {'levels': [{'knob': 'micro_batch', 'value': 2, 'gain_pct': 3, 'sd_pct': 2, 'p_runs': 0.2,
                                'memory': {'direction': 'spends', 'gib': 30}, 'why': 'it ran out of memory'}], 'notes': ''}
        return {'hypotheses': []}
    st, s, be, seen, eng = _bcomp_toy(script)
    st._init(eng)
    v = ('micro_batch', '2')
    assert st.prior.levels[v]['p_runs'] == 0.9
    st.runs['c9'] = {'cfg': {**st.ref, 'micro_batch': 2}, 'kind': 'oom', 'y': None, 'status': 'infeasible',
                     'fidelity': 'probe', 'origin': 'bcomp'}                # (a failure planted: what the call reads)
    st.bc_revise_due = 'a planted failure'
    used = st._llm_used()
    st._bc_revise(eng)
    assert seen[-1][0] == 'revise' and be.kw[-1]['schema'] is B.REVISE_SCHEMA and be.kw[-1]['role'] == 'revise'
    sent = seen[-1][1]
    assert sent['revise'] == [{'knob': 'micro_batch', 'value': 2}] and 'task' in sent and 'guide' in sent
    assert any(r['role'] == 'revise' for r in s.llm_calls) and st._llm_used() == used + 1 <= CALLS_MAX
    e = st.prior.levels[v]
    assert (e['p_runs'], e['mem_gib']) == (0.2, 30) and e['mu'] == pytest.approx(math.log1p(0.03))
    assert st.bc_revisions[-1]['outcome'] == '1 value(s) revised'

    def bad(role, system, payload):                                          # an answer the schema refuses (no levels):
        return script(role, system, payload) if role != 'revise' else {'notes': 'none'}   # nothing changes
    st2, s2, be2, seen2, eng2 = _bcomp_toy(bad)
    st2._init(eng2)
    st2.runs['c9'] = dict(st.runs['c9'])
    st2.bc_revise_due = 'a planted failure'
    before = copy.deepcopy(st2.prior.levels)
    st2._bc_revise(eng2)
    assert st2.prior.levels == before and 'unchanged' in st2.bc_revisions[-1]['outcome']
