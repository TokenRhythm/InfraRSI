"""explore (search/explore.py; V2_SEARCH_DESIGN_20261101, "Shared module API"): the mixin bcomp v2 uses, on a
test finalist with each one's bases (`ExploreMixin, FinalistBase` and `ExploreMixin, FinalistBase, OPRO`) over test_wcs's
knob table and a simulated engine with a card. The design's evidence cases with synthetic data: W1 - a value the prior
rates +2 % (sd 10 %, p_runs 0.75) whose only run hung with a partner the incumbent holds: the contrarian move sends that
partner back (a memory saver added when the move spends memory and the ledger's P is below 0.5), at most two per arm;
W2 - a value out of memory twice and in no fitted run needs a saver covering its shortfall, and after three failures that
held savers it is excluded; a crash pair is a conflict no later candidate holds whole (each value alone allowed); W3 -
never repeat an exact failure (the same big knobs and the cause unchanged), a repair adds the change that removes the
cause; W4 - two full screens without an improvement stall the search: radius 2 (then 3), the admissible candidates
differ from the incumbent in at least that many big knobs, near-duplicates of runs are refused, the escape candidate
honours both, a warm restart from the best run of another class after four stalled screens, momentum after a one-knob
improvement. Also: a move the strategy runs its own way is matched and counted, the parts switch off, the view, no knob,
model or library name in the module."""
import dataclasses
import json
import math
import re
from pathlib import Path

import pytest

from finalist_helpers import CodeProposer
from hls2_helpers import BASE_TEXT, PATCH_FILE, SimEngine, SimSession
from infrarsi.search import explore as X
from infrarsi.search.explore import EX_PARTS, ExploreMixin
from infrarsi.search.finalist_base import FinalistBase
from infrarsi.search.opro import OPRO
from infrarsi.search.wcs import vk
from test_wcs import KNOBS, START

SRC = Path(__file__).resolve().parents[3] / 'infrarsi' / 'search' / 'explore.py'
CAP = 100.0
API = ('_ex_init', '_ex_observe', '_ex_stalled', '_ex_radius', '_ex_admissible', '_ex_attribution', '_ex_contrarian',
       '_ex_escape', '_ex_repair_candidates', '_ex_view', '_ex_on')


class ExB(ExploreMixin, FinalistBase):
    """bcomp v2's bases: the mixin's state from `_ex_init`, its observation after the base's."""
    name = 'exb'

    def observe(self, engine, p, out):
        cls = super().observe(engine, p, out)
        self._ex_observe(engine, p, out)
        return cls


class ExH(ExploreMixin, FinalistBase, OPRO):
    """the OPRO mixin bases (opro behind the shared base)."""
    name = 'exh'

    def observe(self, engine, p, out):
        cls = super().observe(engine, p, out)
        self._ex_observe(engine, p, out)
        return cls


def table_gain(table):
    """A log gain over the reference: the sum of `table`'s values a configuration holds."""
    def g(cfg):
        return sum(table.get((k, cfg.get(k)), 0.0) for k in START if cfg.get(k) != START[k])
    return g


def mem_of(table, base=60.0):
    """GiB per GPU: the reference's `base` plus `table`'s values a configuration holds."""
    def m(cfg):
        return base + sum(table.get((k, cfg.get(k)), 0.0) for k in START if cfg.get(k) != START[k])
    return m


class MemEngine(SimEngine):
    """test_wcs's simulated engine with a `cap` GiB card: `gain(cfg)` the log gain over the reference, `mem(cfg)` its GiB
    per GPU (above the card: out of memory with PyTorch's message, its shortfall the need less the card; a full run that
    fits records its peak); a configuration holding one of `hang` (knob -> value) times out at its start."""

    def __init__(self, gain, mem, *, cap=CAP, hang=(), screens=10):
        s = SimSession(START, knobs=KNOBS, screens=screens, base_files={PATCH_FILE: BASE_TEXT})
        self.mem_of, self.cap = mem, cap

        def effect(base, cfg, overlay):
            if any(all(cfg.get(k) == v for k, v in h.items()) for h in hang):
                return None, 'timeout'
            if mem(cfg) > cap:
                return None, 'oom'
            return gain(cfg) - gain(base.config), None
        super().__init__(s, effect, sigma_runs=[0.01 / math.sqrt(2)])
        s.candidates['c0'].receipt['evidence'] = {'memory_peak_measured_gib': float(mem(START))}

    def evaluate(self, p):
        o = super().evaluate(p)
        c = self.s.candidates.get(o.cid) if o.cid else None
        if c is None:
            return o
        need = self.mem_of(c.config)
        if need > self.cap and o.status in ('infeasible', 'invalid'):
            tried = need - self.cap + 1.5
            msg = (f'completes: exit code 1, 0/30 updates; the log: torch.OutOfMemoryError: CUDA out of memory. Tried to '
                   f'allocate {tried:.2f} GiB. GPU 0 has a total capacity of {self.cap:.2f} GiB of which 1.50 GiB is free. '
                   f'Including non-PyTorch memory, this process has {self.cap - 1.5:.2f} GiB memory in use.')
            return dataclasses.replace(o, reasons=[msg])
        if o.status == 'screened':
            c.verdict['peak_reserved_gib'] = round(need, 3)
        return o


def lv(knob, value, gain, sd, mem=None, p_runs=None):
    row = {'knob': knob, 'value': value, 'gain_pct': gain, 'sd_pct': sd,
           'memory': {'direction': mem[0], 'gib': mem[1]} if mem else {'direction': 'neutral', 'gib': 0}}
    if p_runs is not None:
        row['p_runs'] = p_runs
    return row


def prior(*levels, conflicts=()):
    """A prior over test_wcs's KNOBS (wcs's answer format): the levels given, the declared conflicts, no configurations."""
    return {'levels': list(levels), 'configs': [], 'notes': 'a test prior',
            'interactions': [{'members': [{'knob': a, 'value': va}, {'knob': b, 'value': vb}], 'kind': 'conflicts'}
                             for (a, va), (b, vb) in conflicts]}


def base_levels(comp=(2, 10), comp_mem=None, comp_p=0.75, tf32=(0, 1), tf32_mem=None, wk4=(0.5, 1)):
    """The big knobs: recompute-like rc (+8 / +22 %, spenders), micro-batch-like mb (+8 %, a spender), a fused-loss-like
    liger (+5 %, frees 5); the compiler-like comp as given; small tf32 and wk."""
    return [lv('rc', 'half', 8, 4, ('spends', 12)), lv('rc', 'none', 22, 8, ('spends', 30)),
            lv('mb', 2, 8, 3, ('spends', 15)), lv('liger', True, 5, 3, ('frees', 5)),
            lv('comp', True, comp[0], comp[1], comp_mem, p_runs=comp_p), lv('tf32', True, tf32[0], tf32[1], tf32_mem),
            lv('wk', 0, -1, 1), lv('wk', 4, wk4[0], wk4[1])]


def setup(cls=ExB, guide=None, *, gain=None, mem=None, hang=(), screens=10, check=None):
    eng = MemEngine(gain or table_gain({}), mem or mem_of({}), hang=hang, screens=screens)
    if check is not None:
        eng.s.check_config = check                                          # (the session's legal-table check)
    st = cls(CodeProposer(guide), facts={'cap_gib': CAP}, seed=3)
    st._init(eng)
    st._ex_init(eng)
    return st, eng


def run(st, eng, q):
    o = eng.evaluate(q)
    st.observe(eng, q, o)
    return o


def put(st, eng, change, plan=('full',), p=0.99):
    """A whole configuration (the reference's values unless `change`) through the base's proposal, run and observed."""
    q = st._knob_q(eng, eng.incumbent(), {**START, **change}, kind='t', plan=plan, p=p)
    assert q is not None, change
    return run(st, eng, q)


def knobs_of(st, q):
    return {k: q.change[k] for k in st.knobs}


def cfg(**change):
    return {**START, **change}


def big_off(st, a, b):
    return sum(1 for k in st._ex_big_knobs() if vk(a.get(k)) != vk(b.get(k)))


# ---------------------------------------------------------------- the API, the constants, both bases
def test_the_api_the_constants_and_both_bases():
    assert (X.EX_CONTRA_MAX, X.EX_STALL, X.EX_RADIUS_MAX, X.EX_REPAIR_MAX, X.EX_REVISE_MAX) == (2, 2, 3, 2, 2)
    assert (ExploreMixin.EX_CONTRA_MAX, ExploreMixin.EX_STALL, ExploreMixin.EX_RADIUS_MAX, ExploreMixin.EX_REPAIR_MAX,
            ExploreMixin.EX_REVISE_MAX) == (2, 2, 3, 2, 2)
    assert set(EX_PARTS) == {'contrarian', 'escape', 'repair', 'attribution'}
    assert all(callable(getattr(ExploreMixin, m)) for m in API)
    assert '__init__' not in ExploreMixin.__dict__ and 'observe' not in ExploreMixin.__dict__
    assert ExH.__mro__[1:4] == (ExploreMixin, FinalistBase, OPRO)
    for cls in (ExB, ExH):
        st, eng = setup(cls, prior(*base_levels()))
        assert all(st._ex_on(x) for x in EX_PARTS) and not st._ex_stalled() and st._ex_radius() == 1
        assert st._ex_attribution() == {'needs_saver': {}, 'excluded': {}, 'conflicts': [], 'failures': []}
        assert st._ex_big_knobs(eng) == {'rc', 'mb', 'liger'}          # (|prior gain| >= max(2 %, 2 sigma_pair))
        json.dumps(st._ex_view())
        st.ex_off = {'escape'}
        assert not st._ex_on('escape') and st._ex_on('contrarian')


# ---------------------------------------------------------------- W1 / M1: the contrarian move
def _w1(cls, comp_mem=None, tf32_mem=None):
    """The incumbent holds the fused-loss-like value (x1.05); the compiler-like value (+2 %, sd 10 %, p_runs 0.75) ran
    once, with it, and hung at its start."""
    guide = prior(*base_levels(comp_mem=comp_mem, tf32_mem=tf32_mem, wk4=(1, 6)), lv('tf32', True, -3, 8, tf32_mem))
    st, eng = setup(cls, guide, gain=table_gain({('liger', True): 0.05, ('comp', True): 0.04}),
                    mem=mem_of({('liger', True): -5.0}), hang=({'comp': True, 'liger': True},))
    assert put(st, eng, {'liger': True}).status == 'screened' and eng.incumbent() != 'c0'
    o = put(st, eng, {'liger': True, 'comp': True}, plan=('probe', 'full'))   # (at its probe: no full screen spent)
    assert o.status == 'infeasible' and st.runs[o.cid]['kind'] == 'start_cap'
    return st, eng


@pytest.mark.parametrize('cls', [ExB, ExH])
def test_w1_the_contrarian_move_sends_the_conflicting_partner_back(cls):
    st, eng = _w1(cls)
    att = st._ex_attribution()
    assert [c['members'] for c in att['conflicts']] == [[('comp', 'true'), ('liger', 'true')]]
    inc = eng.incumbent()
    assert st._ex_rules(cfg(comp=True, liger=True)) is not None            # (the incumbent + the value alone: refused)
    q = st._ex_contrarian(eng, inc)
    assert q is not None and q.why == 'contrarian' and q.hypothesis.startswith('contrarian|')
    assert knobs_of(st, q) == cfg(comp=True)                                # the partner back to the reference's value
    row = st.ex_by_hid[q.hypothesis]
    assert row['value'] == 'comp=true' and row['flipped'] == [{'value': 'liger=true', 'to': False, 'source': 'measured'}]
    assert 'saver' not in row and st.ledger[q.hypothesis]['kind'] == 'contrarian'
    assert run(st, eng, q).status == 'screened' and st._ex_used('contrarian') == 1
    view = st._ex_view()
    assert view['screens']['contrarian'] == 1 and view['decisions'][-1]['outcome'] == 'screened'
    json.dumps(view)


def test_w1_eligibility_and_ranking_follow_the_priors_own_numbers():
    st, eng = _w1(ExB)
    inc_cfg = st._knob_cfg(eng, eng.incumbent())
    ranked = [v for _, v, *_ in st._ex_contra_values(eng, inc_cfg)]
    # comp (+2 % + 10 %) x 0.75 before wk 4 (+1 % + 6 %) before tf32 (-3 % + 8 %); wk 0 (-1 % +- 1 %: gain + 2 sd is 1 %,
    # below what one run can tell - no detectable upside); rc / mb / liger are not rated low
    assert ranked == [('comp', 'true'), ('wk', '4'), ('tf32', 'true')]
    st.prior.levels[('comp', 'true')]['p_runs'] = 0.2                       # outlandish by the prior's own estimate
    assert ('comp', 'true') not in [v for _, v, *_ in st._ex_contra_values(eng, inc_cfg)]
    st.prior.levels[('comp', 'true')].update(p_runs=0.75, sd=math.log1p(0.01))   # sd 1 % < 2 %: its sign is not in doubt
    assert ('comp', 'true') not in [v for _, v, *_ in st._ex_contra_values(eng, inc_cfg)]


def test_w1_a_small_rating_with_a_wide_spread_is_contrarian_however_small_its_optimistic_gain():
    """The design's own M1 case (a value rated -0.5 % +- 2 %, measured about +10 % later): eligible, its optimistic gain
    (+1.5 %) below max(2 %, 2 sigma_pair) ranks it, it is no veto; no failed run needed."""
    guide = prior(lv('rc', 'half', 8, 4, ('spends', 12)), lv('rc', 'none', 22, 8, ('spends', 30)),
                  lv('mb', 2, 8, 3, ('spends', 15)), lv('liger', True, 5, 3), lv('comp', True, 5, 3), lv('tf32', True, 5, 3),
                  lv('wk', 4, -0.5, 2, p_runs=0.9))
    st, eng = setup(ExB, guide)
    ranked = st._ex_contra_values(eng, st._knob_cfg(eng, 'c0'))
    assert [v for _, v, *_ in ranked] == [('wk', '4')]
    opt = ranked[0][5]
    assert 100 * math.expm1(opt) == pytest.approx(1.49, abs=0.01) and opt < st._ex_thr(eng)
    q = st._ex_contrarian(eng, 'c0')
    assert q is not None and knobs_of(st, q) == cfg(wk=4) and st.ex_by_hid[q.hypothesis]['value'] == 'wk=4'


REAL_RATINGS = {   # the v2 runs' priors (gain %, sd %, p_runs): what M1's detectable-upside floor must keep
    'G4U': {'compile': (0, 12, 0.75, True), 'tf32': (0, 0.5, 0.98, False), 'nvls0': (0, 0.5, 0.98, False),
            'workers4': (0, 0.5, 0.98, False), 'expandable': (0, 0.7, 0.98, False),
            'workers0': (-1, 1.5, 0.98, True)},                             # (-1 + 2 x 1.5 = 2.0 %: on the floor)
    'Q27U': {'compile': (-2, 8, 0.7, True), 'tf32': (0, 0.4, 0.98, False), 'nvls0': (0, 0.5, 0.98, False),
             'workers4': (0, 0.3, 0.98, False)},
    'M35H2': {'compile': (-3, 8, 0.55, True), 'nccl32': (-0.5, 2.0, None, True), 'nccl16': (0, 1.5, None, True),
              'nvls0': (0, 1.5, None, True), 'tf32': (0, 1.0, None, True)}}


def test_w1_the_detectable_upside_floor_keeps_the_real_priors_compile_and_drops_their_tiny_knobs():
    """gain + 2 sd (the declared percentages) must reach max(2 %, 2 sigma_pair): on G4U and Q27U the tiny knobs (0 +-
    0.3 to 0.7 %) drop and compile stays; on the two-node M35H2 its uncertain NCCL values stay and M1's score ranks
    them (`_ex_m1_score`: optimistic gain x p_runs); a noisier arm raises the floor."""
    st, eng = setup(ExB, prior(*base_levels()))
    assert st._ex_thr(eng) == pytest.approx(0.02)                           # (sigma_pair 1 %: the floor is 2 %)
    for env, rows in REAL_RATINGS.items():
        for name, (g, sd, pr, keep) in rows.items():
            assert st._ex_upside_ok(g, sd, eng) is keep, (env, name)
    m35 = {n: st._ex_m1_score(g, sd, pr) for n, (g, sd, pr, _) in REAL_RATINGS['M35H2'].items()}
    assert sorted(m35, key=lambda n: -m35[n]) == ['compile', 'nccl16', 'nvls0', 'nccl32', 'tf32']
    assert st._ex_m1_score(0, 12, 0.75) == pytest.approx(math.log1p(0.12) * 0.75)
    assert st._ex_m1_score(None, None) == pytest.approx(X.SD_UNKNOWN) and st._ex_upside_ok(None, None, eng)

    class Noisy:                                                            # an arm whose sigma_pair is 2 %: floor 4 %
        @staticmethod
        def sigma_pair():
            return 0.02
    assert not st._ex_upside_ok(-0.5, 2.0, Noisy) and st._ex_upside_ok(0, 12, Noisy)


def test_w1_tiny_knobs_are_no_contrarian_move_and_compile_is_the_pick():
    """A G4U-like guide (the real ratings on this table: compile-like comp 0 +- 12 % with p_runs 0.75, NVLS-like tf32 and
    workers-like wk=4 0 +- 0.5 %): comp is the contrarian pick, the tiny knobs are not eligible - the v2 run spent both
    slots on them (+0.5 % optimistic, one run cannot tell)."""
    guide = prior(lv('rc', 'half', 8, 4, ('spends', 12)), lv('rc', 'none', 22, 8, ('spends', 30)),
                  lv('mb', 2, 8, 3, ('spends', 15)), lv('liger', True, 3, 5, ('frees', 5)),
                  lv('comp', True, 0, 12, None, p_runs=0.75), lv('tf32', True, 0, 0.5, p_runs=0.98),
                  lv('wk', 4, 0, 0.5, p_runs=0.98), lv('wk', 0, 0, 0.5, p_runs=0.98))
    st, eng = setup(ExB, guide, gain=table_gain({('liger', True): 0.05}))
    put(st, eng, {'liger': True})
    ranked = [v for _, v, *_ in st._ex_contra_values(eng, st._knob_cfg(eng, eng.incumbent()))]
    assert ranked == [('comp', 'true')]
    q = st._ex_contrarian(eng, eng.incumbent())
    assert q is not None and st.ex_by_hid[q.hypothesis]['value'] == 'comp=true'


def test_w1_other_jobs_failed_starts_lower_the_plan_not_the_eligibility():
    """The G4U finding: the store's failed starts of other jobs put compile in a five-value set with three values the
    incumbent holds - the pair table makes the move's P 0.5^3 of the ledger's. It stays a contrarian move (not
    outlandish: no measured infeasibility in this job) and is probed first; while S6 holds (two knob-side runs out of
    memory) its gate - a design rule - refuses it."""
    guide = prior(lv('rc', 'half', 8, 4, ('spends', 12)), lv('mb', 2, 8, 3, ('spends', 15)), lv('liger', True, 5, 3),
                  lv('comp', True, 0, 12, None, p_runs=0.75), lv('tf32', True, 0, 0.5), lv('wk', 4, 0, 0.5),
                  lv('wk', 0, 0, 0.5))
    store = [{'members': [{'comp': True}, {'tf32': True}, {'wk': 4}, {'mb': 2}, {'rc': 'half'}], 'id': 'learned-x',
              'message': 'comp, tf32, wk=4, mb=2, rc=half: failed to start together in another job (compiler)'}]
    eng = MemEngine(table_gain({('mb', 2): 0.05, ('wk', 4): 0.002}), mem_of({}))
    st = ExB(CodeProposer(guide), facts={'cap_gib': CAP}, seed=3, experience={'start_failures': store})
    st._init(eng)
    st._ex_init(eng)
    put(st, eng, {'mb': 2, 'tf32': True, 'wk': 4})                          # the incumbent (+5.2 %)
    inc = eng.incumbent()
    move = cfg(mb=2, tf32=True, wk=4, comp=True)
    assert st._pf(move)[0] < X.EX_P_MIN <= st._ex_p_memory(move)            # (0.5^3 x the ledger's P; the ledger's P)
    st.fb_oom_streak = 2                                                    # S6 holds: its gate refuses P < 0.9
    assert st._ex_contrarian(eng, inc) is None and st.fb_counts['s6_refused'] >= 1
    st.fb_oom_streak = 0
    q = st._ex_contrarian(eng, inc)
    assert q is not None and knobs_of(st, q) == move and tuple(q.fidelity) == ('probe', 'full')
    assert 'an earlier arm' in st.ledger[q.hypothesis]['why_feasible']


def test_w1_at_most_two_contrarian_screens_per_arm():
    st, eng = _w1(ExB)
    made = []
    for _ in range(2):
        q = st._ex_contrarian(eng, eng.incumbent())
        made.append(st.ex_by_hid[q.hypothesis]['value'])
        run(st, eng, q)
    assert made == ['comp=true', 'wk=4']
    assert st._ex_stalled()                                                 # (two full screens, no improvement: while
    assert st._ex_contra_list(eng, eng.incumbent()) == []                   #  stalled a one-knob move is not admissible)
    st.ex_bad = 0                                                           # (the plateau set aside: the cap alone)
    assert [m['value'] for m in st._ex_contra_list(eng, eng.incumbent())] == ['tf32=true']   # (still eligible ...)
    assert st._ex_contrarian(eng, eng.incumbent()) is None                  # (... but two contrarian screens ran)
    assert st._ex_contrarian_moves(eng, eng.incumbent()) == []


def test_w1_a_memory_saver_is_added_when_the_move_spends_and_the_ledgers_p_is_below_half():
    st, eng = _w1(ExB, comp_mem=('spends', 45), tf32_mem=('frees', 25))
    inc = eng.incumbent()
    alone = cfg(comp=True)
    assert st._pf(alone)[0] < 0.5                                           # the move alone: 105 +- 23 GiB of a 98 line
    q = st._ex_contrarian(eng, inc)
    assert q is not None and knobs_of(st, q) == cfg(comp=True, tf32=True)   # partner back, the declared saver added
    row = st.ex_by_hid[q.hypothesis]
    assert row['saver'] == {'knob': 'tf32', 'value': True, 'tier': 'declared', 'saving_gib': 25.0}
    assert row['flipped'][0]['value'] == 'liger=true' and q.p_feasible > st._pf(alone)[0]
    assert 'saves memory' in row['note']


def test_w1_a_value_whose_run_in_the_moves_context_was_measured_is_not_contrarian():
    st, eng = _w1(ExB)
    put(st, eng, {'comp': True})                                            # the move's own context measured already
    moves = st._ex_contra_list(eng, eng.incumbent())
    assert all(m['value'] != 'comp=true' for m in moves)


# ---------------------------------------------------------------- W2: the attribution
def test_w2_repeated_memory_failures_need_a_saver_then_are_excluded_after_three():
    """A value that never fits (+55 GiB on a 100 GiB card at 60): out of memory twice -> it needs a saver covering its
    shortfall (refused without one, or with savers that cannot cover it; allowed with one that does); three more
    failures that held savers -> excluded for the rest of the arm."""
    guide = prior(lv('rc', 'none', 18, 6, ('spends', 30), p_runs=0.25), lv('mb', 2, 8, 3, ('spends', 8)),
                  lv('liger', True, 5, 3, ('frees', 6)), lv('tf32', True, 0, 1, ('frees', 4)),
                  lv('comp', True, 2, 5, ('frees', 20)), lv('wk', 4, 0.5, 1))
    m = mem_of({('rc', 'none'): 55.0, ('mb', 2): 8.0, ('liger', True): -6.0, ('tf32', True): -4.0, ('comp', True): -20.0})
    st, eng = setup(ExB, guide, mem=m)
    st.fb_on['fb_s6'] = False                                               # (failures in a row on purpose)
    for change in ({'rc': 'none'}, {'rc': 'none', 'mb': 2}):
        o = put(st, eng, change)
        assert st.runs[o.cid]['kind'] == 'oom'
    att = st._ex_attribution()
    need = att['needs_saver'][('rc', '"none"')]
    assert len(need['failures']) == 2 and need['with_saver'] == [] and need['gap_gib'] == pytest.approx(17.0, abs=0.01)
    assert all(f['cause'] == 'memory' and ('rc', '"none"') in f['suspects'] for f in att['failures'])
    assert 'needs a saver' in st._ex_rules(cfg(rc='none', wk=4))            # no saver
    assert 'needs a saver' in st._ex_rules(cfg(rc='none', liger=True))      # 6 GiB of 17, the ledger says no
    assert st._ex_rules(cfg(rc='none', comp=True)) is None                  # 20 GiB covers 17
    for change in ({'rc': 'none', 'liger': True}, {'rc': 'none', 'tf32': True}):
        put(st, eng, change)
    assert ('rc', '"none"') in st._ex_attribution()['needs_saver']          # two with a saver: not yet
    put(st, eng, {'rc': 'none', 'liger': True, 'tf32': True})
    att = st._ex_attribution()
    assert ('rc', '"none"') in att['excluded'] and ('rc', '"none"') not in att['needs_saver']
    assert len(att['excluded'][('rc', '"none"')]['with_saver']) == 3
    assert 'is excluded' in st._ex_rules(cfg(rc='none', comp=True))         # even with a saver that covers it
    assert st._ex_rules(cfg(mb=2, comp=True)) is None                       # (the others stay allowed)
    assert ('rc', '"none"') in st.ex_excluded                               # (for the rest of the arm)
    assert st._ex_view()['attribution']['excluded'][0]['value'] == 'rc="none"'


def test_w2_a_crash_pair_is_a_conflict_never_held_whole_each_value_alone_allowed():
    """Each of two values ran alone; the run holding both (and a third, untested one) hung: the conflict is the pair of
    values that ran apart; a later candidate holding it whole is refused, each value alone (or with the third) is not;
    the exact failure and a near copy of it are never repeated."""
    guide = prior(*base_levels(comp=(-5, 10)))
    st, eng = setup(ExB, guide, gain=table_gain({('liger', True): 0.05}), hang=({'comp': True, 'liger': True},))
    put(st, eng, {'comp': True})
    put(st, eng, {'liger': True})                                           # (an improvement: not stalled)
    o = put(st, eng, {'comp': True, 'liger': True, 'wk': 4}, plan=('probe', 'full'))
    assert st.runs[o.cid]['kind'] == 'start_cap' and not st._ex_stalled()
    att = st._ex_attribution()
    assert [c['members'] for c in att['conflicts']] == [[('comp', 'true'), ('liger', 'true')]]
    assert att['failures'][-1]['cause'] == 'conflict' and att['failures'][-1]['run'] == o.cid
    for whole in (cfg(comp=True, liger=True), cfg(comp=True, liger=True, tf32=True, rc='half')):
        assert 'failed together' in st._ex_rules(whole)
    for part in (cfg(comp=True), cfg(liger=True, wk=4), cfg(comp=True, wk=4)):
        assert st._ex_rules(part) is None
    inc = st._knob_cfg(eng, eng.incumbent())
    for again in (cfg(comp=True, liger=True, wk=4), cfg(comp=True, liger=True, wk=0)):   # exact, and small knobs only
        ok, why = st._ex_admissible(again, inc)
        assert not ok and why.startswith(f'it repeats {o.cid}')
    assert st._ex_admissible(cfg(comp=True, wk=4), inc)[0]
    assert st.ex_counts['refused_repeat'] == 2


def test_w2_a_failure_with_one_value_off_the_reference_failed_alone():
    st, eng = setup(ExB, prior(*base_levels()), hang=({'comp': True},))
    put(st, eng, {'liger': True})
    o = put(st, eng, {'comp': True})
    att = st._ex_attribution()
    assert att['conflicts'] == [{'members': [('comp', 'true')], 'runs': [o.cid], 'kinds': ['start_cap'], 'alone': True}]
    assert st._ex_contra_list(eng, eng.incumbent()) == [] or all(m['value'] != 'comp=true'
                                                                   for m in st._ex_contra_list(eng, eng.incumbent()))


# ---------------------------------------------------------------- W3: never repeat a failure, the repair material
def test_w3_never_repeat_an_exact_failure_unless_a_saver_changes_its_cause():
    guide = prior(*base_levels(tf32_mem=('frees', 10)))
    st, eng = setup(ExB, guide, mem=mem_of({('rc', 'none'): 30.0, ('mb', 2): 15.0, ('tf32', True): -10.0}))
    o = put(st, eng, {'rc': 'none', 'mb': 2}, plan=('probe', 'full'))
    assert o.fidelity == 'probe' and st.runs[o.cid]['kind'] == 'oom'
    assert not st._ex_stalled()                                             # (never repeat holds at any time)
    inc = st._knob_cfg(eng, eng.incumbent())
    for again in (cfg(rc='none', mb=2), cfg(rc='none', mb=2, wk=0)):
        ok, why = st._ex_admissible(again, inc)
        assert not ok and 'no saver added' in why
    assert st._ex_admissible(cfg(rc='none', mb=2, tf32=True), inc)[0]       # a (small-knob) saver changes the cause
    assert st._ex_admissible(cfg(rc='none', mb=2, liger=True), inc)[0]      # (another big knob: not the same failure)


def test_w3_repair_material_adds_the_change_that_removes_the_cause():
    """A memory failure: its suspect on the incumbent plus the best saver by GiB freed; a conflict: the failed value on
    the incumbent without its partner. Each repair is recognized (`_ex_repair_of`), runs as a candidate tagged 'repair'
    and counts toward EX_REPAIR_MAX."""
    guide = prior(*base_levels(tf32_mem=('frees', 12)))
    st, eng = setup(ExB, guide, gain=table_gain({('liger', True): 0.05}),
                    mem=mem_of({('rc', 'none'): 50.0, ('liger', True): -5.0, ('tf32', True): -12.0}),
                    hang=({'comp': True, 'liger': True},))
    put(st, eng, {'liger': True})                                           # the incumbent
    put(st, eng, {'liger': True, 'rc': 'none'}, plan=('probe', 'full'))     # 105 GiB: out of memory at its probe
    put(st, eng, {'liger': True, 'comp': True}, plan=('probe', 'full'))     # hangs at its probe: the pair
    mats = st._ex_repair_candidates(eng, eng.incumbent())
    by = {m['cause']: m for m in mats}
    assert set(by) == {'memory', 'conflict'} and all(m['why'] == 'repair' for m in mats)
    assert by['memory']['cfg'] == cfg(liger=True, rc='none', tf32=True) and by['memory']['savers'][0]['knob'] == 'tf32'
    assert by['conflict']['cfg'] == cfg(comp=True) and by['conflict']['dropped'] == ['liger=true']
    assert st._ex_repair_of(by['memory']['cfg'])['cause'] == 'memory'
    assert st._ex_repair_of(by['conflict']['cfg'])['cause'] == 'conflict'
    assert st._ex_repair_of(cfg(liger=True, rc='none', wk=4)) is None       # (no saver: the cause unchanged)
    q = st._ex_repair(eng, eng.incumbent())
    assert q is not None and q.why == 'repair' and q.hypothesis.startswith('repair|')
    run(st, eng, q)
    assert st._ex_used('repair') == 1


def _w3_saver_conflict(conflict, tf32_mem=None):
    """The design's W3: the incumbent holds the fused-loss-like value (+5 %); the spender (+50 GiB on a 100 GiB card at
    60) runs out of memory twice with it and no saver; the compiler-like value frees 30 GiB and conflicts with the held
    value - declared by the prior, or measured (a probe of the two together hung)."""
    guide = prior(lv('rc', 'none', 22, 8, ('spends', 30)), lv('mb', 2, 8, 3, ('spends', 15)),
                  lv('liger', True, 5, 3, ('frees', 5)), lv('comp', True, 2, 10, ('frees', 30), p_runs=0.75),
                  lv('tf32', True, 0, 1, tf32_mem), lv('wk', 4, 0.5, 1),
                  conflicts=[(('comp', True), ('liger', True))] if conflict == 'declared' else ())
    m = mem_of({('rc', 'none'): 50.0, ('liger', True): -5.0, ('comp', True): -30.0, ('tf32', True): -10.0})
    st, eng = setup(ExB, guide, gain=table_gain({('liger', True): 0.05, ('wk', 4): 0.002}), mem=m,
                    hang=({'comp': True, 'liger': True},))
    put(st, eng, {'liger': True})                                           # the incumbent
    if conflict == 'measured':
        put(st, eng, {'liger': True, 'comp': True}, plan=('probe', 'full'))  # hangs at its probe: the pair
    for change, fits in (({'liger': True, 'rc': 'none'}, False), ({'liger': True, 'wk': 4}, True),
                         ({'liger': True, 'rc': 'none', 'wk': 4}, False)):
        o = put(st, eng, change, plan=('probe', 'full') if not fits else ('full',))
        assert (st.runs[o.cid]['kind'] == 'oom') is not fits                # 105 GiB: out of memory, no saver
    return st, eng


@pytest.mark.parametrize('conflict', ['declared', 'measured'])
def test_w3_a_repair_sends_back_the_held_value_its_saver_conflicts_with(conflict):
    st, eng = _w3_saver_conflict(conflict)
    need = st._ex_attribution()['needs_saver'][('rc', '"none"')]
    assert len(need['failures']) == 2 and need['with_saver'] == []          # (the held saver in both: no try)
    mats = st._ex_repair_candidates(eng, eng.incumbent())
    assert mats and mats[0]['cause'] == 'memory'
    m0 = mats[0]
    assert m0['cfg'] == cfg(rc='none', comp=True)                           # the spender + the saver, the held value back
    assert [s['knob'] for s in m0['savers']] == ['comp'] and m0['dropped'] == ['liger=true']
    assert 'liger=true back (conflict)' in m0['note']
    assert st._ex_rules(m0['cfg']) is None and st._ex_repair_of(m0['cfg']) == {'run': m0['run'], 'cause': 'memory'}
    q = st._ex_repair(eng, eng.incumbent())
    assert q is not None and q.why == 'repair' and knobs_of(st, q) == cfg(rc='none', comp=True)
    assert st.ex_by_hid[q.hypothesis]['dropped'] == ['liger=true']


def test_w3_a_conflict_free_saver_that_covers_the_gap_sends_nothing_back():
    st, eng = _w3_saver_conflict('declared', tf32_mem=('frees', 10))       # 10 GiB covers the 7 to lose
    m0 = st._ex_repair_candidates(eng, eng.incumbent())[0]
    assert m0['cfg'] == cfg(liger=True, rc='none', tf32=True) and m0['dropped'] == []


def test_w3_a_contrarian_spender_takes_a_saver_that_sends_a_held_value_back():
    """M1's saver step: the only saver of a move that spends memory conflicts with a value the incumbent holds - it is
    added and that value goes back (bcomp's completion), never the move's own value."""
    guide = prior(*base_levels(comp=(2, 10), comp_mem=('frees', 30)), lv('mb', 2, 1, 6, ('spends', 45), p_runs=0.6),
                  conflicts=[(('comp', True), ('liger', True))])
    st, eng = setup(ExB, guide, gain=table_gain({('liger', True): 0.05}), mem=mem_of({('liger', True): -5.0}))
    put(st, eng, {'liger': True})
    mv = st._ex_move(eng, eng.incumbent(), 'mb', '2')
    assert mv['cfg'] == cfg(mb=2, comp=True) and mv['saver']['knob'] == 'comp'
    assert mv['flipped'] == [{'value': 'liger=true', 'to': False, 'source': 'the saver: declared'}]
    assert 'it conflicts with the saver' in mv['note']


def _two_ooms(guide, mem, hang=()):
    """The incumbent holds the fused-loss-like value; the spender runs out of memory twice at its probe (once with a
    small value that fitted elsewhere, a fitting run between them: S6 reset)."""
    st, eng = setup(ExB, guide, gain=table_gain({('liger', True): 0.05, ('wk', 4): 0.002}), mem=mem, hang=hang)
    put(st, eng, {'liger': True})
    for change, fits in (({'liger': True, 'rc': 'none'}, False), ({'liger': True, 'wk': 4}, True),
                         ({'liger': True, 'rc': 'none', 'wk': 4}, False)):
        o = put(st, eng, change, plan=('probe', 'full') if not fits else ('full',))
        assert (st.runs[o.cid]['kind'] == 'oom') is not fits
    return st, eng


def test_w3_a_memory_repair_covers_the_measured_shortfall_or_is_not_offered():
    """A 17 GiB gap (115 GiB on a 100 GiB card): a 10 GiB saver alone does not cover it - no memory repair (and the
    needs-a-saver rule says so); with an 8 GiB saver too, the repair holds both (18 GiB net)."""
    def guide(wk0):
        return prior(lv('rc', 'none', 22, 8, ('spends', 30)), lv('liger', True, 5, 3), lv('tf32', True, 0, 1, ('frees', 10)),
                     lv('wk', 4, 0.5, 1), lv('wk', 0, -1, 1, ('frees', 8) if wk0 else None))
    m = mem_of({('rc', 'none'): 55.0, ('tf32', True): -10.0, ('wk', 0): -8.0})
    st, eng = _two_ooms(guide(False), m)
    assert st._ex_attribution()['needs_saver'][('rc', '"none"')]['gap_gib'] == pytest.approx(17.0, abs=0.01)
    assert [x for x in st._ex_repair_candidates(eng, eng.incumbent()) if x['cause'] == 'memory'] == []
    short = cfg(liger=True, rc='none', tf32=True)
    assert 'needs a saver covering 17.0 GiB' in st._ex_rules(short) and st._ex_repair_of(short) is None
    assert st._ex_repair(eng, eng.incumbent()) is None
    st2, eng2 = _two_ooms(guide(True), m)
    m0 = st2._ex_repair_candidates(eng2, eng2.incumbent())[0]
    assert m0['cfg'] == cfg(liger=True, rc='none', tf32=True, wk=0) and [s['knob'] for s in m0['savers']] == ['tf32', 'wk']
    assert m0['net_gib'] == pytest.approx(18.0, abs=0.1) and eng2.mem_of(m0['cfg']) <= CAP   # (the ledger's tags)
    assert st2._ex_repair_of(m0['cfg']) == {'run': m0['run'], 'cause': 'memory'}


def test_w3_a_saver_that_sends_a_held_saver_back_counts_its_net_and_another_saver_follows():
    """A 12 GiB gap; the 25 GiB saver conflicts with the held value, which saves 20 GiB: net 5 GiB - not the gap; the
    10 GiB saver follows within the two-saver limit: net 15, 95 GiB on the card (no 105 GiB repair)."""
    guide = prior(lv('rc', 'none', 22, 8, ('spends', 70)), lv('liger', True, 5, 3, ('frees', 20)),
                  lv('comp', True, 2, 10, ('frees', 25)), lv('tf32', True, 0, 1, ('frees', 10)), lv('wk', 4, 0.5, 1),
                  conflicts=[(('comp', True), ('liger', True))])
    m = mem_of({('rc', 'none'): 70.0, ('liger', True): -20.0, ('comp', True): -25.0, ('tf32', True): -10.0})
    st, eng = _two_ooms(guide, m)
    mats = [x for x in st._ex_repair_candidates(eng, eng.incumbent()) if x['cause'] == 'memory']
    assert len(mats) == 1
    m0 = mats[0]
    assert m0['gap_gib'] == pytest.approx(12.0, abs=0.01) and m0['net_gib'] == pytest.approx(15.0, abs=0.1)
    assert m0['cfg'] == cfg(rc='none', comp=True, tf32=True) and m0['dropped'] == ['liger=true']
    assert sorted(s['knob'] for s in m0['savers']) == ['comp', 'tf32'] and eng.mem_of(m0['cfg']) == 95.0
    q = st._ex_repair(eng, eng.incumbent())
    assert q is not None and knobs_of(st, q) == m0['cfg']
    o = run(st, eng, q)
    assert o.status in ('screened', 'aborted') and st.runs[o.cid]['kind'] != 'oom'   # (it fits)


def test_w3_a_partner_the_table_requires_counts_before_a_repair_decides_it_needs_no_saver():
    """The table requires mb=2 with rc='none'; mb=2 fitted (75 GiB, the reference stays the incumbent), rc='none' with it
    ran out of memory at its probe (105 GiB: 7 GiB to lose, only rc='none' blamed). The suspect on the incumbent is the
    illegal rc='none', mb=1 - it would look 15 GiB lighter -; completed first it is the failed configuration again, so the
    20 GiB saver is added: the legal 85 GiB repair is offered, proposed and fits."""
    def check(c, c0):
        return (None, 'rc=none needs mb=2') if c.get('rc') == 'none' and c.get('mb') != 2 else ('ok', None)
    guide = prior(lv('rc', 'none', 22, 8, ('spends', 30)), lv('mb', 2, 8, 3, ('spends', 15)), lv('liger', True, 5, 3),
                  lv('tf32', True, 0, 1, ('frees', 20)), lv('wk', 4, 0.5, 1))
    st, eng = setup(ExB, guide, mem=mem_of({('rc', 'none'): 30.0, ('mb', 2): 15.0, ('tf32', True): -20.0}), check=check)
    assert not st._legal(eng, cfg(rc='none')) and st._legal(eng, cfg(rc='none', mb=2))
    put(st, eng, {'mb': 2})
    o = put(st, eng, {'rc': 'none', 'mb': 2}, plan=('probe', 'full'))
    assert o.fidelity == 'probe' and st.runs[o.cid]['kind'] == 'oom' and eng.incumbent() == 'c0'
    f = st._ex_attribution()['failures'][-1]
    assert f['suspects'] == [('rc', '"none"')] and f['gap_gib'] == pytest.approx(7.0, abs=0.01)
    mats = st._ex_repair_candidates(eng, 'c0')
    assert mats and mats[0]['cause'] == 'memory'
    m0 = mats[0]
    assert m0['cfg'] == cfg(rc='none', mb=2, tf32=True) and [s['knob'] for s in m0['savers']] == ['tf32']
    assert m0['net_gib'] == pytest.approx(20.0, abs=0.1) and eng.mem_of(m0['cfg']) == 85.0
    q = st._ex_repair(eng, 'c0')
    assert q is not None and q.why == 'repair' and knobs_of(st, q) == m0['cfg']
    assert run(st, eng, q).status == 'screened'


def test_w3_a_conflict_whose_members_never_ran_apart_gets_both_directions():
    """One probe with both values hung and neither ran apart: a repair keeping each (the other left out), both
    recognized; a configuration holding both, or neither, repairs nothing. When both ran apart, either alone is one."""
    st, eng = setup(ExB, prior(*base_levels()), hang=({'comp': True, 'liger': True},))
    o = put(st, eng, {'comp': True, 'liger': True}, plan=('probe', 'full'))
    assert st.runs[o.cid]['kind'] == 'start_cap'
    mats = st._ex_repair_candidates(eng, 'c0')
    got = sorted((json.dumps(m['cfg'], sort_keys=True), m['dropped']) for m in mats if m['cause'] == 'conflict')
    assert got == sorted([(json.dumps(cfg(comp=True), sort_keys=True), ['liger=true']),
                          (json.dumps(cfg(liger=True), sort_keys=True), ['comp=true'])])
    assert all(m['p'] == 1.0 for m in mats)
    for one in (cfg(comp=True), cfg(liger=True, wk=4)):
        assert st._ex_repair_of(one) == {'run': o.cid, 'cause': 'conflict'}
    assert st._ex_repair_of(cfg(comp=True, liger=True)) is None and st._ex_repair_of(cfg(tf32=True)) is None
    st2, eng2 = setup(ExB, prior(*base_levels()), hang=({'comp': True, 'liger': True},))
    put(st2, eng2, {'comp': True})
    put(st2, eng2, {'liger': True})
    o2 = put(st2, eng2, {'comp': True, 'liger': True}, plan=('probe', 'full'))
    assert st2._ex_repair_of(cfg(comp=True, tf32=True)) == {'run': o2.cid, 'cause': 'conflict'}


# ---------------------------------------------------------------- W4 / M4: leaving a local optimum
def test_w4_the_plateau_is_counted_from_the_first_improvement_on_the_reference():
    """Failed full screens at the start leave no optimum to leave: not stalled, radius 1; after the first valid full
    screen that improved on the reference, two without an improvement stall the search."""
    st, eng = setup(ExB, prior(*base_levels()), gain=table_gain({('liger', True): 0.05}),
                    mem=mem_of({('rc', 'none'): 50.0}), hang=({'comp': True},))
    assert put(st, eng, {'rc': 'none'}).status == 'invalid'                 # out of memory in full
    assert put(st, eng, {'comp': True}).status == 'invalid'                 # hangs in full
    assert not st._ex_stalled() and st._ex_radius() == 1 and st.ex_bad == 0 and not st.ex_found
    put(st, eng, {'tf32': True})                                            # valid, +0 %: no improvement either
    assert not st._ex_stalled() and st.ex_bad == 0
    put(st, eng, {'liger': True})                                           # +5 %: the first improvement
    assert st.ex_found and st.ex_bad == 0 and not st._ex_stalled()
    put(st, eng, {'liger': True, 'wk': 4})
    assert not st._ex_stalled() and st.ex_bad == 1
    put(st, eng, {'liger': True, 'tf32': True})
    assert st._ex_stalled() and st._ex_radius() == 2
    assert st._ex_view()['stall']['counting'] is True



W4_GAIN = {('liger', True): 0.05, ('rc', 'half'): -0.03, ('rc', 'none'): -0.02, ('mb', 2): -0.02, ('comp', True): 0.005,
           ('wk', 4): 0.002, ('wk', 0): -0.005}
W4_MEM = {('rc', 'half'): 12.0, ('rc', 'none'): 30.0, ('mb', 2): 15.0, ('liger', True): -5.0, ('tf32', True): -10.0}


def _w4(cls=ExB):
    guide = prior(*base_levels(comp=(-5, 10), tf32_mem=('frees', 10)))
    st, eng = setup(cls, guide, gain=table_gain(W4_GAIN), mem=mem_of(W4_MEM))
    assert put(st, eng, {'liger': True}).status == 'screened'               # an improvement: the incumbent
    put(st, eng, {'liger': True, 'wk': 4})                                  # +0.2 %: not one
    assert not st._ex_stalled() and st.ex_bad == 1
    o = put(st, eng, {'rc': 'none', 'mb': 2}, plan=('probe', 'full'))       # out of memory at its probe: no full screen
    assert o.fidelity == 'probe' and st.ex_bad == 1
    put(st, eng, {'rc': 'half', 'mb': 2})                                   # -5 %: the second full screen without one
    return st, eng


@pytest.mark.parametrize('cls', [ExB, ExH])
def test_w4_a_stall_raises_the_radius_and_admits_only_far_candidates_and_no_near_duplicates(cls):
    st, eng = _w4(cls)
    assert st._ex_stalled() and st._ex_radius() == 2 and st._ex_big_knobs() == {'rc', 'mb', 'liger', 'comp'}
    inc = eng.incumbent()
    inc_cfg = st._knob_cfg(eng, inc)
    for near in (cfg(liger=True, mb=2), cfg(liger=True, mb=2, wk=0, tf32=True)):   # one big knob off the incumbent
        ok, why = st._ex_admissible(near, inc_cfg)
        assert not ok and why.startswith('trust region: 1 big knob(s)')
    assert st._ex_admissible(cfg(liger=True, rc='half', mb=2), inc_cfg) == (True, '')
    ok, why = st._ex_admissible(cfg(rc='half', mb=2, wk=4), inc_cfg)        # far from the incumbent, but a near
    assert not ok and why.startswith('tabu')                                # duplicate of a run (small knobs only)
    q = st._ex_escape(eng, inc)
    assert q is not None and q.why == 'escape' and q.hypothesis.startswith('escape|')
    got = knobs_of(st, q)
    assert big_off(st, got, inc_cfg) >= 2
    big = st._ex_big_knobs()
    assert all(any(vk(got[k]) != vk(st._ex_kcfg(r['cfg'])[k]) for k in big) for r in st.runs.values()
               if r['kind'] not in ('oom', 'thrash', 'start_cap'))          # (no near-duplicate of a run that ran)
    assert st.ex_by_hid[q.hypothesis]['radius'] == 2
    q2 = st._ex_contrarian(eng, inc)                                        # (a contrarian move is a knob-side candidate
    assert q2 is None or big_off(st, knobs_of(st, q2), inc_cfg) >= 2        #  too: the trust region holds for it)


def test_w4_the_radius_grows_to_three_and_a_warm_restart_follows_four_stalled_screens():
    """The incumbent's class stalls; two runs of other classes follow (the third and fourth stalled full screens): the
    radius is 3 and the warm restart starts from the better of them with its best untried big-knob move - here micro-
    batch-like mb=2, rated +8 % and never measured, so the restart is optimistically faster than the incumbent."""
    guide = prior(lv('rc', 'half', 8, 4, ('spends', 12)), lv('rc', 'none', 2, 1, ('spends', 30)),
                  lv('mb', 2, 8, 3, ('spends', 15)), lv('liger', True, 5, 3), lv('comp', True, -5, 10),
                  lv('tf32', True, 0, 1), lv('wk', 0, -1, 1), lv('wk', 4, 0.5, 1))
    gain = table_gain({('liger', True): 0.05, ('comp', True): 0.01, ('rc', 'none'): -0.02, ('rc', 'half'): -0.03,
                       ('mb', 2): 0.02, ('wk', 4): 0.002, ('wk', 0): -0.005})
    st, eng = setup(ExB, guide, gain=gain, mem=mem_of({}))                  # (memory never binds here)
    put(st, eng, {'liger': True})                                           # +5 %: the incumbent
    put(st, eng, {'liger': True, 'wk': 4})
    put(st, eng, {'liger': True, 'tf32': True})
    assert st._ex_stalled() and st._ex_radius() == 2
    e = put(st, eng, {'comp': True, 'rc': 'none'}).cid                      # -1 %: the third stalled full screen
    assert st._ex_radius() == 3
    put(st, eng, {'rc': 'half', 'wk': 0})                                   # -3.5 %: the fourth
    assert st.ex_bad == 4 and st._ex_radius() == 3 and st.ex_restart_next == 4
    inc_cfg = st._knob_cfg(eng, eng.incumbent())
    q = st._ex_escape(eng, eng.incumbent())
    assert q is not None and st.ex_counts['warm_restarts'] == 1
    row = st.ex_by_hid[q.hypothesis]
    assert row['restart'] == e and 'warm restart' in row['note'] and row['optimistic_pct'] > 0
    got, start = knobs_of(st, q), st._ex_kcfg(st.runs[e]['cfg'])
    moved = [k for k in st.knobs if vk(got[k]) != vk(start[k])]
    assert moved == ['mb'] and 'mb' in st._ex_big_knobs()                   # its best untried big-knob move
    assert big_off(st, got, inc_cfg) >= 3                                   # (radius 3)
    assert st.ex_restart_next == 4 + 2 * X.EX_STALL                         # (the next one after four more)
    put(st, eng, {'liger': True, 'rc': 'half', 'comp': True})               # another non-improving screen
    q2 = st._ex_escape(eng, eng.incumbent())
    assert q2 is None or 'restart' not in st.ex_by_hid[q2.hypothesis]


def test_w4_no_escape_when_every_row_is_predicted_slower_than_the_incumbent():
    """A Q27U-like history: the incumbent near the ceiling, every far row rated well below it (and measured so where it
    ran): stalled, no escape is offered - not even the least pessimistic -, nor a warm restart after four stalled
    screens; the strategy's own path goes on."""
    guide = prior(lv('rc', 'half', -8, 2), lv('rc', 'none', -10, 2), lv('mb', 2, -6, 2), lv('liger', True, 5, 2),
                  lv('comp', True, -5, 2), lv('tf32', True, 0, 0.5), lv('wk', 0, -1, 0.5), lv('wk', 4, 0.5, 0.5))
    gain = table_gain({('liger', True): 0.05, ('rc', 'half'): -0.08, ('rc', 'none'): -0.10, ('mb', 2): -0.06,
                       ('comp', True): -0.05, ('wk', 4): 0.002, ('wk', 0): -0.005})
    st, eng = setup(ExB, guide, gain=gain, mem=mem_of({}))
    put(st, eng, {'liger': True})
    put(st, eng, {'liger': True, 'wk': 4})
    put(st, eng, {'liger': True, 'tf32': True})
    assert st._ex_stalled() and st._ex_radius() == 2
    assert st._ex_escape(eng, eng.incumbent()) is None                      # (every admissible row: optimistic <= 0)
    put(st, eng, {'comp': True, 'rc': 'half'})                              # two runs of other classes, both slower
    put(st, eng, {'comp': True, 'mb': 2})
    assert st.ex_bad == 4 and st.ex_restart_next == 4
    assert st._ex_escape(eng, eng.incumbent()) is None
    assert st.ex_counts['escape'] == 0 and st.ex_counts['warm_restarts'] == 0 and st._ex_view()['screens']['escape'] == 0


def test_w4_an_improvement_resets_the_radius_and_momentum_extends_an_ordered_knob():
    gain = table_gain({('rc', 'half'): 0.05, ('rc', 'none'): 0.02, ('liger', True): 0.0, ('wk', 4): 0.001})
    st, eng = setup(ExB, prior(*base_levels()), gain=gain, mem=mem_of({('rc', 'half'): 12.0, ('rc', 'none'): 30.0}))
    put(st, eng, {'rc': 'half'})                                            # +5 %: one knob moved full -> half
    assert st.ex_momentum == {'run': eng.incumbent(), 'moves': [{'knob': 'rc', 'from': '"full"', 'to': '"half"',
                                                                  'next': '"none"'}]}
    put(st, eng, {'rc': 'half', 'wk': 4})
    put(st, eng, {'rc': 'half', 'tf32': True})
    assert st._ex_stalled() and st._ex_radius() == 2
    q = st._ex_escape(eng, eng.incumbent())
    assert q is not None and knobs_of(st, q)['rc'] == 'none'                # the next value in the same direction
    assert st.ex_by_hid[q.hypothesis]['momentum'] == ['rc="none"'] and st.ex_counts['momentum'] == 1
    assert st.ex_momentum is None                                           # (for the next escape only)
    o = put(st, eng, {'rc': 'half', 'mb': 2, 'liger': True})                # gain +5 %: no improvement (not >= +2 %)
    assert o.status == 'screened' and st._ex_radius() == 3
    st2, eng2 = setup(ExB, prior(*base_levels()), gain=table_gain({('rc', 'half'): 0.05, ('mb', 2): 0.04}))
    put(st2, eng2, {'rc': 'half'})
    put(st2, eng2, {'rc': 'half', 'wk': 4})
    put(st2, eng2, {'rc': 'half', 'wk': 0})
    assert st2._ex_radius() == 2
    put(st2, eng2, {'rc': 'half', 'mb': 2})                                 # +4 % more: an improvement
    assert not st2._ex_stalled() and st2._ex_radius() == 1 and st2.ex_bad == 0
    assert st2._ex_escape(eng2, eng2.incumbent()) is None                   # (no escape unless stalled)


# ---------------------------------------------------------------- the base's gates
def test_s6_the_p_floor_and_the_probe_allowance_hold_for_every_candidate():
    st, eng = _w1(ExB, comp_mem=('spends', 45), tf32_mem=('frees', 25))
    inc = eng.incumbent()
    st.fb_oom_streak = 2                                                    # S6: two knob-side runs in a row out of memory
    q = st._ex_contrarian(eng, inc)                                         # the best move (P about 0.7) is refused, the
    assert st.fb_counts['s6_refused'] >= 1                                  # next one fits with P >= 0.9
    assert q is not None and st.ex_by_hid[q.hypothesis]['value'] == 'wk=4' and st._pf(knobs_of(st, q))[0] >= 0.9
    st.ex_log.clear()
    st.tried.add(st._ckey(knobs_of(st, q), ()))
    st.fb_oom_streak = 0
    real = st._pf
    st._ex_p_memory = lambda c, ids=(), unblamed=False: 0.1                 # the ledger's P below the floor: the runs
    st._pf = lambda c, ids=(): (0.1, 'planted')                             # say no
    assert st._ex_contrarian(eng, inc) is None and st._ex_contra_list(eng, inc) == []
    del st._ex_p_memory
    st._pf = lambda c, ids=(): (0.5, 'planted')                             # doubtful ...
    st.ladder._fund = lambda engine, cost, code=False: False                # ... and no probe left: not run in full
    assert st._ex_contrarian(eng, inc) is None
    st._pf = lambda c, ids=(): (0.8, 'planted')
    q = st._ex_contrarian(eng, inc)
    assert q is not None and tuple(q.fidelity) == ('full',)                 # (fits by the ladder's rule: in full)
    st._pf = real
    st2, eng2 = _w4()
    st2.fb_oom_streak = 2
    q2 = st2._ex_escape(eng2, eng2.incumbent())
    assert q2 is None or st2._pf(knobs_of(st2, q2))[0] >= 0.9


# ---------------------------------------------------------------- the strategy's own path, the parts, the view
def test_an_offered_move_the_strategy_runs_its_own_way_is_matched_and_counted():
    st, eng = _w1(ExH)
    moves = st._ex_contrarian_moves(eng, eng.incumbent())
    assert moves and moves[0]['why'] == 'contrarian' and moves[0]['value'] == 'comp=true'
    q = st._knob_q(eng, eng.incumbent(), moves[0]['cfg'], kind='challenge', plan=('full',), p=0.9)
    run(st, eng, q)                                                         # (a controlled test with explicit observation)
    assert st._ex_used('contrarian') == 1 and st.ex_counts['matched'] == 1
    assert st.ex_by_hid[q.hypothesis]['outcome'] == 'screened'
    q2 = st._knob_q(eng, eng.incumbent(), st._ex_contrarian_moves(eng, eng.incumbent())[0]['cfg'], kind='challenge',
                    plan=('full',), p=0.9)
    st._ex_mark(q2, {'why': 'contrarian', 'cfg': knobs_of(st, q2), 'note': 'marked'}, eng)
    st._ex_mark(q2, {'why': 'contrarian', 'cfg': knobs_of(st, q2), 'note': 'marked'}, eng)   # (twice: one screen)
    assert q2.why == 'contrarian' and sum(1 for r in st.ex_log if r['id'] == q2.hypothesis) == 1
    run(st, eng, q2)
    assert st._ex_used('contrarian') == 2 and st._ex_contrarian(eng, eng.incumbent()) is None


def test_a_proposal_made_and_not_run_frees_its_slot():
    st, eng = _w1(ExB)
    q = st._ex_contrarian(eng, eng.incumbent())
    assert st._ex_used('contrarian') == 1
    put(st, eng, {'tf32': True})                                            # another proposal ran in its step
    assert st.ex_by_hid[q.hypothesis]['outcome'] == 'not run' and st._ex_used('contrarian') == 0


def test_every_part_off():
    st, eng = _w4()
    put(st, eng, {'comp': True, 'liger': True, 'wk': 0})
    st.ex_off = set(EX_PARTS)
    assert st._ex_contrarian(eng, eng.incumbent()) is None and st._ex_escape(eng, eng.incumbent()) is None
    assert st._ex_repair_candidates(eng, eng.incumbent()) == [] and st._ex_attribution()['off'] is True
    assert st._ex_radius() == 1 and st._ex_rules(cfg(rc='none', mb=2)) is None
    inc_cfg = st._knob_cfg(eng, eng.incumbent())
    assert st._ex_admissible(cfg(rc='none', mb=2), inc_cfg) == (True, '')   # (no repeat rule, no trust region)
    view = st._ex_view()
    assert view['off'] == sorted(EX_PARTS) and view['attribution']['off'] is True
    json.dumps(view)
    st.ex_off = {'contrarian'}
    assert st._ex_contrarian(eng, eng.incumbent()) is None and st._ex_escape(eng, eng.incumbent()) is not None


def test_the_view_records_every_decision_for_the_report():
    st, eng = _w4()
    q = st._ex_escape(eng, eng.incumbent())
    run(st, eng, q)
    st._ex_admissible(cfg(liger=True, mb=2), st._knob_cfg(eng, eng.incumbent()))
    view = st._ex_view()
    json.dumps(view)
    assert view['version'] == X.EX_VERSION and view['limits']['contrarian'] == 2 and view['limits']['radius_max'] == 3
    assert view['stall']['stalled'] and view['stall']['improvements'][0]['moved'] == ['liger']
    d = view['decisions'][-1]
    assert d['why'] == 'escape' and d['id'] == q.hypothesis and d['outcome'] in ('screened', 'aborted', 'invalid', 'infeasible')
    assert view['refused'] and view['counts']['refused_trust'] >= 1
    assert view['attribution']['failures'] >= 1


def test_the_module_names_no_knob_model_or_library():
    src = SRC.read_text(encoding='utf-8')
    code = '\n'.join(ln for ln in src.splitlines() if not ln.strip().startswith('#'))
    for word in ('recompute', 'micro_batch', 'liger', 'compile', 'fsdp', 'deepspeed', 'megatron', 'torch', 'qwen', 'gemma',
                 'llama', 'flash', 'expandable', 'checkpointing', 'nccl', 'cudnn', 'triton', 'sdpa', 'hf_trainer', 'nemo',
                 'deepseek', 'transformers', 'automodel', 'channels', 'nvls'):
        assert not re.search(rf'\b{word}\b', code, re.I), word
    assert not re.search(r'\b(Q9U|Q9H|G4U|M35H[24]|Q27U|Q27H2|E16)\b', code)
    assert 'trials_tb' not in src
