"""bcomp (search/bcomp.py; FINALISTS_IMPL_SPEC_20261101 A.1, A.5, B.7 D5, B.8) on the finalists' shared base: the three
starts in order (experience, guide, uniform; pairwise different; the uniform one in full and the random arm's first draw);
the completion (a failed start short by 4.5 GiB + a value measured to free 6.5 GiB -> that run + the value, carrying its
patches; an unknown-direction partner within two completions; two per run, the second with another partner; the original
failures first; a patch-caused out-of-memory run completed with its patch); the challenge (one switch group, the largest
|declared gain| among the vetoed values no run holds, two per arm, probed only below 0.5); the ledger's best fill; the
declared-memory floor and the measurement deciding; isolated_blame by combination; probe thrash by its peak; S6 (the
counter, the 0.9 gate, the uniform start's exemption) over whole runs; the code track - the knob side's empty screen and
the protected code screens, asks by budget within 10 calls / 8 USD that stop at the top -; an engine-level ten-screen run
with its accounting; every component off; the fairness invariants (no knob, model or library name in the module; the
experience from other jobs only; the registration)."""
import dataclasses
import itertools
import json
import math
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cand_helpers import run_kept, toy
from finalist_helpers import CodeProposer
from hls2_helpers import BASE_TEXT, PATCH_FILE, SimEngine, SimSession, snippet
from infrarsi.search import ABLATIONS, BCOMP_COMPONENTS, LLM_STRATEGIES, STRATEGIES
from infrarsi.search import bcomp as B
from infrarsi.search.bcomp import BComp
from infrarsi.search.engine import Outcome
from infrarsi.search.finalist_base import (BLOCKED_VALUE, CARD_RESERVE, CODE_PROBES_MAX, P_FLOOR_DECLARED, S6_P,
                                             THRASH_PEAK, FinalistBase, FinalistLadder)
from infrarsi.search.knob_space import _feasible, choices
from infrarsi.search.llmopt import CALLS_MAX, USD_MAX
from infrarsi.search.toytrain import MEM_KNOBS, make_sim, mem_tasks
from infrarsi.search.wcs import CODE_ASKS as WCS_CODE_ASKS
from infrarsi.search.wcs import WCS
from test_wcs import GAIN, KNOBS, START, answer, engine
from test_xfer import NEAR, pack

SRC = Path(__file__).resolve().parents[3] / 'infrarsi' / 'search' / 'bcomp.py'
TARGET = {'card_gib': 141.0, 'nodes': 1, 'gpus_per_node': 8, 'params_b': 8.95, 'framework': 'hf_trainer'}
OWN = (141.0, 1, 8.95)                   # the target's own numbers (a pack with them is this job's: never a source)
RC = {'full': 0.0, 'half': 12.0, 'none': 36.5}


def mem_a(c):
    """GiB per GPU on a 100 GiB card (the reference 60): recompute half +12, none +36.5; micro-batch 2 +8; the fused loss
    frees 6.5. So recompute none + micro-batch 2 needs 104.5 - short by 4.5 - and fits with the fused loss (98)."""
    return 60.0 + RC[c['rc']] + (8.0 if c['mb'] == 2 else 0.0) - (6.5 if c['liger'] else 0.0)


def gain_of(c):
    return sum(GAIN.get((k, v), 0.0) for k, v in c.items() if v != START[k])


class PeakEngine(SimEngine):
    """test_wcs's simulated engine with memory: `mem(cfg)` GiB per GPU on a `cap` GiB card. A configuration above the card
    runs out of memory (at its probe or its full run) with PyTorch's message - its shortfall is its need less the card -,
    one that fits records its peak; the compiler with the fused loss times out (test_wcs's crash pair); the gain is
    test_wcs's GAIN. `patch_mem(overlay)`: the GiB the patches of an overlay add."""

    def __init__(self, mem=mem_a, cap=100.0, *, screens=10, patch_mem=None, ref_peak=None):
        s = SimSession(START, knobs=KNOBS, screens=screens, base_files={PATCH_FILE: BASE_TEXT})
        self.mem_of, self.cap, self.patch_mem = mem, cap, patch_mem or (lambda overlay: 0.0)

        def effect(base, cfg, overlay):
            if cfg['comp'] and cfg['liger']:
                return None, 'timeout'
            if self.need(cfg, overlay) > cap:
                return None, 'oom'
            pg = 0.05 * sum(1 for t in (overlay or {}).values() if '_m_' in t and '_m_slow' not in t)
            bg = 0.05 * sum(1 for t in (base.overlay or {}).values() if '_m_' in t and '_m_slow' not in t)
            return gain_of(cfg) + pg - gain_of(base.config) - bg, None
        super().__init__(s, effect, sigma_runs=[0.01 / math.sqrt(2)])
        s.candidates['c0'].receipt['evidence'] = {'memory_peak_measured_gib': float(ref_peak if ref_peak is not None
                                                                                   else mem(START))}

    def need(self, cfg, overlay=None):
        return self.mem_of(cfg) + self.patch_mem(overlay)

    def evaluate(self, p):
        o = super().evaluate(p)
        c = self.s.candidates.get(o.cid) if o.cid else None
        if c is None:
            return o
        need = self.need(c.config, c.overlay)
        if need > self.cap and o.status in ('infeasible', 'invalid'):
            tried = need - self.cap + 1.5
            msg = (f'completes: exit code 1, 0/30 updates; the log: torch.OutOfMemoryError: CUDA out of memory. Tried to '
                   f'allocate {tried:.2f} GiB. GPU 0 has a total capacity of {self.cap:.2f} GiB of which 1.50 GiB is free. '
                   f'Including non-PyTorch memory, this process has {self.cap - 1.5:.2f} GiB memory in use.')
            return dataclasses.replace(o, reasons=[msg])
        if o.status == 'screened':
            c.verdict['peak_reserved_gib'] = round(need, 3)
        return o


def prior(configs=(('lean', {'liger': True}, 5),), liger=('frees', 5), extra=(), conflicts=()):
    """A guide over test_wcs's KNOBS: recompute half / none and micro-batch 2 declared spenders, the fused loss as given
    (None: unknown memory direction), the rest declared neutral; its whole configurations as given."""
    lv = [{'knob': 'rc', 'value': 'half', 'gain_pct': 8, 'sd_pct': 4, 'memory': {'direction': 'spends', 'gib': 12}},
          {'knob': 'rc', 'value': 'none', 'gain_pct': 22, 'sd_pct': 8, 'memory': {'direction': 'spends', 'gib': 30}},
          {'knob': 'mb', 'value': 2, 'gain_pct': 8, 'sd_pct': 3, 'memory': {'direction': 'spends', 'gib': 8}},
          {'knob': 'liger', 'value': True, 'gain_pct': 5, 'sd_pct': 3,
           'memory': {'direction': liger[0], 'gib': liger[1]} if liger else {'direction': 'unknown', 'gib': None}},
          {'knob': 'comp', 'value': True, 'gain_pct': -5, 'sd_pct': 5, 'memory': {'direction': 'neutral', 'gib': 0}},
          {'knob': 'tf32', 'value': True, 'gain_pct': 0, 'sd_pct': 1, 'memory': {'direction': 'neutral', 'gib': 0}},
          {'knob': 'wk', 'value': 0, 'gain_pct': -1, 'sd_pct': 1, 'memory': {'direction': 'neutral', 'gib': 0}},
          {'knob': 'wk', 'value': 4, 'gain_pct': 0.5, 'sd_pct': 1, 'memory': {'direction': 'neutral', 'gib': 0}},
          *extra]
    return {'levels': lv, 'interactions': [{'members': [{'knob': a, 'value': va}, {'knob': b, 'value': vb}],
                                            'kind': 'conflicts'} for (a, va), (b, vb) in conflicts],
            'configs': [{'name': n, 'change': ch, 'gain_pct': g} for n, ch, g in configs], 'notes': 'a test guide'}


def winner_pack(cfg, pid='run-near', env=NEAR, t=0.75):
    """Another environment's run pack whose best valid knob-only run is `cfg` (its reference at 1.0 s, it at `t`)."""
    return pack(pid, env, [('c0', {}, 1.0), ('c1', cfg, t)])


def bc(eng=None, guide=None, *, packs=None, answers=None, env=TARGET, **kw):
    eng = eng if eng is not None else PeakEngine()
    st = BComp(CodeProposer(guide, answers), facts={'cap_gib': getattr(eng, 'cap', 100.0)}, seed=3, packs=packs,
               env=env, **kw)
    return st, eng


def run(st, eng, q):
    o = eng.evaluate(q)
    st.observe(eng, q, o)
    return o


def step(st, eng):
    """One engine step: the strategy's proposal, run and observed (None: nothing proposed)."""
    props = st.propose(eng, 16)
    if not props:
        return None, None
    q = props[0]
    assert eng._fresh(q)
    return q, run(st, eng, q)


def knobs_of(st, q):
    return {k: q.change[k] for k in st.knobs} if q.absolute else None


def ph(name, mech=None, mu=0.03):
    return {'mechanism': mech or f'patch {name}', 'change': {}, 'patch': {PATCH_FILE: BASE_TEXT + snippet(name)}, 'mu': mu,
            'sigma': 0.02, 'p_feasible': 0.9, 'site': name}


# ---------------------------------------------------------------- the registration, the constants
def test_registration_components_and_the_spec_constants():
    assert 'bcomp' in STRATEGIES and 'bcomp' in LLM_STRATEGIES
    assert set(BCOMP_COMPONENTS) == {'bcomp_ledger', 'bcomp_complete', 'bcomp_challenge', 'bcomp_isolated_blame',
                                     'bcomp_peak_thrash', 'bcomp_feedback', 'bcomp_revise', 'ex_contrarian', 'ex_escape'}
    assert all(ABLATIONS[k].startswith('bcomp: ') for k in BCOMP_COMPONENTS)
    assert (B.STARTS, B.CONFIG_SCREENS, B.COMPLETE_PER_RUN, B.CHALLENGE_MAX) == (3, 2, 2, 2)
    assert B.COMPLETE_MIN_GAIN == 0.02 and B.CHALLENGE_PROBE_BELOW == 0.5 and B.LEDGER_P == 0.5
    st = BComp(CodeProposer())
    assert issubclass(BComp, FinalistBase) and isinstance(st.ladder, FinalistLadder) and st.ladder.short is False
    assert (st.calls_max, st.usd_max, st.patches_per_ask) == (CALLS_MAX, USD_MAX, 3) and st.on['wcs_code']
    assert st.use_ledger and all(st.fb_on.values()) and all(st.bc_on.values())
    off = BComp(CodeProposer(), off=sorted(BCOMP_COMPONENTS))
    assert not off.use_ledger and not off.fb_on['fb_isolated_blame'] and not off.fb_on['fb_peak_thrash']
    assert off.fb_on['fb_s6'] and off.fb_on['fb_transfer'] and not any(off.bc_on.values())


def test_the_knob_side_owns_five_screens_and_the_code_track_the_rest():
    st, eng = bc(guide=prior())
    st._init(eng)
    assert st.bc_knob_screens == 5 and st.protected_screens == frozenset(range(6, 11))
    st6, eng6 = bc(PeakEngine(screens=6), prior())
    st6._init(eng6)
    assert st6.bc_knob_screens == 3 and st6.protected_screens == frozenset({4, 5, 6})


# ---------------------------------------------------------------- (1) the three starts
def test_1_the_three_starts_are_experience_guide_uniform_pairwise_different_the_third_in_full():
    eng = PeakEngine(cap=200.0)                                              # everything fits: the starts' plans alone
    guide = prior(configs=(('aggressive', {'rc': 'none', 'mb': 2, 'liger': True}, 30),))
    st, _ = bc(eng, guide, packs=[winner_pack({'liger': True, 'tf32': True})])
    made = []
    for _ in range(3):
        q, o = step(st, eng)
        made.append((q.hypothesis.split('|')[0], knobs_of(st, q), tuple(q.fidelity), o.fidelity))
    assert [m[0] for m in made] == ['experience', 'guide', 'uniform']
    assert made[0][1] == {**START, 'liger': True, 'tf32': True}            # the source's winner, mapped by name
    assert made[1][1] == {**START, 'rc': 'none', 'mb': 2, 'liger': True}   # the guide's best (two big knobs off start 1)
    assert all(a[1] != b[1] for a, b in itertools.combinations(made, 2))   # pairwise different
    assert made[2][2] == ('full',) and made[2][3] == 'full'                # the uniform start: no probe
    assert [st.bc_slots[s]['state'] for s in B.START_SLOTS] == ['consumed'] * 3 and eng.full_screens() == 3
    view = st.describe()['bcomp']['bcomp']
    assert view['starts']['experience']['source'] == {'card_gib': 141, 'nodes': 1, 'params_b': 25.81}


def test_1_the_uniform_start_is_the_random_arms_first_draw_on_a_real_session():
    """On the memory toy's real session (its own check): the first new draw of knob_space.random_search with the arm's
    seed - the random arm's first screen unless the experience or guide start already ran it."""
    task = mem_tasks()[3]
    eng, sim, sp, facts = toy(task, 10)
    st = BComp(sp(), facts=facts, seed=7)
    fresh, *_ = toy(task, 10)
    draws, _ = _feasible(fresh.s, choices(fresh.s.task), random.Random(7))
    first = list(itertools.islice(draws, 3))
    got = []
    for _ in range(12):
        q, o = step(st, eng)
        if q is None:
            break
        got.append((q.hypothesis.split('|')[0], knobs_of(st, q)))
        if got[-1][0] == 'uniform':
            break
    starts = [cfg for kind, cfg in got if kind in ('experience', 'guide')]
    expected = next(c for c in first if c not in starts)
    assert got[-1] == ('uniform', expected)


def test_1_without_experience_start_one_is_the_guides_best_and_start_two_its_next_far_one():
    guide = prior(configs=(('a', {'rc': 'half', 'mb': 2}, 20), ('b', {'rc': 'half', 'mb': 2, 'wk': 4}, 15),
                           ('c', {'liger': True, 'tf32': True}, 6)))
    st, eng = bc(guide=guide, packs=None)
    q1, _ = step(st, eng)
    q2, _ = step(st, eng)
    assert q1.hypothesis.startswith('experience|') and knobs_of(st, q1) == {**START, 'rc': 'half', 'mb': 2}
    assert q2.hypothesis.startswith('guide|') and knobs_of(st, q2) == {**START, 'liger': True, 'tf32': True}
    assert any('no usable experience source' in n for n in st.bc_notes)   # ('b' moves no big knob off 'a': skipped)


def test_1_experience_comes_from_other_jobs_only_the_most_similar_source_first():
    far = winner_pack({'mb': 2}, pid='run-far', env=(77.3, 2, 35.0))         # weight 1.5
    near = winner_pack({'liger': True, 'tf32': True}, pid='run-near', env=NEAR)   # weight 3
    own = winner_pack({'rc': 'half', 'liger': True}, pid='run-own', env=OWN, t=0.5)   # this job's numbers: never
    st, eng = bc(PeakEngine(cap=200.0), prior(), packs=[far, own, near])
    st._init(eng)
    opts = st._bc_start_options(eng, 'experience')
    assert [o['cfg'] for o in opts] == [{**START, 'liger': True, 'tf32': True}, {**START, 'mb': 2}]
    assert any("this job's own numbers left out" in n for n in st.bc_notes)
    assert 'run-' not in json.dumps(opts)                                    # never a source's name


# ---------------------------------------------------------------- (2) the completion
def _failed_start_then_guide(liger=('frees', 6.5), configs=(('lean', {'liger': True}, 5),), off=()):
    """Start one (the experience source's recompute none + micro-batch 2) runs out of memory at its probe, short by 4.5
    GiB; start two (the guide's) runs in full."""
    st, eng = bc(guide=prior(configs=configs, liger=liger), packs=[winner_pack({'rc': 'none', 'mb': 2})], off=list(off))
    q1, o1 = step(st, eng)
    assert q1.hypothesis.startswith('experience|') and o1.status == 'infeasible' and o1.fidelity == 'probe'
    f = st.failed_material()[0]
    assert f['run'] == o1.cid and f['kind'] == 'oom' and f['short_gib'] == pytest.approx(4.5, abs=1e-3)
    q2, o2 = step(st, eng)
    assert q2.hypothesis.startswith('guide|') and o2.status == 'screened'
    return st, eng, o1.cid


def test_2_a_start_short_by_4_5_gib_and_a_value_measured_to_free_6_5_is_completed_with_it():
    st, eng, fcid = _failed_start_then_guide()
    tag = st.price_tags()[('liger', 'true')]
    assert tag['source'] == 'measured' and tag['gib'] == pytest.approx(-6.5, abs=0.3)   # (the guide's run measured it)
    q = st.complete(eng, eng.incumbent())
    assert q is not None and q.hypothesis.startswith('complete|')
    assert knobs_of(st, q) == {**START, 'rc': 'none', 'mb': 2, 'liger': True}      # F + that value
    assert st.ideas[q.hypothesis] == frozenset(st.derived_ids(fcid))              # F's patches (and the chain)
    rec = st.bc_completions[-1]
    assert rec['run'] == fcid and rec['partner'] == 'liger=True' and rec['tier'] == 'measured'
    assert rec['saving_gib'] == pytest.approx(6.5, abs=0.3) and rec['gap_gib'] == pytest.approx(4.5, abs=1e-3)
    assert st.ledger[q.hypothesis]['derived_from'] == fcid and tuple(q.fidelity)[0] == 'probe'   # (it holds F whole)
    o = run(st, eng, q)
    assert o.status == 'screened' and eng.incumbent() == o.cid                     # 98 GiB: it fits


def test_2_b8_an_unknown_direction_partner_is_proposed_within_two_completions():
    """B.8's gate: a gap of 4.5 GiB and one partner of unknown memory direction (every other value declared) - F + the
    partner within two completions."""
    st, eng, fcid = _failed_start_then_guide(liger=None, configs=(('safe', {'wk': 4}, 1),))
    assert st.price_tags()[('liger', 'true')]['source'] == 'unknown'
    made = []
    for _ in range(2):
        q = st.complete(eng, eng.incumbent())
        if q is None:
            break
        made.append(knobs_of(st, q))
        run(st, eng, q)
    assert {**START, 'rc': 'none', 'mb': 2, 'liger': True} in made
    assert st.bc_completions[0]['tier'] == 'unknown' and st.bc_completions[0]['run'] == fcid


def test_2_two_completions_per_run_the_second_with_another_partner_the_originals_first():
    """mem_c: the fused loss frees 2 (not enough alone: recompute none + micro-batch 2 is short by 4.5), the compiler frees
    6.5 and times out with the fused loss. Completion one: F + the fused loss (unknown direction, the best guide gain)
    runs out of memory; completion two: F again (an original before the failed completion), with another partner; no
    third for F."""
    def mem_c(c):
        return 60.0 + RC[c['rc']] + (8.0 if c['mb'] == 2 else 0.0) - (2.0 if c['liger'] else 0.0) - (6.5 if c['comp'] else 0.0)
    extra = [{'knob': 'comp', 'value': True, 'gain_pct': 2, 'sd_pct': 2, 'memory': {'direction': 'unknown', 'gib': None}}]
    guide = prior(configs=(('safe', {'wk': 4}, 1),), liger=None)
    guide['levels'] = [r for r in guide['levels'] if r['knob'] != 'comp'] + extra
    # v1 behaviour pinned with the v2 part off (bcomp_feedback: M2 would require a saver covering the gap after two OOMs)
    st, eng = bc(PeakEngine(mem_c), guide, packs=[winner_pack({'rc': 'none', 'mb': 2})], off=['bcomp_feedback'])
    step(st, eng)                                                            # F at its probe
    fcid = st.failed_material()[0]['run']
    step(st, eng)                                                            # the guide's start
    st.fb_oom_streak = 0                                                     # (S6 is not under test here)
    q1 = st.complete(eng, eng.incumbent())
    assert st.bc_completions[-1]['partner'] == 'liger=True'
    o1 = run(st, eng, q1)
    assert o1.status == 'infeasible' and st.runs[o1.cid]['kind'] == 'oom'      # 102.5 GiB: still out of memory
    st.fb_oom_streak = 0
    q2 = st.complete(eng, eng.incumbent())
    assert st.bc_completions[-1]['run'] == fcid and st.bc_completions[-1]['partner'] != 'liger=True'
    run(st, eng, q2)
    assert len(st._bc_done(st.bc_completions, run=fcid)) == B.COMPLETE_PER_RUN
    assert all(f['run'] != fcid for f in st._bc_failures(eng, eng.incumbent()))   # no third for F


def test_2_a_price_ranks_and_never_vetoes_and_a_slower_completion_is_not_proposed():
    st, eng, fcid = _failed_start_then_guide()
    f = next(m for m in st._bc_failures(eng, eng.incumbent()) if m['run'] == fcid)
    big = dict(f, gap_gib=9.0)                                               # a gap the measured 6.5 GiB does not cover:
    first = st._bc_partners(big)[0]                                          # still the first partner (BCOMP_r1 P1; the
    assert (first['knob'], first['tier']) == ('liger', 'measured')           # ledger and the probe decide the rest)
    parts = st._bc_partners(f)
    assert parts and all(f['cfg'][p['knob']] == START[p['knob']] for p in parts)   # F + a partner keeps F's values:
    assert not any(p['knob'] in ('rc', 'mb') for p in parts)                 # never one of F's own knobs changed
    st.surr.mean = lambda: {('rc', '"none"'): 0.2, ('mb', '2'): 0.08, ('liger', 'true'): -0.5}
    assert st._bc_complete_q(eng, 'c0', f, {'knob': 'liger', 'vk': 'true', 'value': True,      # predicted slower than
                                            'tier': 'measured', 'saving_gib': 6.5}) is None   # the incumbent: none


def test_2_bcomp_r1_an_under_declared_saver_is_completed_and_the_completion_fits():
    """BCOMP_r1 P1's scenario: the fused loss really frees 6.5 GiB but the guide declares 2.2, F is short by 4.5 and the
    guide's start (the loader) leaves the fused loss unmeasured: it is still F's partner (a price ranks, it never vetoes),
    the ledger admits F + it (P about 0.25, a probe first), and the completion fits (98 GiB) and becomes the incumbent."""
    st, eng, fcid = _failed_start_then_guide(liger=('frees', 2.2), configs=(('safe', {'wk': 4}, 1),))
    tag = st.price_tags()[('liger', 'true')]
    assert tag['source'] == 'declared' and tag['gib'] == pytest.approx(-2.2, abs=0.05)
    q = st.complete(eng, eng.incumbent())
    assert q is not None and knobs_of(st, q) == {**START, 'rc': 'none', 'mb': 2, 'liger': True}
    rec = st.bc_completions[-1]
    assert rec['run'] == fcid and rec['tier'] == 'declared' and rec['saving_gib'] < rec['gap_gib']
    assert 0.15 <= q.p_feasible < 0.7 and tuple(q.fidelity) == ('probe', 'full')
    o = run(st, eng, q)
    assert o.status == 'screened' and eng.incumbent() == o.cid and eng.need(knobs_of(st, q)) == pytest.approx(98.0)


def test_2_bcomp_r1_a_frees_declaration_without_gib_is_an_unpriced_partner():
    """BCOMP_r1 P1: a value the guide declares to free memory without saying how much (direction frees, gib None) is a
    partner of its own tier (before the values of unknown direction) - completed when the ledger admits it."""
    st, eng, fcid = _failed_start_then_guide(liger=('frees', None), configs=(('safe', {'wk': 4}, 1),))
    assert st.prior.levels[('liger', 'true')]['mem_dir'] == 'frees' and st.prior.levels[('liger', 'true')]['mem_gib'] is None
    f = next(m for m in st._bc_failures(eng, eng.incumbent()) if m['run'] == fcid)
    parts = st._bc_partners(f)
    assert parts[0]['knob'] == 'liger' and parts[0]['tier'] == 'declared, no price'
    q = st.complete(eng, eng.incumbent())
    assert q is not None and knobs_of(st, q) == {**START, 'rc': 'none', 'mb': 2, 'liger': True}
    assert st.bc_completions[-1]['tier'] == 'declared, no price' and 'no price given' in st.ledger[q.hypothesis]['why_feasible']


def test_2_a_completion_the_runs_rule_out_is_not_proposed_and_a_conflicting_value_goes_back():
    st, eng, fcid = _failed_start_then_guide()
    f = next(m for m in st._bc_failures(eng, eng.incumbent()) if m['run'] == fcid)
    st.pairs[frozenset({('liger', 'true'), ('mb', '2')})] = {'p': 0.05, 'why': 'failed to start together'}
    q = st._bc_complete_q(eng, eng.incumbent(), f, {'knob': 'liger', 'vk': 'true', 'value': True, 'tier': 'measured',
                                                    'saving_gib': 6.5})
    assert q is not None and knobs_of(st, q) == {**START, 'rc': 'none', 'liger': True}   # micro-batch 2 back (F5c)
    assert st.bc_completions[-1]['swapped'] == ['mb=2']
    real = st._pf
    st._pf = lambda cfg, ids=(): (0.1, 'planted')                            # below the declared floor: the runs say no
    assert st._bc_complete_q(eng, eng.incumbent(), f, {'knob': 'tf32', 'vk': 'true', 'value': True,
                                                        'tier': 'unknown'}) is None
    st._pf = real


# ---------------------------------------------------------------- (3) declared memory: the floor, then the measurement
def test_3_a_value_declared_to_spend_13_gib_keeps_p_at_least_015_until_a_peak_is_measured():
    extra = [{'knob': 'tf32', 'value': True, 'gain_pct': 3, 'sd_pct': 2, 'memory': {'direction': 'spends', 'gib': 13}}]
    guide = prior()
    guide['levels'] = [r for r in guide['levels'] if r['knob'] != 'tf32'] + extra
    st, eng = bc(PeakEngine(cap=40.0, ref_peak=35.0, mem=lambda c: 35.0 + (13.0 if c['tf32'] else 0.0)), guide)
    st._init(eng)
    cfg = {**START, 'tf32': True}
    assert st.mledger.p_feasible(cfg) < P_FLOOR_DECLARED and st._pf(cfg)[0] == pytest.approx(P_FLOOR_DECLARED)
    sup = {**cfg, 'wk': 4}
    st.mledger.observe_run(cfg, 'ok', peak=37.0, source='cX')               # one measured peak: it decides
    assert st._pf(sup)[0] == pytest.approx(st.mledger.p_feasible(sup)) and st._pf(sup)[0] > 0.9
    st2, eng2 = bc(PeakEngine(cap=40.0, ref_peak=35.0), guide)
    st2._init(eng2)
    st2.mledger.observe_run(cfg, 'oom', oom={'need_gib': 48.0}, source='cY')
    assert st2._pf(sup)[0] == pytest.approx(st2.mledger.p_feasible(sup)) and st2._pf(sup)[0] < P_FLOOR_DECLARED


# ---------------------------------------------------------------- (4) isolated_blame by combination
def test_4_three_ooms_sharing_a_value_whose_others_never_fitted_leave_its_p_unchanged():
    def mem_mb(c):                                                           # micro-batch 2 with any other change: over
        return 60.0 + (45.0 if c['mb'] == 2 and (c['rc'] != 'full' or c['tf32']) else 0.0)
    st, eng = bc(PeakEngine(mem_mb), prior())
    st._init(eng)
    st.fb_on['fb_s6'] = False                                                # (three in a row on purpose)
    for extra in ({'rc': 'half'}, {'rc': 'none'}, {'tf32': True}):
        o = run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, **extra}, kind='t', plan=('full',), p=0.9))
        assert st.runs[o.cid]['kind'] == 'oom'
    probe = {**START, 'mb': 2, 'wk': 4}
    assert st.isolated_blame() == {}
    on = st._pf(probe)[0]
    st.fb_on['fb_isolated_blame'] = False
    assert st._pf(probe)[0] == pytest.approx(on)                             # unchanged by the rule
    assert ('mb', '2') in WCS._blocked_values(st)                            # (wcs's rule would have lowered it)
    off = BComp(CodeProposer(), off=['bcomp_isolated_blame'])
    assert not off.fb_on['fb_isolated_blame']
    st.fb_on['fb_isolated_blame'] = True
    for extra in ({'rc': 'half'}, {'rc': 'none'}, {'tf32': True}):          # the others fit alone: now it is blamed
        run(st, eng, st._knob_q(eng, 'c0', {**START, **extra}, kind='t', plan=('full',), p=0.9))
    assert set(st.isolated_blame()) == {('mb', '2')}
    p = st._pf(probe)[0]
    st.fb_on['fb_isolated_blame'] = False
    assert p == pytest.approx(BLOCKED_VALUE * st._pf(probe)[0])


# ---------------------------------------------------------------- (5) the challenge
def test_5_the_challenge_changes_one_switch_group_to_the_largest_vetoed_value_no_run_holds():
    extra = [{'knob': 'tf32', 'value': True, 'gain_pct': 4, 'sd_pct': 2, 'p_runs': 0.2,     # declared not to fit
              'memory': {'direction': 'neutral', 'gib': 0}}]
    guide = prior(configs=(), conflicts=[(('liger', True), ('wk', 4))])     # the fused loss declared to conflict
    guide['levels'] = [r for r in guide['levels'] if r['knob'] != 'tf32'] + extra
    for r in guide['levels']:
        if r['knob'] == 'comp':
            r['gain_pct'] = -9                                               # the largest |declared gain| of the vetoed
    st, eng = bc(PeakEngine(cap=200.0), guide)
    st._init(eng)
    q = st.challenge(eng, 'c0')
    assert q is not None and q.hypothesis.startswith('challenge|') and knobs_of(st, q) == {**START, 'comp': True}
    rec = st.bc_challenges[-1]
    assert rec['changed'] == ['comp'] and rec['vetoed'].startswith('the guide declares -9')
    assert tuple(q.fidelity) == ('full',)                                    # in full: P >= 0.5
    run(st, eng, q)
    q2 = st.challenge(eng, eng.incumbent())                                  # the next: the fused loss (5 %), then tf32
    assert st.bc_challenges[-1]['value'] == 'liger=True' and st.bc_challenges[-1]['vetoed'] == 'the guide declares a conflict'
    run(st, eng, q2)
    assert st.challenge(eng, eng.incumbent()) is None                        # CHALLENGE_MAX = 2
    assert not any(r['value'] == 'wk=0' for r in st.bc_challenges)          # (-1 %: within the noise, no challenge)
    st2, eng2 = bc(PeakEngine(cap=200.0), guide)
    st2._init(eng2)
    run(st2, eng2, st2._knob_q(eng2, 'c0', {**START, 'comp': True, 'wk': 0}, kind='t', plan=('full',), p=0.9))
    st2.challenge(eng2, eng2.incumbent())
    assert st2.bc_challenges[-1]['value'] == 'liger=True'                    # the compiler ran: no longer a challenge


def test_5_a_challenge_below_p_05_is_probed_first_and_skipped_without_a_probe():
    guide = prior(configs=())
    st, eng = bc(PeakEngine(cap=200.0), guide)
    st._init(eng)
    real = st._pf
    st._pf = lambda cfg, ids=(): (0.3, 'planted') if cfg.get('comp') else real(cfg, ids)
    q = st.challenge(eng, 'c0')
    assert knobs_of(st, q) == {**START, 'comp': True} and tuple(q.fidelity) == ('probe', 'full')
    st.bc_challenges.clear()
    st.tried.clear()
    st.ladder._fund = lambda engine, cost, code=False: False                # no probe left: below 0.5, not in full
    q = st.challenge(eng, 'c0')
    assert q is None or knobs_of(st, q) != {**START, 'comp': True}


def test_5_bcomp_r1_a_small_claimed_loss_never_hides_a_conflict_or_a_memory_veto():
    """BCOMP_r1 P2: each veto reason on its own. (1) The compiler declared at -1 % (within the noise: no gain veto) and to
    run with P 0.2 (declared not to fit) is challenged for the memory veto; (2) the loader at 0 workers (-1 %) and tf32
    (0 %), declared to conflict, are challenged for the conflict, the larger |claim| first; the loader at -1 % alone is
    never a challenge (the first instance)."""
    g1 = prior(configs=())
    for r in g1['levels']:
        if r['knob'] == 'comp':
            r.update(gain_pct=-1, p_runs=0.2)
    st, eng = bc(PeakEngine(cap=200.0), g1)
    st._init(eng)
    q = st.challenge(eng, 'c0')
    assert q is not None and knobs_of(st, q) == {**START, 'comp': True}
    assert st.bc_challenges[-1]['vetoed'] == 'the guide declares it does not fit'
    run(st, eng, q)
    assert st.challenge(eng, eng.incumbent()) is None                        # (wk=0's -1 % alone: no challenge)
    g2 = prior(configs=(), conflicts=[(('tf32', True), ('wk', 0))])
    for r in g2['levels']:
        if r['knob'] == 'comp':
            r['gain_pct'] = 0                                                # (no veto of its own)
    st2, eng2 = bc(PeakEngine(cap=200.0), g2)
    st2._init(eng2)
    run(st2, eng2, st2.challenge(eng2, 'c0'))
    run(st2, eng2, st2.challenge(eng2, eng2.incumbent()))
    assert [r['value'] for r in st2.bc_challenges] == ['wk=0', 'tf32=True']
    assert all(r['vetoed'] == 'the guide declares a conflict' and r['changed'] == [r['value'].split('=')[0]]
               for r in st2.bc_challenges)


# ---------------------------------------------------------------- (6) probe thrash by its peak
def _edge_task():
    rows = [dict(zip(MEM_KNOBS, v)) for v in itertools.product(*MEM_KNOBS.values())]
    for t in mem_tasks():
        if t.antagonism:
            continue
        sim = make_sim(t)
        hi = [c for c in rows if THRASH_PEAK * sim.cap <= sim.memory(c) < sim.cap]
        if hi:
            return t, hi
    raise AssertionError('no edge task')


def test_6_a_probe_at_95_percent_of_the_card_is_thrash_and_not_extended():
    task, hi = _edge_task()
    eng, sim, sp, facts = toy(task, 10)
    st = BComp(sp(), facts=facts, seed=1)
    st._init(eng)
    o = run(st, eng, st._knob_q(eng, 'c0', hi[0], kind='t', plan=('probe', 'full'), p=0.5))
    assert o.status == 'infeasible' and o.fidelity == 'probe' and o.failed_checks == ['probe_thrash']
    assert eng.full_screens() == 0 and st.runs[o.cid]['kind'] == 'thrash' and st.fb_thrash[0]['share'] >= THRASH_PEAK
    m = st.failed_material()[0]
    assert m['kind'] == 'thrash' and m['probe_peak'] and st._bc_gap(m) == pytest.approx(m['gap_gib'])
    off = BComp(sp(), facts=facts, seed=1, off=['bcomp_peak_thrash'])
    eng2, *_ = toy(task, 10)
    off._init(eng2)
    o2 = run(off, eng2, off._knob_q(eng2, 'c0', hi[0], kind='t', plan=('probe', 'full'), p=0.5))
    assert 'probe_thrash' not in (o2.failed_checks or []) and off.fb_thrash == []


# ---------------------------------------------------------------- (7) the code track within the budget
def test_7_the_code_asks_go_on_within_the_budget_and_stop_at_its_top():
    """Every patch the model writes fails at its probe (no screen spent): the asks go on, one per run, past wcs's three,
    until the ten calls are spent (the prior's included) - then none, and the code screens are filled by the knob side.
    Base f1.1 (F1 (b)): the code-probe pool pays the first four proofs, the allowance's code share (and its borrow) the
    rest, as in f1.0."""
    def effect(base, cfg, overlay):
        if any('_m_bad' in t for t in (overlay or {}).values()):
            return None, 'crash'
        return gain_of(cfg) - gain_of(base.config), None
    eng = engine(effect=effect)
    st = BComp(CodeProposer(prior(), answers=lambda ctx, n: [ph(f'bad{n}')]), facts={'cap_gib': 100.0}, seed=3)
    trace = eng.run(st)
    assert eng.full_screens() == 10 and trace.stop_reason.startswith('budget')
    assert st.proposer.calls == st._llm_used() == CALLS_MAX                  # the prior + every ask, to the top
    assert st.asks == CALLS_MAX - 1 > WCS_CODE_ASKS and not st._budget_ok(eng)
    assert [x['kind'] for x in st.ladder.code_log] == ['patch'] * CODE_PROBES_MAX      # the pool's proofs first, ...
    assert len(st.ladder.code_share_log) == st.asks - CODE_PROBES_MAX >= 1  # ... then the code share's
    assert [a['why'] for a in st.fb_asks].count(None) == 0
    n_last = st.fb_asks[-1]['screens_done']
    assert all(b['screens_done'] <= n_last or b['pick'] != 'code ask' for b in st.bc_steps)   # none after the top


def test_7_the_knob_sides_empty_screen_goes_to_code_and_the_code_screens_are_protected(monkeypatch):
    st, eng = bc(guide=None, answers=lambda ctx, n: [ph(f'k{n}')])          # a flat guide: nothing but the uniform start
    q, _ = step(st, eng)
    assert q.hypothesis.startswith('uniform|') and st.bc_slots['experience']['state'] == 'none'
    q, _ = step(st, eng)                                                     # screen 2: the knob side has nothing
    assert q.hypothesis.startswith('patch|') and st.fb_asks[-1]['why'].startswith('no knob candidate')
    st2, eng2 = bc(PeakEngine(cap=200.0), prior(), answers=lambda ctx, n: [ph(f'm{n}')])
    st2._init(eng2)
    eng2.s.ledger.used['screens'] = 5                                        # the code screens: an ask there whatever
    monkeypatch.setattr(FinalistBase, '_knob_strong', lambda self, engine, inc: {'predicted_gain_pct': 9.0})
    st2.fb_runs = 1
    q = st2._bc_step(eng2, 'c0', 5)
    assert q.hypothesis.startswith('patch|') and 'protected' in st2.fb_asks[-1]['why']


# ---------------------------------------------------------------- (8) a patch-caused out-of-memory run, completed
def test_8_the_completion_of_a_patch_caused_oom_carries_that_patch():
    eng = PeakEngine(cap=100.0, patch_mem=lambda ov: 48.0 if any('_m_big' in t for t in (ov or {}).values()) else 0.0)
    st, _ = bc(eng, prior(configs=(), liger=('frees', 10)))              # (the declared saver frees more than the 8 short)
    st._init(eng)
    qa = st._patch_proposal(eng, 'c0', ph('big', mech='a big fused kernel', mu=0.2))
    oa = run(st, eng, qa)
    assert oa.status == 'infeasible' and oa.fidelity == 'probe' and st.runs[oa.cid]['kind'] == 'oom'
    pid = qa.hypothesis.split('|')[1]
    assert st.failed_material()[0]['pids'] == [pid] and st.chain == []
    q = st.complete(eng, 'c0')
    assert q is not None and st.ideas[q.hypothesis] == frozenset({pid})      # the patch carried (S3)
    assert '_m_big()' in q.patch[PATCH_FILE] and knobs_of(st, q)['liger'] is True   # with the declared saver
    assert st.bc_completions[-1]['patches'] == [pid] and st.ledger[q.hypothesis]['proof_owed'] == [pid]
    assert tuple(q.fidelity)[0] == 'probe'                                   # its patch proves itself first


def test_8_f2_a_patch_carrying_completion_takes_its_probe_from_the_code_pool_when_the_knob_allowance_is_spent():
    """FINALISTS_IMPL_SPEC B.9 F2 (base f1.1): a completion carrying an unproven patch owes that patch's proof, which the
    code pool pays - so it is proposed, with its probe, although the knob side's allowance is spent (before F2 bcomp's own
    check dropped it below P 0.7); its probe is logged in the pool, not charged to the allowance."""
    eng = PeakEngine(cap=100.0, patch_mem=lambda ov: 44.0 if any('_m_big' in t for t in (ov or {}).values()) else 0.0)
    st, _ = bc(eng, prior(configs=(), liger=('frees', 10)))
    st._init(eng)
    qa = st._patch_proposal(eng, 'c0', ph('big', mech='a big fused kernel', mu=0.2))
    oa = run(st, eng, qa)
    pid = qa.hypothesis.split('|')[1]
    assert st.runs[oa.cid]['kind'] == 'oom' and [x['kind'] for x in st.ladder.code_log] == ['patch']
    st.ladder.lowfi_share = 0.0                                              # the knob side's allowance: none left
    assert st.ladder.pool and not st.ladder.affordable(eng, 'probe') and st.ladder.code_probe_ok(eng)
    q = st.complete(eng, 'c0')
    assert q is not None and st.ideas[q.hypothesis] == frozenset({pid}) and knobs_of(st, q)['liger'] is True
    assert tuple(q.fidelity) == ('probe', 'full') and st.ledger[q.hypothesis]['probe_pool'] == 'code'
    assert q.p_feasible < 0.7                                                # (the case the old check dropped)
    o = run(st, eng, q)                                                      # 60 + 44 - 6.5 GiB: it fits
    assert o.status == 'screened' and [x['kind'] for x in st.ladder.code_log] == ['patch', 'derived']
    assert st.ladder.spent(eng) == pytest.approx(0.0)                        # (nothing charged to the knob side)
    ids = (pid,)                                                             # the pool spent: as derived_q plans it, the
    for i in range(st.ladder.code_probes_left()):                            # knob side's allowance pays the proof
        st.ladder.note_code_probe(f'cx{i}', f'patch|x{i}', 0.0, 'patch')
    assert st.ladder.code_probes_left() == 0 and not st._bc_probe_payable(eng, ids, 0.45)   # (neither has one)
    st.ladder.lowfi_share = 0.5
    assert st.ladder.affordable(eng, 'probe') and st._bc_probe_payable(eng, ids, 0.45)


def test_2_f2_a_p06_completion_runs_in_full_where_a_probe_costs_over_06_and_is_probed_where_it_costs_less():
    """FINALISTS_IMPL_SPEC B.9 F2 (base f1.1's ratio rule): a knob-only completion at P 0.6 - its run's conflicting value
    back to the reference's (F5c), so it does not hold that run's whole failed combination - runs in full where a probe
    costs more than 0.6 of a full run (probe_below 0.5), even with the knob side's allowance spent; where a probe costs
    0.6 or less (probe_below 0.7) it is probed first. A completion that holds its run whole is still probed (hard rule
    5, the base's), whatever the ratio."""
    def at(ratio, *, swap=True, spent=False):
        st, eng, fcid = _failed_start_then_guide()
        full = next(r['gpu_hours'] for r in eng.s.screens if r.get('fidelity') == 'full' and r.get('valid'))
        eng.s.screens.append({'fidelity': 'probe', 'gpu_hours': ratio * full, 'valid': True})   # the job's probe / full
        f = next(m for m in st._bc_failures(eng, eng.incumbent()) if m['run'] == fcid)
        if swap:
            st.pairs[frozenset({('liger', 'true'), ('mb', '2')})] = {'p': 0.05, 'why': 'failed to start together'}
        real = st._pf
        st._pf = lambda cfg, ids=(): (0.6, 'planted') if cfg.get('liger') else real(cfg, ids)
        if spent:
            st.ladder.lowfi_share = 0.0
        q = st._bc_complete_q(eng, eng.incumbent(), f, {'knob': 'liger', 'vk': 'true', 'value': True,
                                                         'tier': 'measured', 'saving_gib': 6.5})
        return st, eng, q
    st, eng, q = at(0.8)
    assert st.ladder.probe_ratio()[0] == pytest.approx(0.8) and st.ladder.probe_below == 0.5
    assert knobs_of(st, q) == {**START, 'rc': 'none', 'liger': True} and tuple(q.fidelity) == ('full',)
    st, eng, q = at(0.8, spent=True)                                         # the allowance spent: still in full
    assert q is not None and tuple(q.fidelity) == ('full',) and not st.ladder.affordable(eng, 'probe')
    st, eng, q = at(0.5)
    assert st.ladder.probe_below == 0.7 and tuple(q.fidelity) == ('probe', 'full')
    st, eng, q = at(0.8, swap=False)                                         # F whole: hard rule 5's probe
    assert knobs_of(st, q) == {**START, 'rc': 'none', 'mb': 2, 'liger': True} and tuple(q.fidelity) == ('probe', 'full')
    assert st.fb_counts['combo_probed'] >= 1


# ---------------------------------------------------------------- the ledger's best, the configuration order
def test_the_ledger_best_takes_p_at_least_05_and_a_big_knob_off_every_run():
    st, eng = bc(PeakEngine(cap=200.0), prior(configs=()))
    st._init(eng)
    q = st._bc_ledger_best(eng, 'c0')
    rec = st.bc_by_hid[q.hypothesis]
    assert rec['what'] == 'ledger' and q.p_feasible >= B.LEDGER_P and rec['predicted_gain_pct'] > 0
    big = st._big_knobs()
    cfg = knobs_of(st, q)
    assert all(any(cfg[k] != (r['cfg'] or {}).get(k) for k in big) for r in st.runs.values())
    real = st._pf
    st._pf = lambda c, ids=(): (0.45, 'planted')
    assert st._bc_ledger_best(eng, 'c0') is None                             # nothing at P >= 0.5
    st._pf = real


def test_the_configuration_screens_alternate_completion_and_challenge_then_the_ledgers_best():
    st, eng, fcid = _failed_start_then_guide(off=['ex_contrarian'])         # v1 pinned, v2 part off (M1 takes the slot)
    st.bc_slots['uniform']['state'] = 'none'
    q = st._bc_knob(eng, eng.incumbent())
    assert q.hypothesis.startswith('complete|')                              # screen 4: a completion first
    run(st, eng, q)
    assert st.bc_config == ['complete']
    q = st._bc_knob(eng, eng.incumbent())
    assert q.hypothesis.startswith('challenge|')                             # screen 5: the other kind
    st.bc_on.update(bcomp_complete=False, bcomp_challenge=False)
    q = st._bc_knob(eng, eng.incumbent())
    assert q is None or q.hypothesis.startswith('ledger|')


# ---------------------------------------------------------------- S6 over whole runs (B.7 D5)
def test_s6_no_knob_candidate_below_09_follows_two_ooms_but_the_uniform_start():
    """Every micro-batch-2 configuration runs out of memory and the experience and guide starts hold it: every knob
    candidate proposed while two knob-side runs in a row had run out of memory had P(fits) >= 0.9 - the uniform start
    alone excepted - and a third out-of-memory knob run in a row came only from such a candidate."""
    def mem_s(c):
        return 60.0 + (50.0 if c['mb'] == 2 else 0.0) + RC[c['rc']] * 0.2
    guide = prior(configs=(('big', {'mb': 2, 'rc': 'half'}, 20),))
    st, eng = bc(PeakEngine(mem_s), guide, packs=[winner_pack({'mb': 2, 'wk': 4})], answers=lambda ctx, n: [ph(f's{n}')])
    made, real = [], BComp._knob_q

    def knob_q(self, engine, inc, cfg, **kw):
        active, p = self.s6_active(), self._pf(cfg, tuple(self.chain))[0]
        q = real(self, engine, inc, cfg, **kw)
        if q is not None:
            made.append((active, p, bool(kw.get('s6_exempt'))))
        return q
    BComp._knob_q = knob_q
    try:
        eng.run(st)
    finally:
        BComp._knob_q = real
    gated = [(p, exempt) for active, p, exempt in made if active]
    assert gated and all(p >= S6_P or exempt for p, exempt in gated), gated
    assert st.fb_oom_streak_max >= 2 and eng.full_screens() == 10


def test_s6_the_counter_holds_a_start_below_09_open_runs_the_uniform_start_and_offers_it_again():
    """B.7 D5 at bcomp's starts: two knob-side runs in a row out of memory (the counter at two) - the experience and guide
    starts (P below 0.9) are refused and stay open, the uniform start runs in full regardless; a knob-side run that fits
    resets the counter, and the held experience start is offered again."""
    def mem_s(c):
        return 60.0 + (45.0 if c['mb'] == 2 and c['tf32'] else 0.0) + RC[c['rc']]
    guide = prior(configs=(('deep', {'rc': 'none'}, 22),))
    # v1 behaviour pinned with the v2 parts off (bcomp_feedback: M2's needs-a-saver rule; ex_escape: M4's stall gate)
    st, eng = bc(PeakEngine(mem_s), guide, packs=[winner_pack({'rc': 'none', 'mb': 2})], off=['bcomp_feedback', 'ex_escape'])
    st._init(eng)
    for extra in ({'wk': 0}, {'wk': 4}):                                     # two knob-side OOMs: the counter at two
        run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, 'tf32': True, **extra}, kind='t', plan=('full',), p=0.9))
    assert st.fb_oom_streak == 2 and st.s6_active()
    p1 = st._pf({**START, 'rc': 'none', 'mb': 2})[0]
    assert p1 < S6_P
    q, o = step(st, eng)
    assert q.hypothesis.startswith('uniform|') and tuple(q.fidelity) == ('full',)   # the S6-exempt start, in full
    assert st.bc_slots['experience']['state'] == 'open' and st.bc_slots['guide']['state'] == 'open'
    assert st.fb_counts['s6_refused'] >= 2
    if st.s6_active():                                                       # (the draw ran out of memory too: end the
        run(st, eng, st._knob_q(eng, 'c0', {**START, 'wk': 4}, kind='t', plan=('full',), p=0.99))   # streak by a fit)
    assert not st.s6_active()
    q, _ = step(st, eng)
    assert q.hypothesis.startswith('experience|') and knobs_of(st, q) == {**START, 'rc': 'none', 'mb': 2}


# ---------------------------------------------------------------- the engine-level run and its accounting
def bcomp_llm_toy(script, *, screens=10, usd_per_call=1.0, off=(), packs=None, task=None, retries=0):
    """bcomp with an LLMProposer over a priced scripted backend (each call `usd_per_call`, capped by what the budget
    gives it; `retries`: the backend's own re-asks per call, every one it is granted used and counted as an attempt) on a
    memory toy, run for `screens` full screens."""
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.agent.budget import VirtualClock
    from infrarsi.search import toybench as tb
    from infrarsi.search.proposers import LLMProposer
    task = task or mem_tasks()[1]
    sim = tb.make_sim(task)
    clock = VirtualClock()
    runner = tb.ToyRunner(sim, clock)
    ref_cfg = sim.reference()
    probe_task = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                             'gpu_hours': 1e9}))
    ref = runner.run(probe_task, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=tb.FULL)
    c_full = ref['cost']['gpu_seconds'] / 3600
    env = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': screens, 'llm_tokens': 10 ** 7, 'wall_seconds': 1e12,
                                                      'gpu_hours': 4 * screens * c_full}))
    hooks = tb.MemoryHooks(env, sim.memory_facts, base_files={'patches.txt': ''})
    s = tb.SearchSession(task=env, budget=env.search_budget(), runner=runner, judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']),
                         run_id='toy-b', arm='b', start_config=ref_cfg, start_receipt=ref, reference_receipt=ref, hooks=hooks,
                         clock=clock, stop_after_same_failures=10 ** 6)
    eng = tb.Engine(s, full_steps=tb.FULL, short_steps=tb.SHORT, probe_steps=tb.PROBE, job_gpu_hours=None, multi_fidelity=True)
    seen = []

    def wrapped(role, system, prompt):
        payload = json.loads(prompt)
        seen.append((role, payload, s.ledger.used['screens']))
        return script(role, system, payload)

    class Priced(ScriptedBackend):
        def __init__(self):
            super().__init__(wrapped, tokens_per_call=2000)
            self.max_retries, self.kw, self.attempts_made, self.granted = retries, [], 0, []

        def complete(self, system, prompt, **kwargs):
            self.kw.append(dict(kwargs))
            res = super().complete(system, prompt, **{k: v for k, v in kwargs.items() if k in ('schema', 'call_dir', 'role',
                                                                                               'expect_json')})
            n = 1 + int(self.max_retries or 0)           # (every re-ask it was granted, used)
            self.granted.append((kwargs.get('role'), int(self.max_retries or 0)))
            self.attempts_made += n
            cap = kwargs.get('max_usd')
            return dataclasses.replace(res, llm_usd=usd_per_call if cap is None else min(usd_per_call, float(cap)), attempts=n)
    be = Priced()
    prop = LLMProposer(be, hooks, search_calibrates=True, features=BComp.proposer_features(off))
    st = BComp(prop, facts=sim.memory_facts, seed=1, off=off, packs=packs, env=tb.toy_env(sim))
    trace = eng.run(st)
    return SimpleNamespace(st=st, s=s, be=be, seen=seen, trace=trace, eng=eng, sim=sim)


def _toy_script():
    state = {'n': 0}

    def script(role, system, payload):
        if role == 'prior':
            return {'levels': [{'knob': 'fused_loss', 'value': True, 'gain_pct': 2, 'sd_pct': 2,
                                'memory': {'direction': 'frees', 'gib': 3}},
                               {'knob': 'micro_batch', 'value': 2, 'gain_pct': 8, 'sd_pct': 3,
                                'memory': {'direction': 'spends', 'gib': 10}},
                               {'knob': 'compile', 'value': True, 'gain_pct': -4, 'sd_pct': 3}],
                    'interactions': [], 'configs': [{'name': 'a', 'change': {'micro_batch': 2, 'fused_loss': True},
                                                     'gain_pct': 9}], 'notes': ''}
        if role == 'code_review':
            return {'verdict': 'patch', 'summary': 'x', 'findings': []}
        state['n'] += 1
        n = state['n']
        return {'hypotheses': [{'mechanism': f'patch x{n}a', 'change': {}, 'patch': {'patches.txt': f'x{n}a\nproof:x{n}a\n'},
                                'predicted_gain_pct': 3, 'site': f'x{n}a'},
                               {'mechanism': f'patch x{n}b', 'change': {}, 'patch': {'patches.txt': f'x{n}b\n'},
                                'predicted_gain_pct': 3, 'site': f'x{n}b'}]}
    return script


def test_ten_screens_on_the_memory_toy_and_their_accounting():
    """bcomp end to end on the memory toy with a priced scripted model (0.5 USD a call) and another task's packs: ten full
    screens, probes not counted as screens, no short run, every backend attempt within the ten calls and every dollar
    within the eight (each call capped), the starts first, the code track on the protected screens, the low-fidelity
    account within its allowance (base f1.1: the code-probe pool apart, in three lines)."""
    from infrarsi.search import toybench as tb
    task = mem_tasks()[1]
    r = bcomp_llm_toy(_toy_script(), usd_per_call=0.5, packs=tb.toy_packs(task, 2), task=task)
    st, s, be, seen = r.st, r.s, r.be, r.seen
    d = st.describe()['bcomp']
    fid = [x.get('fidelity') for x in s.screens]
    assert r.trace.stop_reason.startswith('budget') and s.ledger.used['screens'] == 10 == fid.count('full')
    assert fid.count('short') == 0 and fid.count('probe') >= 1
    assert st._llm_used() == be.attempts_made == d['llm_calls'] <= CALLS_MAX
    assert s.ledger.used['llm_usd'] <= USD_MAX + 1e-9 and all(k['max_usd'] <= USD_MAX + 1e-9 for k in be.kw)
    roles = [x[0] for x in seen]
    assert roles[0] == 'prior' and roles.count('prior') == 1 and 'configs' not in roles   # (no configuration call)
    assert [st.bc_slots[x]['state'] for x in B.START_SLOTS] == ['consumed'] * 3
    order = [x['pick'] for x in d['bcomp']['steps'] if x['pick'] in B.START_SLOTS]
    assert order == list(B.START_SLOTS)
    hyp = [(p, n) for role, p, n in seen if role == 'hypotheses']
    assert hyp and all('measured_history' in p for p, _ in hyp)
    acc = st.ladder.describe(r.eng)['account']                              # f1.1: pool, code share, knob side
    probe = max(x.get('gpu_hours') or 0.0 for x in s.screens if x.get('fidelity') == 'probe')
    inside = acc['knob_lowfi_gpu_hours'] + acc['code_share_gpu_hours']      # (the pool's probes are outside the allowance)
    assert inside <= acc['allowance_gpu_hours'] + probe
    assert acc['lowfi_gpu_hours'] == pytest.approx(inside + acc['code_pool_gpu_hours'], abs=1e-3)
    assert acc['code_pool_probes'] == len(st.ladder.code_log) == CODE_PROBES_MAX   # (the pool first, then the share)
    assert d['finalist']['protected_screens'] == list(range(6, 11)) and 'bcomp' in d
    json.dumps(st.describe(), default=str)                                   # (the report serializes)


def test_re_asks_count_against_the_ten_calls_and_the_eight_dollars():
    """A backend that uses every re-ask it is granted (one per call): every attempt counts (BudgetDriver), the run still
    makes at most ten attempts and spends at most eight dollars, and the last call is granted no re-ask it cannot pay."""
    r = bcomp_llm_toy(_toy_script(), usd_per_call=1.0, retries=1)
    st, s, be = r.st, r.s, r.be
    assert s.ledger.used['screens'] == 10 and st._llm_used() == be.attempts_made <= CALLS_MAX
    assert s.ledger.used['llm_usd'] <= USD_MAX + 1e-9
    assert all(g <= 1 for _, g in be.granted) and st._llm_used() >= 2 * len(be.granted) - 1


@pytest.mark.parametrize('off', [[c] for c in BCOMP_COMPONENTS] + [sorted(BCOMP_COMPONENTS)])
def test_every_component_off_runs(monkeypatch, off):
    row, st = run_kept(monkeypatch, mem_tasks()[2], lambda sp, facts, sim: BComp(sp, facts=facts, off=off, seed=5), budget=6)
    d = st.describe()['bcomp']
    assert row['screens'] == 6 and d['bcomp']['off'] == sorted(off) and row['shorts'] == 0


def test_a_failed_prior_and_a_failed_patch_call_degrade_and_never_crash():
    def boom(ctx, n):
        raise RuntimeError('the backend fell over')
    eng = PeakEngine()
    st = BComp(CodeProposer(None, boom), facts={'cap_gib': 100.0}, seed=3)
    trace = eng.run(st)                                                      # (no exception, no loop)
    assert st.prior.source == 'flat' and st.asks >= 1 and all(a['sent'] is False for a in st.fb_asks)
    assert eng.full_screens() >= 3 and (trace.stop_reason.startswith('budget') or trace.stop_reason == 'strategy_done')
    assert len(st.fb_asks) <= eng.full_screens() + len([x for x in eng.s.screens if x.get('fidelity') == 'probe']) + 1


def test_the_toybench_arms():
    from infrarsi.search import toybench as tb
    names = set(tb.arms(SimpleNamespace(task=SimpleNamespace(seed=1), memory_facts=None)))
    assert {'bcomp', 'bcomp-src', 'random', 'wcs', 'xfer', 'xfer-src', 'opro'} <= names
    row = tb.run_one(mem_tasks(6)[0], 'bcomp-src', 10, job_factor=None, proposer={'memory': True}, bind='screens')
    assert row['screens'] == 10 and row['shorts'] == 0


# ---------------------------------------------------------------- the fairness invariants
def test_the_module_names_no_knob_model_library_or_environment():
    src = SRC.read_text(encoding='utf-8')
    code = '\n'.join(ln for ln in src.splitlines() if not ln.strip().startswith('#'))
    for word in ('recompute', 'micro_batch', 'liger', 'compile', 'fsdp', 'deepspeed', 'megatron', 'torch', 'qwen', 'gemma',
                 'llama', 'flash', 'expandable', 'checkpointing', 'nccl', 'cudnn', 'triton', 'sdpa', 'hf_trainer', 'nemo',
                 'deepseek', 'transformers', 'automodel'):
        assert not re.search(rf'\b{word}\b', code, re.I), word
    assert not re.search(r'\b(Q9U|Q9H|G4U|M35H[24]|Q27U|Q27H2|E16)\b', code)
    assert 'trials_tb' not in src                                            # never read at run time


def test_the_random_arm_and_the_verifier_are_untouched():
    from infrarsi.search import knob_space
    assert STRATEGIES['random'].__closure__ is not None
    assert any(c.cell_contents is knob_space.random_search for c in STRATEGIES['random'].__closure__)
    assert B.FinalistBase is FinalistBase and BComp.observe is not WCS.observe
