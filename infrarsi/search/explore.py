"""explore: shared search changes (V2_SEARCH_DESIGN_20261101, "Shared module API") -
M1 contrarian moves, M4 leaving a local optimum, and the failure attribution that M2 (bcomp's failure feedback) and M3
(repair combinations) read - as a mixin on the finalists' shared base: `class BComp(ExploreMixin, FinalistBase)`.
It can also be combined with OPRO. The mixin defines no __init__, observe or propose: the strategy calls
`_ex_init` once (its init / first propose) and `_ex_observe` from its observe (after the base's, so the run is recorded);
the state is made on first use. It reads only the base's state - the table (`knobs`, `ref`, `_legal`, `_complete`,
`_candidates`), the prior's levels and pairs, the memory ledger (`_pf`, `price_tags`, `failed_material`), the pair table,
the runs, the time model (`surr`), the ladder and S6 - and every candidate it makes is the base's own whole-configuration
proposal (`_knob_q`: S6's gate, hard rule 5's probe, the ladder's plan), so nothing here runs what the base would refuse.

    attribution  (`_ex_attribution`, part 'attribution') from the runs so far, rebuilt after every observation:
                 memory     a run out of memory or thrashing (any rung): its suspects are the values it holds that no
                            fitted run holds; a value in EX_NEEDS = 2 memory failures or more and in no fitted run
                            `needs a saver` - it is composed only with savers that cover its measured shortfall (the
                            leanest failure's gap: the savers' freed GiB, else the ledger's own P >= 0.5 with one held);
                            after EX_EXCLUDE = 3 memory failures that tried a saver (one its first failure did not hold:
                            a saver every failure held, the incumbent's own, is no try) it is `excluded` for the rest
                            of the arm
                 conflict   a run that failed for a reason a combination can cause (a crash, the compiler, a collective,
                            a timeout at the start or later: hls3's TOGETHER): the smallest set of its values (a pair,
                            else a triple, else all of them) that no run which ran (ok, slower, numerics, thrash, a slow
                            probe) holds whole - a run with one value off the reference failed alone; sets of one size
                            ranked by the prior's declared conflicts, the pair table, how many such failures hold it, how
                            many of its values ran somewhere (the rest tied: `ambiguous`); no later candidate holds a
                            conflict whole, each value alone stays allowed (`_ex_rules`)
    M1 contrarian (`_ex_contrarian`, part 'contrarian') a value v of the prior's levels the incumbent does not hold is
                 eligible when the prior rates it low (declared gain <= max(2 %, 2 sigma_pair), a declared conflict, or
                 declared not to fit: p_runs < 0.5, or a declared spender the ledger puts below P 0.5 on the incumbent),
                 its sign is uncertain under the prior's own numbers (sd >= |gain|, the time model's SD_UNKNOWN for a
                 missing sd; or a conflict / a memory veto a partner change could remove), its upside is detectable
                 (`_ex_upside_ok`: gain + 2 sd in the declared percentages reaches max(2 %, 2 sigma_pair), the arm's
                 noise - a value rated 0 +- 0.5 % is no contrarian move, one run could not tell its change), it is not
                 outlandish (p_runs >= 0.3; no memory failure that holds v and every other value of its failure inside
                 the move, unless the move frees the failure's gap; not excluded, not failed alone; the move's P read
                 as the ledger's, `_ex_p_memory`, at least EX_P_MIN - the pair table's start suspicions, other jobs'
                 failed starts among them, lower its plan's P, never its eligibility: no measured infeasibility in
                 this job); ranked by the optimistic gain (gain + sd) x p_runs (`_ex_m1_score`), the best first. The
                 move:
                 the incumbent with v; every value it holds that conflicts with v (declared and not contradicted by a run,
                 an attributed conflict, the pair table below 0.5) goes back to the reference's value (the other side of
                 a boolean); when v spends memory (measured, else declared) and the ledger's P of the move is below
                 EX_SAVER_P = 0.5, the best memory saver is added (measured tags by GiB, declared priced savers by GiB,
                 declared savers without a price, as bcomp's completion ranks them: the first that brings P to 0.5
                 sending nothing back, else one that sends back a held value it conflicts with - never v -, else the
                 first by rank); the coupled knobs completed by the legal table. Untested: no run holds v with the
                 move's other big-knob values (a failure whose cause the move changes - its partner sent back - does
                 not count). At most
                 EX_CONTRA_MAX = 2 contrarian screens per arm (at ten screens; F1 below), each through the ladder
                 (probed first below its P rule)
    M4 escape    (`_ex_escape`, part 'escape') plateau (ReduceLROnPlateau): the best-so-far full screen not improved by
                 max(1 %, 2 sigma_pair) over the last EX_STALL = 2 full screens, counted from the first valid full
                 screen that improved on the reference by as much (a local optimum to leave; before it the search is
                 never stalled, radius 1); trust region: radius 1, while stalled
                 2 + the stalled screens beyond EX_STALL, at most EX_RADIUS_MAX = 3, an improvement resets it - a
                 knob-side candidate differs from the incumbent in at least r big knobs (a big knob: some value's prior
                 |gain|, its time-model mean or a measured one-knob difference >= max(2 %, 2 sigma_pair); no big knob:
                 every knob; r at most their number); tabu: while stalled, no candidate equal to an already-run
                 configuration on the big knobs (a near-duplicate; of a failed run, never-repeat judges: one that removes
                 its cause is no near-duplicate); warm restart (SGDR): from 2 x EX_STALL stalled full screens on (again
                 every 2 x EX_STALL), the next escape starts from the best run of a different class (>= 2 big knobs off
                 the incumbent) with its best untried one-big-knob move; momentum (light): after an improvement that
                 changed one knob (or one and its coupled partner), the next escape prefers a candidate holding the next
                 value of that ordered knob (three values or more: numbers by value, else the table's order). The
                 escape's score: the time model's gain over the incumbent plus its spread (optimistic), x P; an escape
                 (a warm restart too) only when that optimistic gain is above 0 - none predicted even optimistically
                 faster: no escape, the strategy's own path goes on
    M3 material  (`_ex_repair_candidates`, part 'repair') from each attributed failure: memory - one repair, the suspects
                 on the incumbent (their own conflicts with what it holds sent back; the failed configuration when every
                 value fitted somewhere), its coupled knobs completed first (a partner the table requires counts before
                 the draft can look covered), plus saver(s) in bcomp's order while the measured shortfall is not covered
                 (EX_SAVERS = 2 at most): covered is the completed configuration's net GiB against the failed one -
                 every value it changed, a value sent back and a coupled knob included - reaching the gap, else the
                 ledger's own P >= 0.5; first with the savers that send nothing back, then allowing one that conflicts
                 with a value the incumbent holds, that value sent back to the value that removes the conflict (the
                 design's W3); a saver that does not raise the net is passed over; not covered: no repair (and M2's
                 needs-a-saver rule holds for repairs too); a conflict - one repair per direction: a non-empty proper
                 subset of the set kept (holding a value that has not run outside the failure, when there is one), the
                 rest left out (sent back when the incumbent holds it); ranked by the time model's gain x P; none once
                 EX_REPAIR_MAX = 2 repair screens ran (at ten screens; F1 below). `_ex_repair_of` recognizes a
                 configuration by the same rules
    admissible   (`_ex_admissible`) never repeat a failure (part 'repair'): a candidate equal on the big knobs to a failed
                 configuration whose cause it leaves unchanged - no saver added (a changed value that frees at least
                 EX_SAVE_MIN of the card, the net saving positive), the conflict set still whole - is refused; while
                 stalled (part 'escape') the trust region and the tabu list. For many candidates at once (a whole
                 table): `ctx = _ex_context(inc_cfg, engine)` once, then `_ex_adm(cfg, ctx, ids)` each - the same rules,
                 no refusal recorded (a stable part of the API)
    floors       every candidate: legal, untried, P >= EX_P_MIN = 0.15 (only measurements push P below the declared-memory
                 floor; a move that adds a saver reads it without isolated_blame's factor, as bcomp's completion does),
                 a doubtful one (below the ladder's probe_below) only while a probe is affordable, S6 and hard rule 5 by
                 `_knob_q`, a screen left; the attribution's rules (`_ex_rules`, whole) for every candidate; admissible -
                 so while stalled a contrarian move or a repair, knob-side candidates too, differs from the incumbent in
                 at least r big knobs
    F1 rates     (the open wave, FABLE_OPEN_DESIGN_20261102 section 1) EX_CONTRA_MAX, EX_REPAIR_MAX and EX_REVISE_MAX are
                 per-arm caps at ten screens: at `_ex_init` each becomes EX_*_RATE = 0.2 per screen of the arm's budget
                 (finalist_base.rate_screens: counted at least ten, so ten screens or fewer keep today's 2; 4 at 20), set
                 on the instance (`_ex_rates`); a cap the strategy or its class set to another value is kept
    v4           (FABLE_CODE_DIAGNOSIS_20261102, the base's `_v4` gate: the code track open, beyond ten screens) F3 - a
                 failed run that carried patches beyond the adopted chain, or one isolated_blame gives a patch, is that
                 patch's failure: never-repeat refuses a configuration for it only when the configuration carries those
                 patches (`_ex_patches_of`, the context's `patches`); F9 - M1's moves ranked by the optimistic gain x the
                 move's own P (the ledger's) instead of the prior's p_runs (`score_v4_pct`)

Candidates are the base's proposals (wcs's Proposal: the engine's object, handed to the ladder as bcomp's `_bc_whole`
does) with the kind - its id's prefix and the ledger row's `kind` - and the attribute `why` set to 'contrarian',
'escape' or 'repair' (`_ex_candidate`, the one place they are made; a strategy may override it). The plain moves are
available too (`_ex_contrarian_moves`, `_ex_move`, the repair material): a strategy that runs one through its own path
(an archive item and `_run_item`) is matched by the configuration at its observation, or marks it (`_ex_mark`).
Every decision is in `_ex_view` (for describe(): which mechanism proposed each screen and why).

    parts (`_ex_on`): 'contrarian', 'escape', 'repair', 'attribution' - on unless the strategy lists a part in `ex_off`
    or overrides `_ex_on` (its component switches mapped onto them)

Nothing here names a knob, a model or a library: every value comes from the table, the prior, the ledger and the runs.
"""
from __future__ import annotations

import itertools
import math

from .finalist_base import (FITTED, KNOB_SCAN_MAX, MEM_FAIL, P_FLOOR_DECLARED, RATE_SCREENS, STRONG_MIN, FinalistBase,
                            per_screen, rate_screens, vk)
from .hls3 import TOGETHER
from .wcs import SD_UNKNOWN, Surrogate

EX_VERSION = 'x1.1'             # v3: M1's detectable upside and the ledger's P as its floor; no pessimistic escape
EX_CONTRA_MAX = 2               # M1: contrarian screens per arm at most
EX_STALL = 2                    # M4: full screens without an improvement before the search is stalled
EX_RADIUS_MAX = 3               # M4: the trust region's radius at most (big knobs off the incumbent)
EX_REPAIR_MAX = 2               # M3: repair screens per arm at most
EX_REVISE_MAX = 2               # M2: bcomp's revise calls per arm at most (the strategy's; kept here with the others)
EX_CONTRA_RATE = EX_REPAIR_RATE = EX_REVISE_RATE = 0.2   # F1: the three caps above per screen of the budget (at ten
#                                 screens or fewer: the caps above; `_ex_rates`)
EX_LOW = STRONG_MIN            # M1: the prior rates a value low at or below max(2 %, 2 sigma_pair); a big knob from there
EX_IMPROVE = 0.01               # M4: an improvement of the best-so-far is at least max(1 %, 2 sigma_pair)
EX_P_RUNS_MIN = 0.3             # M1: not outlandish - the prior's p_runs at least this
EX_P_RUNS_LOW = 0.5             # M1: declared not to fit - the prior's p_runs below this (bcomp's P_RUNS_LOW)
EX_SAVER_P = 0.5                # M1: a move that spends memory gets a saver when the ledger's P is below this
EX_P_MIN = P_FLOOR_DECLARED     # every candidate: P at least this (only measurements push P below the declared floor)
EX_NEEDS = 2                    # attribution: memory failures (no fitted run) before a value needs a saver
EX_EXCLUDE = 3                  # attribution: memory failures that held a saver before a value is excluded
EX_SAVE_MIN = 0.005             # a value saves memory when it frees at least this share of the card (bcomp's SAVE_MIN)
EX_SAVERS = 2                   # M3: savers a memory repair adds at most
EX_SET_MAX = 3                  # attribution: a conflict is a pair, else a triple, else the whole failed set
EX_SPREAD_TOP = 64              # M4: admissible rows (best mean first) whose spread is read
EX_LOG_MAX = 40                 # the view's decisions at most
EX_PARTS = {
    'contrarian': 'M1: contrarian moves on the incumbent (conflicts resolved, a saver added when needed)',
    'escape': 'M4: the plateau, the trust region, the tabu list, the warm restart and momentum (escape candidates)',
    'repair': 'M3: repair material (a failed value plus the change that removes its cause) and never repeating a failure',
    'attribution': 'M2 / M3: the failure attribution (needs a saver, excluded, conflicts)'}
RAN = ('ok', 'slower', 'numerics', 'thrash', 'slow_probe')   # runs that started and ran updates (no crash, no timeout)
DONE = ('screened', 'aborted', 'invalid', 'infeasible', 'probed', 'short')   # outcomes of a run that took place
_MISS = object()


def _pct(x) -> float | None:
    return None if x is None else round(100 * math.expm1(x), 2)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class ExploreMixin:
    """M1, M4 and the failure attribution (the module docstring) for a strategy on FinalistBase."""
    CHECKPOINT_LIVE = ('ex_eng',)                        # search-state checkpoints: the engine last seen is the resuming run's
    EX_CONTRA_MAX = EX_CONTRA_MAX
    EX_STALL = EX_STALL
    EX_RADIUS_MAX = EX_RADIUS_MAX
    EX_REPAIR_MAX = EX_REPAIR_MAX
    EX_REVISE_MAX = EX_REVISE_MAX

    # ------------------------------------------------------------ state, parts
    def _ex_state(self) -> None:
        """The mixin's state, made once on first use (the mixin has no __init__ of its own)."""
        d = self.__dict__
        if d.get('_ex_made'):
            return
        d['_ex_made'] = True
        d.setdefault('ex_off', set())                    # parts the strategy switched off (or it overrides _ex_on)
        self.ex_ready = False                            # _ex_init made
        self.ex_counts = {'contrarian': 0, 'escape': 0, 'repair': 0, 'warm_restarts': 0, 'restart_no_base': 0,
                          'momentum': 0, 'refused_repeat': 0, 'refused_trust': 0, 'refused_tabu': 0, 'matched': 0}
        self.ex_log: list[dict] = []                     # every candidate made (or matched), with why and its outcome
        self.ex_by_hid: dict[str, dict] = {}             # proposal id -> its row
        self.ex_offered: dict[tuple, dict] = {}          # configuration key -> a plain move handed to the strategy
        self.ex_refused: list[dict] = []                 # the admissibility refusals the strategy asked about
        self.ex_seen: set = set()                        # outcomes observed (an observation is counted once)
        self.ex_full = 0                                 # full screens observed
        self.ex_anchor, self.ex_anchor_cid = 0.0, 'c0'   # M4: the best-so-far at the last improvement (log gain)
        self.ex_found = False                            # M4: a full screen improved on the reference (a local optimum
        #                                                  to leave: the plateau is counted from then on)
        self.ex_bad = 0                                  # M4: full screens since that improvement
        self.ex_restart_next = 2 * self.EX_STALL         # M4: the stalled screens at which the next warm restart is due
        self.ex_improved: list[dict] = []                # M4: each improvement (the run, its gain, what it moved)
        self.ex_momentum: dict | None = None             # M4: the last improving move's extension (until an escape)
        self.ex_excluded: set = set()                    # attribution: values excluded for the rest of the arm
        self.ex_attr = None                              # (stamp, attribution)
        self.ex_big = None                               # (stamp, big knobs)
        self.ex_tags = None                              # (stamp, price tags)
        self.ex_eng = None                               # the engine last seen (`_ex_admissible` takes none)

    def _ex_on(self, name: str) -> bool:
        """Whether a part is on ('contrarian', 'escape', 'repair', 'attribution'): unless the strategy lists it in
        `ex_off` (a strategy may override this with its own component switches)."""
        return name not in (self.__dict__.get('ex_off') or ())

    def _ex_bind(self, engine) -> None:
        self._ex_state()
        if engine is not None:
            self.ex_eng = engine

    def _ex_forget(self) -> None:
        """The caches (attribution, big knobs, price tags) go: for a change the observations do not stamp - the prior's
        levels revised in place (bcomp's revise call)."""
        self._ex_state()
        self.ex_attr = self.ex_big = self.ex_tags = None

    def _ex_engine(self):
        return self.__dict__.get('ex_eng') or getattr(self, 'engine', None)

    def _ex_init(self, engine, *, skip: str | None = None) -> None:
        """Once, from the strategy's init / first propose: F1's caps (`_ex_rates`), and the best-so-far from the runs
        already made - the best valid full screen when it improved on the reference by max(1 %, 2 sigma_pair), else the
        reference's 0.0 (`skip`: the run being observed, when the first call comes from `_ex_observe`)."""
        self._ex_bind(engine)
        if self.ex_ready:
            return
        self.ex_ready = True
        self._ex_rates(engine)
        thr = self._ex_thr(engine, EX_IMPROVE)
        for cid, r in self.runs.items():
            y = self._ex_y(r)
            if cid != skip and r.get('origin') != 'reference' and y is not None and y > max(self.ex_anchor, thr):
                self.ex_anchor, self.ex_anchor_cid, self.ex_found = y, cid, True

    def rebudget(self, engine) -> None:
        """Resume (FinalistBase.rebudget): the base's caps, then F1's contrarian / repair / revise caps again."""
        super().rebudget(engine)
        if self.__dict__.get('_ex_made') and self.ex_ready:
            for name in ('EX_CONTRA_MAX', 'EX_REPAIR_MAX', 'EX_REVISE_MAX'):
                self.__dict__.pop(name, None)
            self._ex_rates(engine)

    def _ex_rates(self, engine) -> None:
        """F1 (the module docstring): from `_ex_init`, the contrarian, repair and revise caps as per-screen rates of the
        arm's screen budget, set on the instance; ten screens or fewer: today's caps, nothing is set; a cap the strategy or
        its class set to another value than the module's is kept."""
        b = rate_screens(engine)
        if b <= RATE_SCREENS:
            return
        for name, ref, rate in (('EX_CONTRA_MAX', EX_CONTRA_MAX, EX_CONTRA_RATE),
                                ('EX_REPAIR_MAX', EX_REPAIR_MAX, EX_REPAIR_RATE),
                                ('EX_REVISE_MAX', EX_REVISE_MAX, EX_REVISE_RATE)):
            if getattr(self, name) == ref:
                setattr(self, name, per_screen(rate, b))

    # ------------------------------------------------------------ small readings
    def _ex_sp(self, engine=None) -> float:
        eng = engine or self._ex_engine()
        try:
            return float(eng.sigma_pair()) if eng is not None else 0.0
        except Exception:                                # noqa: BLE001 - no noise known yet: the fixed floors alone
            return 0.0

    def _ex_thr(self, engine=None, share: float = EX_LOW) -> float:
        """max(share, 2 sigma_pair) as a log gain (share EX_LOW: rated low, a big knob; EX_IMPROVE: an improvement)."""
        return max(math.log1p(share), 2.0 * self._ex_sp(engine))

    @staticmethod
    def _ex_y(r: dict) -> float | None:
        """A run's measured log gain over the reference when it is a valid full screen (or the reference)."""
        if r.get('y') is None or r.get('bound') or r.get('kind') == 'thrash':
            return None
        if r.get('origin') == 'reference' or (r.get('status') == 'screened' and r.get('fidelity') in (None, 'full')):
            return float(r['y'])
        return None

    def _ex_kcfg(self, cfg: dict | None) -> dict:
        cfg = cfg or {}
        return {k: cfg.get(k, self.ref.get(k)) for k in self.knobs}

    def _ex_vals(self, cfg: dict | None, ids=()) -> frozenset:
        """The values off the reference a configuration holds (knob values, carried patches as patch:<id>)."""
        return frozenset(self._levels(self._m_of(self._ex_kcfg(cfg), tuple(ids or ()))))

    def _ex_run_vals(self, r: dict) -> frozenset:
        return frozenset(self._levels(r.get('cfg') or {}))

    def _ex_val(self, k: str, key: str):
        return next((x for x in self.knobs.get(k) or [] if vk(x) == key), _MISS)

    @staticmethod
    def _ex_word(v) -> str:
        return f'{v[0]}={v[1]}'

    def _ex_words(self, vals) -> list[str]:
        return [self._ex_word(v) for v in sorted(vals)]

    def _ex_change(self, cfg: dict, base: dict | None = None) -> dict:
        """The knob values of `cfg` that differ from `base` (the reference by default)."""
        base = self.ref if base is None else base
        return {k: cfg.get(k) for k in self.knobs if vk(cfg.get(k)) != vk(base.get(k))}

    def _ex_tag_map(self) -> dict:
        led = self.mledger
        stamp = (getattr(self, 'fb_obs', 0), len(led.obs) if led is not None else -1)
        if self.ex_tags is None or self.ex_tags[0] != stamp:
            self.ex_tags = (stamp, self.price_tags())
        return self.ex_tags[1]

    def _ex_cap(self) -> float:
        led = self.mledger
        return float(self.mem.cap or (led.cap if led is not None else None) or 0.0)

    def _ex_floor(self) -> float:
        return max(1e-6, EX_SAVE_MIN * self._ex_cap())

    def _ex_gib(self, k: str, key: str) -> float:
        """A value's GiB per GPU against the reference's value (+ spends, - frees): the ledger's price tag where an
        observation or a declaration gives it, else the prior's declaration, else 0."""
        if k in self.knobs and key == vk(self.ref.get(k)):
            return 0.0
        tag = self._ex_tag_map().get((k, key))
        if tag is not None and tag.get('source') in ('measured', 'declared') and _num(tag.get('gib')):
            return float(tag['gib'])
        e = (self.prior.levels.get((k, key)) if self.prior else None) or {}
        if e.get('mem_dir') == 'spends' and e.get('mem_gib'):
            return float(e['mem_gib'])
        if e.get('mem_dir') == 'frees' and e.get('mem_gib'):
            return -float(e['mem_gib'])
        return 0.0

    def _ex_measured(self, k: str, key: str) -> bool:
        tag = self._ex_tag_map().get((k, key))
        return tag is not None and tag.get('source') == 'measured'

    def _ex_saver(self, k: str, key: str) -> bool:
        """The value frees memory: by its GiB (measured, else declared), or declared to free without a price."""
        if self._ex_gib(k, key) <= -self._ex_floor():
            return True
        e = (self.prior.levels.get((k, key)) if self.prior else None) or {}
        return not self._ex_measured(k, key) and e.get('mem_dir') == 'frees' and not e.get('mem_gib')

    def _ex_spender(self, k: str, key: str) -> bool:
        """The value spends memory: measured (its tag), else declared (with or without a price)."""
        if self._ex_measured(k, key):
            return self._ex_gib(k, key) >= self._ex_floor()
        e = (self.prior.levels.get((k, key)) if self.prior else None) or {}
        return e.get('mem_dir') == 'spends' or self._ex_gib(k, key) >= self._ex_floor()

    def _ex_saved(self, new: dict, old: dict) -> tuple[float, bool]:
        """(the GiB `new` frees against `old` by the values' GiB - net -, whether a changed value frees at least the
        saving floor): a saver added."""
        floor, net, one = self._ex_floor(), 0.0, False
        for k in self.knobs:
            a, b = vk(old.get(k, self.ref.get(k))), vk(new.get(k, self.ref.get(k)))
            if a == b:
                continue
            d = self._ex_gib(k, a) - self._ex_gib(k, b)
            net += d
            one = one or d >= floor
        return net, one and net > 0.0

    # ------------------------------------------------------------ big knobs (M1 untested, M4 radius and tabu)
    def _ex_big_knobs(self, engine=None) -> set:
        """The big knobs: a value whose prior |gain|, time-model mean or measured one-knob difference (two measured runs
        with the same patches that differ only in it) reaches max(2 %, 2 sigma_pair)."""
        self._ex_state()
        thr = self._ex_thr(engine)
        stamp = (getattr(self, 'fb_obs', 0), len(self.runs), round(thr, 6), id(self.prior), len(self.surr.obs))
        if self.ex_big is not None and self.ex_big[0] == stamp:
            return self.ex_big[1]
        out = set()
        for (k, key), e in (self.prior.levels.items() if self.prior else ()):
            if k in self.knobs and _num(e.get('mu')) and abs(e['mu']) >= thr:
                out.add(k)
        try:
            mean = self.surr.mean() if self.surr.names else {}
        except Exception:                                # noqa: BLE001 - a posterior that fails: the prior and the runs
            mean = {}
        for f, mu in mean.items():
            if not isinstance(f, tuple) or abs(mu) < thr:
                continue
            if f[0] in self.knobs:
                out.add(f[0])
            elif f[0] == 'pair':
                out.update(k for k, _ in f[1] if k in self.knobs)
        meas = []
        for cid, r in self.runs.items():
            y = self._ex_y(r)
            if y is not None:
                cfg = r.get('cfg') or {}
                meas.append((self._ex_kcfg(cfg), frozenset(x for x, v in cfg.items() if str(x).startswith('patch:') and v), y))
        for (ca, pa, ya), (cb, pb, yb) in itertools.combinations(meas, 2):
            if pa != pb or abs(ya - yb) < thr:
                continue
            d = [k for k in self.knobs if vk(ca.get(k)) != vk(cb.get(k))]
            if len(d) == 1:
                out.add(d[0])
        self.ex_big = (stamp, out)
        return out

    def _ex_cmp(self, engine=None) -> tuple:
        """The knobs a near-duplicate, the trust region and an untested move are read on: the big knobs, else every
        knob (a flat prior with nothing measured)."""
        big = self._ex_big_knobs(engine)
        return tuple(sorted(big)) if big else tuple(sorted(self.knobs))

    def _ex_sig(self, cfg: dict, cmp: tuple) -> tuple:
        return tuple(vk(cfg.get(k, self.ref.get(k))) for k in cmp)

    # ------------------------------------------------------------ M4: the plateau, the radius
    def _ex_stalled(self) -> bool:
        """The best-so-far full screen has not improved by max(1 %, 2 sigma_pair) over the last EX_STALL full screens."""
        self._ex_state()
        return self.ex_bad >= self.EX_STALL

    def _ex_radius(self) -> int:
        """The trust region's radius: 1, or 2..EX_RADIUS_MAX while stalled (one more per further stalled full screen)."""
        if not self._ex_stalled() or not self._ex_on('escape'):
            return 1
        return int(min(self.EX_RADIUS_MAX, 2 + self.ex_bad - self.EX_STALL))

    # ------------------------------------------------------------ the attribution
    def _ex_attribution(self) -> dict:
        """{'needs_saver': {value: rec}, 'excluded': {value: rec}, 'conflicts': [rec], 'failures': [rec]} from the runs so
        far (the module docstring); a value is (knob, value key) as the base's levels give it. Part 'attribution' off:
        empty (and 'off': True)."""
        self._ex_state()
        if not self._ex_on('attribution'):
            return {'needs_saver': {}, 'excluded': {}, 'conflicts': [], 'failures': [], 'off': True}
        stamp = (getattr(self, 'fb_obs', 0), len(self.runs), len(getattr(self, 'pairs', {}) or {}), id(self.prior))
        if self.ex_attr is not None and self.ex_attr[0] == stamp:
            return self.ex_attr[1]
        fitted, ran, mem, other = set(), [], [], []
        for cid, r in self.runs.items():
            kind, vals = r.get('kind'), self._ex_run_vals(r)
            if kind in FITTED:
                fitted |= vals
            if kind in RAN:
                ran.append(vals)
            if kind in MEM_FAIL:
                mem.append((cid, vals, r))
            elif kind in TOGETHER:
                other.append((cid, vals, r))
        material = {m['run']: m for m in self.failed_material()} if mem else {}
        failures, holds, with_saver, first = [], {}, {}, {}
        for cid, vals, r in mem:
            savers = sorted(v for v in vals if not v[0].startswith('patch:') and self._ex_saver(*v))
            suspects = sorted(vals - fitted)
            gap = (material.get(cid) or {}).get('gap_gib')
            failures.append({'run': cid, 'kind': r.get('kind'), 'cause': 'memory', 'fidelity': r.get('fidelity'),
                             'values': sorted(vals), 'suspects': suspects, 'savers': savers,
                             'gap_gib': round(gap, 3) if _num(gap) else None})
            for v in suspects:                           # (an attempt with a saver: a saver its first failure did not
                holds.setdefault(v, []).append(cid)      #  hold - a saver every failure held, the incumbent's, is no try)
                base = first.setdefault(v, frozenset(s for s in savers if s != v))
                if len(holds[v]) > 1 and any(s != v and s not in base for s in savers):
                    with_saver.setdefault(v, []).append(cid)
        needs, excluded = {}, {}
        for v, cids in sorted(holds.items()):
            if len(cids) < EX_NEEDS and v not in self.ex_excluded:
                continue
            known = [(material[c]['gap_gib'], c) for c in cids if _num((material.get(c) or {}).get('gap_gib'))]
            gap, lean = min(known) if known else (None, cids[0])
            rec = {'failures': list(cids), 'with_saver': list(with_saver.get(v, [])),
                   'gap_gib': round(gap, 3) if gap is not None else None, 'leanest': lean}
            if len(rec['with_saver']) >= EX_EXCLUDE or v in self.ex_excluded:
                self.ex_excluded.add(v)                  # (for the rest of the arm, whatever runs later)
                excluded[v] = rec
            else:
                needs[v] = rec
        for v in sorted(self.ex_excluded - set(excluded)):
            excluded[v] = {'failures': list(holds.get(v, [])), 'with_saver': list(with_saver.get(v, [])),
                           'gap_gib': None, 'leanest': None}
        sets = [vals for _, vals, _ in other]
        conflicts: dict = {}
        for cid, vals, r in other:
            got = self._ex_conflict_of(vals, ran, sets)
            if got is None:
                continue
            members, info = got
            failures.append({'run': cid, 'kind': r.get('kind'), 'cause': 'conflict', 'fidelity': r.get('fidelity'),
                             'values': sorted(vals), 'set': sorted(members), **info})
            rec = conflicts.setdefault(members, {'members': sorted(members), 'runs': [], 'kinds': [], **info})
            rec['runs'].append(cid)
            rec['kinds'].append(r.get('kind'))
        order = {cid: i for i, cid in enumerate(self.runs)}
        failures.sort(key=lambda f: order.get(f['run'], -1))   # (in run order, whatever the cause: the latest last)
        out = {'needs_saver': needs, 'excluded': excluded, 'conflicts': list(conflicts.values()), 'failures': failures}
        self.ex_attr = (stamp, out)
        return out

    def _ex_conflict_of(self, vals: frozenset, ran: list, sets: list):
        """(the conflict set, its notes) of a failure that is not memory's: the smallest subset of its values - a pair,
        else a triple, else all of them - that no run which ran holds whole; one value off the reference: it failed
        alone. Sets of one size by: the prior's declared conflict, the pair table, the failures holding it, its values
        that ran somewhere (the others tied: `ambiguous`)."""
        vals = sorted(vals)
        if not vals:
            return None
        if len(vals) == 1:
            return frozenset(vals), {'alone': True}
        declared = [frozenset(pr['members']) for pr in (self.prior.pairs if self.prior else [])
                    if pr.get('kind') == 'conflicts']
        table = [pr for pr, rec in (getattr(self, 'pairs', {}) or {}).items() if (rec or {}).get('p', 1.0) < 1.0]
        for size in range(2, min(EX_SET_MAX, len(vals)) + 1):
            rows = []
            for s in itertools.combinations(vals, size):
                s = frozenset(s)
                if any(s <= r for r in ran):
                    continue
                score = (any(s <= d for d in declared), any(pr <= s for pr in table), sum(1 for o in sets if s <= o),
                         sum(1 for v in s if any(v in r for r in ran)))
                rows.append((score, sorted(s)))
            if rows:
                top = max(sc for sc, _ in rows)
                tied = sorted(m for sc, m in rows if sc == top)
                info = {'declared': bool(top[0])}
                if len(tied) > 1:
                    info['ambiguous'] = [self._ex_words(m) for m in tied[1:4]]
                return frozenset(tied[0]), info
        return frozenset(vals), {'whole': True}

    def _ex_covered(self, held: frozenset, v, rec: dict, cfg: dict | None = None, ids=()) -> bool:
        """Whether a configuration holding the needs-a-saver value `v` holds savers that cover its shortfall: savers the
        leanest failure did not hold, the configuration's net GiB against that failure's (`_ex_saved`: every value it
        changed - the savers, a value sent back, a coupled knob; without the configuration, the savers' GiB alone)
        reaching its gap - or, the gap unknown or a saver without a price, the memory ledger's own P of the
        configuration at least 0.5 (bcomp's completion: a price ranks, the ledger decides). The ledger is read directly,
        never the strategy's `_pf`: a strategy whose `_pf` applies `_ex_rules` does not recurse through here (no
        ledger: the price alone, an unknown gap covered by a saver held)."""
        lean = self.runs.get(rec.get('leanest')) or {}
        had = self._ex_run_vals(lean) if lean else frozenset()
        savers = [u for u in held if u != v and u not in had and not u[0].startswith('patch:') and self._ex_saver(*u)]
        if not savers:
            return False
        if cfg is not None and lean:
            net = self._ex_saved(self._ex_kcfg(cfg), self._ex_kcfg(lean.get('cfg')))[0]
        else:
            net = sum(max(0.0, -self._ex_gib(*u)) for u in savers)
        if rec.get('gap_gib') is not None and net + 1e-9 >= rec['gap_gib']:
            return True
        led = self.mledger
        if cfg is None or led is None:
            return rec.get('gap_gib') is None
        return led.p_feasible(self._ex_kcfg(cfg), ids=self._led_ids(tuple(ids or ()))) >= 0.5

    def _ex_rules(self, cfg: dict, ids=None, *, saver: bool = True) -> str | None:
        """The attribution's rules for a candidate (M2's; every candidate made here): why it breaks one - it holds an
        excluded value, holds a conflict whole, or (`saver`) holds a value that needs a saver without savers covering
        its shortfall - else None. It never reads the strategy's `_pf` (the ledger directly: `_ex_covered`), so a
        strategy's `_pf` may apply it without recursing."""
        att = self._ex_attribution()
        if att.get('off'):
            return None
        ids = tuple(self.chain) if ids is None else tuple(ids)
        held = self._ex_vals(cfg, ids)
        for v in sorted(held):
            if v in att['excluded']:
                n = len(att['excluded'][v].get('with_saver') or [])
                return f'{self._ex_word(v)} is excluded ({n} memory failures with a saver)'
        for c in att['conflicts']:
            if set(c['members']) <= held:
                return (f'it holds {", ".join(self._ex_words(c["members"]))} whole, which failed together '
                        f'({", ".join(c["runs"][:3])}: {c["kinds"][0]})')
        if saver:
            for v, rec in sorted(att['needs_saver'].items()):
                if v in held and not self._ex_covered(held, v, rec, cfg, ids):
                    gap = f'{rec["gap_gib"]:.1f} GiB' if rec.get('gap_gib') is not None else 'its shortfall'
                    return f'{self._ex_word(v)} needs a saver covering {gap} (out of memory in {", ".join(rec["failures"][:3])})'
        return None

    # ------------------------------------------------------------ admissibility (trust region, tabu, never repeat)
    def _ex_context(self, inc_cfg: dict | None, engine=None) -> dict:
        """What the admissibility of many candidates reads, built once: the compared knobs, the runs' and the failures'
        signatures on them, the incumbent's, the radius."""
        cmp = self._ex_cmp(engine)
        att = self._ex_attribution()
        sets = {f['run']: frozenset(f['set']) for f in att.get('failures', []) if f.get('cause') == 'conflict'}
        runs, fails = set(), {}
        own = self._ex_v4('v4_repeat', engine)
        for cid, r in self.runs.items():
            cfg = self._ex_kcfg(r.get('cfg'))
            sig = self._ex_sig(cfg, cmp)
            kind = r.get('kind')
            if kind in MEM_FAIL or kind in TOGETHER:     # (a failed run's near copies: never-repeat judges them - one
                cause = 'memory' if kind in MEM_FAIL else 'conflict'   # that removes the cause is no near-duplicate)
                fails.setdefault(sig, []).append({'run': cid, 'kind': kind, 'cause': cause, 'cfg': cfg,
                                                  'set': sets.get(cid, self._ex_run_vals(r)),
                                                  **({'patches': self._ex_patches_of(cid, r)} if own else {})})
            else:
                runs.add(sig)
        inc = self._ex_kcfg(inc_cfg) if inc_cfg is not None else None
        stalled = self._ex_stalled() and self._ex_on('escape')
        return {'cmp': cmp, 'runs': runs, 'fails': fails, 'inc': inc, 'inc_sig': self._ex_sig(inc, cmp) if inc else None,
                'stalled': stalled, 'r': min(self._ex_radius(), len(cmp)) if stalled else 0}

    def _ex_repeat(self, cfg: dict, ctx: dict, ids=()) -> str | None:
        """Never repeat a failure: a failed configuration equal to `cfg` on the compared knobs whose cause `cfg` leaves
        unchanged (memory: no saver added; another failure: its conflict set still whole)."""
        hits = ctx['fails'].get(self._ex_sig(cfg, ctx['cmp'])) if ctx['cmp'] else None
        if not hits:
            return None
        held = self._ex_vals(cfg, ids)
        for f in hits:
            if f.get('patches') and not f['patches'] <= set(ids or ()):
                continue                                 # (v4 F3: that failure is its unproven patches', not repeated here)
            if f['cause'] == 'memory':
                if self._ex_saved(cfg, f['cfg'])[1]:
                    continue
                return f'it repeats {f["run"]} ({f["kind"]}): the same big-knob values and no saver added'
            if f['set'] <= held:
                return (f'it repeats {f["run"]} ({f["kind"]}): the same big-knob values with '
                        f'{", ".join(self._ex_words(f["set"]))} still together')
        return None

    def _ex_v4(self, part: str, engine=None) -> bool:
        """The base's v4 gate (finalist_base: `_v4` - its fix on, the code track open, beyond ten screens); False on a base
        without it."""
        v4 = getattr(self, '_v4', None)
        return bool(v4 is not None and v4(part, engine if engine is not None else self._ex_engine()))

    def _ex_patches_of(self, cid: str, r: dict) -> frozenset:
        """v4 F3: the patches that make a failed run that patch's failure - those it carried beyond the adopted chain (a
        patch's probe on the incumbent's values), and a patch isolated_blame names for it; never-repeat refuses a
        configuration for that run only when the configuration carries them all."""
        pids = set((getattr(self, 'pids', None) or {}).get(cid) or ()) or {
            k[len('patch:'):] for k, v in (r.get('cfg') or {}).items() if isinstance(k, str) and k.startswith('patch:') and v}
        out = pids - set(self.chain)
        blame = getattr(self, 'isolated_blame', None)
        for (k, _v), runs in (blame() if callable(blame) else {}).items():
            if isinstance(k, str) and k.startswith('patch:') and cid in runs:
                out.add(k[len('patch:'):])
        return frozenset(out)

    def _ex_adm(self, cfg: dict, ctx: dict, ids=()) -> tuple[bool, str]:
        """`_ex_admissible`'s rules for one of many candidates, read against a context built once (`_ex_context`): no
        refusal recorded (the bulk form a strategy may use to filter a whole table)."""
        if self._ex_on('repair'):
            why = self._ex_repeat(cfg, ctx, ids)
            if why:
                return False, why
        if ctx['stalled'] and ctx['cmp']:
            sig = self._ex_sig(cfg, ctx['cmp'])
            if ctx['inc_sig'] is not None:
                d = sum(1 for a, b in zip(sig, ctx['inc_sig']) if a != b)
                if d < ctx['r']:
                    return False, f'trust region: {d} big knob(s) off the incumbent, radius {ctx["r"]}'
            if sig in ctx['runs']:
                return False, 'tabu: it differs from an already-run configuration only in small knobs'
        return True, ''

    def _ex_admissible(self, cfg: dict, inc_cfg: dict | None = None) -> tuple[bool, str]:
        """(admissible, why not): never repeat a failure (part 'repair') and, while stalled (part 'escape'), the trust
        region (at least r big knobs off the incumbent `inc_cfg`) and the tabu list (no near-duplicate of a run). A
        refusal is counted and kept for the view."""
        self._ex_state()
        if inc_cfg is None and self._ex_engine() is not None:
            eng = self._ex_engine()
            inc_cfg = self._knob_cfg(eng, eng.incumbent())
        ok, why = self._ex_adm(self._ex_kcfg(cfg), self._ex_context(inc_cfg), tuple(self.chain))
        if not ok:
            key = 'refused_repeat' if why.startswith('it repeats') else 'refused_trust' if why.startswith('trust') \
                else 'refused_tabu'
            self.ex_counts[key] += 1
            self.ex_refused.append({'change': self._ex_change(self._ex_kcfg(cfg)), 'why': why[:200]})
            del self.ex_refused[:-20]
        return ok, why

    # ------------------------------------------------------------ the candidates
    def _ex_used(self, why: str) -> int:
        """The screens of a kind made so far: its candidates that ran, or are proposed and not observed yet."""
        return sum(1 for r in self.ex_log if r['why'] == why and r.get('outcome') not in
                   ('duplicate', 'infra', 'rejected', 'not run', 'refused'))

    def _ex_p_unblamed(self, cfg: dict, ids=()) -> float:
        """S4's P without isolated_blame's factor: what the floor reads for a move that adds a saver - it compensates the
        blamed value, which the ledger's own estimate weighs (bcomp's `_bc_p_unblamed`); its proposal keeps the full P."""
        if not (getattr(self, 'fb_on', None) or {}).get('fb_isolated_blame') or not self.isolated_blame():
            return self._pf(cfg, ids)[0]
        self.fb_on['fb_isolated_blame'] = False          # (a read: the switch back right after)
        try:
            return self._pf(cfg, ids)[0]
        finally:
            self.fb_on['fb_isolated_blame'] = True

    def _ex_p_memory(self, cfg: dict, ids=(), *, unblamed: bool = False) -> float:
        """S4's P read without the pair table's start suspicions (`_fb_starts`: the prior's declared pairs, other jobs'
        failed starts from the store, this job's pair marks): what M1 reads as the ledger's P - its not-outlandish floor
        (the design: p_runs and no measured infeasibility in this job), declared not to fit, the saver decision. The
        memory ledger with its declared floor and this job's failed runs (`_fb_bad`; the attribution's conflicts and
        values that failed alone keep their own rules); `unblamed`: without isolated_blame's factor too. Read from the
        base's `_pf` (no strategy cache; M2's rules apply apart, `_ex_rules`); without the ledger, the strategy's P. The
        proposal keeps the full P (the ladder's plan, S6)."""
        if self.mledger is None:
            return self._pf(cfg, ids)[0]
        fb = getattr(self, 'fb_on', None) or {}
        anta, blame = self.on.get('wcs_antagonism'), fb.get('fb_isolated_blame')
        self.on['wcs_antagonism'] = False                # (a read: the switches back right after)
        if unblamed and blame:
            fb['fb_isolated_blame'] = False
        try:
            return FinalistBase._pf(self, cfg, ids)[0]
        finally:
            self.on['wcs_antagonism'] = anta
            if unblamed and blame:
                fb['fb_isolated_blame'] = True

    def _ex_gates(self, engine, cfg: dict, *, unblamed: bool = False, memory: bool = False) -> tuple[float, str] | None:
        """(P, why) of a candidate the floors admit - a screen left, legal, untried, P >= EX_P_MIN (`unblamed`: read
        without isolated_blame's factor; `memory`: M1's, the ledger's P without the pair table's start suspicions), a
        doubtful one only while a probe is affordable - else None."""
        ids = tuple(self.chain)
        if self._left(engine) < 1 or self._ckey(cfg, ids) in self.tried or not self._legal(engine, cfg):
            return None
        p, why = self._pf(cfg, ids)
        if p < EX_P_MIN and memory:
            floor = self._ex_p_memory(cfg, ids, unblamed=unblamed)
        else:
            floor = self._ex_p_unblamed(cfg, ids) if unblamed and p < EX_P_MIN else p
        if p <= 0.0 or floor < EX_P_MIN:
            return None
        if p < self.ladder.probe_below and not self.ladder.affordable(engine, 'probe'):
            return None                                  # (doubtful and no probe left: not run in full, as the draw)
        return p, why

    def _ex_candidate(self, engine, inc, move: dict):
        """A move as the ladder candidate (wcs's Proposal by the base's `_knob_q`: S6's gate, hard rule 5's probe, the
        ladder's plan - probed first below its P rule), tagged with `why` (its kind and the attribute); logged. None when
        a floor, S6 or hard rule 5 refuses it. (A strategy may override it to build its own object.)"""
        self._ex_bind(engine)
        cfg = self._ex_kcfg(move['cfg'])
        got = self._ex_gates(engine, cfg, unblamed=bool(move.get('saver') or move.get('savers')),
                             memory=move.get('why') == 'contrarian')
        if got is None:
            return None
        p, pwhy = got
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None, short=False)
        q = self._knob_q(engine, inc, cfg, kind=move['why'], plan=plan, p=p,
                         why=(move.get('note', move['why']) + (f'; {pwhy}' if pwhy else ''))[:300])
        if q is None or not engine._fresh(q):
            return None
        return self._ex_mark(q, move, engine)

    def _ex_mark(self, q, move: dict, engine=None):
        """A proposal made from a move (here, or by the strategy's own path): tagged (`q.why`) and logged, so that its
        observation is attributed to the move's kind and counted. Returns q."""
        self._ex_state()
        eng = engine or self._ex_engine()
        q.why = move['why']
        if isinstance(move.get('cfg'), dict):            # (an offered move, now a proposal: matched by its id)
            self.ex_offered.pop(self._ckey(self._ex_kcfg(move['cfg']), ()), None)
        if q.hypothesis in self.ex_by_hid:               # (marked already: one screen, one row)
            return q
        row = self._ex_row(move, eng, hid=q.hypothesis, p=q.p_feasible, plan=list(q.fidelity or ()))
        self.ex_log.append(row)
        self.ex_by_hid[q.hypothesis] = row
        self.ex_counts[move['why']] = self.ex_counts.get(move['why'], 0) + 1
        if move['why'] == 'escape':
            if move.get('restart'):
                self.ex_counts['warm_restarts'] += 1
                self.ex_restart_next = self.ex_bad + 2 * self.EX_STALL
            if move.get('momentum'):
                self.ex_counts['momentum'] += 1
            self.ex_momentum = None                      # (momentum is for the next escape only)
        if self.ledger.get(q.hypothesis) is not None:
            self.ledger[q.hypothesis]['explore'] = move['why']
        return q

    def _ex_row(self, move: dict, engine, *, hid=None, p=None, plan=None, outcome: str = 'proposed') -> dict:
        row = {'why': move['why'], 'id': hid, 'screens_done': engine.full_screens() if engine is not None else None,
               'change': move.get('change'), 'outcome': outcome}
        for k in ('value', 'flipped', 'saver', 'run', 'cause', 'failed', 'savers', 'dropped', 'net_gib', 'restart',
                  'momentum', 'radius', 'moved', 'score_pct', 'optimistic_pct', 'predicted_gain_pct'):
            if move.get(k) not in (None, [], {}):
                row[k] = move[k]
        row['note'] = str(move.get('note') or '')[:240]
        if p is not None:
            row['p_feasible'] = round(float(p), 3)
        if plan:
            row['plan'] = plan
        return row

    def _ex_offer(self, moves: list[dict]) -> list[dict]:
        """Plain moves handed to the strategy: matched by their configuration at the observation (the strategy may run
        one through its own path)."""
        for mv in moves:
            self.ex_offered[self._ckey(self._ex_kcfg(mv['cfg']), ())] = mv
        while len(self.ex_offered) > 24:
            self.ex_offered.pop(next(iter(self.ex_offered)))
        return moves

    # ------------------------------------------------------------ M1: contrarian moves
    def _ex_away(self, k: str, key: str):
        """The value that removes a conflict of (k, key): the reference's value, or the other side of a boolean."""
        ref = vk(self.ref.get(k))
        if key != ref:
            return self.ref.get(k)
        vals = self.knobs.get(k) or []
        if len(vals) == 2 and all(isinstance(x, bool) for x in vals):
            return next(x for x in vals if vk(x) != key)
        return _MISS

    def _ex_conflicts(self, v, held: set, ran: list | None = None) -> list[tuple]:
        """[(partner value, source)] of the values in `held` that conflict with v: the prior's declared conflicts no run
        contradicted (a run that ran held both), the attributed conflicts (v with the rest of a set), the pair table
        below 0.5."""
        att = self._ex_attribution()
        if ran is None:
            ran = [self._ex_run_vals(r) for r in self.runs.values() if r.get('kind') in RAN]
        out = []
        for pr in (self.prior.pairs if self.prior else []):
            if pr.get('kind') != 'conflicts' or v not in pr['members']:
                continue
            for m in pr['members']:
                if m != v and m in held and not any({v, m} <= r for r in ran):
                    out.append((m, 'declared'))
        for c in att.get('conflicts', []):
            ms = set(c['members'])
            if v in ms and len(ms) >= 2 and ms - {v} <= held:
                out.append((sorted(ms - {v})[0], 'measured'))
        for pair, rec in (getattr(self, 'pairs', {}) or {}).items():
            if v in pair and (rec or {}).get('p', 1.0) < 0.5:
                for m in pair:
                    if m != v and m in held:
                        out.append((m, 'pair table'))
        seen, uniq = set(), []
        for m, src in out:
            if m not in seen and not m[0].startswith('patch:'):
                seen.add(m)
                uniq.append((m, src))
        return uniq

    def _ex_savers(self, cfg: dict, *, skip=(), avoid=()) -> list[dict]:
        """The memory savers a move may add: a value of a knob `cfg` holds at the reference's value (not in `skip`), none
        excluded or in conflict with a value of `avoid` (the values the move must keep; one in conflict with another
        value held sends it back: `_ex_add_saver`); measured tags first by GiB freed, then declared savers with a price
        by GiB, then declared savers without one by the prior's gain (bcomp's completion order)."""
        att = self._ex_attribution()
        floor = self._ex_floor()
        tags = self._ex_tag_map()
        levels = self.prior.levels if self.prior else {}
        avoid = set(avoid)
        tiers: dict = {'measured': [], 'declared': [], 'declared, no price': []}
        for k, vals in self.knobs.items():
            ref = vk(self.ref.get(k))
            if k in skip or vk(cfg.get(k)) != ref:
                continue
            for x in vals:
                key = vk(x)
                if key == ref or (k, key) in att['excluded'] or self._ex_conflicts((k, key), avoid):
                    continue
                tag, e = tags.get((k, key)), levels.get((k, key)) or {}
                row = {'knob': k, 'vk': key, 'value': x}
                if tag is not None and tag.get('source') == 'measured':
                    save = -float(tag['gib'])
                    if save >= floor:
                        tiers['measured'].append((-save, k, key, {**row, 'tier': 'measured', 'saving_gib': round(save, 3)}))
                elif e.get('mem_dir') == 'frees' and e.get('mem_gib'):
                    save = -float(tag['gib']) if tag is not None and tag.get('source') == 'declared' else float(e['mem_gib'])
                    if save >= floor:
                        tiers['declared'].append((-save, k, key, {**row, 'tier': 'declared', 'saving_gib': round(save, 3)}))
                elif e.get('mem_dir') == 'frees':
                    tiers['declared, no price'].append((-(e.get('mu') or 0.0), k, key, {**row, 'tier': 'declared, no price'}))
        out = []
        for t in ('measured', 'declared', 'declared, no price'):
            out += [r[3] for r in sorted(tiers[t], key=lambda r: r[:3])]
        return out

    def _ex_add_saver(self, engine, cfg: dict, s: dict, keep: set, protect, ids) -> tuple | None:
        """`cfg` with the saver `s`: a value it holds that conflicts with the saver (declared and not contradicted by a
        run, an attributed conflict, the pair table below 0.5) goes back to the value that removes the conflict - the
        reference's, the other side of a boolean (bcomp's completion, hls3 F5c) -, the coupled knobs completed by the
        legal table: (the configuration, the values sent back, the knobs kept), or None when the saver conflicts with a
        value the move must keep (in `protect`, or of a knob in `keep`) or no legal configuration holds them."""
        new = {**cfg, s['knob']: s['value']}
        kept, sent = set(keep) | {s['knob']}, []
        for (pk, pkey), src in self._ex_conflicts((s['knob'], s['vk']), set(self._ex_vals(new, ids))):
            if pk not in self.knobs or (pk, pkey) in protect or pk in kept:
                return None
            away = self._ex_away(pk, pkey)
            if away is _MISS:
                return None
            new[pk] = away
            kept.add(pk)
            sent.append({'value': f'{pk}={pkey}', 'to': away, 'source': f'the saver: {src}'})
        trial = self._complete(engine, dict(new), kept)
        if trial is None or any(vk(trial.get(x)) != vk(new.get(x)) for x in kept):
            return None
        return self._ex_kcfg(trial), sent, kept

    def _ex_upside_ok(self, gain_pct, sd_pct, engine=None) -> bool:
        """M1's detectable-upside floor for a value the prior rated, in its declared percentages (e.g. 2.0 and 10.0; an
        unknown gain 0, an unknown spread the time model's SD_UNKNOWN): gain + 2 sd reaches max(2 %, 2 sigma_pair) - the
        arm's noise -, so that one run could tell the change it hopes for (0 +- 0.5 % is no contrarian move). A public
        helper of the API (for ranking controlled tests)."""
        g = float(gain_pct) / 100.0 if _num(gain_pct) else 0.0
        s = float(sd_pct) / 100.0 if _num(sd_pct) else math.expm1(SD_UNKNOWN)
        return g + 2.0 * s + 1e-9 >= max(EX_LOW, 2.0 * self._ex_sp(engine))

    def _ex_m1_score(self, gain_pct, sd_pct, p_runs=None) -> float:
        """M1's score for a value the prior rated: the optimistic gain - the declared gain plus its one-sigma spread, as
        log gains (an unknown gain 0, an unknown spread SD_UNKNOWN) - x p_runs (unknown: 1), what M1 ranks by, best
        first. A public helper of the API (for comparing controlled tests with contrarian moves)."""
        mu = math.log1p(max(-0.99, float(gain_pct) / 100.0)) if _num(gain_pct) else 0.0
        s = math.log1p(max(0.0, float(sd_pct) / 100.0)) if _num(sd_pct) else SD_UNKNOWN
        return (mu + s) * (float(p_runs) if _num(p_runs) else 1.0)

    def _ex_contra_values(self, engine, inc_cfg: dict) -> list[tuple]:
        """(score, value, entry, reasons) of the values M1 may move to (eligible by the prior; the module docstring),
        the best optimistic gain x p_runs first."""
        if not self.prior:
            return []
        thr = self._ex_thr(engine)
        att = self._ex_attribution()
        alone = {c['members'][0] for c in att.get('conflicts', []) if c.get('alone')}
        declared = {m for pr in self.prior.pairs if pr.get('kind') == 'conflicts' for m in pr['members']}
        ids = tuple(self.chain)
        rows = []
        any_saver = None
        for (k, key), e in self.prior.levels.items():
            if k not in self.knobs or vk(inc_cfg.get(k)) == key or (k, key) in att['excluded'] or (k, key) in alone:
                continue
            v = self._ex_val(k, key)
            if v is _MISS:
                continue
            mu, sd, pr = e.get('mu'), e.get('sd'), e.get('p_runs')
            if _num(pr) and pr < EX_P_RUNS_MIN:
                continue                                 # (outlandish by the prior's own estimate)
            reasons = []
            if _num(mu) and mu <= thr:
                reasons.append(f'declared {_pct(mu):+.1f} %')
            conflict = (k, key) in declared
            if conflict:
                reasons.append('declared a conflict')
            unfit = _num(pr) and pr < EX_P_RUNS_LOW or (
                e.get('mem_dir') == 'spends' and self._ex_p_memory({**inc_cfg, k: v}, ids) < EX_SAVER_P)
            if unfit:
                reasons.append('declared not to fit')
            if not reasons:
                continue
            g_pct = 100.0 * math.expm1(mu) if _num(mu) else None     # (the declared percentages)
            s_pct = 100.0 * math.expm1(sd) if _num(sd) else None
            if not self._ex_upside_ok(g_pct, s_pct, engine):
                continue                                 # (no detectable upside: gain + 2 sd below what one run can tell)
            gain, spread = (g_pct or 0.0) / 100.0, math.expm1(sd if _num(sd) else SD_UNKNOWN)
            band = not _num(mu) or spread + 1e-12 >= abs(gain)
            if not band and unfit and any_saver is None:
                any_saver = bool(self._ex_savers(self.ref))
            if not (band or conflict or unfit and any_saver):
                continue                                 # (its sign is not in doubt under the prior's own numbers)
            opt = (mu if _num(mu) else 0.0) + (sd if _num(sd) else SD_UNKNOWN)   # (the ranking only: optimistic x p_runs)
            rows.append((self._ex_m1_score(g_pct, s_pct, pr), (k, key), v, e, reasons, opt))
        rows.sort(key=lambda t: (-t[0], t[1]))
        return rows

    def _ex_move(self, engine, inc, k: str, key: str, *, why: str = 'contrarian', inc_cfg: dict | None = None):
        """The move of one value on the incumbent (M1's build): v set, its conflicting partners sent back, the best saver
        added when v spends memory and the ledger's P is below EX_SAVER_P, the coupled knobs completed - a plain dict
        (its 'cfg', the partners `flipped`, the `saver`, P, the predicted gain), or None when no legal configuration
        holds them. No eligibility, attribution rule or admissibility is checked here."""
        self._ex_bind(engine)
        v = self._ex_val(k, key)
        if v is _MISS:
            return None
        inc_cfg = self._ex_kcfg(inc_cfg if inc_cfg is not None else self._knob_cfg(engine, inc))
        ids = tuple(self.chain)
        cfg = {**inc_cfg, k: v}
        keep, flipped = {k}, []
        for (pk, pkey), src in self._ex_conflicts((k, key), set(self._ex_vals(cfg, ids))):
            if pk == k or pk not in self.knobs:
                continue
            away = self._ex_away(pk, pkey)
            if away is _MISS:
                return None
            cfg[pk] = away
            keep.add(pk)
            flipped.append({'value': f'{pk}={pkey}', 'to': away, 'source': src})
        done = self._complete(engine, dict(cfg), keep)   # (the coupled knobs first: the saver decision reads the
        if done is None or any(vk(done.get(x)) != vk(cfg.get(x)) for x in keep):   # configuration that would run)
            return None
        cfg = self._ex_kcfg(done)
        saver, sent = None, []
        if self._ex_spender(k, key) and self._ex_p_memory(cfg, ids) < EX_SAVER_P:   # (the ledger's P on the move)
            protect, cands = {(k, key)}, []              # (a saver in conflict with another value held sends it back)
            for s in self._ex_savers(cfg, skip=keep, avoid=protect):
                got = self._ex_add_saver(engine, cfg, s, keep, protect, ids)
                if got is not None and not self._ex_rules(got[0], ids, saver=False):
                    cands.append((s, *got))

            def fits(c):
                return self._ex_p_memory(c[1], ids) >= EX_SAVER_P
            pick = (next((c for c in cands if not c[2] and fits(c)), None)   # the best that brings P to EX_SAVER_P
                    or next((c for c in cands if c[2] and fits(c)), None)    # sending nothing back, else sending a
                    or next((c for c in cands if not c[2]), None)            # value back; else the best by rank
                    or (cands[0] if cands else None))
            if pick is not None:
                saver, cfg, sent, keep = pick
        legal = self._complete(engine, dict(cfg), keep)
        if legal is None or any(vk(legal.get(x)) != vk(cfg.get(x)) for x in keep):
            return None
        legal = self._ex_kcfg(legal)
        if self._ckey(legal, ()) == self._ckey(inc_cfg, ()):
            return None
        p, _ = self._pf(legal, ids)
        mu, _ = self._mu(engine, inc, legal, ids)
        word = f'{k}={key}'
        note = (f'{why}: {word} on {inc}' + (f'; {", ".join(f["value"] for f in flipped)} back (conflict)' if flipped
                                             else '') + (f'; {s_word(saver)} added (saves memory)' if saver else '')
                + (f'; {", ".join(x["value"] for x in sent)} back (it conflicts with the saver)' if sent else ''))
        flipped = flipped + sent
        return {'why': why, 'value': word, 'key': [k, key], 'cfg': legal, 'change': self._ex_change(legal),
                'moved': self._ex_change(legal, inc_cfg), 'flipped': flipped,
                'saver': None if saver is None else {k2: saver[k2] for k2 in ('knob', 'value', 'tier', 'saving_gib')
                                                     if saver.get(k2) is not None},
                'p': round(p, 3), 'predicted_gain_pct': _pct(mu), 'note': note}

    def _ex_untested(self, move: dict, cmp: tuple) -> bool:
        """M1: no run holds the move's value with the move's other big-knob values (a failure whose cause the move
        changed - its partner sent back, a big saver added - holds other ones)."""
        k, key = move['key']
        others = [b for b in cmp if b != k]
        for r in self.runs.values():
            cfg = self._ex_kcfg(r.get('cfg'))
            if vk(cfg.get(k)) == key and all(vk(cfg.get(b)) == vk(move['cfg'].get(b)) for b in others):
                return False
        return True

    def _ex_kept_failure(self, move: dict) -> str | None:
        """M1's ledger test: a memory failure holding the move's value whose every value the move keeps, the gap not
        freed by what the move adds (a failure whose cause the move changes does not count)."""
        att = self._ex_attribution()
        v = tuple(move['key'])
        held = self._ex_vals(move['cfg'], tuple(self.chain))
        for f in att.get('failures', []):
            if f.get('cause') != 'memory' or v not in set(map(tuple, f['values'])):
                continue
            if not set(map(tuple, f['values'])) <= held:
                continue
            fcfg = self._ex_kcfg((self.runs.get(f['run']) or {}).get('cfg'))
            net, _ = self._ex_saved(move['cfg'], fcfg)
            if net + 1e-9 < max(self._ex_floor(), f.get('gap_gib') or 0.0):
                return f['run']
        return None

    def _ex_contra_list(self, engine, inc) -> list[dict]:
        if not self._ex_on('contrarian') or not self.prior or not self.knobs:
            return []
        inc_cfg = self._knob_cfg(engine, inc)
        ctx = self._ex_context(inc_cfg, engine)
        cmp = ctx['cmp']
        ids = tuple(self.chain)
        out = []
        ledger_p = self._ex_v4('v4_ledger_p', engine)
        for score, (k, key), v, e, reasons, opt in self._ex_contra_values(engine, inc_cfg):
            mv = self._ex_move(engine, inc, k, key, why='contrarian', inc_cfg=inc_cfg)
            if mv is None or self._ckey(mv['cfg'], ids) in self.tried:
                continue
            if not self._ex_untested(mv, cmp) or self._ex_kept_failure(mv) or self._ex_rules(mv['cfg'], ids):
                continue
            if not self._ex_adm(mv['cfg'], ctx, ids)[0] or mv['p'] < EX_P_MIN and \
                    self._ex_p_memory(mv['cfg'], ids, unblamed=bool(mv['saver'])) < EX_P_MIN:
                continue                                 # (not outlandish: the ledger's P - other jobs' failed starts
                #                                          lower its plan's P, not its eligibility)
            mv.update(score_pct=round(100 * score, 2), reasons=reasons,
                      prior={'gain_pct': _pct(e.get('mu')), 'sd_pct': _pct(e.get('sd')), 'p_runs': e.get('p_runs')})
            mv['note'] = (mv['note'] + f' ({"; ".join(reasons)}; optimistic {100 * math.expm1(opt):+.1f} %'
                          + (f' x p_runs {e["p_runs"]:.2f}' if _num(e.get('p_runs')) else '') + ')')
            if ledger_p:                                 # (v4 F9: ranked by the optimistic gain x the ledger's P instead)
                mv['score_v4_pct'] = round(100 * opt * min(1.0, max(0.0, float(mv['p'] or 0.0))), 2)
            out.append(mv)
        if ledger_p:
            out.sort(key=lambda mv: -mv['score_v4_pct'])
        return out

    def _ex_contrarian_moves(self, engine, inc) -> list[dict]:
        """M1's eligible moves on the incumbent, best first, as plain dicts (each with its 'cfg' and 'why':
        'contrarian'); [] once EX_CONTRA_MAX contrarian screens ran. A strategy that runs one through its own path is
        matched by its configuration at the observation."""
        self._ex_bind(engine)
        if self._ex_used('contrarian') >= self.EX_CONTRA_MAX:
            return []
        return self._ex_offer(self._ex_contra_list(engine, inc))

    def _ex_contrarian(self, engine, inc):
        """M1: the best eligible move as a ladder candidate (why 'contrarian'), or None (none eligible, EX_CONTRA_MAX
        contrarian screens ran, or every floor refused)."""
        self._ex_bind(engine)
        if self._ex_used('contrarian') >= self.EX_CONTRA_MAX:
            return None
        for mv in self._ex_contra_list(engine, inc):
            q = self._ex_candidate(engine, inc, mv)
            if q is not None:
                return q
        return None

    # ------------------------------------------------------------ M4: escape candidates
    def _ex_restart_base(self, engine, inc, inc_cfg: dict, cmp: tuple):
        """The warm restart's start: the best valid full screen of a different class (at least two compared knobs off
        the incumbent - all of them when fewer), or None."""
        need = min(2, len(cmp))
        best = None
        for cid, r in self.runs.items():
            y = self._ex_y(r)
            if cid == inc or y is None or r.get('origin') == 'reference':
                continue
            cfg = self._ex_kcfg(r.get('cfg'))
            d = sum(1 for k in cmp if vk(cfg.get(k)) != vk(inc_cfg.get(k)))
            if d >= need and (best is None or y > best[0]):
                best = (y, cid, cfg)
        return best

    def _ex_spread(self, x: dict, x_inc: dict, cmp: tuple) -> float:
        """The time model's posterior sd of a configuration's gain over the incumbent from the compared (big) knobs'
        values it moves: d' Sigma d with d its features less the incumbent's on those knobs (what both hold cancels; a
        small knob moved adds its mean to the gain, no spread to the optimism)."""
        names = self.surr.names
        if not names:
            return 0.0
        _, L = self.surr.posterior()
        big = set(cmp)
        d = []
        for f in names:
            if isinstance(f, tuple) and (f[0] in big or f[0] == 'pair'):
                d.append(float(f in x) - float(f in x_inc))
            else:
                d.append(0.0)
        y = [0.0] * len(d)
        for i in range(len(d)):
            y[i] = (d[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
        return math.sqrt(sum(t * t for t in y))

    def _ex_momentum_values(self) -> set:
        m = self.ex_momentum
        return {(x['knob'], x['next']) for x in (m or {}).get('moves', [])}

    def _ex_escape_rows(self, engine, inc, inc_cfg: dict, ctx: dict, base: tuple | None) -> list[dict]:
        """The escape's scored candidates, best first: the untried legal table (a warm restart: the base run's
        configuration with one compared knob moved, its coupled knobs completed), admissible, within the attribution's
        rules, by the time model's gain over the incumbent plus its spread, x P (P >= EX_P_MIN); momentum's first."""
        ids = tuple(self.chain)
        mean = self.surr.mean()
        x_inc = self._incumbent_feats(engine, inc)
        y_inc = Surrogate.value(x_inc, mean)
        if base is not None:
            rows, bcfg = [], base[2]
            for k in ctx['cmp']:
                for x in self.knobs.get(k) or []:
                    if vk(x) == vk(bcfg.get(k)):
                        continue
                    c = self._complete(engine, {**bcfg, k: x}, {k})
                    if c is not None and vk(c.get(k)) == vk(x):
                        rows.append(self._ex_kcfg(c))
        else:
            rows = [cfg for cfg, _ in self._candidates(engine, inc)]
        scored, seen = [], set()
        for cfg in rows:
            key = self._ckey(cfg, ids)
            if key in self.tried or key in seen:
                continue
            seen.add(key)
            scored.append((Surrogate.value(self._feats(cfg, ids), mean) - y_inc, len(scored), cfg))
        scored.sort(key=lambda t: (-t[0], t[1]))
        mom = self._ex_momentum_values()
        big = set(ctx['cmp'])
        groups: dict = {}                                # big-knob signature -> (small knobs moved, -mean, order, row)
        for mu, i, cfg in scored:                        # (an escape moves big knobs: of the rows with the same big
            if not self._ex_adm(cfg, ctx, ids)[0] or self._ex_rules(cfg, ids):   # knobs, the one that keeps most of
                continue                                 #  the incumbent's small ones, then the best mean)
            sig = self._ex_sig(cfg, ctx['cmp'])
            small = 0 if base is not None else sum(1 for k in self.knobs if k not in big
                                                   and vk(cfg.get(k)) != vk(inc_cfg.get(k)))
            row = (small, -mu, i, cfg)
            if sig not in groups or row[:3] < groups[sig][:3]:
                groups[sig] = row
        picked = []
        for small, neg, i, cfg in sorted(groups.values(), key=lambda t: (t[1], t[2]))[:EX_SPREAD_TOP]:
            sd = self._ex_spread(self._feats(cfg, ids), x_inc, ctx['cmp'])
            picked.append((-neg + sd, -neg, sd, cfg))
        picked.sort(key=lambda t: -t[0])
        out = []
        for opt, mu, sd, cfg in picked[:KNOB_SCAN_MAX]:
            if opt <= 0.0:                               # (not even optimistically faster than the incumbent: no escape
                break                                    #  - the strategy's own path goes on; picked is best first)
            p, _ = self._pf(cfg, ids)
            if p < EX_P_MIN:
                continue
            held = self._ex_vals(cfg, ids)
            m = sorted(f'{a}={b}' for a, b in mom if (a, b) in held)
            score = opt * p
            out.append({'why': 'escape', 'cfg': cfg, 'change': self._ex_change(cfg), 'moved': self._ex_change(cfg, inc_cfg),
                        'score': score, 'p': round(p, 3), 'predicted_gain_pct': _pct(mu), 'optimistic_pct': _pct(opt),
                        'score_pct': round(100 * score, 2), 'momentum': m or None})
        out.sort(key=lambda r: (not r['momentum'], -round(r['score'], 6), len(r['moved'])))
        return out

    def _ex_escape(self, engine, inc):
        """M4: while stalled, an escape candidate (why 'escape') - admissible (radius r big knobs off the incumbent, no
        near-duplicate, no repeated failure), within the attribution's rules, its optimistic gain over the incumbent above
        0, the best optimistic gain x P; a warm restart when due, momentum preferred - or None (no row is even
        optimistically faster: the strategy's own path goes on)."""
        self._ex_bind(engine)
        if not self._ex_on('escape') or not self._ex_stalled() or not self.knobs or self._left(engine) < 1:
            return None
        inc_cfg = self._knob_cfg(engine, inc)
        ctx = self._ex_context(inc_cfg, engine)
        base = None
        if self.ex_bad >= self.ex_restart_next:
            base = self._ex_restart_base(engine, inc, inc_cfg, ctx['cmp'])
            if base is None:
                self.ex_counts['restart_no_base'] += 1
                self.ex_restart_next = self.ex_bad + 2 * self.EX_STALL
        for b in ((base, None) if base is not None else (None,)):
            for mv in self._ex_escape_rows(engine, inc, inc_cfg, ctx, b):
                mv['radius'] = ctx['r']
                if b is not None:
                    mv['restart'] = b[1]
                mv['note'] = (f'escape (stalled {self.ex_bad} full screens, radius {ctx["r"]})'
                              + (f': a warm restart from {b[1]} (x{math.exp(b[0]):.3f}) with one big knob moved'
                                 if b is not None else f': {len(mv["moved"])} knob(s) off {inc}')
                              + (f'; momentum {", ".join(mv["momentum"])}' if mv.get('momentum') else '')
                              + f'; optimistic {mv["optimistic_pct"]:+.1f} % over {inc}, P {mv["p"]:.2f}')
                q = self._ex_candidate(engine, inc, mv)
                if q is not None:
                    return q
        return None

    # ------------------------------------------------------------ M3: repair material
    def _ex_send_back(self, cfg: dict, values, protect, keep: set, ids) -> tuple | None:
        """`cfg` with every value it holds that conflicts with one of `values` sent back to the value that removes the
        conflict (M1's resolution): (the configuration, the values sent back, the knobs kept), or None when a partner is
        one the move must keep."""
        cfg, kept, back = dict(cfg), set(keep), []
        for v in values:
            for (pk, pkey), src in self._ex_conflicts(v, set(self._ex_vals(cfg, ids))):
                if pk not in self.knobs or (pk, pkey) in protect or pk in kept:
                    return None
                away = self._ex_away(pk, pkey)
                if away is _MISS:
                    return None
                cfg[pk] = away
                kept.add(pk)
                back.append({'value': f'{pk}={pkey}', 'to': away, 'source': src})
        return cfg, back, kept

    def _ex_repair_covers(self, cfg: dict, f: dict, ids=()) -> tuple[bool, float]:
        """Whether a configuration removes a memory failure's cause: its net GiB against the failed configuration
        (`_ex_saved`: every value it changed - the savers, a value sent back, a coupled knob, a spender left out) reach
        the measured gap (a positive net when the gap is unknown), else the memory ledger's own P of it is at least 0.5
        (a price ranks, the ledger decides; read directly, never the strategy's `_pf`): (covered, the net GiB)."""
        kcfg = self._ex_kcfg(cfg)
        net = self._ex_saved(kcfg, self._ex_kcfg((self.runs.get(f['run']) or {}).get('cfg')))[0]
        gap = f.get('gap_gib')
        if (net + 1e-9 >= gap) if gap is not None else (net > 1e-9):
            return True, net
        led = self.mledger
        return led is not None and led.p_feasible(kcfg, ids=self._led_ids(tuple(ids or ()))) >= 0.5, net

    def _ex_repair_savers(self, engine, base: dict, keep: set, protect, f: dict, ids, *, sendback: bool) -> tuple:
        """A memory repair's savers, by bcomp's order while the failure's shortfall is not covered (`_ex_repair_covers`
        on the completed configuration; EX_SAVERS at most), each with `_ex_add_saver` (`sendback`: a saver in conflict
        with a value held - not a failed one - may send it back); a saver that does not raise the net GiB (its send-back
        costs what it frees) is passed over: (the configuration, the savers, the values sent back, the net GiB, the
        knobs kept, covered)."""
        cur, kept, got, sent = dict(base), set(keep), [], []
        covered, net = self._ex_repair_covers(cur, f, ids)
        for s in self._ex_savers(base, skip=keep, avoid=protect):
            if covered or len(got) >= EX_SAVERS:
                break
            if s['knob'] in kept or vk(cur.get(s['knob'])) != vk(self.ref.get(s['knob'])):
                continue
            add = self._ex_add_saver(engine, cur, s, kept, protect, ids)
            if add is None or add[1] and not sendback or self._ex_rules(add[0], ids, saver=False):
                continue
            c2, n2 = self._ex_repair_covers(add[0], f, ids)
            if not c2 and n2 <= net + 1e-9:
                continue
            cur, back, kept = add
            got.append(s)
            sent += back
            covered, net = c2, n2
        return cur, got, sent, net, kept, covered

    @staticmethod
    def _ex_partitions(members: list, untested: set) -> list[list]:
        """A conflict's repair directions: the non-empty proper subsets of its values to keep (the rest left out), each
        holding a value that has not run outside the failure when there is one (the value whose run failed)."""
        out = []
        for n in range(1, len(members)):
            for kept in itertools.combinations(sorted(members), n):
                if not untested or set(kept) & untested:
                    out.append(list(kept))
        return out

    def _ex_repair_drafts(self, engine, f: dict, inc_cfg: dict, ran: list, ids) -> list[tuple]:
        """A failure's repair drafts, (the move, its configuration, the knobs it keeps): memory - one, its savers covering
        the gap (none when no saver does); a conflict - one per direction (`_ex_partitions`), on the incumbent."""
        fcfg = self._ex_kcfg((self.runs.get(f['run']) or {}).get('cfg'))
        if f['cause'] == 'memory':
            failed = [tuple(v) for v in f['suspects'] if not v[0].startswith('patch:')]
            base = dict(inc_cfg) if failed else dict(fcfg)
            for k, key in failed:
                base[k] = self._ex_val(k, key)
            if any(x is _MISS for x in base.values()):
                return []
            keep = {k for k, _ in failed} if failed else {k for k in self.knobs if vk(fcfg.get(k)) != vk(self.ref.get(k))}
            protect = set(failed) if failed else set(self._ex_vals(fcfg, ids))
            got = self._ex_send_back(base, failed, protect, keep, ids)
            if got is None:
                return []                                # (a failed value's conflict with a value it must keep)
            base, back, keep = got
            done = self._complete(engine, dict(base), keep)   # (the coupled knobs first: a partner the table requires
            if done is None or any(vk(done.get(x)) != vk(base.get(x)) for x in keep):   # counts before a draft looks
                return []                                #  covered and takes no saver)
            base = self._ex_kcfg(done)
            pick = self._ex_repair_savers(engine, base, keep, protect, f, ids, sendback=False)
            if not pick[5]:                              # (the savers that send nothing back do not cover the gap)
                pick = self._ex_repair_savers(engine, base, keep, protect, f, ids, sendback=True)
            cfg, savers, sent, net, keep, covered = pick
            if not covered:
                return []                                # (no saver covers its shortfall: no repair)
            gap = f.get('gap_gib')
            dropped = [x['value'] for x in back + sent]
            note = (f'repair of {f["run"]} ({f["kind"]}' + (f', {gap:.1f} GiB to lose' if gap is not None else '')
                    + f'): {", ".join(self._ex_words(failed)) or "its configuration"}'
                    + (' on the incumbent' if failed else '')
                    + (' + ' + ', '.join(f'{s["knob"]}={s["vk"]}' for s in savers) + ' (saves memory)' if savers else '')
                    + (f'; {", ".join(dropped)} back (conflict)' if dropped else ''))
            mv = {'why': 'repair', 'run': f['run'], 'cause': 'memory', 'failed': self._ex_words(failed),
                  'savers': [{k2: s[k2] for k2 in ('knob', 'value', 'tier', 'saving_gib') if s.get(k2) is not None}
                             for s in savers], 'dropped': dropped, 'gap_gib': gap, 'note': note}
            return [(mv, cfg, keep)]
        members = [tuple(v) for v in f['set']]
        if f.get('alone') or len(members) < 2:
            return []                                    # (a value that failed alone: nothing to take away)
        untested = {v for v in members if not any(v in r for r in ran)}
        out = []
        for kept_vals in self._ex_partitions(members, untested):
            removed = [v for v in members if v not in kept_vals]
            cfg, keep, ok = dict(inc_cfg), set(), True
            for k, key in kept_vals:
                cfg[k] = self._ex_val(k, key)
                keep.add(k)
            for k, key in removed:                       # (a value left out that the incumbent holds: sent back)
                if vk(cfg.get(k)) != key:
                    continue
                away = self._ex_away(k, key)
                if away is _MISS or k in keep:
                    ok = False
                    break
                cfg[k] = away
                keep.add(k)
            if not ok or any(x is _MISS for x in cfg.values()):
                continue
            got = self._ex_send_back(cfg, kept_vals, set(kept_vals), keep, ids)   # (the kept values' other conflicts)
            if got is None:
                continue
            cfg, back, keep = got
            dropped = self._ex_words(removed) + [x['value'] for x in back if x['value'] not in self._ex_words(removed)]
            note = (f'repair of {f["run"]} ({f["kind"]}): {", ".join(self._ex_words(kept_vals))} on the incumbent without '
                    f'{", ".join(dropped)} (the conflict)')
            out.append(({'why': 'repair', 'run': f['run'], 'cause': 'conflict', 'failed': self._ex_words(kept_vals),
                         'dropped': dropped, 'note': note}, cfg, keep))
        return out

    def _ex_repair_list(self, engine, inc) -> list[dict]:
        att = self._ex_attribution()
        if att.get('off') or not att['failures']:
            return []
        inc_cfg = self._knob_cfg(engine, inc)
        ctx = self._ex_context(inc_cfg, engine)
        ids = tuple(self.chain)
        ran = [self._ex_run_vals(r) for r in self.runs.values() if r.get('kind') in RAN]
        out, seen = [], set()
        for f in reversed(att['failures']):
            for mv, cfg, keep in self._ex_repair_drafts(engine, f, inc_cfg, ran, ids):
                legal = self._complete(engine, dict(cfg), keep)
                if legal is None or any(vk(legal.get(x)) != vk(cfg.get(x)) for x in keep):
                    continue
                legal = self._ex_kcfg(legal)
                key = self._ckey(legal, ids)
                if key in seen or key in self.tried or self._ckey(legal, ()) == self._ckey(inc_cfg, ()):
                    continue
                if mv['cause'] == 'memory':              # (the completed configuration: its coupled knobs counted too)
                    covered, net = self._ex_repair_covers(legal, f, ids)
                    if not covered:
                        continue
                    mv['net_gib'] = round(net, 3)
                seen.add(key)
                if self._ex_rules(legal, ids) or not self._ex_adm(legal, ctx, ids)[0]:
                    continue                             # (M2's rules whole: a needs-a-saver value uncovered is refused)
                p, _ = self._pf(legal, ids)
                if p < EX_P_MIN and (mv['cause'] != 'memory' or self._ex_p_unblamed(legal, ids) < EX_P_MIN):
                    continue                             # (a saver compensates the blamed value: bcomp's completion)
                mu, _ = self._mu(engine, inc, legal, ids)
                mv.update(cfg=legal, change=self._ex_change(legal), moved=self._ex_change(legal, inc_cfg),
                          p=round(p, 3), predicted_gain_pct=_pct(mu), score=(mu if mu > 0 else mu / max(p, 1e-6)) * p)
                out.append(mv)
        out.sort(key=lambda r: -r['score'])
        return out

    def _ex_repair_candidates(self, engine, inc) -> list:
        """M3 material: the repairs of the attributed failures (the failed value(s) plus the change that removes the
        cause: memory - one per failure, savers covering its shortfall by net GiB (`net_gib`); a conflict - one per
        direction, the values left out), plain dicts with their 'cfg', 'why': 'repair', the failure (`run`, `cause`,
        `failed`), the `savers` and the values `dropped` (left out or sent back), P and the predicted gain; best (the
        time model's gain x P) first; [] once EX_REPAIR_MAX repair screens ran or the attribution is off. Matched by
        configuration at the observation when the strategy runs one."""
        self._ex_bind(engine)
        if not self._ex_on('repair') or self._ex_used('repair') >= self.EX_REPAIR_MAX or not self.knobs:
            return []
        return self._ex_offer(self._ex_repair_list(engine, inc))

    def _ex_repair(self, engine, inc):
        """The best repair as a ladder candidate (why 'repair'), or None."""
        self._ex_bind(engine)
        if not self._ex_on('repair') or self._ex_used('repair') >= self.EX_REPAIR_MAX or not self.knobs:
            return None
        for mv in self._ex_repair_list(engine, inc):
            q = self._ex_candidate(engine, inc, mv)
            if q is not None:
                return q
        return None

    def _ex_repair_of(self, cfg: dict) -> dict | None:
        """The attributed failure a configuration repairs, by the builder's own rules - memory: it holds the failure's
        failed value(s) and covers its shortfall (`_ex_repair_covers`: the net GiB against the failed configuration, else
        the ledger's own P >= 0.5); a conflict: it keeps a non-empty proper subset of the set, with a value that has not
        run outside the failure when there is one (`_ex_partitions`) - as {'run', 'cause'}, the latest failure first,
        else None (whether a batch already holds a repair)."""
        att = self._ex_attribution()
        if att.get('off'):
            return None
        ids = tuple(self.chain)
        held = self._ex_vals(cfg, ids)
        ran = [self._ex_run_vals(r) for r in self.runs.values() if r.get('kind') in RAN]
        for f in reversed(att['failures']):
            if f['cause'] == 'memory':
                failed = {tuple(v) for v in f['suspects']} or {tuple(v) for v in f['values']}
                if failed and failed <= held and self._ex_repair_covers(cfg, f, ids)[0]:
                    return {'run': f['run'], 'cause': 'memory'}
            elif not f.get('alone'):
                members = {tuple(v) for v in f['set']}
                kept = members & held
                untested = {v for v in members if not any(v in r for r in ran)}
                if kept and not members <= held and (not untested or kept & untested):
                    return {'run': f['run'], 'cause': 'conflict'}
        return None

    # ------------------------------------------------------------ outcomes
    def _ex_observe(self, engine, p, out) -> None:
        """From the strategy's observe (after the base's): the outcome of a candidate made here (or of a move the
        strategy ran its own way, matched by configuration), the plateau and the radius (full screens), momentum, the
        attribution's sticky exclusions."""
        self._ex_bind(engine)
        if not self.ex_ready and self.knobs:
            self._ex_init(engine, skip=out.cid)
        hid = p.hypothesis or ''
        key = (hid, out.cid, out.status, out.fidelity)
        if key in self.ex_seen:
            return
        self.ex_seen.add(key)
        for r in self.ex_log:                            # (one proposal runs per step: another made then did not run)
            if r.get('outcome') == 'proposed' and r.get('id') != hid:
                r['outcome'] = 'not run'
        row = self.ex_by_hid.get(hid)
        run = self.runs.get(out.cid) if out.cid else None
        if row is None and run is not None and out.status in DONE:
            mv = self.ex_offered.pop(self._ckey(self._ex_kcfg(run.get('cfg')), ()), None)
            if mv is not None:
                row = self._ex_row(mv, engine, hid=hid, p=getattr(p, 'p_feasible', None), plan=list(p.fidelity or ()))
                self.ex_log.append(row)
                self.ex_by_hid[hid] = row
                self.ex_counts[mv['why']] = self.ex_counts.get(mv['why'], 0) + 1
                self.ex_counts['matched'] += 1
        if row is not None:
            row.update(outcome=out.status, candidate=out.cid, fidelity=out.fidelity)
            if run is not None:
                row['kind'] = run.get('kind')
                if out.status == 'screened' and self._ex_y(run) is not None:
                    row['measured_gain_pct'] = _pct(self._ex_y(run))
        del self.ex_log[:-200]
        if out.fidelity == 'full' and out.cid and out.status in ('screened', 'aborted', 'invalid'):
            self.ex_full += 1
            y = self._ex_y(run) if run is not None else None
            if y is not None and y > self.ex_anchor + self._ex_thr(engine, EX_IMPROVE):
                prev = self.ex_anchor_cid
                self.ex_momentum = self._ex_momentum_of(prev, out.cid)
                self.ex_improved.append({'full_screen': self.ex_full, 'run': out.cid, 'gain_pct': _pct(y),
                                         'over_pct': _pct(y - self.ex_anchor), 'previous': prev,
                                         'moved': sorted(self._ex_change(self._ex_kcfg(run.get('cfg')), self._ex_kcfg(
                                             (self.runs.get(prev) or {}).get('cfg') or self.ref)))})
                self.ex_anchor, self.ex_anchor_cid, self.ex_bad, self.ex_found = y, out.cid, 0, True
                self.ex_restart_next = 2 * self.EX_STALL
            elif self.ex_found:                          # (no improvement on the reference yet: no optimum to leave,
                self.ex_bad += 1                         #  the plateau is not counted)
        self._ex_attribution()                           # (the sticky exclusions follow every observation)

    def _ex_order(self, k: str) -> list:
        vals = list(self.knobs.get(k) or [])
        if vals and all(_num(x) for x in vals):
            vals = sorted(vals)
        return vals

    def _ex_momentum_of(self, prev: str, cid: str) -> dict | None:
        """The extension of an improving move that changed one knob (or one and its coupled partner): for each changed
        knob of three values or more, the next value in the same direction."""
        a = self._ex_kcfg((self.runs.get(prev) or {}).get('cfg') or self.ref)
        b = self._ex_kcfg((self.runs.get(cid) or {}).get('cfg'))
        changed = [k for k in self.knobs if vk(a.get(k)) != vk(b.get(k))]
        if not 1 <= len(changed) <= 2:
            return None
        moves = []
        for k in changed:
            order = [vk(x) for x in self._ex_order(k)]
            if len(order) < 3 or vk(a.get(k)) not in order or vk(b.get(k)) not in order:
                continue
            i, j = order.index(vk(a.get(k))), order.index(vk(b.get(k)))
            n = j + (1 if j > i else -1)
            if 0 <= n < len(order):
                moves.append({'knob': k, 'from': order[i], 'to': order[j], 'next': order[n]})
        return {'run': cid, 'moves': moves} if moves else None

    # ------------------------------------------------------------ report
    def _ex_view(self) -> dict:
        """The mixin's account for describe(): its parts and limits, the screens each mechanism proposed and why, the
        plateau, the radius, the warm restarts and momentum, the attribution, the refusals."""
        self._ex_state()
        att = self._ex_attribution() if self.knobs else {'needs_saver': {}, 'excluded': {}, 'conflicts': [], 'failures': []}

        def rec(v, r):
            return {'value': self._ex_word(v), 'failures': r.get('failures'), 'with_saver': r.get('with_saver'),
                    'gap_gib': r.get('gap_gib')}
        return {'version': EX_VERSION, 'off': sorted(x for x in EX_PARTS if not self._ex_on(x)),
                'limits': {'contrarian': self.EX_CONTRA_MAX, 'repair': self.EX_REPAIR_MAX, 'stall': self.EX_STALL,
                           'radius_max': self.EX_RADIUS_MAX, 'revise': self.EX_REVISE_MAX},
                'screens': {w: self._ex_used(w) for w in ('contrarian', 'escape', 'repair')},
                'counts': dict(self.ex_counts),
                'stall': {'stalled': self._ex_stalled(), 'radius': self._ex_radius(), 'stalled_screens': self.ex_bad,
                          'counting': self.ex_found, 'full_screens': self.ex_full,
                          'best_gain_pct': _pct(self.ex_anchor), 'best_run': self.ex_anchor_cid,
                          'warm_restart_at': self.ex_restart_next, 'improvements': self.ex_improved[-10:],
                          'momentum': self.ex_momentum},
                'attribution': {'off': bool(att.get('off')),
                                'needs_saver': [rec(v, r) for v, r in sorted(att['needs_saver'].items())],
                                'excluded': [rec(v, r) for v, r in sorted(att['excluded'].items())],
                                'conflicts': [{'values': self._ex_words(c['members']), 'runs': c['runs'],
                                               'kinds': c['kinds'], **({'alone': True} if c.get('alone') else {}),
                                               **({'ambiguous': c['ambiguous']} if c.get('ambiguous') else {})}
                                              for c in att['conflicts']][:12],
                                'failures': len(att['failures'])},
                'decisions': [dict(r) for r in self.ex_log[-EX_LOG_MAX:]], 'refused': list(self.ex_refused[-20:])}


def s_word(s: dict | None) -> str:
    return f'{s["knob"]}={s["vk"]}' if s else ''
