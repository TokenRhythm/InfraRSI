"""bcomp v2: bundle completion over the memory ledger (FINALISTS_IMPL_SPEC_20261101 A.1, A.5; V2_SEARCH_DESIGN_20261101 M1,
M2, M4) on the finalists' shared base (search/finalist_base: S1 the code track by budget, S2 patch memory, S3 derived candidates
carry their run's patches, S4 the memory ledger as the only feasibility, S5 probe thrash by its peak, S6 no out-of-memory
streaks, the D5 ladder) and the shared exploration moves (search/explore, ExploreMixin: M1's contrarian moves, M4's stall,
trust region, tabu list and escape candidates, and the failure attribution M2 reads). v1 learned nothing from a failure but
its completion; v2 feeds every failure back (M2: rules from the attribution, a revise call of the guide), keeps its knob
screens busy when the code track is closed (the knob-only schedule), tests the guide's doubtful vetoes with conflict-resolved,
memory-completed moves (M1 in place of the challenge) and leaves a plateau (M4).

Screens (a budget of B; the knob side owns the first KNOB = min(STARTS + CONFIG_SCREENS, (B + 1) // 2) = 5 of 10, the code
track the rest, its screens protected - with the code track closed, the knob-only gate of finalist_base (code.mutable empty:
`_can_code()` False), the knob side keeps every screen):

    1     start one: the experience store's whole configuration - the arm's home's run packs (or the packs given), one
          source per numeric environment other than this job's, each source's best valid knob-only run mapped onto this
          table by xfer's _map_winner, the most similar source first (Stoll's best-first); no usable source: the guide's
          best; probed first below the base's probe_below (0.7; 0.5 where a probe costs over 0.6 of a full run)
    2     start two: the guide's (wcs's prior call's) best whole configuration that differs from start one in at least one
          big knob, else its next one (no whole configuration: the time model's best row by the guide's values); probed
          first below the base's probe_below
    3     start three: the first new draw of knob_space.random_search with the arm's seed that M2's rules allow (a draw
          that breaks one is skipped for the next legal draw: the control never repeats a failure the attribution
          explains), in full (the control start, as random draws it: no probe, whatever S4's P, the only S6 exemption and
          M4's: it never has to be admissible; hard rule 5 still probes a measured failed combination)
    4     a completion (first) or a contrarian move (M1; v1's challenge when ex_contrarian is off)
    5     the other kind (or the same when the other has none); neither: an escape candidate while stalled (M4), then the
          ledger's best untried configuration - the time model's gain x P, P >= 0.5, at least one big knob off every run
    6-10  the code track open (v1, unchanged): code (S1); nothing to run there: llmopt's tail bundle, then the
          feasibility-weighted draw. The code track closed (the knob-only schedule): in order, a completion of a failure,
          a contrarian move (M1), an escape candidate while stalled (M4), the ledger's best untried configuration, and
          only then the tail bundle and the draw - where v1 went straight to the tail and the draw, which kept composing
          the values that had just failed
A start or a configuration screen whose probe fails spends no screen: the start is consumed (its failure is completion
material) and the next step takes the screen; a knob side with nothing to propose gives its screen to the code track (an
ask, as S1 rules; none when it is closed), then the tail and the draw; S6 refuses a knob candidate below P 0.9 after two
knob-side runs out of memory in a row (the start stays open, the screen goes to code); a start M4's stall gate refuses
stays open too (its next option is tried; the refused one comes back once the stall lets it through).

completion  `complete()`: a run that failed for memory (out of memory or thrash, at a probe too: failed_material) whose
            configuration the time model rates (an unmeasured patch it carries by the model's own prediction, shrunk) at
            COMPLETE_MIN_GAIN = max(2 %, 2 sigma_pair) or more over the incumbent and whose gap the ledger knows (out of
            memory: its shortfall; thrash: its peak less THRASH_LO of the card; else the ledger's need over the card), plus
            one partner - a value of a knob the run holds at the reference's value (the run + the partner keeps every value
            of the run): the values whose measured price tag frees memory, by the GiB they free, then the declared savers
            with a price, by the GiB, then the declared savers without one and the values of unknown memory direction, by
            the guide's gain (a price ranks, it never vetoes - until M2 says a value of the run needs a saver: then only
            savers covering its shortfall complete it, M2's rule below); a held value the pair table says conflicts with
            the partner goes back to the reference's (hls3 F5c), the coupled knobs completed (wcs _complete); predicted
            faster than the incumbent;
            carrying the run's patches with the chain (S3, derived_q); the ledger's P (below COMPLETE_P_MIN = 0.15, the
            declared-memory floor, isolated_blame's factor aside, the runs rule it out: not proposed), probed first below
            the base's probe_below (0.7; 0.5 under its ratio rule) on the knob side's allowance - a carried unproven patch's
            proof from the base's code pool (B.9 F2); COMPLETE_PER_RUN = 2 per run (the second with another partner). v2
            (bcomp_feedback): the completion queue - every failed run x its first PARTNER_SCAN partners - is re-ranked at
            each pick with the updated ledger: the original failures before failed completions, then the predicted gain x
            P; a queued completion that breaks an M2 rule is dropped (`bc_dropped`, with the rule). v1's order (the first
            that passes) with bcomp_feedback off
feedback (M2)  the failure attribution (the shared `_ex_attribution`, rebuilt after every observation) turned into rules
            every knob candidate obeys, read with S4's feasibility (`_pf`: the shared `_ex_rules`; a candidate that breaks
            one is ruled out - P 0 - and says which rule, after M2_TAG): a value in EX_NEEDS = 2 or more memory failures (out
            of memory, thrash, at a probe too) and in no fitted run needs a saver - a candidate holding it must hold savers
            the leanest of those failures lacked whose freed GiB (the ledger's price tags, else the guide's declared GiB)
            cover that failure's gap, or with the gap unknown or a saver without a price the ledger's P (S4's, without M2's
            rules) at 0.5 or more; a value in EX_EXCLUDE = 3 memory failures that held a saver is excluded for the rest of
            the arm; the smallest set of values a failure other than memory pins (a crash, a hang, a start timeout: a pair,
            else a triple, absent from every run that ran) is a conflict no later candidate holds whole - each value alone
            stays allowed (a value that failed alone is one)
revise (M2)  the failures fed back into the guide: after the second failed knob-side run or probe since the last revise
            call (or one failure of a value the guide declared to run with p_runs >= REVISE_P_RUNS = 0.7 and that no
            fitted run holds), one model call with the guide's levels of the knobs involved (the failures' values that no
            fitted run holds - all of a failure's values when each fitted somewhere -, at most REVISE_VALUES) and the
            measured history (the base's history rows: failures with their cause, successes with their speed, peaks),
            answering revised levels (gain, sd, p_runs, memory) for those values; schema-checked as the prior call
            (REVISE_SCHEMA; the proposer's `knob_revise` when it has one, else an LLMProposer's driver as its prior call
            makes it: the session's tokens, the call's allowances, the proposer's call allowance, role 'revise'); merged
            into the guide the program uses - the prior's levels (M1 and the partners read them), the time model's prior
            for each value (wcs's rule: sd at least half the gain), the ledger's declared GiB, the memory model's declared
            moves; at most EX_REVISE_MAX = 2 calls per arm, inside the arm's calls and dollars (BudgetDriver; while the
            code track can use them S1's last CODE_CALLS calls stay its own, as for a configuration call); any failure of
            the call - none made, an error, no usable answer, an answer that does not merge - leaves every estimate as it
            was
contrarian  (M1) the shared `_ex_contrarian`: the value the guide rates low with a sign its own spread leaves open (or a
            declared conflict or memory veto a partner change removes), not outlandish (p_runs >= 0.3, no measured
            infeasibility in a context it keeps), untested with the incumbent's other big values - built on the incumbent
            with its conflicts resolved and, when it spends memory, the best saver added; ranked by (gain + sd) x p_runs;
            EX_CONTRA_MAX = 2 per arm through the ladder (the module's proposal by the base's `_knob_q`). It replaces v1's
            `challenge()` (the vetoed value with the largest |declared gain| that no run holds, on the incumbent with its
            switch group alone - M1's special case: one knob, no conflict resolution, no saver; kept with its
            CHALLENGE_MAX = 2 for ex_contrarian off)
escape      (M4) the shared stall tracker (`_ex_stalled`: the best has not improved by max(1 %, 2 sigma_pair) over the last
            EX_STALL = 2 full screens), trust region, tabu list and escape candidates: while stalled every knob candidate -
            the starts but the uniform one, completions, contrarian and escape moves, the ledger's best and, with the code
            track closed, the tail and the draw - is admissible (at least the radius in big knobs off the incumbent, no
            near-duplicate of a run, never a failure repeated) before it is run: a candidate built here by
            `_ex_admissible` (`_knob_q`, `derived_q`: a refusal counted in both views), the table's rows the ledger's best
            and the draw read by the module's own bulk test against one context per step (`_candidates`, `_ex_adm`); a
            tail bundle a refusal held back waits for the stall to end; the draw reads only the admissible rows that reach
            its own floors (P_MIN; with no probe left the ladder's probe_below; under S6, S6_P) - the draw keeps a row below
            them only when none reaches them, which M4's filter alone must never cause - and when none does it runs with M4
            relaxed, M2's rules, S6 and the feasibility kept (the arm never ends on admissibility); the escape candidate
            (`_ex_escape`: a move of the radius, a warm restart from the best run of another class, momentum) takes its
            place in the order above. The plateau counts from the first full screen that improved on the reference (the
            module's). With the code track open, screens 6-10 stay v1's
isolated_blame  the base's (S4, `FinalistBase.isolated_blame`): a value is lowered alone only when its every failed run
            holds no other value that never fitted

bcomp v4 (FABLE_CODE_DIAGNOSIS_20261102; the base's v4 gate: the code track open, beyond ten screens - finalist_base's module
docstring): with the code track open and nothing on it, the tail and the draw run under F1's floor (`_bc_tail_draw` through
the base's `_fb_v4_step`): a pick below the two-sigma floor or P 0.5 gives its screen to the code track - a pending patch,
F2's stack test, an ask -, and F6: a step with nothing to run (an empty ask, S6 holding the knob side) goes to the code track
too, asked again while an ask brings nothing to run and makes a call, so the arm no longer ends with screens and money left
while calls remain; the knob side's own screens (the starts, the configuration screens) are v3's. F2-F5 and F7-F10 are the
base's and the shared module's.

Every decision is in the report (`bcomp_view`): which mechanism proposed each full screen and why (`screens`), the
contrarian and escape moves, the completion queue and what M2's rules dropped, the rules themselves, each revise call
(why, the values asked, what changed), the admissibility refusals, and the shared module's own view.

    bcomp_ledger          the memory ledger as the feasibility (S4); off: wcs's
    bcomp_complete        the completions; off: the contrarian moves (or the challenges) and the ledger's best fill the
                          configuration screens
    bcomp_challenge       v1's challenges - only with ex_contrarian off (M1 takes their slot otherwise)
    bcomp_isolated_blame  S4's isolated_blame (the base's fb_isolated_blame)
    bcomp_peak_thrash     S5's probe thrash by its peak (the base's fb_peak_thrash)
    bcomp_feedback        M2's rules on every knob candidate and the re-ranked completion queue (the module's part
                          'attribution'); off: v1's completions and feasibility
    bcomp_revise          M2's revise calls; off: the guide stays the prior call's
    ex_contrarian         M1's contrarian moves in the challenge's slot (the module's part 'contrarian'); off: v1's
                          challenges
    ex_escape             M4: the admissibility while stalled and the escape candidates (the module's parts 'escape' and
                          'repair' - the latter's never-repeat rule; bcomp uses no repair material); off: neither

Nothing here names a knob, a model or a library: every value comes from the table, the guide, the ledger, the home's packs
and the runs.
"""
from __future__ import annotations

import itertools
import json
import math
import random

from . import BCOMP_COMPONENTS, knob_space
from .explore import ExploreMixin
from .finalist_base import (CARD_RESERVE, CODE_CALLS, FITTED, KNOB_SCAN_MAX, MEM_FAIL, P_FLOOR_DECLARED, RATE_SCREENS, S6_P,
                            STRONG_MIN, FinalistBase, rate_screens, vk)
from .hls3 import DIRS, TOGETHER
from .ladder import screens_budget
from .ledger import SD_DECLARED, SD_DECLARED_MIN
from .ledger import SD_UNKNOWN as LEDGER_SD_UNKNOWN
from .llmopt import P_MIN
from .proposers import LLMProposer, _allowances, _say_error
from .wcs import SD_FLOOR, SD_UNKNOWN, Surrogate, _coerce, _num
from .xfer import XFer, sources_from_packs

BCOMP_VERSION = 'v2'
STARTS, CONFIG_SCREENS = 3, 2   # the starts (experience, guide, uniform) and the configuration screens after them
COMPLETE_MIN_GAIN = STRONG_MIN  # a failed run is completed when the time model rates it max(2 %, 2 sigma_pair) over the inc.
COMPLETE_PER_RUN = 2            # completions per failed run at most (the second with another partner)
COMPLETE_P_MIN = P_FLOOR_DECLARED   # a completion the ledger puts below this (isolated_blame's factor aside: the completion
#                                 compensates the blamed value) is not proposed: only measurements push P below the
#                                 declared-memory floor - the runs say it does not fit
CHALLENGE_MAX = 2               # v1's challenges per arm at most (ex_contrarian off)
CHALLENGE_PROBE_BELOW = 0.5     # a challenge is probed first below this P (it is a controlled run: in full otherwise)
LEDGER_P = 0.5                  # the ledger's best untried configuration: P at least this
P_RUNS_LOW = 0.5                # the guide says the value runs with P below this: declared not to fit
SAVE_MIN = 0.005                # a partner saves memory when it frees at least this share of the card (the ledger's tags)
PARTNER_SCAN = 8                # partners tried per failed run and step at most (the best first)
UNIFORM_SCAN = 1000             # draws read for the first new legal one at most
EX_PART_OF = {'contrarian': 'ex_contrarian', 'escape': 'ex_escape', 'repair': 'ex_escape',   # the shared module's parts
              'attribution': 'bcomp_feedback'}                                                # (`_ex_on`) by component
REVISE_EVERY = 2                # M2: a revise call after this many failed knob-side runs or probes since the last one
REVISE_P_RUNS = 0.7             # M2: ... or after one failure of a value the guide declared to run with p_runs >= this
REVISE_VALUES = 12              # M2: the values one revise call asks about, at most
STEPS_SHOWN = 6                 # the report's picks per screen of the budget (finalist_base.rate_screens: 60 at ten screens);
#                                 its per-screen record (`screens`) keeps every full screen (F1, the open wave's 20)
FAIL_KINDS = MEM_FAIL + TOGETHER + ('numerics',)   # the failed runs (a probe too) that feed the revise call
M2_TAG = 'M2: '                 # (the reason of a candidate M2's rules rule out begins with this)
START_SLOTS = ('experience', 'guide', 'uniform')
CONFIG_KINDS = ('complete', 'challenge', 'contrarian', 'escape', 'ledger')
KNOB_ONLY_ORDER = ('complete', 'contrarian', 'escape', 'ledger')   # screens 6-10 with the code track closed, in order
REVISE_SYSTEM = """You revise your own earlier estimates (the guide) for some values of one training job's knobs, in the
light of what the search has measured since. Given: the job (task view), the knob table of the knobs involved (each
knob's values, the reference's marked), `guide` - your earlier estimate for each of their values (gain_pct, sd_pct, p_runs,
memory) -, `measured` - every configuration run so far, best first: its knob values that differ from the reference and
its outcome (the measured speed over the reference, or how it failed: out of memory and by how many GiB, thrash at the
memory edge, a slow start, a crash) with its measured memory peak per GPU when one was read -, and `revise`: the values
whose estimates the failures put in question. For each value in `revise` give your revised `gain_pct` (its expected speed
gain over the reference when it runs, in percent), `sd_pct` (your one-sigma uncertainty), `p_runs` (the probability that
it runs at all), `memory`: {"direction": "spends" | "frees" | "neutral" | "unknown", "gib": the GiB per GPU it frees or
adds, null when you cannot say} and `why` (one line: the measurement that moved it). Keep an estimate the measurements do
not speak to; say "unknown" (null) rather than guess. Reply with exactly one JSON object: {"levels":[{"knob","value",
"gain_pct","sd_pct","p_runs","memory":{"direction","gib"},"why"}],"notes":""}."""
REVISE_SCHEMA = {'type': 'object', 'required': ['levels'], 'properties': {
    'levels': {'type': 'array', 'items': {'type': 'object', 'required': ['knob', 'value'], 'properties': {
        'knob': {'type': 'string'}, 'gain_pct': {'type': ['number', 'null']}, 'sd_pct': {'type': ['number', 'null']},
        'p_runs': {'type': ['number', 'null']}, 'memory': {'type': ['object', 'null']}, 'why': {'type': ['string', 'null']}}}},
    'notes': {'type': ['string', 'null']}}}


def _llm_revise(prop, engine, payload: dict) -> dict | None:
    """M2's revise call through an LLMProposer, as its prior call is made (proposers.LLMProposer.knob_prior): no call
    without a model or without the session's tokens; the call's allowances (call_allowances); the proposer's own call
    allowance (`_spend`, which the strategy sets from its calls left); the schema-checked answer; charged to the session as
    role 'revise'. The answer as the model gave it, None without one."""
    s = engine.s
    led = getattr(s, 'ledger', None)
    if prop.driver is None or led is None or led.used['llm_tokens'] >= led.budget.llm_tokens:
        return None
    body = {'task': prop.hooks.task_view(), **payload}
    facts = getattr(prop.hooks, 'memory_facts', None)
    if callable(facts):
        body['memory_facts'] = facts()
    prompt = json.dumps(body, ensure_ascii=False, default=str)
    kw = _allowances(s, prop.driver, REVISE_SYSTEM + prompt, prop.max_tokens)
    if kw is None or not prop._spend():
        return None
    prop.calls += 1
    res = prop.driver.complete(REVISE_SYSTEM, prompt, schema=REVISE_SCHEMA, role='revise', **kw)
    s.charge_llm(res, 'revise')
    _say_error(engine, res, 'revise call')
    if not isinstance(res.action, dict):
        prop.failed += 1
        return None
    return res.action


class BComp(ExploreMixin, FinalistBase):
    """bcomp v2 (the module docstring). `off`: its components switched off (BCOMP_COMPONENTS); every other keyword goes to
    the shared base (packs / brief / env given instead of the home's, the model budget, facts, experience, seed, ...)."""
    name = 'bcomp'
    supports_token_free = True

    # xfer's mapping of another environment's winner onto this table (its registry labels in `labels`), borrowed as wcs
    # borrows hls3's methods: no inheritance, no other xfer state
    _mech, _state, _map_winner = XFer._mech, XFer._state, XFer._map_winner

    def __init__(self, proposer, *, off=(), **kw):
        off = set(off or ())
        self.bc_on = {c: c not in off for c in BCOMP_COMPONENTS}
        fb_off = [fb for fb, c in (('fb_isolated_blame', 'bcomp_isolated_blame'), ('fb_peak_thrash', 'bcomp_peak_thrash'))
                  if not self.bc_on[c]]
        super().__init__(proposer, fb_off=fb_off, ledger=self.bc_on['bcomp_ledger'], **kw)
        self.labels: dict = {}                           # knob -> its registry label (xfer's _map_winner reads it)
        self.bc_knob_screens = STARTS + CONFIG_SCREENS   # the knob side's screens (set from the budget at the start)
        self.bc_slots = {s: {'state': 'open'} for s in START_SLOTS}   # open / proposed / consumed / none
        self.bc_options: dict[str, list] = {}            # a start's options (experience: the mapped sources' winners)
        self.bc_by_hid: dict[str, dict] = {}             # proposal id -> what bcomp proposed it as
        self.bc_cid_what: dict[str, str] = {}            # candidate -> what it ran as (start, complete, contrarian, ...)
        self.bc_config: list[str] = []                   # the configuration screens' kinds, in order (full runs)
        self.bc_completions: list[dict] = []             # every completion proposed
        self.bc_challenges: list[dict] = []              # every challenge proposed (v1's, ex_contrarian off)
        self.bc_contras: list[dict] = []                 # every contrarian move proposed (M1)
        self.bc_escapes: list[dict] = []                 # every escape candidate proposed (M4)
        self.bc_steps: list[dict] = []                   # each step's pick
        self.bc_screens: list[dict] = []                 # each full screen: the mechanism that proposed it, why, its outcome
        self.bc_notes: list[str] = []
        # M2: the failures fed back
        self.bc_queue: list[dict] = []                   # the last completion queue, re-ranked (its head)
        self.bc_dropped: list[dict] = []                 # queued completions the rules dropped, with the rule
        self.bc_fails = 0                                # failed knob-side runs and probes observed
        self.bc_fails_at_revise = 0                      # ... at the last revise call
        self.bc_revise_due: str | None = None            # why a revise call is due (a failure sets it, the next step makes it)
        self.bc_revise_calls = 0                         # revise calls made (a model call spent or attempted)
        self.bc_revisions: list[dict] = []               # every revise call: why, the values asked, what changed, outcome
        self.bc_contradicted: set = set()                # values whose failure contradicted a declared p_runs >= 0.7
        # M4: the admissibility while stalled
        self._bc_gate = False                            # (inside a knob-side step while stalled: candidates admissible)
        self._bc_pool_floor: float | None = None         # (the stalled draw: its pool's rows at this P or more)
        self._bc_admit_memo: dict = {}
        self._bc_admit_stamp = None
        self.bc_admit_refused = False                    # (the last gated proposal was refused as not admissible)
        self.bc_start_held = False                       # (the last start step kept an option M4's gate refused)
        self.bc_refusals = 0                             # knob candidates refused while stalled
        self.bc_refused: list[dict] = []                 # ... the last ones, with why

    # ------------------------------------------------------------ set-up
    def _init(self, engine) -> None:
        super()._init(engine)
        b = int(getattr(engine.s.budget, 'screens', 0) or 0)
        self.bc_knob_screens = max(1, min(STARTS + CONFIG_SCREENS, (b + 1) // 2))
        self.protected_screens = frozenset(range(self.bc_knob_screens + 1, b + 1))   # S1: the code track's screens
        self._ex_init(engine)

    def rebudget(self, engine) -> None:
        """Resume (FinalistBase.rebudget): the caps of the base and of the shared module, then the knob side's screens and the
        protected code screens from the resumed run's budget, as `_init` sets them."""
        super().rebudget(engine)
        b = int(getattr(engine.s.budget, 'screens', 0) or 0)
        self.bc_knob_screens = max(1, min(STARTS + CONFIG_SCREENS, (b + 1) // 2))
        self.protected_screens = frozenset(range(self.bc_knob_screens + 1, b + 1))

    def _ex_on(self, name: str) -> bool:
        """The shared module's parts by bcomp's components (EX_PART_OF): 'contrarian' - ex_contrarian; 'escape' and
        'repair' (its never-repeat rule; bcomp runs no repair material) - ex_escape; 'attribution' - bcomp_feedback."""
        comp = EX_PART_OF.get(name)
        return self.bc_on.get(comp, True) if comp else super()._ex_on(name)

    def _knob_strong(self, engine, inc):
        """bcomp's knob side has its own screens and gives one to the code track only when it has nothing to propose
        (A.1: a knob side without candidates early goes to code); on the code screens an ask is protected - so no knob
        candidate holds an ask back."""
        return None

    # ------------------------------------------------------------ proposals
    def propose(self, engine, k):
        from .token_governor import TokenGovernor
        tokens_spent = TokenGovernor(engine.s.ledger).exhausted
        if tokens_spent and self.prior is None:
            while self.pending:
                q = self.pending.pop(0)
                if engine._fresh(q):
                    return [q]
            return []
        if self.prior is None:
            self._init(engine)
        self.engine = engine
        left = self._left(engine)
        if left <= 0:
            return []
        inc = engine.incumbent()
        while self.pending:
            q = self.pending.pop(0)
            if engine._fresh(q):
                return [q]
        tokens_spent = TokenGovernor(engine.s.ledger).exhausted
        q = self._bc_zero_token(engine, inc, left) if tokens_spent else self._bc_step(engine, inc, left)
        return [q] if q is not None else []

    def _bc_zero_token(self, engine, inc, left):
        """Spend remaining screens on prepared patches and measured memory/exploration moves, without a model call."""
        if self.on['wcs_code']:
            q = self._code_step(engine, inc, left, ask=False)
            if q is not None:
                return self._bc_note(engine, 'code', q, why='queued before the token budget was spent')
        for move in (self.complete, self._bc_contrarian, self._bc_escape):
            q = move(engine, inc)
            if q is not None and engine._fresh(q):
                return q
        return None

    def _bc_step(self, engine, inc, left):
        self._bc_revise(engine)                          # M2: a revise call when a failure made one due
        knob_only = not self._can_code()
        if engine.full_screens() < self.bc_knob_screens:     # the knob side's screens
            q = self._bc_gated(self._bc_knob, engine, inc)
            if q is not None:
                return q
            q = self._bc_code(engine, inc, left, 'the knob side has nothing to propose')
        elif knob_only:                                      # v2: the knob-only schedule (no code track)
            q = self._bc_gated(self._bc_knob_only, engine, inc)
        else:                                                # the code track's screens (v1)
            q = self._bc_code(engine, inc, left, 'a code screen')
        if q is not None:
            return q
        if knob_only:
            return self._bc_rest(engine, inc, left)
        return self._fb_v4_step(engine, inc, left, lambda: self._bc_tail_draw(engine, inc, left))

    def _bc_tail_draw(self, engine, inc, left):
        """With the code track open and nothing on it: llmopt's tail bundle, then the draw - v4 under F1's floor (the base's
        `_fb_v4_step`: a step the floor empties goes to the code track, F6 an empty one too)."""
        q = self._tail_step(engine, inc, left)
        if q is not None:
            return self._bc_note(engine, 'tail', q, why="llmopt's tail bundle on the incumbent")
        return self._bc_note(engine, 'draw', self._fallback(engine, inc), why='the feasibility-weighted draw')

    def _fb_v4_note(self, engine, q, why: str) -> None:
        """v4: a screen the code track took for the knob side (F1's floor, F6), in the steps."""
        self._bc_note(engine, 'code (v4)', q, why=why)

    def _bc_rest(self, engine, inc, left):
        """With the code track closed, after bcomp's own mechanisms: the tail bundle, then the draw - admissible ones only
        while stalled (M4); a tail bundle a refusal held back is offered again once the stall ends. While stalled the draw
        reads the admissible rows that reach its own floors (`_bc_draw_floor`: the draw keeps a row below them only when
        none in its pool reaches them - M4's filter must never be what leaves a doubtful row to run in full, V2B_r1
        P1-2); when no admissible row reaches them the draw runs with M4 relaxed - M2's rules, S6 and the feasibility kept
        (the arm never ends on admissibility)."""
        self.bc_admit_refused = False
        q = self._bc_gated(self._tail_step, engine, inc, left)
        if q is None and self.bc_admit_refused and self.tail_base is not None and self.tail_base[2] == 'waiting':
            self.tail_base = (self.tail_base[0], self.tail_base[1], 'ready')
        if q is not None:
            return self._bc_note(engine, 'tail', q, why="nothing of bcomp's own to run: llmopt's tail bundle on the incumbent")
        if not self._bc_stalled():
            return self._bc_note(engine, 'draw', self._fallback(engine, inc),
                                 why="nothing of bcomp's own to run and no tail: the feasibility-weighted draw")
        floor = self._bc_draw_floor(engine)
        self._bc_pool_floor = floor
        try:
            q = self._bc_gated(self._fallback, engine, inc)
        finally:
            self._bc_pool_floor = None
        if q is not None:
            return self._bc_note(engine, 'draw', q, why='stalled: the feasibility-weighted draw over the admissible rows '
                                                        f'(M4) at P {floor:.2f} or more')
        return self._bc_note(engine, 'draw', self._fallback(engine, inc),
                             why=f'stalled and no row of the table admissible at P {floor:.2f} or more (M4): the draw with '
                                 'M4 relaxed - M2, S6 and the feasibility kept')

    def _bc_draw_floor(self, engine) -> float:
        """The draw's own floors as one P (FinalistBase._fb_draw over llmopt's weighted_draw): P_MIN; with no probe left
        the ladder's probe_below (a doubtful row would run in full); under S6, S6_P."""
        p = P_MIN
        if not self.ladder.affordable(engine, 'probe'):
            p = max(p, self.ladder.probe_below)
        if self.s6_active():
            p = max(p, S6_P)
        return p

    def _bc_code(self, engine, inc, left, why: str):
        """The code track (S1): a queued patch, else an ask when one is due."""
        if not self.on['wcs_code']:
            return None
        n0 = self.asks
        q = self._code_step(engine, inc, left)
        if self.asks != n0:
            self.bc_steps.append({'screens_done': engine.full_screens(), 'pick': 'code ask', 'why': why})
        return self._bc_note(engine, 'code', q, why=why)

    def _bc_note(self, engine, pick: str, q, **kw):
        if q is not None:
            self.bc_steps.append({'screens_done': engine.full_screens(), 'pick': pick, 'id': q.hypothesis, **kw,
                                  **({'stalled': True} if self._bc_stalled() else {})})
        return q

    def _bc_knob(self, engine, inc):
        """The knob side's step: the open starts in order (one an S6 refusal held back stays open), then a configuration
        screen."""
        for slot in START_SLOTS:
            st = self.bc_slots[slot]
            if st['state'] != 'open':
                continue
            self.fb_s6_refused = False
            q = self._bc_start(engine, inc, slot)
            if q is not None:
                st.update(state='proposed', id=q.hypothesis)
                return q
            if self.fb_s6_refused or self.bc_start_held:
                continue                                 # (S6 or M4's stall gate held it back: offered again later)
            st['state'] = 'none'
            self.bc_steps.append({'screens_done': engine.full_screens(), 'pick': f'{slot} start: nothing to propose'})
        return self._bc_config(engine, inc)

    def _bc_record(self, engine, q, what: str, **kw):
        self.bc_by_hid[q.hypothesis] = {'what': what, 'id': q.hypothesis, 'screens_done': engine.full_screens(),
                                        'plan': list(q.fidelity), 'p_feasible': round(q.p_feasible, 3), **kw}
        self.bc_steps.append({'screens_done': engine.full_screens(), 'pick': what, 'id': q.hypothesis,
                              **({'why': str(kw['why'])[:200]} if kw.get('why') else {}),
                              **({'stalled': True} if self._bc_stalled() else {})})
        return q

    # ------------------------------------------------------------ M4: the admissibility while stalled
    def _bc_stalled(self) -> bool:
        """M4: the shared stall tracker says the search is stalled (ex_escape on)."""
        return bool(self.bc_on['ex_escape'] and self.prior is not None and self._ex_stalled())

    def _bc_gated(self, fn, *a, **kw):
        """`fn`'s knob candidates - admissible ones only while stalled (M4): `_knob_q`, `derived_q` and the table's rows
        (`_candidates`) read the gate."""
        old, self._bc_gate = self._bc_gate, self._bc_stalled()
        try:
            return fn(*a, **kw)
        finally:
            self._bc_gate = old

    def _bc_admissible(self, engine, inc, cfg: dict, *, row: bool = False) -> tuple[bool, str]:
        """The shared admissibility of a knob configuration against the incumbent's (the trust region, the tabu list, never
        a failure repeated), read once per configuration and observation: a candidate built here by `_ex_admissible` (a
        refusal counted and kept in the module's view); a row of the table (`row`) by the module's own test against one
        context built per step (`_ex_context`, `_ex_adm`: what its escape and contrarian lists read - a table's thousands
        of rows neither rebuild the context nor fill the refusal log)."""
        if self._bc_admit_stamp != (self.fb_obs, inc):
            inc_cfg = self._knob_cfg(engine, inc)
            self._bc_admit_memo, self._bc_admit_stamp = {'inc': inc_cfg, 'ctx': None}, (self.fb_obs, inc)
        memo = self._bc_admit_memo
        kcfg = {k: cfg.get(k, self.ref.get(k)) for k in self.knobs}
        key = (row, self._ckey(kcfg, ()))
        got = memo.get(key)
        if got is None:
            if row:
                if memo['ctx'] is None:
                    memo['ctx'] = self._ex_context(dict(memo['inc']), engine)
                ok, why = self._ex_adm(kcfg, memo['ctx'], tuple(self.chain))
            else:
                ok, why = self._ex_admissible(kcfg, dict(memo['inc']))
            got = memo[key] = (bool(ok), str(why or ''))
        return got

    def _bc_refuse(self, engine, kind: str, cfg: dict, why: str) -> None:
        self.bc_admit_refused = True
        self.bc_refusals += 1
        self.bc_refused = (self.bc_refused + [{'screens_done': engine.full_screens(), 'kind': kind,
                                               'change': self._words_of(cfg), 'why': why[:200]}])[-20:]

    def _knob_q(self, engine, inc, cfg: dict, **kw):
        """The base's proposal of a whole configuration; while stalled (M4's gate) only an admissible one - the uniform
        start (S6's exemption, `s6_exempt`) apart."""
        if self._bc_gate and not kw.get('s6_exempt'):
            ok, why = self._bc_admissible(engine, inc, cfg)
            if not ok:
                self._bc_refuse(engine, str(kw.get('kind') or ''), cfg, why)
                return None
        return super()._knob_q(engine, inc, cfg, **kw)

    def derived_q(self, engine, inc, cfg: dict, cid: str, **kw):
        """S3's derived proposal (a completion); while stalled only an admissible one (M4)."""
        if self._bc_gate:
            ok, why = self._bc_admissible(engine, inc, cfg)
            if not ok:
                self._bc_refuse(engine, str(kw.get('kind') or ''), cfg, why)
                return None
        return super().derived_q(engine, inc, cfg, cid, **kw)

    def _candidates(self, engine, inc) -> list[tuple[dict, tuple]]:
        """wcs's untried rows of the table; while stalled (M4's gate) the admissible ones only - what the ledger's best,
        the draw and the shared moves read; for the stalled draw (`_bc_pool_floor`) only those at its floors' P."""
        out = super()._candidates(engine, inc)
        if not self._bc_gate:
            return out
        out = [(cfg, ids) for cfg, ids in out if self._bc_admissible(engine, inc, cfg, row=True)[0]]
        if self._bc_pool_floor is not None:
            out = [(cfg, ids) for cfg, ids in out if self._pf(cfg, ids)[0] >= self._bc_pool_floor]
        return out

    # ------------------------------------------------------------ the starts
    def _bc_start(self, engine, inc, slot: str):
        """A start's next option as a proposal; None when none is left - or none could run now: S6 held one back
        (`fb_s6_refused`), or M4's stall gate refused some (`bc_start_held`: kept for a later step - a temporary refusal,
        as S6's, never consumes a start)."""
        self.bc_start_held = False
        if slot == 'uniform':
            return self._bc_uniform(engine, inc)
        for opt in self._bc_start_options(engine, slot):
            if opt.get('used'):
                continue
            opt['used'] = True
            self.bc_admit_refused = False
            q = self._bc_whole(engine, inc, opt['cfg'], kind=slot, why=opt['why'])
            if q is not None:
                self.bc_slots[slot].update(cfg=dict(opt['cfg']), why=opt['why'][:200],
                                           **({'source': opt['source']} if opt.get('source') else {}))
                return self._bc_record(engine, q, slot, change=self._words_of(opt['cfg']), source=opt.get('source'),
                                       why=opt['why'][:200])
            if self.fb_s6_refused:
                opt['used'] = False                      # (S6: the same option later)
                return None
            if self.bc_admit_refused:
                opt['used'] = False                      # (M4: offered again once the stall lets it through; the next
                self.bc_start_held = True                #  option may pass now)
        return None

    def _bc_whole(self, engine, inc, cfg: dict, *, kind: str, why: str, probe: bool | None = None):
        """A whole configuration with the chain, the ladder's plan (probe first below probe_below unless `probe` says); None when
        it is illegal, ran, is ruled out, doubtful with no probe left, or S6 / hard rule 5 / M4's gate refuse it."""
        ids = tuple(self.chain)
        cfg = {k: cfg.get(k, self.ref.get(k)) for k in self.knobs}
        if self._ckey(cfg, ids) in self.tried or not self._legal(engine, cfg):
            return None
        p, pwhy = self._pf(cfg, ids)
        if p <= 0.0:
            return None
        if probe is None and p < self.ladder.probe_below and not self.ladder.affordable(engine, 'probe'):
            return None                                  # (doubtful and no probe left: not run in full, as the draw)
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None, probe=probe, short=False)
        q = self._knob_q(engine, inc, cfg, kind=kind, plan=plan, p=p, why=(why + (f'; {pwhy}' if pwhy else ''))[:300])
        return q if q is not None and engine._fresh(q) else None

    def _bc_start_options(self, engine, slot: str) -> list[dict]:
        """A start's options, built once: experience - the mapped sources' winners, the most similar source first (none:
        the guide's configurations stand in); guide - the guide's configurations that differ from start one in a big knob
        first, then the others that differ from it at all."""
        if slot in self.bc_options:
            return self.bc_options[slot]
        if slot == 'experience':
            opts = self._bc_experience(engine)
            if not opts:
                opts = self._bc_guide(engine)
                if opts:
                    self.bc_notes.append('no usable experience source: start one is the guide\'s best configuration')
        else:
            s1 = self.bc_slots['experience'].get('cfg')
            ranked = [o for o in self._bc_guide(engine) if s1 is None or self._ckey(o['cfg'], ()) != self._ckey(s1, ())]
            if s1 is not None:
                big = self._big_knobs()
                far = [o for o in ranked if not big or any(vk(o['cfg'].get(k)) != vk(s1.get(k)) for k in big)]
                ranked = far + [o for o in ranked if o not in far]
            opts = ranked
        self.bc_options[slot] = opts
        return opts

    def _bc_packs(self) -> list:
        """The run packs start one reads: those given, else the home's that the experience plan loaded (xfer's loader)."""
        if self.fb_packs_in is not None:
            return list(self.fb_packs_in)
        if self.exp:
            return XFer._home_packs(self)
        return []

    def _bc_experience(self, engine) -> list[dict]:
        """Each experience source's best valid knob-only run mapped onto this table (xfer: sources_from_packs, its
        _map_winner; a source with this job's numbers is never one), the heaviest source first; [] when none maps."""
        try:
            packs = self._bc_packs()
            if not packs:
                self.bc_notes.append('no run pack in the experience: no experience start')
                return []
            target = self.fb_target or self._fb_target_env(engine)
            sources, notes = sources_from_packs(packs, target)
            self.bc_notes += notes
            self.labels = XFer._labels(self, engine)
            out, seen = [], set()
            for s in sorted(sources, key=lambda s: (-s['w0'], -((s['winner'] or {}).get('gain') or 0.0))):
                if not s.get('winner'):
                    continue
                cfg, _ = self._map_winner(engine, s['winner']['states'])
                key = self._ckey(cfg, ()) if cfg is not None else None
                if cfg is None or key in seen:
                    continue
                seen.add(key)
                env = {k: s['env'][k] for k in ('card_gib', 'nodes', 'params_b') if s['env'].get(k) is not None}
                out.append({'cfg': cfg, 'source': env, 'why': f'the best run of an experience source (weight {s["w0"]:.2f}, '
                                                              f'x{math.exp(s["winner"]["gain"]):.3f} there)'})
            return out
        except Exception as exc:                         # noqa: BLE001 - an unreadable experience: no experience start
            self.bc_notes.append(f'the experience could not be read: {str(exc)[:120]}')
            return []

    def _bc_guide(self, engine) -> list[dict]:
        """The guide's whole configurations by their stated gain (else the time model's), legal; without any: the time
        model's best rows (the guide's per-value gains) predicted faster than the reference."""
        if not self.prior:
            return []
        mean = self.surr.mean()
        out = []
        for i, c in enumerate(self.prior.configs or []):
            cfg = {**{k: self.ref.get(k) for k in self.knobs}, **{k: v for k, v in c['change'].items() if k in self.knobs}}
            if self._ckey(cfg, ()) == self._ckey(self.ref, ()) or not self._legal(engine, cfg):
                continue
            mu = Surrogate.value(self._feats(cfg), mean)
            out.append((-(c['mu'] if c.get('mu') is not None else mu), i, cfg, f"the guide's '{c.get('name')}' configuration"))
        if not out:
            rows = []
            for cfg in self._table(engine):
                mu = Surrogate.value(self._feats(cfg), mean)
                if mu > 0:
                    rows.append((-mu, len(rows), cfg))
            rows.sort(key=lambda t: t[:2])
            out = [(r[0], r[1], r[2], "the guide's best configuration by its values' gains") for r in rows[:4]]
        out.sort(key=lambda t: t[:2])
        return [{'cfg': cfg, 'why': why} for _, _, cfg, why in out]

    def _bc_draws(self, engine):
        """knob_space.random_search's draws with the arm's seed (the session's check and its candidates: the same draw as
        the random arm's first), each a whole legal configuration not proposed yet; on a session without the check, the
        same order of the product with the strategy's own legality."""
        s = engine.s
        space = knob_space.choices(s.task)
        if not space:
            return
        rng = random.Random(self.seed)
        if callable(getattr(s, 'check_config', None)) and isinstance(getattr(s, 'by_key', None), dict):
            draws = knob_space._feasible(s, space, rng)[0]
        else:
            draws = self._bc_plain_draws(engine, space, rng)
        for cfg in itertools.islice(draws, UNIFORM_SCAN):
            yield {k: cfg.get(k, self.ref.get(k)) for k in self.knobs}

    def _bc_plain_draws(self, engine, space: dict, rng: random.Random):
        names, n = list(space), math.prod(len(v) for v in space.values())
        order = rng.sample(range(n), n) if n <= knob_space.CHECK_CAP else (rng.randrange(n) for _ in range(UNIFORM_SCAN))
        for i in order:
            cfg = {}
            for name in names:
                i, j = divmod(i, len(space[name]))
                cfg[name] = space[name][j]
            if self._legal(engine, cfg):
                yield cfg

    def _bc_uniform(self, engine, inc):
        """Start three: the first new draw that M2's rules allow, in full (uniform_start_q: S6-exempt and never held to
        M4's admissibility; hard rule 5 still applies). A draw that breaks an M2 rule - a value that needs a saver held
        without one, an excluded value, a conflict held whole - is skipped for the next legal draw (V2B_r1 P1-1: the
        control start never repeats a failure the attribution explains); the skips are kept on the slot."""
        ids = tuple(self.chain)
        slot = self.bc_slots['uniform']
        for cfg in self._bc_draws(engine):
            if self._ckey(cfg, ids) in self.tried:
                continue
            rule = self._bc_m2_rule(cfg, ids)
            if rule:
                slot['skipped'] = slot.get('skipped', 0) + 1
                slot.setdefault('skipped_why', rule[len(M2_TAG):][:200])
                continue
            q = self.uniform_start_q(engine, inc, cfg)
            if q is not None and engine._fresh(q):
                slot['cfg'] = dict(cfg)
                skipped = f'; {slot["skipped"]} earlier draw(s) skipped by M2' if slot.get('skipped') else ''
                return self._bc_record(engine, q, 'uniform', change=self._words_of(cfg),
                                       why='the first new draw of the legal table with the arm\'s seed (the control '
                                           f'start){skipped}')
        return None

    # ------------------------------------------------------------ the configuration screens, the knob-only schedule
    def _bc_config(self, engine, inc):
        """A configuration screen: the kind the last one was not (a completion first) - a completion or a contrarian move
        (v1's challenge with ex_contrarian off) -, then the other, then an escape candidate while stalled, then the
        ledger's best untried configuration."""
        last = self.bc_config[-1] if self.bc_config else None
        other = 'contrarian' if self.bc_on['ex_contrarian'] else 'challenge'
        order = (other, 'complete') if last == 'complete' else ('complete', other)
        for kind in order + ('escape', 'ledger'):
            q = self._bc_mech(engine, inc, kind)
            if q is not None:
                return q
        return None

    def _bc_knob_only(self, engine, inc):
        """Screens 6-10 with the code track closed (v2): a completion of a failure, a contrarian move, an escape candidate
        while stalled, the ledger's best untried configuration - in this order; None: the tail and the draw follow."""
        for kind in KNOB_ONLY_ORDER:
            q = self._bc_mech(engine, inc, kind)
            if q is not None:
                return q
        return None

    def _bc_mech(self, engine, inc, kind: str):
        if kind == 'complete':
            return self.complete(engine, inc)
        if kind == 'challenge':
            return self.challenge(engine, inc)
        if kind == 'contrarian':
            return self._bc_contrarian(engine, inc)
        if kind == 'escape':
            return self._bc_escape(engine, inc)
        return self._bc_ledger_best(engine, inc)

    def _bc_done(self, log: list, **match) -> list[dict]:
        """The entries of a log matching `match` that did not come back as a duplicate."""
        return [r for r in log if r.get('outcome') != 'duplicate' and all(r.get(k) == v for k, v in match.items())]

    # ------------------------------------------------------------ M1 and M4's candidates (the shared module's)
    def _bc_as_q(self, engine, inc, got, kind: str):
        """A shared move as bcomp's proposal: the module's own proposal (`_ex_candidate`: the base's `_knob_q` - S6, hard
        rule 5, the ladder's plan and M4's gate applied), or a plain move ({'cfg', 'note'}) built as a start is
        (`_bc_whole`) and marked for the module (`_ex_mark`); None when there is none or it is not fresh."""
        if got is None:
            return None
        if hasattr(got, 'hypothesis'):
            return got if engine._fresh(got) else None
        cfg = got.get('cfg') if isinstance(got, dict) else None
        if not isinstance(cfg, dict):
            return None
        q = self._bc_whole(engine, inc, cfg, kind=kind, why=str(got.get('note') or kind))
        return self._ex_mark(q, {**got, 'why': kind}, engine) if q is not None else None

    def _bc_move_rec(self, engine, inc, q) -> dict:
        """What the report keeps of a shared move: its change against the incumbent, why the module proposed it (its note:
        the value, the partners sent back, the saver added; the stall, the radius, a warm restart, momentum), its gain."""
        cfg = self._cfg_of(q)
        inc_cfg = self._knob_cfg(engine, inc)
        row = (self.__dict__.get('ex_by_hid') or {}).get(q.hypothesis) or {}
        why = row.get('note') or (self.ledger.get(q.hypothesis) or {}).get('why_feasible') or q.mechanism
        return {'change': self._words_of(cfg), 'why': str(why or '')[:240],
                'changed': sorted(k for k in self.knobs if vk(cfg.get(k)) != vk(inc_cfg.get(k))),
                'predicted_gain_pct': round(100 * math.expm1(q.mu), 2),
                **{k: row[k] for k in ('value', 'flipped', 'saver', 'restart', 'momentum') if row.get(k) not in (None, [], {})}}

    def _bc_contrarian(self, engine, inc):
        """M1 (the challenge's slot): the shared `_ex_contrarian`'s best eligible move as a ladder candidate, at most
        EX_CONTRA_MAX per arm; None when off, spent or there is none."""
        if not self.bc_on['ex_contrarian'] or not self.prior or len(self._bc_done(self.bc_contras)) >= self.EX_CONTRA_MAX:
            return None
        q = self._bc_as_q(engine, inc, self._ex_contrarian(engine, inc), 'contrarian')
        if q is None:
            return None
        rec = self._bc_move_rec(engine, inc, q)
        self.bc_contras.append({**rec, 'id': q.hypothesis})
        return self._bc_record(engine, q, 'contrarian', **rec)

    def _bc_escape(self, engine, inc):
        """M4: while stalled, the shared `_ex_escape`'s candidate; None when off, not stalled or there is none."""
        if not self._bc_stalled():
            return None
        q = self._bc_as_q(engine, inc, self._ex_escape(engine, inc), 'escape')
        if q is None:
            return None
        rec = {**self._bc_move_rec(engine, inc, q), 'radius': self._ex_radius()}
        self.bc_escapes.append({**rec, 'id': q.hypothesis})
        return self._bc_record(engine, q, 'escape', **rec)

    # ------------------------------------------------------------ the completions
    def _bc_gap(self, m: dict) -> float | None:
        """The ledger's gap of a failed run, the GiB it must lose: out of memory - the shortfall its message gave; thrash -
        its peak (else the card) less the cliff's lower edge, THRASH_LO of the card (failed_material's); else the ledger's
        need over the card (at least the card's reserve); None without a card."""
        num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)   # noqa: E731
        if m['kind'] == 'oom' and num(m.get('short_gib')):
            return max(0.0, float(m['short_gib']))
        if m['kind'] == 'thrash' and num(m.get('gap_gib')):
            return float(m['gap_gib'])
        led = self.mledger
        if led is None or not led.cap:
            return None
        try:
            mu, _ = led.predict(m['cfg'], ids=self._led_ids(m.get('pids') or ()))
        except Exception:                                # noqa: BLE001 - a fit that fails: no gap known
            return None
        return max(CARD_RESERVE * led.cap, mu - led.cap)

    def _bc_mu(self, engine, inc, cfg: dict, ids) -> float:
        """The predicted log gain over the incumbent: the time model's, plus the model's own (shrunk) prediction of each
        carried patch the time model has not measured (a patch whose run failed has no observation of its own)."""
        mu, _ = self._mu(engine, inc, cfg, ids)
        names = set(self.surr.names)
        return mu + sum((self.patches.get(x) or {}).get('mu') or 0.0 for x in ids
                        if x not in self.chain and (f'patch:{x}', 'true') not in names)

    def _bc_failures(self, engine, inc) -> list[dict]:
        """The failed runs a completion may start from (the module docstring): the original failures before failed
        completions, the fastest by the time model first."""
        thr = max(math.log1p(COMPLETE_MIN_GAIN), 2 * engine.sigma_pair())
        rows = []
        for m in self.failed_material():
            cid = m['run']
            if len(self._bc_done(self.bc_completions, run=cid)) >= COMPLETE_PER_RUN:
                continue
            mu = self._bc_mu(engine, inc, m['cfg'], self.derived_ids(cid))
            if mu < thr:
                continue
            gap = self._bc_gap(m)
            if gap is None:
                continue                                 # (the ledger knows no gap: no card)
            rows.append((self.bc_cid_what.get(cid) == 'complete', -mu, len(rows), dict(m, gap_gib=gap, mu=mu)))
        rows.sort(key=lambda t: t[:3])
        return [r[3] for r in rows]

    def _bc_partners(self, f: dict) -> list[dict]:
        """A failed run's partners (the module docstring), in order: the values whose measured price tag frees memory, by
        the GiB they free; the declared savers with a price, by the GiB; the declared savers without one, then the values
        of unknown memory direction, by the guide's gain. A price ranks, it never vetoes: whether the completed
        configuration fits is the ledger's and its probe's to decide (BCOMP_r1 P1: an under-declared saver still closes
        the gap) - unless M2 says a value of the run needs a saver (S4's feasibility then rules out a partner that does not
        cover the gap). A value measured or declared not to free memory is no partner; those already proposed for this
        run are left out."""
        tags = self.price_tags()
        levels = self.prior.levels if self.prior else {}
        cap = self.mem.cap or (self.mledger.cap if self.mledger is not None else None) or 0.0
        floor = max(1e-6, SAVE_MIN * cap)                # (what tells a measured saver from a neutral value)
        used = {tuple(r['partner_key']) for r in self.bc_completions if r['run'] == f['run']}
        tiers = {'measured': [], 'declared': [], 'declared, no price': [], 'unknown': []}
        for k, vals in self.knobs.items():
            ref = vk(self.ref.get(k))
            if vk(f['cfg'].get(k)) != ref:
                continue                                 # (a knob the run moved: changing it is no completion of the run)
            for v in vals:
                key = vk(v)
                if key == ref or (k, key) in used:
                    continue
                tag, e = tags.get((k, key)), levels.get((k, key)) or {}
                row = {'knob': k, 'vk': key, 'value': v}
                g = e.get('mu') if e.get('mu') is not None else 0.0
                guide = {'guide_gain_pct': round(100 * math.expm1(g), 2)}
                if tag is not None and tag['source'] == 'measured':
                    save = -tag['gib']
                    if save >= floor:
                        tiers['measured'].append((-save, k, key, {**row, 'tier': 'measured', 'saving_gib': round(save, 3)}))
                elif e.get('mem_dir') == 'frees' and e.get('mem_gib'):
                    save = -tag['gib'] if tag is not None else e['mem_gib']
                    tiers['declared'].append((-save, k, key, {**row, 'tier': 'declared', 'saving_gib': round(save, 3)}))
                elif e.get('mem_dir') == 'frees':
                    tiers['declared, no price'].append((-g, k, key, {**row, 'tier': 'declared, no price', **guide}))
                elif e.get('mem_dir') is None:
                    tiers['unknown'].append((-g, k, key, {**row, 'tier': 'unknown', **guide}))
        out = []
        for t in ('measured', 'declared', 'declared, no price', 'unknown'):
            out += [r[3] for r in sorted(tiers[t], key=lambda r: r[:3])]
        return out

    def complete(self, engine, inc):
        """A completion (the module docstring): with bcomp_feedback, the best of the re-ranked completion queue (the
        original failures before failed completions, then the predicted gain x the ledger's P; a queued candidate an M2 rule
        rules out dropped and recorded); without it, v1's first failed run with a partner whose candidate is legal,
        untried, predicted faster than the incumbent, feasible by the ledger, the probe it needs paid
        (`_bc_probe_payable`). None when none."""
        if not self.bc_on['bcomp_complete']:
            return None
        if not self.bc_on['bcomp_feedback']:
            for f in self._bc_failures(engine, inc):
                for partner in self._bc_partners(f)[:PARTNER_SCAN]:
                    q = self._bc_complete_q(engine, inc, f, partner)
                    if q is not None:
                        return q
            return None
        queue = []
        for f in self._bc_failures(engine, inc):
            for partner in self._bc_partners(f)[:PARTNER_SCAN]:
                c = self._bc_complete_plan(engine, inc, f, partner)
                if isinstance(c, str):
                    if c.startswith(M2_TAG):
                        self._bc_drop(engine, f, partner, c)
                    continue
                queue.append((self.bc_cid_what.get(f['run']) == 'complete', -c['mu'] * c['p'], len(queue), c))
        queue.sort(key=lambda t: t[:3])
        self.bc_queue = [{'run': c['f']['run'], 'partner': c['rec']['partner'], 'p_feasible': round(c['p'], 3),
                          'predicted_gain_pct': c['rec']['predicted_gain_pct']} for *_, c in queue[:8]]
        for *_, c in queue:
            q = self._bc_complete_build(engine, inc, c)
            if q is not None:
                return q
        return None

    def _bc_drop(self, engine, f: dict, partner: dict, why: str) -> None:
        """A queued completion an M2 rule ruled out, recorded once (the report's `dropped`)."""
        word = f'{partner["knob"]}={partner["value"]}'
        if any(r['run'] == f['run'] and r['partner'] == word for r in self.bc_dropped):
            return
        self.bc_dropped = (self.bc_dropped + [{'screens_done': engine.full_screens(), 'run': f['run'], 'partner': word,
                                               'tier': partner.get('tier'), 'rule': why[len(M2_TAG):][:200]}])[-24:]

    def _bc_complete_q(self, engine, inc, f: dict, partner: dict):
        """The completion of failed run `f` by `partner` as a proposal, or None (not a candidate, or not proposed)."""
        c = self._bc_complete_plan(engine, inc, f, partner)
        return None if isinstance(c, str) else self._bc_complete_build(engine, inc, c)

    def _bc_complete_plan(self, engine, inc, f: dict, partner: dict):
        """A completion before it is proposed - {'f', 'cfg', 'ids', 'mu', 'p', 'why', 'rec'} - or why it is no candidate
        (a str; an M2 rule's begins with M2_TAG)."""
        k, key, v = partner['knob'], partner['vk'], partner['value']
        ids = self.derived_ids(f['run'])
        new = {**f['cfg'], k: v}
        held = set(self._levels(self._m_of(new, ids)))
        swapped = []
        for pair, rec in sorted(self.pairs.items(), key=lambda t: sorted(t[0])):   # a held value that conflicts with the
            if rec['p'] < 0.5 and (k, key) in pair and pair <= held:              # partner goes back (hls3 F5c's swap)
                for kk, kv in sorted(pair - {(k, key)}):
                    if (kk, kv) in held and not kk.startswith('patch:') and kk in self.knobs:
                        new[kk] = self.ref.get(kk)
                        held.discard((kk, kv))
                        swapped.append(f'{kk}={json.loads(kv)}')
                        break
        legal = self._complete(engine, new, {k})
        if legal is None or vk(legal.get(k)) != key:
            return 'no legal configuration holds it'
        legal = {kk: legal.get(kk) for kk in self.knobs}
        if self._ckey(legal, ids) in self.tried:
            return 'it ran or was proposed'
        mu = self._bc_mu(engine, inc, legal, ids)
        if mu <= 0.0:
            return 'no faster than the incumbent'    # (the partner costs what the run gains)
        p, pwhy = self._pf(legal, ids)
        if p <= 0.0:
            return pwhy or 'ruled out by the runs'
        if self._bc_p_unblamed(legal, ids, p) < COMPLETE_P_MIN:
            return 'below the declared-memory floor: the runs say it does not fit'
        if not self._bc_probe_payable(engine, ids, p):
            return 'the probe it needs is not paid'
        word = f'{k}={v}'
        why = (f'{f["run"]} ' + ('thrashed' if f['kind'] == 'thrash' else 'ran out of memory')
               + (' at its probe' if f.get('at_probe') else '') + f', {f["gap_gib"]:.1f} GiB to lose; {word}: '
               + (f'frees {partner["saving_gib"]:.1f} GiB ({partner["tier"]})' if partner.get('saving_gib') is not None
                  else 'declared to free memory, no price given' if partner['tier'] == 'declared, no price'
                  else 'its memory direction is unknown') + (f'; {", ".join(swapped)} back' if swapped else '')
               + (f'; {pwhy}' if pwhy else ''))
        rec = {'run': f['run'], 'partner': word, 'partner_key': [k, key], 'tier': partner['tier'],
               'saving_gib': partner.get('saving_gib'), 'gap_gib': round(f['gap_gib'], 3), 'swapped': swapped,
               'run_gain_pct': round(100 * math.expm1(f['mu']), 2), 'predicted_gain_pct': round(100 * math.expm1(mu), 2)}
        return {'f': f, 'cfg': legal, 'ids': ids, 'mu': mu, 'p': p, 'why': why, 'rec': rec}

    def _bc_complete_build(self, engine, inc, c: dict):
        q = self.derived_q(engine, inc, c['cfg'], c['f']['run'], kind='complete', p=c['p'], why=c['why'][:300])
        if q is None or not engine._fresh(q):
            return None
        rec = c['rec']
        self.bc_completions.append({**rec, 'id': q.hypothesis, 'patches': sorted(self.ideas.get(q.hypothesis) or ())})
        return self._bc_record(engine, q, 'complete', **rec, why=c['why'][:200])

    def _bc_probe_payable(self, engine, ids, p: float) -> bool:
        """Whether the probe a completion needs can be paid (FINALISTS_IMPL_SPEC B.9 F2, base f1.1). One that carries an
        unproven patch owes that patch's proof, which derived_q plans on the base's code pool while it has a probe -
        whatever the knob side's allowance holds - and beyond it (or without the pool) on the knob side's allowance, as in
        f1.0. A knob-only one is probed only below the base's `probe_below` (0.7; 0.5 under the ratio rule, so from 0.5 it
        runs in full where a probe costs over 0.6 of a full run), paid by the knob side's allowance. Hard rule 5's probe
        of a completion that holds its run's whole failed combination stays the base's to plan, and to refuse when
        nothing pays it."""
        if any(x not in self.chain for x in ids):
            return self.ladder.code_probes_left() >= 1 or self.ladder.affordable(engine, 'probe')
        return p >= self.ladder.probe_below or self.ladder.affordable(engine, 'probe')

    def _bc_p_unblamed(self, cfg: dict, ids, p: float) -> float:
        """S4's P without isolated_blame's factor (`p`: with it): what COMPLETE_P_MIN reads - a completion is the
        compensation of its run's blamed value, which the ledger's own estimate weighs; its proposal keeps the full P."""
        if not self.fb_on['fb_isolated_blame'] or not self.isolated_blame():
            return p
        self.fb_on['fb_isolated_blame'] = False          # (a read: the switch back right after)
        try:
            return self._pf(cfg, ids)[0]
        finally:
            self.fb_on['fb_isolated_blame'] = True

    # ------------------------------------------------------------ v1's challenge (ex_contrarian off)
    def _bc_unfit(self, inc_cfg: dict, k: str, key: str, v, e: dict) -> bool:
        """The guide declares the value does not fit: it says the value runs with P < 0.5, or it declares a spender that
        the feasibility puts below 0.5 on the incumbent."""
        pr = e.get('p_runs')
        if isinstance(pr, (int, float)) and pr < P_RUNS_LOW:
            return True
        return e.get('mem_dir') == 'spends' and self._pf({**inc_cfg, k: v}, tuple(self.chain))[0] < 0.5

    def challenge(self, engine, inc):
        """v1's challenge (the module docstring), or None - M1's contrarian move takes its slot unless ex_contrarian is
        off."""
        if not self.bc_on['bcomp_challenge'] or not self.prior or len(self._bc_done(self.bc_challenges)) >= CHALLENGE_MAX:
            return None
        ran = set()
        for r in self.runs.values():
            ran |= set(self._levels(r.get('cfg') or {}))
        tried = {tuple(r['value_key']) for r in self.bc_challenges}
        inc_cfg = self._knob_cfg(engine, inc)
        conflicts = {m for pr in self.prior.pairs if pr['kind'] == 'conflicts' for m in pr['members']}
        thr = max(math.log1p(COMPLETE_MIN_GAIN), 2 * engine.sigma_pair())   # (a claimed effect one run can tell)
        vetoed = []
        for (k, key), e in self.prior.levels.items():
            if k not in self.knobs or (k, key) in ran or (k, key) in tried or vk(inc_cfg.get(k)) == key:
                continue
            v = next((x for x in self.knobs[k] if vk(x) == key), None)
            mu = e.get('mu')
            reasons = []                                 # (each veto on its own - BCOMP_r1 P2: a small claimed loss
            if mu is not None and mu <= 0.0 and -mu >= thr:   # hides neither a conflict nor a memory veto)
                reasons.append(f'the guide declares {100 * math.expm1(mu):+.1f} %')
            if (k, key) in conflicts:
                reasons.append('the guide declares a conflict')
            if self._bc_unfit(inc_cfg, k, key, v, e):
                reasons.append('the guide declares it does not fit')
            if not reasons:
                continue                                 # (a declared loss within the noise alone: one run cannot refute it)
            vetoed.append((-abs(mu or 0.0), k, key, v, '; '.join(reasons)))
        ids = tuple(self.chain)
        for _, k, key, v, vwhy in sorted(vetoed, key=lambda t: t[:3]):
            cfg = self._complete(engine, {**inc_cfg, k: v}, {k})
            if cfg is None or vk(cfg.get(k)) != key:
                continue
            cfg = {kk: cfg.get(kk) for kk in self.knobs}
            if self._ckey(cfg, ids) in self.tried:
                continue
            p, pwhy = self._pf(cfg, ids)
            if p <= 0.0:
                continue
            probe = p < CHALLENGE_PROBE_BELOW
            if probe and not self.ladder.affordable(engine, 'probe'):
                continue                                 # (no probe left: not run in full below 0.5)
            plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None, probe=probe, short=False)
            word = f'{k}={v}'
            changed = sorted(kk for kk in self.knobs if vk(cfg.get(kk)) != vk(inc_cfg.get(kk)))
            q = self._knob_q(engine, inc, cfg, kind='challenge', plan=plan, p=p,
                             why=(f'a challenge of the guide: {word} ({vwhy}), on {inc} alone' + (f'; {pwhy}' if pwhy else ''))[:300])
            if q is None or not engine._fresh(q):
                continue
            rec = {'value': word, 'value_key': [k, key], 'vetoed': vwhy, 'changed': changed}
            self.bc_challenges.append({**rec, 'id': q.hypothesis})
            return self._bc_record(engine, q, 'challenge', **rec, why=f'a challenge of the guide: {word} ({vwhy})')
        return None

    def _bc_ledger_best(self, engine, inc):
        """The ledger's best untried configuration: the time model's gain over the incumbent x P, P >= LEDGER_P, at least
        one big knob off every run (wcs's weak diversity); the feasibility read for at most KNOB_SCAN_MAX rows (while
        stalled: the admissible rows only, `_candidates`)."""
        mean = self.surr.mean()
        y_inc = Surrogate.value(self._incumbent_feats(engine, inc), mean)
        rows = []
        for cfg, ids in self._candidates(engine, inc):
            mu = Surrogate.value(self._feats(cfg, ids), mean) - y_inc
            if mu > 0.0:
                rows.append((mu, len(rows), cfg, ids))
        rows.sort(key=lambda t: (-t[0], t[1]))
        big = sorted(self._big_knobs())                  # (wcs's _diverse 'weak', its big knobs read once)
        ran = [r.get('cfg') or {} for r in self.runs.values()]
        need = LEDGER_P if self.ladder.affordable(engine, 'probe') else max(LEDGER_P, self.ladder.probe_below)
        best, best_v, reads = None, 0.0, 0              # (no probe left: only what is predicted to fit, as the draw)
        for mu, _, cfg, ids in rows:
            if mu <= best_v or reads >= KNOB_SCAN_MAX:
                break
            if big and not all(any(vk(cfg.get(k)) != vk(rc.get(k)) for k in big) for rc in ran):
                continue
            reads += 1
            p, why = self._pf(cfg, ids)
            if p >= need and mu * p > best_v:
                best, best_v = (cfg, p, why, mu), mu * p
        if best is None:
            return None
        cfg, p, why, mu = best
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None, short=False)
        text = f"the ledger's best untried configuration (x{math.exp(mu):.3f} predicted, P {p:.2f})"
        q = self._knob_q(engine, inc, cfg, kind='ledger', plan=plan, p=p, why=(text + (f'; {why}' if why else ''))[:300])
        if q is None or not engine._fresh(q):
            return None
        return self._bc_record(engine, q, 'ledger', change=self._words_of(cfg), predicted_gain_pct=round(100 * math.expm1(mu), 2),
                               why=text)

    # ------------------------------------------------------------ M2: the rules from the failures
    def _bc_kvals(self, cfg: dict) -> list:
        """A configuration's knob values off the reference ((knob, value key); patches are not M2's revise call's)."""
        return [(k, vk(cfg.get(k, self.ref.get(k)))) for k in self.knobs
                if vk(cfg.get(k, self.ref.get(k))) != vk(self.ref.get(k))]

    @staticmethod
    def _bc_words(values) -> str:
        return ', '.join(f'{k}={v}' for k, v in sorted(values))

    def _bc_m2_rule(self, cfg: dict, ids=()) -> str | None:
        """The M2 rule a configuration breaks (bcomp_feedback; the shared `_ex_rules`: a value excluded, a conflict set
        held whole, a value that needs a saver held without savers covering its shortfall), as its reason after M2_TAG;
        None when it breaks none (or no rule exists yet: nothing to read per row of a draw)."""
        if not self.bc_on['bcomp_feedback'] or self.prior is None:
            return None
        att = self._ex_attribution()
        if not (att.get('needs_saver') or att.get('excluded') or att.get('conflicts')):
            return None
        hit = self._ex_rules(cfg, tuple(ids or ()))      # (it reads the ledger itself, never this `_pf`: no recursion)
        return M2_TAG + hit if hit else None

    def _pf(self, cfg: dict, ids=()) -> tuple[float, str]:
        """S4's feasibility (the base's), and 0 for a configuration that breaks one of M2's rules (`_bc_m2_rule`)."""
        p, why = super()._pf(cfg, ids)
        if p <= 0.0:
            return p, why
        hit = self._bc_m2_rule(cfg, ids)
        return (0.0, hit) if hit else (p, why)

    # ------------------------------------------------------------ M2: the revise call
    def _bc_revise_max(self) -> int:
        return int(getattr(self, 'EX_REVISE_MAX', 2))

    def _bc_fitted(self) -> set:
        out = set()
        for r in self.runs.values():
            if r.get('kind') in FITTED:
                out |= set(self._bc_kvals(r.get('cfg') or {}))
        return out

    def _bc_failed(self, engine, p, out, cls) -> None:
        """A failed knob-side run or probe (FAIL_KINDS): counted, and a revise call made due - after REVISE_EVERY failures
        since the last one, or at once when the run holds a value the guide declared to run with p_runs >= REVISE_P_RUNS
        that no fitted run holds."""
        hid = p.hypothesis or ''
        if not out.cid or out.status in ('duplicate', 'infra', 'rejected') or hid.startswith(('patch|', 'stack|')):
            return                                       # (v4 F2: a stack test is the code track's, not the knob side's)
        kind = (cls or {}).get('kind')
        if kind not in FAIL_KINDS:
            return
        self.bc_fails += 1
        if not self.bc_on['bcomp_revise'] or self.bc_revise_calls >= self._bc_revise_max() or not self.prior:
            return
        fitted = self._bc_fitted()
        held = self._bc_kvals((self.runs.get(out.cid) or {}).get('cfg') or {})
        hit = [v for v in held if v not in fitted and v not in self.bc_contradicted
               and isinstance((self.prior.levels.get(v) or {}).get('p_runs'), (int, float))
               and self.prior.levels[v]['p_runs'] >= REVISE_P_RUNS]
        if hit:
            self.bc_contradicted |= set(hit)
            self.bc_revise_due = (f'{out.cid} failed ({kind}) holding {self._bc_words(hit)}, which the guide declared to '
                                  f'run with p_runs >= {REVISE_P_RUNS}')
        elif self.bc_fails - self.bc_fails_at_revise >= REVISE_EVERY:
            self.bc_revise_due = (f'{self.bc_fails - self.bc_fails_at_revise} failed knob-side runs or probes since the last '
                                  f'revise call (the latest {out.cid}: {kind})')

    def _bc_involved(self) -> list:
        """The values a revise call asks about: each failed run's values that no fitted run holds (all of its values when
        each fitted somewhere: a combination), the latest failures first, at most REVISE_VALUES."""
        fitted, out = self._bc_fitted(), []
        for cid in reversed(list(self.runs)):
            r = self.runs[cid]
            if cid == 'c0' or r.get('kind') not in FAIL_KINDS:
                continue
            vals = self._bc_kvals(r.get('cfg') or {})
            for v in [x for x in vals if x not in fitted] or vals:
                if v not in out:
                    out.append(v)
        return out[:REVISE_VALUES]

    def _bc_level_row(self, k: str, v) -> dict:
        """The guide's level of a value as the revise call shows it (the prior call's fields)."""
        e = (self.prior.levels.get((k, vk(v))) if self.prior else None) or {}
        pct = lambda x: None if x is None else round(100 * math.expm1(x), 2)   # noqa: E731
        return {'knob': k, 'value': v, 'gain_pct': pct(e.get('mu')), 'sd_pct': pct(e.get('sd')), 'p_runs': e.get('p_runs'),
                'memory': {'direction': e.get('mem_dir') or 'unknown', 'gib': e.get('mem_gib')}, 'why': e.get('why') or None}

    def _bc_val(self, k: str, key: str):
        return next((x for x in self.knobs[k] if vk(x) == key), None)

    def _bc_revise_payload(self, engine, vals: list) -> dict:
        knobs = sorted({k for k, _ in vals})
        return {'reference': {'config': {k: self.ref.get(k) for k in self.knobs}},
                'knob_table': {k: {'reference': self.ref.get(k), 'values': list(self.knobs[k])} for k in knobs},
                'guide': [self._bc_level_row(k, v) for k in knobs for v in self.knobs[k] if vk(v) != vk(self.ref.get(k))],
                'revise': [{'knob': k, 'value': self._bc_val(k, key)} for k, key in vals],
                'measured': self._history(engine)}

    def _bc_revise_call(self, engine, payload: dict):
        """One revise call: the proposer's `knob_revise` (engine, base, payload) when it has one, else an LLMProposer's
        driver (`_llm_revise`); within the arm's budget, S1's last calls held back for the code track as a configuration
        call holds them (the proposer's allowance capped meanwhile). The answer, or None."""
        self._allowance()
        full, hold = self.calls_max, self._fb_code_can_use_calls(engine)
        if hold:
            self.calls_max = full - CODE_CALLS
            self._allowance()
        try:
            fn = getattr(self.proposer, 'knob_revise', None)
            if callable(fn):
                return fn(engine, 'c0', payload)
            if isinstance(self.proposer, LLMProposer):
                return _llm_revise(self.proposer, engine, payload)
            return None
        finally:
            if hold:
                self.calls_max = full
                self._allowance()

    def _bc_revise(self, engine) -> None:
        """M2's revise call when a failure made one due (the module docstring); recorded in `bc_revisions`."""
        from .token_governor import TokenGovernor
        if not TokenGovernor(engine.s.ledger).allow_revise:
            return
        why, self.bc_revise_due = self.bc_revise_due, None
        if why is None or not self.bc_on['bcomp_revise'] or not self.prior or self.bc_revise_calls >= self._bc_revise_max():
            return
        vals = self._bc_involved()
        rec = {'screens_done': engine.full_screens(), 'why': why[:240],
               'values': [f'{k}={self._bc_val(k, key)}' for k, key in vals]}
        self.bc_revisions.append(rec)
        self.bc_fails_at_revise = self.bc_fails
        if not vals:
            rec['outcome'] = 'nothing to revise: the failures hold no knob value off the reference'
            return
        if not self._budget_ok(engine) or not self._fb_knob_call_ok(engine):
            rec['outcome'] = 'not made: no model call left for it in the budget (S1 keeps the last ones for the code track)'
            return
        before, err, ans = self._llm_used(), None, None
        try:
            ans = self._bc_revise_call(engine, self._bc_revise_payload(engine, vals))
        except Exception as exc:                         # noqa: BLE001 - a failed call leaves the estimates unchanged
            err = str(exc)[:200]
            (getattr(engine, 'log', None) or (lambda m: None))(f'[search] {self.name}: the revise call failed: {err}')
        self.llm_calls = self._llm_used()
        rec['calls'] = [before, self._llm_used()]
        made = self._llm_used() > before or err is not None or ans is not None
        if made:
            self.bc_revise_calls += 1                    # (a call made or attempted: one of EX_REVISE_MAX)
        if err is not None:
            rec['outcome'] = f'failed: {err}; the estimates are unchanged'
            return
        if not made:
            rec['outcome'] = 'not made: no model to ask (or no tokens left); the estimates are unchanged'
            return
        if not isinstance(ans, dict):
            rec['outcome'] = 'no usable answer: the estimates are unchanged'
            return
        try:
            changes = self._bc_merge(ans, vals)
        except Exception as exc:                         # noqa: BLE001 - an answer that does not merge: nothing changed
            rec['outcome'] = f'the answer could not be merged ({str(exc)[:120]}): the estimates are unchanged'
            return
        rec['changes'] = changes
        rec['outcome'] = f'{len(changes)} value(s) revised' if changes else 'the answer revised none of the values asked'

    def _bc_parse_revision(self, ans: dict, vals: list) -> dict:
        """The revised levels of the values asked, parsed as wcs.Prior parses the prior call's (`_coerce`, `_num`, the
        same bounds): (knob, value key) -> the fields given (mu, sd, p_runs, mem_dir, mem_gib, why); a value not asked, a
        row of the wrong shape or a field not given leaves the guide's as it is."""
        want, out = set(vals), {}
        rows = ans.get('levels')
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not isinstance(row.get('knob'), str) or row['knob'] not in self.knobs:
                continue
            k = row['knob']
            v = _coerce(row.get('value'), self.knobs[k])
            if v is None or (k, vk(v)) not in want or (k, vk(v)) in out:
                continue
            e = {}
            g = _num(row.get('gain_pct'), -90.0, 400.0)
            if g is not None:
                e['mu'] = math.log1p(g / 100.0)
            s = _num(row.get('sd_pct'), 0.0, 200.0)
            if s is not None:
                e['sd'] = math.log1p(s / 100.0)
            pr = _num(row.get('p_runs'), 0.0, 1.0)
            if pr is not None:
                e['p_runs'] = pr
            m = row.get('memory') if isinstance(row.get('memory'), dict) else None
            if m is not None:
                d = str(m.get('direction') or '').strip().lower()
                if d in DIRS:
                    gib = _num(m.get('gib'), 0.0, 10000.0)
                    e.update(mem_dir=d, mem_gib=gib if d in ('spends', 'frees') and gib else None)
                elif d in ('unknown', ''):
                    e.update(mem_dir=None, mem_gib=None)
            if e:
                e['why'] = str(row.get('why') or '')[:160]
                out[(k, vk(v))] = e
        return out

    def _bc_merge(self, ans: dict, vals: list) -> list[dict]:
        """The revised levels merged into the guide the program uses - all of them or none: the prior's levels, the time
        model's prior of each value (wcs's rule: its spread at least half its gain, SD_FLOOR at least), the ledger's
        declared GiB of each value (the ledger's own rule: half of them as the spread), the memory model's declared moves;
        every cache that read them goes. Returns what changed, as the report shows it."""
        new = self._bc_parse_revision(ans, vals)
        if not new:
            return []
        shown = lambda e: {x: y for x, y in self._bc_level_row(*e).items() if x not in ('knob', 'value', 'why')}   # noqa: E731
        plan = []
        for (k, key), e in new.items():
            lv = dict(self.prior.levels.get((k, key)) or {'mu': None, 'sd': None, 'mem_dir': None, 'mem_gib': None,
                                                           'warmup': None, 'p_runs': None, 'why': ''})
            lv.update({x: e[x] for x in ('mu', 'sd', 'p_runs', 'mem_dir', 'mem_gib') if x in e})
            lv['why'] = ((e.get('why') or lv.get('why') or '')[:150] + ' [revised]').strip()
            plan.append(((k, key), lv, 'mu' in e or 'sd' in e, 'mem_dir' in e))
        s, led = self.surr, self.mledger                 # (what the merge touches, kept: any error puts it all back)
        held = ({key: self.prior.levels.get(key) for key, *_ in plan}, list(s.names), list(s.mu0), list(s.sd0),
                [dict(b) for b in led.basis] if led is not None else None, self.declared)
        changes = []
        try:
            for (k, key), lv, timing, memory in plan:
                v = self._bc_val(k, key)
                before = shown((k, v))
                self.prior.levels[(k, key)] = lv
                if timing:
                    self._bc_surr_prior((k, key), lv)
                if memory:
                    self._bc_ledger_prior(k, key, lv)
                changes.append({'value': f'{k}={v}', 'before': before, 'after': shown((k, v)), 'why': lv['why'][:160]})
            if self.on['wcs_feasibility']:
                self.declared = self.prior.declared(self.ref)
        except Exception:
            levels, s.names[:], s.mu0[:], s.sd0[:], basis, self.declared = held
            for key, e in levels.items():
                if e is None:
                    self.prior.levels.pop(key, None)
                else:
                    self.prior.levels[key] = e
            if led is not None:
                for b, old in zip(led.basis, basis):
                    b.clear()
                    b.update(old)
                led._fit, led._pred = None, {}
            s._post = None
            raise
        self._forget()
        self._fb_forget()
        self._ex_forget()                                # (the shared module's attribution, big knobs and price tags: a
        self._bc_admit_stamp = None                      #  change no observation stamps; the admissibility read again)
        return changes

    def _bc_surr_prior(self, name, e: dict) -> None:
        """The time model's prior of a value from its (revised) level, by wcs's rule (wcs._init)."""
        mu = e['mu'] if e.get('mu') is not None else 0.0
        sd = e['sd'] if e.get('sd') is not None else SD_UNKNOWN
        if e.get('mu') is not None:
            sd = max(sd, 0.5 * abs(mu))
        s = self.surr
        if name not in s.names:
            s.ensure(name, mu, sd)
            return
        i = s.names.index(name)
        s.mu0[i], s.sd0[i] = float(mu), max(SD_FLOOR, float(sd))
        s._post = None                                   # (the posterior is solved again on its next read)

    def _bc_ledger_prior(self, k: str, key: str, e: dict) -> None:
        """The ledger's declared memory of a value from its (revised) level, by the ledger's own rule (add_value_terms):
        signed GiB, half of them as the spread; neutral: 0; unknown: 0 with the wide spread."""
        led = self.mledger
        if led is None:
            return
        b = next((b for b in led.basis if b['kind'] == 'value' and b.get('knob') == k and b.get('vk') == key), None)
        if b is None:
            return
        scale = led.cap or 100.0
        d, g = e.get('mem_dir'), e.get('mem_gib')
        if d in ('spends', 'frees') and g:
            mu, sd = (float(g) if d == 'spends' else -float(g)), max(SD_DECLARED * float(g), SD_DECLARED_MIN * scale)
        elif d == 'neutral':
            mu, sd = 0.0, SD_DECLARED_MIN * scale
        else:
            mu, sd = 0.0, LEDGER_SD_UNKNOWN * scale
        b.update(mu=mu, sd=sd, declared=d)
        led._fit, led._pred = None, {}                   # (fitted again on its next read)

    # ------------------------------------------------------------ outcomes
    def observe(self, engine, p, out):
        cls = super().observe(engine, p, out)
        self._ex_observe(engine, p, out)                 # (the shared module: the stall, the tabu list, the attribution)
        self._bc_failed(engine, p, out, cls)
        self._bc_screen(engine, p, out, cls)
        rec = self.bc_by_hid.get(p.hypothesis or '')
        if rec is None:
            return cls
        dup = out.status == 'duplicate'
        rec.update(outcome=out.status, fidelity=out.fidelity, candidate=out.cid, kind=(cls or {}).get('kind'))
        what = rec['what']
        for log in (self.bc_completions, self.bc_challenges, self.bc_contras, self.bc_escapes):
            for r in log:
                if r['id'] == rec['id']:
                    r.update(outcome=out.status, fidelity=out.fidelity, candidate=out.cid, kind=(cls or {}).get('kind'))
        if what in START_SLOTS:                          # a start is consumed by its run (a duplicate: the next option)
            self.bc_slots[what].update(state='open' if dup else 'consumed', outcome=out.status, candidate=out.cid)
        if out.cid and not dup:
            self.bc_cid_what[out.cid] = what
        if what in CONFIG_KINDS and out.fidelity == 'full' and out.cid and out.status in ('screened', 'aborted', 'invalid'):
            self.bc_config.append(what)                  # (a configuration screen spent; a failed probe spends none)
        return cls

    def _bc_screen(self, engine, p, out, cls) -> None:
        """The report's account of a full screen spent: which mechanism proposed it and why, and how it came out."""
        if out.fidelity != 'full' or not out.cid or out.status not in ('screened', 'aborted', 'invalid'):
            return
        hid = p.hypothesis or ''
        rec = self.bc_by_hid.get(hid) or {}
        step = next((s for s in reversed(self.bc_steps) if s.get('id') == hid), {})   # (as it was when proposed)
        by = rec.get('what') or step.get('pick') or {'patch': 'code'}.get(hid.split('|', 1)[0], hid.split('|', 1)[0] or '?')
        why = rec.get('why') or step.get('why') or (self.ledger.get(hid) or {}).get('why_feasible')
        self.bc_screens.append({'screen': engine.full_screens(), 'by': by, 'id': hid, 'candidate': out.cid,
                                'outcome': (cls or {}).get('kind') or out.status, **({'why': str(why)[:200]} if why else {}),
                                **({'stalled': True} if step.get('stalled') else {})})

    # ------------------------------------------------------------ report
    def _bc_ex_view(self) -> dict | None:
        try:
            return self._ex_view() if self.prior is not None else None
        except Exception as exc:                         # noqa: BLE001 - the report never fails the arm
            return {'error': str(exc)[:160]}

    def _bc_feedback_view(self) -> dict | None:
        """M2's rules as they stand (the shared attribution) and what they did to the completion queue."""
        if self.prior is None or not self.bc_on['bcomp_feedback']:
            return None
        try:
            att = self._ex_attribution()
        except Exception as exc:                         # noqa: BLE001 - the report never fails the arm
            return {'error': str(exc)[:160]}
        return {'needs_saver': [{'value': self._bc_words([v]), 'failures': r.get('failures'),
                                 'with_saver': r.get('with_saver'), 'gap_gib': r.get('gap_gib')}
                                for v, r in sorted(att.get('needs_saver', {}).items())][:12],
                'excluded': [self._bc_words([v]) for v in sorted(att.get('excluded', {}))][:12],
                'conflicts': [{'values': self._bc_words(c['members']), 'runs': c.get('runs'), 'kinds': c.get('kinds')}
                              for c in att.get('conflicts', [])][:12],
                'queue': self.bc_queue, 'dropped': self.bc_dropped[-12:]}

    def bcomp_view(self) -> dict:
        def row(r):
            return {k: v for k, v in r.items() if k not in ('partner_key', 'value_key') and v not in (None, [], {})}
        eng = getattr(self, 'engine', None)
        b = (screens_budget(eng) if eng is not None else 0) or RATE_SCREENS
        later = 'screens 6-10'                           # (v3's words at ten screens or fewer; above: the real range)
        if b > RATE_SCREENS:
            later = f'screens {self.bc_knob_screens + 1}-{b}'
        return {'version': BCOMP_VERSION, 'off': sorted(k for k, v in self.bc_on.items() if not v),
                'knob_screens': self.bc_knob_screens,
                'schedule': f'code track open: {later} are the code track\'s (v1)' if self._can_code() else
                            f'knob-only: {later} are completions, contrarian moves, escapes, the ledger\'s best, then the '
                            'tail and the draw',
                'starts': {s: {k: v for k, v in st.items() if k != 'cfg'} | (
                    {'change': self._words_of(st['cfg'])} if st.get('cfg') else {}) for s, st in self.bc_slots.items()},
                'config_screens': list(self.bc_config), 'screens': list(self.bc_screens),   # (F1: every full screen)
                'completions': [row(r) for r in self.bc_completions[-12:]],
                'challenges': [row(r) for r in self.bc_challenges[-6:]],
                'contrarian': [row(r) for r in self.bc_contras[-6:]], 'escapes': [row(r) for r in self.bc_escapes[-6:]],
                'feedback': self._bc_feedback_view(),
                'revisions': {'calls': self.bc_revise_calls, 'max': self._bc_revise_max(), 'failures_seen': self.bc_fails,
                              'log': self.bc_revisions[-4:]},
                'admissibility': {'refused': self.bc_refusals, 'last': self.bc_refused[-8:]},
                'explore': self._bc_ex_view(), 'steps': self.bc_steps[-STEPS_SHOWN * rate_screens(eng):],
                'notes': self.bc_notes[-12:]}

    def describe(self) -> dict:
        out = super().describe()
        out[self.name]['bcomp'] = self.bcomp_view()
        return out
