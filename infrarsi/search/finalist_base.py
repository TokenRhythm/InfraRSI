"""What the finalists (bcomp, beam, cards; FINALISTS_IMPL_SPEC_20261101 section A.0, S1-S5) share, on the
candidates' base (search/llmopt: one budget of 10 model calls / 8 USD through BudgetDriver at ten screens - F1 below -,
wcs's prior call, table, surrogate, pair table and code track; the memory ledger; the rung ladder). FinalistBase is a
cooperative subclass of LLMOptBase: a finalist subclasses it (bcomp, beam, cards), or puts it before another LLMOptBase
strategy in its bases (for example, a subclass combining FinalistBase and OPRO). Nothing in llmopt, wcs, cand_base, ledger, ladder, proof
or proposers changes: every other strategy runs as before.

    S1  code track   by budget, not by a count: an ask when no patch is pending, at least one screen is left after this
                     one, a run was made since the last ask, and either the knob side has no candidate predicted at
                     max(2 %, 2 sigma_pair) or more with P >= 0.5 (`_knob_strong`: the time model's untried rows, S4's P)
                     or the next screen is one the strategy protects (`protected_screens`); at most 3 patches per ask; the
                     first ask brings the code review (wcs's, unchanged), whose no_patch does not close the track (for each
                     ask the verdict reads REVIEW_VERDICT, put back right after); while code can still be asked, the last
                     CODE_CALLS = 2 calls are the code track's (a configuration call is held back). Every patch call
                     carries three additive, self-describing fields through wcs's `context` hook: `measured_history`
                     (HISTORY_MAX = 40 rows at ten screens, F1's 4 per screen beyond - with peaks, out-of-memory
                     shortfalls and patch results), `transfer` (other
                     jobs' proven patches by weighted gain, one per site, and their inert ones as do-not-repeat with what
                     their proof lines printed) and `fix_inert` (the last patch whose probe did not prove it took effect:
                     its self-proof record and added lines, "fix this one rather than rewrite")
    S2  patch memory at the start, the home's transfer brief (xfer's loader) and the `patches` sections other jobs' packs
                     carry (packs the experience plan loaded, this job's own numbers left out); at the end this arm's proven
                     and inert patches as describe()'s strategy-neutral `patches` section (mechanism, canonical site, gain,
                     inert reason, scanner lines), which runs.write_back puts on the arm's pack header
    S3  derived      a candidate derived from a failed run (a rescue, a completion, a repair) carries that run's patches
                     with the adopted chain (`derived_ids`, `derived_q`: the run's files rebased onto what is carried
                     now); wcs's edge rescue with those ids (`_rescue`)
    S4  memory price the memory ledger is the only feasibility source: P(peak <= the card less 2 % of it); a run's own
                     result decides its whole configuration (out of memory, thrash: 0; fitted: 1); declared GiB (the
                     ledger's priors, SD_DECLARED 0.5) never push P below P_FLOOR_DECLARED = 0.15 while their values are
                     unmeasured; isolated_blame: a value is lowered alone (x BLOCKED_VALUE 0.25) only when, in every run
                     that failed for memory holding it, every other value off the reference appeared in a run that fitted
                     (wcs's _blocked_values, its measured dominance and the store's warnings are not used); the brief's
                     hard rule 5 in the plans, not in P: a configuration holding the whole combination of values (and
                     patches) a run failed for memory with is probed before any full run, and not run in full when the
                     allowance has no probe for it
    S5  probe thrash a probe whose measured peak reaches THRASH_PEAK = 0.95 of the card is thrash: no full run, the ledger
                     (its interval), the completion material (`failed_material`); full runs keep wcs's jitter rule
    S6  no OOM runs  a counter of knob-side runs in a row out of memory (probes and full runs count, any other result
                     resets it, patch runs do not touch it); after S6_STREAK = 2 the next knob candidate needs P(fits) >=
                     S6_P = 0.9 (S4's, the ledger's) - else it is refused and the screen goes to the code track (an ask is
                     due; the draw takes only such configurations; none in the table: the code track's patch, else no
                     proposal at all); the uniform-draw start alone runs in full regardless (`uniform_start_q`) and
                     counts; a patch that ran out of
                     memory is never retried - a later patch of the same mechanism (its words without tags and digits, or
                     its files) is not queued
    ladder           RungLadder's rungs and constants: LOWFI_SHARE 0.5, PROBE_BELOW 0.7; a knob configuration is probed
                     first only below 0.7 (within the allowance), a patch always (its self-proof); no short rung unless
                     asked (FinalistLadder): the allowance is LOWFI_SHARE x the screens x the mean full run, never a share
                     of the GPU-hour cap (D5: a raised gpu_hours buys no probe). Base f1.1 (FINALISTS_IMPL_SPEC B.9 P2,
                     F1 (b)): the code proofs' probes - a patch's self-proof, a fix_inert repair's, a derived candidate's
                     that carries an unadopted patch - are paid first by their own pool of CODE_PROBES_MAX = 4 per arm
                     (at ten screens; F1's 0.4 per screen beyond) outside the allowance; beyond it exactly as in f1.0: a
                     patch's probe by the allowance's code share
                     (CODE_RESERVE, the knob side's unspent part borrowed: fb_code_borrow), a derived candidate's on the
                     knob side - so f1.1 never pays the code track less than f1.0; the knob side never draws on the pool;
                     the GPU-hours in three lines (pool, code share, knob side); an ask only while a proof can be paid;
                     on a job where a probe costs more than RATIO_MAX = 0.6 of a full run the knob side probes only below
                     RATIO_PROBE_BELOW = 0.5 (P >= 0.5 runs in full; forced probes - hard rule 5, a proof, a rescue -
                     unchanged). f1.0 and f1.1 differ only in this probe accounting
    F1  rates        (the open wave, FABLE_OPEN_DESIGN_20261102 section 1) the per-arm caps are rates of the arm's screen
                     budget B, read at the start as bcomp reads its knob screens (`rate_screens`: the session's budget,
                     counted at least RATE_SCREENS = 10) and set before the prior call (`_fb_rates`): the model calls and
                     dollars CALLS_RATE = 1 and USD_RATE = 0.8 per screen (llmopt's CALLS_MAX / USD_MAX at ten screens;
                     the asks they bound follow the calls), the history rows HISTORY_RATE = 4 (llmopt's HISTORY_MAX), the
                     code-probe pool CODE_PROBES_RATE = 0.4 (CODE_PROBES_MAX); search/explore scales its contrarian, repair
                     and revise caps the same way (0.2 per screen). At ten screens or fewer every cap is today's constant
                     (nothing changes); at 20: 20 calls / 16 USD, 80 rows, 8 probes, 4 / 4 / 4. A cap given to the
                     constructor or changed before the start is kept. llmopt.py itself is unchanged (the shared files stay
                     byte for byte the pristine tree's)
    v4               (FABLE_CODE_DIAGNOSIS_20261102 F1-F10; tree pc_code_o2) every fix applies only while the
                     code track is open to the arm (`_can_code()`: a knob-only arm runs v3's knob side byte for byte) and the
                     screen budget is beyond ten screens (the open wave's arms; at ten screens or fewer every v3 suite runs
                     as before, as F1's rates); each can be switched off (`v4_off`, V4_PARTS):
        F1 floor     a knob-side pick of a held kind (`fb_floor_kinds`: the tail and the draw)
                     runs only when its two-sigma upside - the time model's predicted gain + FLOOR_SD sd - reaches
                     max(2 %, 2 sigma_pair) and its P is at least STRONG_P, while the code track can take the screen (a
                     full screen ran, ASK_LEFT screens left, a call and dollars, a patch probe affordable); a refused
                     screen goes to the code track (`_fb_floor_code`: a pending patch, F2's stack test, an ask without the
                     knob side's hold-back); when the code track has nothing to run either, the floor is lifted for that
                     step (`_fb_v4_step`). The controlled tests and the repairs keep their own floors (v3's detectable
                     upside on the declared rating, M3's coverage and cap): they are what covers a wrong prior
        F2 stack     a proven patch not adopted with a positive gain (0 < gain <= sigma_pair) joins a pending stack; when
                     the stack's summed measured gain (ln) reaches max(2 %, 2 sigma_pair), one full screen runs the chain
                     with the stack merged on the incumbent (each patch's own edit against the files it ran on; their own
                     lines within the per-patch limit together; one whose edit no longer merges is left out) - its members
                     adopted when it measures more than sigma_pair over its base (or the engine makes it the incumbent);
                     the lock re-run stays the arbiter
        F3 repeat    never-repeat (search/explore) refuses a configuration for a failed run only when the configuration
                     carries that run's unproven patches: a failure that carried a patch beyond the chain, or whose
                     isolated_blame names a patch, is that patch's
        F4 lines     the session's per-patch line limit counts the new patch's own added and removed lines against the
                     candidate it is proposed on, the self-proof's lines left out, the whole overlay under OVERLAY_CEILING x
                     the limit (agent/search.py `patch_lines_own`, set on the arm's session at the start)
        F5 timeout   a model call that hit its wall is charged the usage its stream reported, priced at the model's rates
                     (`stream_charge`; the cap only when the stream reported none): CallGuard inside the budget driver; a
                     patch ask that timed out is asked again once with a slim payload (no transfer field); the per-call
                     wall the spec sets (llm.timeout_s, `_fb_rates`) on the backend
        F6 never     a step with nothing on the knob side goes to the code track before it ends: a pending patch, the
           done      stack test, an ask - with no run since the last ask when that ask's patches all ended before a run
                     (their refusals in the ledger it shows), again while it brings nothing to run and a call is left; the
                     arm ends with screens left only when the calls (or dollars) are spent or nothing is legal
        F7 slow      a patch that declares warmup_updates or active_from_update of SLOW_START or more probes with the
           start     launch timeout as its cap (no start cap)
        F8 store     a contrarian move holding a value that failed to start together with others on an earlier arm (wcs's
                     `from_store`) is probed first, never run in full first; with no probe affordable it waits (eligible)
        F9 ledger P  the controlled test's score is the optimistic gain x the ledger's P (the exploration module's M1
                     ranking); a move at P below STRONG_P does not displace a feasible test
        F10 slim     the transfer field's do-not-repeat list goes in the first ask that sends the field only
        budget       (FABLE_V4_BUDGET_20261102 section 1) the caps a spec sets in its manifest, read where the base
                     reads its budget (`_fb_rates`, `spec_budget`; the code track open; a key absent: v3's value, a cap
                     given to the constructor kept): search.budget.llm_usd (the session's own budget too) -> the dollars,
                     search.budget.llm_calls -> the model calls and the asks they bound, search.caps.patches_per_ask,
                     search.caps.code_probes (the code-proof pool), search.caps.fix_calls and fix_calls_per_screen (the
                     fix calls, below; 0: none), llm.max_usd_per_call and llm.timeout_s (one call's dollars and wall, on
                     the backend); search.budget.llm_tokens is the session's
        fix call     a queued patch the guard refused (its message with the lines it counted) or whose probe failed (out
                     of memory, thrash, slower than its base, no update before its start cap: the run's reasons and
                     numbers) gets one patch call carrying that finding (`fix_refused`, FIX_REFUSED_RULE), asking for the
                     patch revised or a replacement - before any screen is spent on it; at most fix_calls_per_screen per
                     screen and fix_calls per arm, within the calls and dollars; a patch measured in full is not fixed
                     (its result goes into the next ask, F6's re-ask path)
    v4.1             (the user's rule of 11-03, tree pc_code_o2b) `bounded_asks`: what a model call shows of
                     the past is bounded, as late calls had grown to several times an early one - the measured history
                     (both calls), the ledger of the patch call (every entry, no longer its last 40), wcs's failed runs
                     and inert patches (every one, no longer the last 10 / 6): the rows that matter most whole within a
                     character budget, the rest folded into one summary that keeps its information (search/fold.py); a
                     list that fits is shown as before. The stacking loop's fix is the session's (agent/search.py
                     `_inherited_overlay`); `rebudget` and the search-state checkpoints serve `infrarsi run --resume`

Hooks of a finalist: `protected_screens` (1-based screen numbers on which the code track may ask whatever the knob side
has), `_knob_strong` (the knob side's best candidate, by default from the table), `derived_ids` / `derived_q` (S3),
`failed_material`, `price_tags`, `isolated_blame`, `oom_combinations` (S4 / S5), `s6_active`, `uniform_start_q` (S6),
`finalist_view` and `patch_section` (its report and the write-back), `run_engine` (its registration on the Engine); its
switches: FB_PARTS (`fb_off`).

Nothing here names a knob, a model or a library: every value comes from the table, the ledger, the home, the model's
answers and the runs.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import math
import re
from pathlib import Path

from ..agent.state import Stateful
from . import llmopt as _llmopt
from .engine import Proposal
from .fold import (FAILURE_CHARS, INERT_CHARS, failure_label, failure_line, fold_history, fold_records, inert_label,
                   inert_line)
from .ladder import PROBE_BELOW, RungLadder, lowfi_spent, screens_budget
from .ledger import THRASH_LO
from .llmopt import CALLS_MAX, HISTORY_MAX, USD_MAX, LLMOptBase
from .hls3 import GATED_STEPS
from .proof import PROBE_STEPS_MAX, events_of, files_of, parsed, proof_of
from .proof import check as proof_check
from .strategies import merge_patches
from .wcs import P_ALONE_FAILED, RESCUE_MAX, RESCUE_SHORT, WCS, Surrogate, _coerce
from .wcs import vk as _wcs_vk
from .wcs import proposer_flags as wcs_flags
from .xfer import XFer, brief_entries, env_numbers, same_env, similarity

CODE_CALLS = 2                  # S1: the last model calls are the code track's while it can still ask
PATCHES_PER_ASK = 3             # S1: patches queued per ask at most
ASK_LEFT = 2                    # S1: an ask needs this many screens left (this one and at least one after it)
STRONG_MIN, STRONG_P = 0.02, 0.5   # S1: a knob candidate predicted >= max(2 %, 2 sigma_pair) with P >= 0.5 holds asks back
KNOB_SCAN_MAX = 400             # S1: the time model's best untried rows whose feasibility is read, at most
CARD_RESERVE = 0.02             # S4: a configuration fits below the card less this share of it
P_FLOOR_DECLARED = 0.15         # S4: declared memory never pushes the feasibility below this
BLOCKED_VALUE = 0.25            # S4: isolated_blame's factor per value blamed alone
THRASH_PEAK = 0.95              # S5: a probe whose measured peak reaches this share of the card is thrash
S6_STREAK, S6_P = 2, 0.9        # S6: after this many knob-side runs in a row out of memory (probes count), the next knob
#                                 candidate needs P(fits) >= S6_P, else the screen goes to the code track
SAME_FAILURE_STOP = 6           # the session's same-failure stop (D5, Fable on SHARED_r1 P2: random's rule): six full runs
#                                 in a row failing with one signature end the arm (probe verdicts never reach the counter)
FB_VERSION = 'f1.1'             # the shared base's version (f1.0 -> f1.1: only the probe accounting, B.9 P2)
CODE_PROBES_MAX = 4             # f1.1 (B.9 P2): the code proofs' probes of an arm, outside the low-fidelity allowance
RATE_SCREENS = 10               # F1: the caps are per-screen rates of the screen budget B counted at least this (today's
#                                 budget: every cap below and llmopt's is today's constant at ten screens or fewer)
CALLS_RATE, USD_RATE = 1, 0.8   # F1: llmopt's CALLS_MAX / USD_MAX (10 / 8.0 at ten screens) per screen of the budget
HISTORY_RATE = 4                # F1: llmopt's HISTORY_MAX (40 at ten screens) per screen: the history rows a prompt shows
CODE_PROBES_RATE = 0.4          # F1: CODE_PROBES_MAX (4 at ten screens) per screen
V4_VERSION = 'v4'               # the diagnosis fixes (module docstring, v4): code track open, beyond ten screens
FLOOR_SD = 2                    # v4 F1: a held knob-side pick's upside is its predicted gain + FLOOR_SD sd (the time model's)
FLOOR_KINDS = frozenset({'tail', 'draw'})   # v4 F1: the knob-side picks the floor holds (a finalist adds its own kinds)
OVERLAY_CEILING = 3             # v4 F4: the whole overlay at most this x the per-patch line limit
SPEC_INT = ('calls', 'patches_per_ask', 'code_probes', 'fix_calls', 'fix_calls_per_screen')   # v4: the counts among the
#                                 spec's caps (`spec_budget`)
FIX_KINDS = ('oom', 'thrash', 'slow_probe', 'start_slow', 'start_cap', 'hang')   # v4: a probe failing so gets a fix call
FIX_FINDING_MAX = 400           # v4: the characters of one line of a fix call's finding
CACHE_1H = 2.0                  # v4 F5: a one-hour cache write at this x the input rate (the price table's: five minutes)
SLOW_START = 2                  # v4 F7: a declared warmup_updates / active_from_update from which the probe has no start cap
RATIO_MAX = 0.6                 # f1.1: a probe dearer than this share of a full run ...
RATIO_PROBE_BELOW = 0.5         # ... and the knob side probes first only below this P (P >= 0.5 runs in full)
PORT_MAX, INERT_MAX = 12, 10    # S1: the transfer field's proven entries (one per site) and do-not-repeat entries
FIX_LINES = 40                  # S1: the added lines of the inert patch `fix_inert` shows, at most
MEM_FAIL = ('oom', 'thrash')    # the runs that failed for memory (at any rung)
FITTED = ('ok', 'slower', 'numerics')   # the runs that fitted in memory (wcs's _record_memory: status ok)
UNTESTED = ('queued', 'open', 'superseded', 'dropped', 'bundled', 'deferred', 'probed', 'proposed', 'duplicate')
INCONCLUSIVE = 1.0              # S2: a measured gain at least this (percent) not adopted: neither proven nor inert
MECH_MAX, SITE_MAX, REASON_MAX, PROOF_MAX, LINE_MAX = 200, 160, 200, 120, 200
_GONE = object()                # (a derived candidate's patches no longer apply on what is carried now)
_VK: dict = {}                  # vk's memo: plain scalars only


def vk(v) -> str:
    """wcs's value key (json, sorted keys), memoised for plain scalars only - str, int, bool, None, and floats by repr
    (so 1, 1.0 and True, or 0.0 and -0.0, never share a key); anything else is built each time. A draw over a ~25k-row
    table builds millions of keys from a few values (hard rule 6: decisions in seconds)."""
    t = type(v)
    if t is str or t is int or t is bool or v is None:
        key = (t, v)
    elif t is float:
        key = (t, repr(v))
    else:
        return _wcs_vk(v)
    out = _VK.get(key)
    if out is None:
        out = _VK[key] = _wcs_vk(v)
    return out


def rate_screens(engine) -> int:
    """F1: the screen budget B the caps are rates of - the session's (ladder.screens_budget's reading, as bcomp reads its
    knob screens at the start), counted at least RATE_SCREENS; RATE_SCREENS without a session budget."""
    budget = getattr(getattr(engine, 's', None), 'budget', None)
    return max(RATE_SCREENS, int(getattr(budget, 'screens', 0) or 0))


def per_screen(rate: float, b: int) -> int:
    """F1: a count cap of `rate` per screen over `b` screens, rounded down (exact at ten and twenty screens)."""
    return int(math.floor(rate * b + 1e-9))

# the base's parts a finalist may switch off (`fb_off`; its own components map onto them)
FB_PARTS = {
    'fb_isolated_blame': 'S4 isolated_blame: a value whose every failed run holds no other value that never fitted is '
                         'lowered alone (x0.25)',
    'fb_peak_thrash': 'S5: a probe whose measured peak reaches 95 % of the card is thrash (no full run; the ledger and the '
                      'completion material)',
    'fb_transfer': "S2: the home's transfer brief and the patches other jobs' packs wrote back, read at the start and shown "
                   'in every patch call (`transfer`)',
    'fb_review_lift': "S1: a code review's no_patch does not close the code track (each ask shows the code again)",
    'fb_code_borrow': "ladder: beyond the code-probe pool (f1.1: 4 proofs outside the allowance, spent first), a patch's "
                      "probe the allowance's code share (CODE_RESERVE) cannot pay may draw on the knob side's unspent "
                      'allowance (f1.0\'s borrow); the two together stay within LOWFI_SHARE',
    'fb_s6': 'S6: after two knob-side runs in a row out of memory (probes count) the next knob candidate needs P(fits) >= '
             '0.9, else the screen goes to the code track; a patch mechanism that ran out of memory is never retried'}

# v4's fixes (module docstring, v4), each on unless listed in `v4_off`; all of them only with the code track open beyond ten
# screens
V4_PARTS = {
    'v4_floor': "F1: a held knob-side pick below the two-sigma floor or P 0.5 gives its screen to the code track",
    'v4_stack': 'F2: proven patches within the noise stacked and tested together once their summed gain passes the floor',
    'v4_repeat': "F3: never-repeat reads a failure that carried an unproven patch (or that isolated_blame gives a patch) as "
                 "that patch's",
    'v4_lines': "F4: the per-patch line limit on the new patch's own lines, the overlay under a ceiling",
    'v4_timeout': 'F5: a timed-out call charged its streamed usage, a slim re-ask once, the per-call wall from the spec',
    'v4_never_done': 'F6: a step with nothing on the knob side goes to the code track (re-asks while calls are left)',
    'v4_slow_start': 'F7: a patch declaring a slow start probes with the launch timeout as its cap',
    'v4_no_start_cap': "the runs' time limit as Auto-Research's (the user, 11-02 22:2x): no probe has a start cap - the "
                       "job's launch timeout bounds every run, as it bounds each Auto-Research experiment",
    'v4_store': 'F8: a contrarian move holding a value that failed to start on an earlier arm is probed first',
    'v4_ledger_p': "F9: the controlled test scored with the ledger's P; a move below P 0.5 does not displace a feasible test",
    'v4_brief_once': "F10: the transfer field's do-not-repeat list in the first ask that sends the field only",
    'v4_fix': 'fix calls: a patch the guard refused or whose probe failed gets one call with the finding (the spec\'s '
              'search.caps.fix_calls / fix_calls_per_screen; none set: no fix call)'}


def _diff_count(old: str, new: str) -> int:
    """Added + removed lines of a unified diff (the hooks' count of a patch's lines)."""
    n = 0
    for line in difflib.unified_diff(str(old or '').splitlines(), str(new or '').splitlines(), lineterm='', n=0):
        if not line.startswith(('+++', '---', '@@')):
            n += line[:1] in ('+', '-')
    return n


def _own_count(old: str, new: str) -> int:
    """v4: a patch's own lines as the session's F4 count reads them (agent/search.py `_own_lines_check`): added + removed,
    the self-proof's print lines (OWN_PROOF_MARK) left out."""
    from ..agent.search import OWN_PROOF_MARK

    def bare(text) -> str:
        return '\n'.join(ln for ln in str(text or '').splitlines() if OWN_PROOF_MARK not in ln)
    return _diff_count(bare(old), bare(new))


def spec_budget(engine) -> dict:
    """v4 (FABLE_V4_BUDGET_20261102 section 1): the caps the arm's spec set in its manifest - search.budget.llm_usd (as
    the session's Budget holds it) and llm_calls, search.caps.patches_per_ask, code_probes, fix_calls and
    fix_calls_per_screen, llm.max_usd_per_call and timeout_s - as name -> a positive number (the counts as int); a key
    absent or not a positive number is absent: v3's value stays."""
    s = getattr(engine, 's', None)
    raw = getattr(getattr(s, 'task', None), 'raw', None)
    man = raw.get('manifest') if isinstance(raw, dict) and isinstance(raw.get('manifest'), dict) else {}
    sub = lambda d, k: d.get(k) if isinstance(d.get(k), dict) else {}   # noqa: E731
    search, llm = sub(man, 'search'), sub(man, 'llm')
    budget, caps = sub(search, 'budget'), sub(search, 'caps')
    usd = getattr(getattr(s, 'budget', None), 'llm_usd', None)
    got = {'usd': budget.get('llm_usd') if usd is None else usd, 'calls': budget.get('llm_calls'),
           'patches_per_ask': caps.get('patches_per_ask'), 'code_probes': caps.get('code_probes'),
           'fix_calls': caps.get('fix_calls'), 'fix_calls_per_screen': caps.get('fix_calls_per_screen'),
           'call_usd': llm.get('max_usd_per_call'), 'call_wall_s': llm.get('timeout_s')}
    out = {}
    for k, v in got.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0:
            out[k] = max(1, int(v)) if k in SPEC_INT else float(v)
    return out


def stream_charge(call_dir, model) -> tuple[float | None, str]:
    """v4 F5: the dollars a call's stream reported (stdout.jsonl in its call directory): per assistant message its input,
    cache-write (one-hour writes at CACHE_1H x the input rate) and cache-read tokens (the message's largest usage), the
    output as the larger of the messages' output tokens and the thinking estimates the stream counted (`thinking_tokens`
    events) plus the text the assistant wrote (3.5 bytes a token), priced at the model's rates (the API backend's
    table). (None, why) when nothing usable was reported."""
    from ..agent.backends.anthropic_api import PRICES
    price = PRICES.get(str(model or ''))
    path = Path(call_dir) / 'stdout.jsonl' if call_dir else None
    if price is None or path is None or not path.is_file():
        return None, 'no price or no stream'
    msgs, thinking, text = {}, 0, 0
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return None, 'the stream could not be read'
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        if e.get('type') == 'system' and e.get('subtype') == 'thinking_tokens':
            d = e.get('estimated_tokens_delta')
            thinking += int(d) if isinstance(d, int) and not isinstance(d, bool) and d > 0 else 0
        m = e.get('message') if e.get('type') == 'assistant' and isinstance(e.get('message'), dict) else None
        if m is None:
            continue
        for b in m.get('content') or []:
            if isinstance(b, dict) and b.get('type') == 'text' and isinstance(b.get('text'), str):
                text += len(b['text'].encode('utf-8'))
        u = m.get('usage') if isinstance(m.get('usage'), dict) else {}
        cc = u.get('cache_creation') if isinstance(u.get('cache_creation'), dict) else {}
        row = msgs.setdefault(str(m.get('id') or len(msgs)), {})
        for k in ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'output_tokens'):
            v = u.get(k)
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                row[k] = max(row.get(k, 0), v)
        v = cc.get('ephemeral_1h_input_tokens')
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            row['h1'] = max(row.get('h1', 0), v)
    if not msgs and not thinking:
        return None, 'the stream reported no usage'
    g = lambda k: sum(r.get(k, 0) for r in msgs.values())   # noqa: E731
    h1 = min(g('h1'), g('cache_creation_input_tokens'))
    out = max(g('output_tokens'), thinking + math.ceil(text / 3.5))
    usd = (g('input_tokens') * price[0] + out * price[1] + g('cache_read_input_tokens') * price[2]
           + (g('cache_creation_input_tokens') - h1) * price[3] + h1 * CACHE_1H * price[0]) / 1e6
    return usd, (f'the stream: {len(msgs)} message(s), {out} output tokens ({thinking} thinking), '
                 f'{g("cache_creation_input_tokens")} cache-write, {g("cache_read_input_tokens")} cache-read, '
                 f'{g("input_tokens")} input')


class CallGuard(Stateful):
    """v4 F5: the backend inside the strategy's budget driver (BudgetDriver's inner): a call that hit its wall (TIMEOUT -
    the backend then charges its dollar cap, the usage unknown to it) is charged what its stream reported (`stream_charge`),
    never more than the cap; the cap when the stream reported nothing. The call's receipt stays the backend's; the charge
    is noted on the result (`meta`: charged_usd, upper_bound_usd, charged_from). Every other attribute is the backend's
    (read and set)."""
    CHECKPOINT_LIVE = ('_inner', '_owner')               # (its last status and the re-priced calls are its state)

    def __init__(self, inner, owner):
        self.__dict__.update(_inner=inner, _owner=owner, last_status=None, repriced=[])

    def __getattr__(self, name):
        return getattr(self.__dict__['_inner'], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.__dict__['_inner'], name, value)

    def complete(self, system, prompt, **kw):
        st = self.__dict__['_owner']
        res = self.__dict__['_inner'].complete(system, prompt, **kw)
        meta = getattr(res, 'meta', None) if isinstance(getattr(res, 'meta', None), dict) else {}
        self.__dict__['last_status'] = meta.get('status')
        cap = getattr(res, 'llm_usd', None)
        if meta.get('status') != 'TIMEOUT' or not st._v4('v4_timeout') or not isinstance(cap, (int, float)):
            return res
        usd, how = stream_charge(meta.get('call_dir'), getattr(res, 'model', None))
        rec = {'call_id': meta.get('call_id'), 'upper_bound_usd': round(float(cap), 4), 'charged_from': how}
        if usd is not None and usd < cap:
            res = dataclasses.replace(res, llm_usd=float(usd), meta={**meta, 'charged_usd': round(usd, 4),
                                                                     'upper_bound_usd': float(cap), 'charged_from': how})
            rec['charged_usd'] = round(usd, 4)
        else:
            rec['charged_usd'] = round(float(cap), 4)
        self.__dict__['repriced'].append(rec)
        return res

class PeakGate:
    """S5's peak rule first, then the gate it wraps (`_fb_gated`): a probe's gate that a search-state checkpoint can keep."""

    def __init__(self, owner, inner):
        self.owner, self.inner = owner, inner

    def __call__(self, engine, cid, base, row):
        why = self.owner._fb_peak_gate(engine, cid, row)
        if why:
            return why
        return self.inner(engine, cid, base, row) if self.inner is not None else None


class ProofGate:
    """A derived candidate's probe gate (`_fb_proof_probe`): the gate it wraps (the peak and speed gates), then the patch's
    self-proof against the chain's carrier (failing it: `patch_inert`, no full run)."""

    def __init__(self, owner, inner, carrier, files, h):
        self.owner, self.inner, self.carrier, self.files, self.h = owner, inner, carrier, files, h

    def __call__(self, eng, cid, base, row):
        why = self.inner(eng, cid, base, row) if self.inner is not None else None
        if why:
            return why
        fb: list = []
        got = proof_check(eng, self.carrier, row, self.files, self.h, whole=True, deferred=True, feedback=fb)
        self.owner.note_feedback += [{'candidate': cid, **x} for x in fb]
        if got:
            self.owner.proof_fail[cid] = got
            return got['reason'], 'patch_inert'
        return None


REVIEW_VERDICT = 'no_patch (the code is shown again: the search keeps asking for patches within its model budget)'
HISTORY_RULE = (
    'MEASURED HISTORY: `rows` lists what this search measured, best first: each configuration\'s knob values off the '
    'reference (`change`), the code patches it carried (`code_patches`), its measured speed over the reference or how it '
    'failed (out of memory and by how much, thrash at the memory edge, a slow start, a patch that did not take effect), '
    'its measured peak memory per GPU (`peak_gib`) and, for a run that tested a patch, that patch\'s result '
    '(`patch_result`); then the probes, the failures and the proposals not run, with why. Patch from what was measured.')
TRANSFER_RULE = (
    'TRANSFER: `proven` lists code patches that measured faster on other training jobs, the best weighted gain first, one '
    'per site: what each did (`mechanism`), the function it patched (`site`, the source\'s own words in `where`), its '
    'measured gain there (`gain_pct`), how much it counts here (`weight`: that job\'s card, nodes and size against this '
    'one\'s), what its proof line printed there (`proof`), that job\'s numbers (`source`) and how many entries of that site '
    'were proven (`runs`). Where this job runs the same code, a port - the same mechanism at this job\'s own call site, '
    'with its proof lines - may be among your patches. `do_not_repeat` lists variants that did not take effect, gave no '
    'measurable gain, were slower or failed on other jobs, with why and what their proof lines printed: do not propose '
    'them again as they were.')
FIX_RULE = (
    'FIX THIS ONE: this is the last patch of this search whose probe did not prove that it took effect: its id, mechanism '
    'and site, why (`reason`), the proof lines its probe printed (`printed`), what the scanner read of them per site '
    '(`parsed`: found, set, active, deferred) and the lines it added (`added_lines`). Fix this one rather than rewrite it: '
    'the same mechanism at the same site with the cause the reason names removed (the hook the job really calls, a found '
    'value that differs from the set one, an `active` line printed when the patched code runs). At most one of your '
    'patches is that fix; begin its `mechanism` with `[fix <id>]`.')
FIX_REFUSED_RULE = (
    'FIX THIS PATCH: this patch of this search did not reach a full run - the patch guard refused it (`stage` "guard": '
    'its message, with the lines it counted), or its probe failed (`stage` "probe": out of memory, thrash at the memory '
    'edge, slower than the configuration it was applied to, or no update before its start cap; what the probe measured '
    'and the end of its log). `finding` gives that, `added_lines` the lines it added. Reply with one patch: this one '
    'revised so that the finding no longer holds (fewer changed lines, less memory, the cause the log names removed), or '
    'a replacement for the same goal; begin its `mechanism` with `[fix <id>]`.')


# ---------------------------------------------------------------- the canonical site (as build_transfer_brief writes it)
GENERIC = {'forward', '__call__', '__init__', 'backward', 'step', 'apply', 'run', 'call', 'enable'}
DOTTED = r'\*?[A-Za-z_][\w*]*(?:\.\*?[A-Za-z_][\w*]*)+'
IDENT = r'\*?[A-Za-z_][\w*]*'
LEAD_TOKEN = r'^([A-Z][a-z]+)(\d+(?:_\d+)*)([A-Z].*)$'   # a class name with a versioned leading token
FILE_EXT = {'py', 'json', 'yaml', 'yml', 'txt', 'md', 'sh', 'cfg', 'toml', 'log', 'jsonl'}
VERSIONED = r'\d_\d'


def _cls(word: str) -> bool:
    w = word.lstrip('*')
    return (len(w) >= 5 and w[:1].isupper() and any(c.islower() for c in w) and sum(c.isupper() for c in w) >= 2
            and '_' not in w.strip('_')) or bool(re.match(LEAD_TOKEN, w))


def _strip_lead(name: str) -> str:
    name = name.lstrip('*')
    m = re.match(LEAD_TOKEN, name)
    return m.group(3) if m else name


def _plain(word: str) -> bool:
    return word in GENERIC or (word.startswith('__') and word.endswith('__')) or (word.islower() and '_' not in word)


def canonical(token: str | None) -> str | None:
    """A code path as its canonical site: its last component, with the one before it when that is a generic method or
    a plain word; a class name without a versioned leading token."""
    if not token:
        return None
    parts = [p.lstrip('*') for p in token.split('.') if p.lstrip('*')]
    if not parts:
        return None
    last = parts[-1]
    if _cls(last):
        return _strip_lead(last)
    if _plain(last) and len(parts) >= 2:
        prev = parts[-2]
        return f'{_strip_lead(prev) if _cls(prev) else prev}.{last}'
    return last


def _strong(token: str) -> bool:
    parts = [p.lstrip('*') for p in token.split('.')]
    return len(parts) >= 3 or any(_cls(p) or ('_' in p.strip('_') and not p.isupper()) for p in parts)


def _tokens(text: str | None, kind: str) -> list[str]:
    if not text:
        return []
    if kind in ('dotted', 'weak'):
        paths = [m.group(0) for m in re.finditer(DOTTED, text) if m.group(0).rsplit('.', 1)[-1].lower() not in FILE_EXT]
        return [p for p in paths if _strong(p)] if kind == 'dotted' else paths
    out = []
    for tok in re.findall(IDENT, text):
        t = tok.lstrip('*')
        if kind == 'camel' and _cls(t) or kind == 'snake' and '_' in t.strip('_') and not t.isupper():
            out.append(t)
    return out


def _modules(*texts) -> set:
    out = set()
    for text in texts:
        for m in re.finditer(DOTTED, str(text or '')):
            out |= {p.lstrip('*').lower() for p in m.group(0).split('.')[:-1]}
    return out


def site_of(proof_key: str | None, site: str | None, mechanism: str | None, found: str | None = None) -> str | None:
    """The canonical site of a patch (build_transfer_brief's rule): the first acceptable name of a dotted code path of the
    self-proof's site name, of its `found` text, of the `site` text, of the mechanism; then a class or snake_case name of
    the self-proof's site name, of the `site` text, a weak dotted path of it, a snake_case or class name of the mechanism.
    Never a file name, nor a single name that is a versioned module or a package component of the record's paths."""
    mechanism = re.sub(r'^review: [^:]{1,80}: ', '', str(mechanism or ''))
    mods = _modules(proof_key, found, site, mechanism)
    for text, kind in ((proof_key, 'dotted'), (found, 'dotted'), (site, 'dotted'), (mechanism, 'dotted'),
                       (proof_key, 'camel'), (proof_key, 'snake'), (site, 'snake'), (site, 'camel'), (site, 'weak'),
                       (mechanism, 'snake'), (mechanism, 'camel')):
        for tok in _tokens(text, kind):
            c = canonical(tok)
            if not c or '.' not in c and not _cls(c) and (re.search(VERSIONED, c) or c.lower() in mods):
                continue
            return c
    return None


def one_line(text, limit: int) -> str:
    s = ' '.join(str(text or '').split())
    s = re.sub(r'^review: [^:]{1,80}: ', '', s)
    return s if len(s) <= limit else s[:limit - 3].rstrip() + '...'


def inert_reason(why: str | None, kind: str | None = None) -> str:
    w = one_line(why, REASON_MAX - 20)
    low = (why or '').lower()
    if 'no `active` line' in low or 'never ran' in low:
        return f'not activated: {w}'
    if 'deferred' in low:
        return f'deferred to the original: {w}'
    if kind in ('inert_at_start', 'inert at its probe', 'patch_inert') or 'still reported' in low or 'found' in low:
        return f'self-proof failed: {w}'
    return f'failed at its run ({kind}): {w}' if kind else f'failed at its run: {w}'


def measured_reason(g: float | None, slower: bool = False) -> str:
    if g is None:
        return 'slower: aborted against the incumbent' if slower else 'no measurable gain'
    return (f'slower: {g:+.2f} % at its full run' if g < 0 or slower else
            f'no measurable gain: {g:+.2f} % at its full run')


def _shown_env(env: dict) -> dict:
    return {k: env[k] for k in ('card_gib', 'nodes', 'params_b') if env.get(k) is not None}


def _site_key(e: dict) -> str:
    return (e.get('site') or '').split('.')[0].lower() or f'#{e["id"]}'


def _gain(x) -> float | None:
    return round(float(x), 2) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def pack_patch_entries(packs) -> tuple[list[dict], list[dict], int]:
    """(proven, inert, packs read) of the `patches` sections runs.write_back put on the headers of a list of packs (each a
    list of pack lines): each entry with its pack's environment numbers (its `env` line), checked on its own as the
    brief's are (a proven entry needs a positive gain); a pack without numbers gives none."""
    proven, inert, n = [], [], 0
    for lines in packs or []:
        head = lines[0] if isinstance(lines, list) and lines and isinstance(lines[0], dict) else {}
        sec = head.get('patches')
        if not isinstance(sec, dict):
            continue
        env = env_numbers(next((x for x in lines if isinstance(x, dict) and x.get('kind') == 'env'), None))
        if env is None:
            continue
        n += 1
        for kind, out in (('proven', proven), ('inert', inert)):
            rows = sec.get(kind)
            for r in rows if isinstance(rows, list) else []:
                if not isinstance(r, dict) or not isinstance(r.get('mechanism'), str) or not r['mechanism'].strip():
                    continue
                g = _gain(r.get('gain_pct'))
                if kind == 'proven' and (g is None or g <= 0):
                    continue
                site = r.get('site') if isinstance(r.get('site'), str) and r['site'].strip() else None
                proof = r.get('proof') if isinstance(r.get('proof'), dict) else None
                printed = [one_line(x, LINE_MAX) for x in r.get('printed') or [] if isinstance(x, str)][:3]
                out.append({'id': f'{kind[0]}h{len(out) + 1:03d}', 'mechanism': one_line(r['mechanism'], MECH_MAX),
                            'site': site[:80] if site else None,
                            'where': one_line(r.get('site_text'), SITE_MAX) or None, 'gain_pct': g,
                            'reason': one_line(r.get('reason'), REASON_MAX) or None,
                            'proof': {k: str(v)[:PROOF_MAX] for k, v in proof.items() if k in ('found', 'set')} if proof else None,
                            'printed': printed or None, 'env': env, 'pack': str(head.get('pack_id') or '')[:80]})
    return proven, inert, n


def _pos(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


POOL_NOTE = ('Base f1.1: the code proofs\' probes are paid first by the code-probe pool (at most {n} per arm, outside the '
             'low-fidelity allowance), then as in f1.0 by the allowance\'s code share (CODE_RESERVE, the knob side\'s unspent '
             'part borrowed); three lines: code_pool_gpu_hours, code_share_gpu_hours, knob_lowfi_gpu_hours.')


class FinalistLadder(RungLadder):
    """RungLadder's rungs, constants and split (LOWFI_SHARE of the allowance for probes and short runs, CODE_RESERVE of
    it the code proofs' share) with the finalists' changes:

    - the allowance is LOWFI_SHARE x the screen budget x the engine's mean full run, never a share of the GPU-hour cap
      (ladder.allowance_of takes the smaller of that and gpu_hours / (1 + LOWFI_SHARE)): raising a job's gpu_hours (D5)
      buys no probe;
    - with `borrow` (fb_code_borrow) a code proof's probe its own share (and RungLadder's stranded-remainder rule) cannot
      pay takes what it lacks from the knob side's remainder, moved and logged as a transfer (f1.0); the two shares
      always sum to the allowance;
    - base f1.1 (FINALISTS_IMPL_SPEC B.9 P2, Fable's F1 (b)): the code proofs' probes - a patch's self-proof, a
      fix_inert repair's, a derived candidate's that carries an unadopted patch - are paid first by the code-probe pool
      of `code_probes_max` (CODE_PROBES_MAX = 4; F1: `size_pool`, 0.4 per screen of the budget beyond ten screens) per
      arm outside the allowance (`code_log`); only once it is spent does
      a patch's probe fall back to the code share and the borrow above (`code_share_log`), and a derived candidate's to
      the knob side, as in f1.0 - so the code track is never paid less than in f1.0; a knob probe is never paid by the
      pool. The probes are counted when they ran, with the GPU-hours of their probe rows, in three lines (`spent_pool`,
      `spent(code=True)`, `spent()`); `code_probes_max = 0` (`pool=False`) is f1.0's accounting;
    - base f1.1's ratio rule (`ratio_rule`): when a probe costs more than RATIO_MAX of a full run (`probe_ratio`), the
      knob side's threshold `probe_below` - a property, so the ladder's plans, the draw and a strategy's own checks all
      read it - is at most RATIO_PROBE_BELOW: P >= 0.5 runs in full, P < 0.5 is probed first; the probes a rule forces
      (hard rule 5, a proof, a rescue) are not touched."""

    def __init__(self, *, borrow: bool = True, pool: bool = True, ratio_rule: bool = True,
                 code_probes_max: int = CODE_PROBES_MAX, **kw):
        self._engine = None                              # (the engine the threshold reads its ratio from: `bind`)
        self._ratio_memo = None
        self.borrow, self.ratio_rule = bool(borrow), bool(ratio_rule)
        self.pool_size = int(code_probes_max)            # (the pool's probes when it is on; F1: `size_pool`)
        self.code_probes_max = self.pool_size if pool else 0
        self.code_log: list[dict] = []                   # f1.1: the code proofs' probes the pool paid
        self.code_share_log: list[dict] = []             # ... and those the allowance's code share paid (beyond the pool)
        super().__init__(**kw)

    @property
    def pool(self) -> bool:
        """Whether the code-probe pool is on (f1.1; off: f1.0's accounting)."""
        return self.code_probes_max > 0

    @pool.setter
    def pool(self, on) -> None:
        self.code_probes_max = self.pool_size if on else 0

    def size_pool(self, n: int) -> None:
        """F1: the code-probe pool's size for the arm's screen budget (`n`: CODE_PROBES_RATE x B), unless a size other
        than CODE_PROBES_MAX was set; a pool switched off stays off."""
        if self.pool_size != CODE_PROBES_MAX or self.code_probes_max not in (0, CODE_PROBES_MAX):
            return
        self.pool_size = int(n)
        if self.code_probes_max:
            self.code_probes_max = self.pool_size

    # ------------------------------------------------------------ the knob side's threshold (the ratio rule)
    @property
    def probe_below(self) -> float:
        base = self.__dict__.get('_probe_below', PROBE_BELOW)
        if not self.ratio_rule or self._engine is None:
            return base
        r = self.probe_ratio()[0]
        return min(base, RATIO_PROBE_BELOW) if r is not None and r > RATIO_MAX else base

    @probe_below.setter
    def probe_below(self, value) -> None:
        self.__dict__['_probe_below'] = float(value)

    def bind(self, engine) -> None:
        self._engine = engine

    @staticmethod
    def _ref_gpu_hours(engine) -> float | None:
        c = engine.s.candidates.get('c0')
        cost = ((getattr(c, 'receipt', None) or {}).get('cost') or {}) if c is not None else {}
        g = cost.get('gpu_seconds') if isinstance(cost, dict) else None
        return g / 3600.0 if _pos(g) else None

    @staticmethod
    def _ref_ratio(engine) -> float | None:
        """Before any probe: the reference run's start - launch to the end of its first update, its wall less the span of
        its stamped updates - plus the probe's other updates at its step time, against its wall; None without a timed
        reference run."""
        c = engine.s.candidates.get('c0')
        rec = (getattr(c, 'receipt', None) or {}) if c is not None else {}
        ev = rec.get('evidence') if isinstance(rec.get('evidence'), dict) else {}
        cost = rec.get('cost') if isinstance(rec.get('cost'), dict) else {}
        wall = ev.get('wall_s') if _pos(ev.get('wall_s')) else cost.get('wall_seconds')
        ts = [r['t'] for r in ev.get('per_step') or [] if isinstance(r, dict) and _pos(r.get('t'))]
        st = ev.get('step_time') if _pos(ev.get('step_time')) else None
        if not _pos(wall) or len(ts) < 2 or st is None:
            return None
        start = max(0.0, wall - (ts[-1] - ts[0]))
        n = int(getattr(engine, 'probe_steps', 2) or 2)
        return min(1.0, (start + max(0, n - 1) * st) / wall)

    def probe_ratio(self, engine=None) -> tuple[float | None, str]:
        """(a probe's cost over a full run's, where it was read): the first valid probe against the first valid full run
        (else the reference's); before a valid probe, the reference's start share (`_ref_ratio`); (None, why) without
        either."""
        engine = engine if engine is not None else self._engine
        if engine is None:
            return None, 'no engine'
        rows = engine.s.screens
        memo = self._ratio_memo
        if memo is not None and memo[0] is engine and memo[1] == len(rows):
            return memo[2]
        got: tuple = (None, 'no valid probe and no timed reference run')
        probe = next((r['gpu_hours'] for r in rows if r.get('fidelity') == 'probe' and r.get('valid')
                      and _pos(r.get('gpu_hours'))), None)
        if probe is not None:
            full = next((r['gpu_hours'] for r in rows if r.get('fidelity', 'full') == 'full' and r.get('valid')
                         and _pos(r.get('gpu_hours'))), None) or self._ref_gpu_hours(engine)
            if full:
                got = (probe / full, 'the first valid probe against the first valid full run')
        if got[0] is None:
            est = self._ref_ratio(engine)
            if est is not None:
                got = (est, "the reference run's start share (no valid probe yet)")
        self._ratio_memo = (engine, len(rows), got)
        return got

    # ------------------------------------------------------------ the allowance (the knob side's) and the code pool
    def base_allowance(self, engine) -> float:
        """LOWFI_SHARE x the screen budget's full runs at the engine's mean full-run cost (GPU-hours)."""
        return self.lowfi_share * screens_budget(engine) * engine.costs()['full']

    def reserve(self, engine) -> float:
        if not self.code_reserve:
            return 0.0
        a = self.base_allowance(engine)
        return min(a, max(self.code_reserve * a, self.next_cost(engine, 'probe')))

    def allowance(self, engine, *, code: bool = False) -> float:
        a = self.base_allowance(engine)
        r = min(a, max(0.0, self.reserve(engine) + self.moved))
        return r if code else a - r

    def code_probes_left(self) -> int:
        """The code-probe pool's probes left (the pool alone; whether a proof can be paid at all: `code_probe_ok`)."""
        return max(0, self.code_probes_max - len(self.code_log))

    def note_code_probe(self, cid: str, hid: str, gpu_hours: float, kind: str) -> None:
        """A code proof's probe that ran: the pool's while it has one left, else the allowance's code share's (a patch's;
        a derived candidate's beyond the pool is the knob side's, as in f1.0, and is not noted)."""
        row = {'candidate': cid, 'id': hid, 'kind': kind, 'gpu_hours': round(float(gpu_hours or 0.0), 6)}
        if self.code_probes_left() >= 1:
            self.code_log.append(row)
        elif kind != 'derived':
            self.code_share_log.append(row)

    def spent_pool(self) -> float:
        """The code-probe pool's GPU-hours (outside the allowance)."""
        return sum(x['gpu_hours'] for x in self.code_log)

    def spent(self, engine, *, code: bool = False) -> float:
        """What the allowance's code share (`code`) or its knob side has spent: the code proofs' probes the share paid,
        or every other probe and short run but the pool's."""
        c = sum(x['gpu_hours'] for x in self.code_share_log)
        return c if code else max(0.0, lowfi_spent(engine) - self.spent_pool() - c)

    def _fund(self, engine, cost: float, *, code: bool = False) -> bool:
        self._engine = engine
        if code and self.code_probes_left() >= 1:
            return True                                  # (the pool first: a count, the allowance untouched)
        if super()._fund(engine, cost, code=code):       # (f1.0: the share's own remainder, a stranded one moved once)
            return True
        if not code or not self.code_reserve or not self.borrow:
            return False
        need = cost - max(0.0, self.allowance(engine, code=True) - self.spent(engine, code=True))
        other = self.allowance(engine) - self.spent(engine)
        if need > 0 and other + 1e-12 >= need:          # (f1.0's borrow of the knob side's remainder)
            self.moved += need
            self.transfers.append({'to': 'code', 'gpu_hours': round(need, 4), 'borrowed': True})
            return True
        return False

    def code_probe_ok(self, engine) -> bool:
        """Whether a code proof's probe can be paid now: the pool's probes left, else (f1.0) the code share, a stranded
        remainder of the knob side's or - with `borrow` - the knob side's remainder; read without moving anything."""
        self._engine = engine
        if self.code_probes_left() >= 1:
            return True
        cost = self.next_cost(engine, 'probe')
        own = self.allowance(engine, code=True) - self.spent(engine, code=True)
        if own + 1e-12 >= cost:
            return True
        other = self.allowance(engine) - self.spent(engine)
        if not self.code_reserve:
            return False
        return own + max(0.0, other) + 1e-12 >= cost if self.borrow else 1e-12 < other < cost - 1e-12 and \
            own + other + 1e-12 >= cost

    def plan(self, engine, p_feasible: float, **kw) -> tuple:
        self._engine = engine
        return super().plan(engine, p_feasible, **kw)

    def describe(self, engine=None) -> dict:
        if engine is not None:
            self._engine = engine
        out = super().describe(engine)
        r, src = self.probe_ratio(engine)
        out.update(pool=self.pool, borrow=self.borrow,
                   allowance_basis='LOWFI_SHARE x the screens x the mean full run (no GPU-hour cap)',
                   probe_below_base=self.__dict__.get('_probe_below'),
                   probe_ratio={'value': None if r is None else round(r, 3), 'source': src, 'max': RATIO_MAX,
                                'rule_on': self.ratio_rule, 'applies': bool(self.ratio_rule and r is not None and r > RATIO_MAX)},
                   code_pool={'max': self.code_probes_max, 'used': len(self.code_log), 'left': self.code_probes_left(),
                              'gpu_hours': round(self.spent_pool(), 4),
                              'probes': self.code_log[-max(CODE_PROBES_MAX, self.code_probes_max):]},
                   code_share={'probes': len(self.code_share_log),
                               'gpu_hours': round(sum(x['gpu_hours'] for x in self.code_share_log), 4),
                               'last': self.code_share_log[-4:]})
        if engine is not None:
            acc = out['account']
            acc.update(allowance_gpu_hours=round(self.base_allowance(engine), 4),
                       code_share_allowance_gpu_hours=round(self.allowance(engine, code=True), 4),
                       knob_allowance_gpu_hours=round(self.allowance(engine), 4),
                       code_pool_gpu_hours=round(self.spent_pool(), 4), code_pool_probes=len(self.code_log),
                       code_share_gpu_hours=round(self.spent(engine, code=True), 4),
                       code_share_probes=len(self.code_share_log), knob_lowfi_gpu_hours=round(self.spent(engine), 4),
                       note=(acc.get('note', '') + ' ' + POOL_NOTE.format(n=self.code_probes_max)).strip())
        return out


# ---------------------------------------------------------------- the base
class FinalistBase(LLMOptBase):
    """The finalists' shared parts (the module docstring). `fb_off`: base parts switched off (FB_PARTS); `brief`, `packs`,
    `env`: the transfer brief, the packs and this job's numbers given (else read from the home at the start, as xfer does);
    `short`: the ladder's short rung (off unless asked). Every other keyword goes to the next class (LLMOptBase's, or the
    strategy's own when FinalistBase comes first in its bases)."""
    name = 'finalist'
    protected_screens: frozenset = frozenset()          # S1: 1-based screen numbers the code track may always ask on

    def checkpoint_parts(self) -> dict:
        """Search-state checkpoints: the budget driver's counters (llmopt) and the call guard inside it (F5)."""
        d = getattr(self, '_bdriver', None)
        g = d.__dict__.get('_inner') if d is not None else None
        return {**super().checkpoint_parts(), **({'call_guard': g} if isinstance(g, CallGuard) else {})}
    history_max = HISTORY_MAX                           # the history rows a prompt shows (F1: `_fb_rates` beyond ten screens)
    fb_floor_kinds: frozenset = FLOOR_KINDS             # v4 F1: the knob-side picks the floor holds
    bounded_asks = True                                 # v4.1: what a call shows of the past is bounded (search/fold.py)

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        """wcs's flags (the code track's patch fields, the self-proof, the library source)."""
        return wcs_flags([])

    def __init__(self, proposer, *, fb_off=(), brief=None, packs=None, env=None, short: bool = False, v4_off=(), **kw):
        off = set(fb_off or ())
        self.fb_on = {k: k not in off for k in FB_PARTS}
        self.v4_on = {k: k not in set(v4_off or ()) for k in V4_PARTS}   # v4: its fixes (module docstring)
        self.fb_rate_given = frozenset(k for k in ('calls_max', 'usd_max') if k in kw)   # F1: caps given are kept
        self.fb_rate_screens: int | None = None          # F1: the budget the caps were set from (`_fb_rates`)
        # v4's state (the module docstring, v4)
        self.v4_counts = {'floor_refused': 0, 'floor_to_code': 0, 'floor_lifted': 0, 'empty_to_code': 0, 'reasks': 0,
                          'slim_reasks': 0, 'stack_tests': 0, 'stack_adopted': 0, 'stack_dropped': 0, 'repeat_lifted': 0,
                          'slow_start_probes': 0, 'store_probes': 0, 'store_held': 0, 'test_kept': 0, 'fix_calls': 0}
        self.v4_floor: list[dict] = []                   # F1: the picks the floor refused (the last ones), lifted or not
        self.v4_code: list[dict] = []                    # F1 / F6: the screens handed to the code track and what they got
        self.v4_stack: list[str] = []                    # F2: proven patches within the noise waiting for a stack test
        self.v4_stacked: set = set()                     # F2: patches a stack test carried (each stacked once)
        self.v4_stacks: list[dict] = []                  # F2: every stack test: its members, summed gain, outcome
        self.v4_notes: list[str] = []
        self.v4_brief_sent = False                       # F10: an ask sent the transfer field already
        self._fb_floor_lift = False                      # (F1: the floor lifted for this step: `_fb_v4_step`)
        self.fb_floor_refused = False                    # (F1: the last held pick was refused by the floor)
        self._fb_force_ask = False                       # (F1 / F6: an ask the code track makes for a screen it was handed)
        self._fb_force_why = ''
        self._fb_slim = False                            # (F5: the slim re-ask after a timed-out patch call)
        self._v4_stack_q = None                          # (F2: the stack test proposed, until it is observed)
        self._v4_store: list | None = None               # (F8: the earlier arms' failed-start sets, read once)
        self.fb_spec_caps: dict = {}                     # v4: the caps the spec's manifest set (`spec_budget`, `_fb_rates`)
        self.v4_fix_max = self.v4_fix_per_screen = 0     # v4: the fix calls per arm / per screen (the spec's; 0: none)
        self.v4_fix_pending: list[dict] = []             # v4: patches refused or failed at their probe, waiting for a fix
        self.v4_fix_log: list[dict] = []                 # v4: every fix call: the patch, why, what it queued
        self._fb_fixing: dict | None = None              # (v4: the fix call in progress: its `fix_refused` field)
        self.v4_fix_authorized: set = set()              # v4: patches a fix call asked for (`[fix <id>]`, files changed):
        #                                                  the failed patch's S6 ban does not drop them
        self._fb_events_seen = 0                         # (v4: the session's events read: a refusal's finding is new)
        super().__init__(proposer, **kw)
        if self.bounded_asks and self.proposer is not None:   # v4.1: the patch call folds the ledger (search/fold.py)
            self.proposer.bounded_ledger = True
        if getattr(self, '_bdriver', None) is not None:  # (F5: the backend inside the budget driver, guarded)
            self._bdriver.__dict__['_inner'] = CallGuard(self._bdriver.__dict__['_inner'], self)
        old = self.ladder                                # (the candidates' ladder; a strategy may have set probes off)
        self.ladder = FinalistLadder(borrow=self.fb_on['fb_code_borrow'], short=bool(short),
                                     probe_below=min(old.probe_below, PROBE_BELOW), promote_z=old.promote_z,
                                     lowfi_share=old.lowfi_share,
                                     code_reserve=old.code_reserve if self._fb_l1_open() else 0.0)   # (no proofs to hold for)
        self.fb_code_hids: set = set()                   # f1.1: derived candidates whose probe is a code proof (the pool's)
        self._fb_rows_seen = 0                           # (the session's rows observed: an evaluation's own rows follow)
        self.code_asks = self.calls_max                  # S1: the asks are bounded by the model budget, not by a count
        self.patches_per_ask = PATCHES_PER_ASK
        self.fb_brief_in, self.fb_packs_in, self.fb_env_in = brief, packs, env
        self.fb_target: dict | None = None               # this job's numbers (S2)
        self.fb_proven: list[dict] = []                  # other jobs' proven patches (the brief, the packs' sections)
        self.fb_inert: list[dict] = []                   # ... and their inert ones
        self.fb_sources = {'brief': 0, 'packs': 0}       # entries read from each
        self.fb_notes: list[str] = []
        self.fb_asks: list[dict] = []                    # S1: every ask, with why it was made and what it showed
        self.fb_obs = 0                                  # observations so far (the caches' stamp)
        self.fb_runs = 0                                 # runs observed (a probe, a short run, a full run)
        self.fb_runs_at_ask = -1                         # ... at the last ask (an ask needs a run since)
        self.fb_code_idle = False                        # an ask that made no model call: the reserve is released
        self.fb_strong: dict | None = None               # the last knob candidate that held an ask back
        self.fb_inert_log: list[dict] = []               # patches inert at their probe: pid, candidate, base, shown
        self.fb_patch_cid: dict[str, str] = {}           # patch id -> the candidate that tested it
        self.fb_cid_patch: dict[str, str] = {}           # ... and back
        self.fb_fixes: dict[str, str] = {}               # a patch -> the inert patch it says it fixes (`[fix <id>]`)
        self.fb_thrash: list[dict] = []                  # S5: the probes recorded as thrash by their peak
        self.fb_counts = {'held_by_knob_side': 0, 'knob_calls_held': 0, 'combo_probed': 0, 'combo_refused': 0,
                          'floor_reads': 0, 'derived': 0, 'derived_dropped': 0, 'protected_asks': 0,
                          's6_refused': 0, 's6_code_asks': 0, 's6_no_knob': 0, 's6_patches_dropped': 0}
        self.fb_oom_streak = 0                           # S6: knob-side runs in a row that ran out of memory
        self.fb_oom_streak_max = 0
        self.fb_s6_bans: list[dict] = []                 # S6: patch mechanisms that ran out of memory (never retried)
        self.fb_s6_refused = False                       # (the last knob proposal was refused by S6's gate)
        self.fb_refused = False                          # the last knob proposal was refused (S4: no probe for a combination)
        self._fb_combo_out = False                       # (a redraw outside the combinations: their members weigh 0)
        self._fb_asking = False                          # (during an ask: the patch call's extra fields)
        self._fb_shown: dict = {}                        # ... what the call's fields showed
        self._fb_iso_stamp = self._fb_combo_stamp = self._fb_meas_stamp = None
        self._fb_iso: dict = {}
        self._fb_iso_ctx: dict = {}                      # S4: blamed value -> the contexts where it fitted
        self._fb_combos: list = []
        self._fb_measured: set = set()

    # S4: the ledger's line is the card less CARD_RESERVE, set whenever a ledger is attached (whichever class builds it)
    @property
    def mledger(self):
        return self.__dict__.get('_fb_mledger')

    @mledger.setter
    def mledger(self, led) -> None:
        if led is not None:
            led.margin = 1.0 - CARD_RESERVE
        self.__dict__['_fb_mledger'] = led

    # ------------------------------------------------------------ set-up
    def _init(self, engine) -> None:
        self._fb_rates(engine)                           # (F1: before the prior call, which the budget driver caps)
        self.engine = engine                             # (v4: its gate reads the budget from the start)
        self._v4_init(engine)                            # (v4 F4 / F5: before the prior call)
        super()._init(engine)
        self.ladder.bind(engine)                         # (f1.1: the knob side's threshold reads the probe / full ratio)
        self._fb_rows_seen = len(engine.s.screens)
        self._fb_transfer_init(engine)

    def _fb_rates(self, engine) -> None:
        """F1 (the module docstring): the caps as per-screen rates of the arm's screen budget, set at the start - the model
        calls (the asks they bound follow) and dollars, the history rows, the code-probe pool. Ten screens or fewer:
        today's constants, nothing is set; a cap given to the constructor or changed before the start is kept (a second
        call changes nothing)."""
        b = rate_screens(engine)
        self.fb_rate_screens = b
        spec = self.fb_spec_caps = (spec_budget(engine) if self.on.get('wcs_code', True) and self._can_code() else {})
        rated = b > RATE_SCREENS                         # (v4: the spec's caps, the code track open; v3 without them)
        if 'calls_max' not in self.fb_rate_given and self.calls_max == CALLS_MAX and (rated or 'calls' in spec):
            calls = spec.get('calls') or per_screen(CALLS_RATE, b)
            if self.code_asks == self.calls_max:
                self.code_asks = calls                   # (S1: the asks are bounded by the model budget)
            self.calls_max = calls
        if 'usd_max' not in self.fb_rate_given and self.usd_max == USD_MAX and (rated or 'usd' in spec):
            self.usd_max = spec.get('usd') or USD_RATE * b
        if rated and self.history_max == HISTORY_MAX:
            self.history_max = per_screen(HISTORY_RATE, b)
        if rated:
            self.ladder.size_pool(per_screen(CODE_PROBES_RATE, b))
        if 'code_probes' in spec:                        # (v4: the spec's pool; a pool switched off stays off)
            self.ladder.pool_size = spec['code_probes']
            if self.ladder.code_probes_max:
                self.ladder.code_probes_max = self.ladder.pool_size
        if 'patches_per_ask' in spec and self.patches_per_ask == PATCHES_PER_ASK:
            self.patches_per_ask = spec['patches_per_ask']
            if hasattr(self.proposer, 'patch_hypotheses'):   # (v4 r2: the request asks for as many)
                self.proposer.patch_hypotheses = self.patches_per_ask
        self.v4_fix_max, self.v4_fix_per_screen = spec.get('fix_calls', 0), spec.get('fix_calls_per_screen', 0)
        d = getattr(self, '_bdriver', None)              # (one call's dollars and wall, on the backend)
        inner = d.__dict__.get('_inner') if d is not None else None
        for cap, attr in (('call_wall_s', 'timeout_seconds'), ('call_usd', 'max_usd_per_call')):
            if cap in spec and inner is not None and hasattr(inner, attr):
                setattr(inner, attr, float(spec[cap]))
        if spec:
            self.v4_notes.append('the spec set: ' + ', '.join(f'{k} {v:g}' for k, v in sorted(spec.items())))

    def rebudget(self, engine) -> None:
        """`infrarsi run --resume` (project/resume.py): the caps that follow the budget and the spec - the model calls and
        dollars, the asks they bound, the history rows, the code-probe pool, the patches per ask, the fix calls, one call's
        dollars and wall - are derived again from the resumed run's budget and manifest by the rules of `_fb_rates`; what the
        search has used (the calls made, the dollars spent, the probes of the pool) is state and stays. With the budget and the
        manifest unchanged every cap comes out as it was."""
        given = self.fb_rate_given
        if 'calls_max' not in given:
            self.calls_max = CALLS_MAX
            self.code_asks = CALLS_MAX
        if 'usd_max' not in given:
            self.usd_max = USD_MAX
        self.history_max = HISTORY_MAX
        self.ladder.pool_size = CODE_PROBES_MAX
        if self.ladder.code_probes_max:
            self.ladder.code_probes_max = CODE_PROBES_MAX
        self.patches_per_ask = PATCHES_PER_ASK
        vars(self.proposer).pop('patch_hypotheses', None)
        n = len(self.v4_notes)
        self._fb_rates(engine)
        self.v4_notes[n:] = ['resume: ' + x for x in self.v4_notes[n:]]

    def _fb_target_env(self, engine) -> dict:
        """This job's numbers: given, else the experience plan's fingerprint, else the memory facts and the manifest."""
        if self.fb_env_in is not None:
            return env_numbers(self.fb_env_in) or env_numbers({})
        fp = (self.exp or {}).get('fingerprint') if isinstance(self.exp, dict) else None
        fp = fp if isinstance(fp, dict) else {}
        raw = getattr(getattr(engine.s, 'task', None), 'raw', None)
        man = raw.get('manifest') if isinstance(raw, dict) and isinstance(raw.get('manifest'), dict) else {}
        card = fp.get('memory_gib') or self.facts.get('cap_gib')
        return env_numbers({'card_gib': card, 'nodes': fp.get('nodes'), 'gpus_per_node': fp.get('gpus_per_node'),
                            'params_b': (fp.get('model') or {}).get('params_b') or self.facts.get('params_b'),
                            'framework': fp.get('framework') or man.get('framework')})

    def _fb_transfer_init(self, engine) -> None:
        """S2's read: the transfer brief and the packs' `patches` sections - given, else from the arm's home with the
        experience plan on (xfer's loaders: only the packs the plan loaded) -, this job's own numbers left out, each entry
        once."""
        if not self.fb_on['fb_transfer']:
            self.fb_notes.append('the transfer part is off: no brief and no written-back patch read')
            return
        self.fb_target = self._fb_target_env(engine)
        brief, packs, why = self.fb_brief_in, self.fb_packs_in, None
        if brief is None or packs is None:
            if self.exp:
                try:
                    if brief is None:
                        brief, why = XFer._home_brief(self)
                    if packs is None:
                        packs = XFer._home_packs(self)
                except Exception as exc:                 # noqa: BLE001 - an unreadable home: nothing transferred
                    why = f'the home could not be read: {str(exc)[:120]}'
            elif brief is None and packs is None:
                why = 'no experience plan: no home read'
        proven, inert = [], []
        if brief is not None:
            bp, bi, bad = brief_entries(brief)
            if bad:
                self.fb_notes.append(f'the transfer brief is unusable ({bad})')
            else:
                proven += bp
                inert += bi
                self.fb_sources['brief'] = len(bp) + len(bi)
        elif why:
            self.fb_notes.append(f'no transfer brief ({why})')
        try:
            pp, pi, n = pack_patch_entries(packs or [])
        except Exception as exc:                         # noqa: BLE001 - packs of an unexpected shape: none of theirs
            pp, pi, n = [], [], 0
            self.fb_notes.append(f'the packs\' patch sections could not be read: {str(exc)[:120]}')
        self.fb_sources['packs'] = len(pp) + len(pi)
        if n:
            self.fb_notes.append(f'{n} pack(s) carried a patch section ({len(pp)} proven, {len(pi)} inert)')
        own = [e for e in proven + inert + pp + pi if same_env(e['env'], self.fb_target)]
        if own:
            self.fb_notes.append(f'{len(own)} entries with this job\'s own numbers left out')
        seen = set()
        for kind, rows, out in (('proven', proven + pp, self.fb_proven), ('inert', inert + pi, self.fb_inert)):
            for e in rows:
                key = (kind, e['mechanism'].lower(), (e.get('site') or '').lower(), json.dumps(e['env'], sort_keys=True))
                if same_env(e['env'], self.fb_target) or key in seen:
                    continue
                seen.add(key)
                out.append(e)

    # ------------------------------------------------------------ S1: the code track by budget
    def _fb_protected(self, engine) -> bool:
        return engine.full_screens() + 1 in set(self.protected_screens or ())

    def _knob_strong(self, engine, inc) -> dict | None:
        """The knob side's best candidate when one is predicted at max(2 %, 2 sigma_pair) or more over the incumbent with
        P >= 0.5 (it holds the code track back), else None: the time model's untried rows of the table with the chain,
        best first, the feasibility read for at most KNOB_SCAN_MAX of them. A strategy with its own candidate list
        overrides it."""
        thr = max(math.log1p(STRONG_MIN), 2 * engine.sigma_pair())
        mean = self.surr.mean()
        y_inc = Surrogate.value(self._incumbent_feats(engine, inc), mean)
        rows = []
        for cfg, ids in self._candidates(engine, inc):
            mu = Surrogate.value(self._feats(cfg, ids), mean) - y_inc
            if mu >= thr:
                rows.append((mu, cfg, ids))
        rows.sort(key=lambda t: -t[0])
        for mu, cfg, ids in rows[:KNOB_SCAN_MAX]:
            p, _ = self._pf(cfg, ids)
            if p >= STRONG_P:
                return {'change': self._words_of(cfg), 'predicted_gain_pct': round(100 * math.expm1(mu), 2),
                        'p_feasible': round(p, 3)}
        return None

    def _fb_code_probe_ok(self, engine) -> bool:
        """Whether a patch's probe (its self-proof) can be paid now - f1.1: the code-probe pool, else the allowance's code
        share and its borrow as in f1.0 - read without moving anything."""
        return self.ladder.code_probe_ok(engine)

    def _fb_ask_due(self, engine, inc, left) -> str | None:
        """Why an ask is due now (S1), else None."""
        if not self.on['wcs_code'] or left < ASK_LEFT or not self._can_code() or not self._budget_ok(engine):
            return None
        if self.fb_runs <= self.fb_runs_at_ask and not (self._fb_force_ask and self._v4('v4_never_done', engine)
                                                         and not self.queue):
            return None                                  # (no run since the last ask: never two asks in a row - v4 F6:
            #                                              unless the code track was handed the screen and the last ask's
            #                                              patches all ended before a run)
        if not self._fb_code_probe_ok(engine):
            return None                                  # (no patch could prove itself: its probe is not affordable)
        if self.s6_active():
            self.fb_counts['s6_code_asks'] += 1
            return 'two knob-side runs in a row ran out of memory: the screen goes to the code track (S6)'
        if self._fb_protected(engine):
            self.fb_counts['protected_asks'] += 1
            return f'screen {engine.full_screens() + 1} is a protected code screen'
        if self._fb_force_ask:                           # v4 F1 / F6: the knob side gave its screen to the code track
            return f'the code track was handed the screen: {self._fb_force_why}'
        strong = self._knob_strong(engine, inc)
        if strong is not None:
            self.fb_strong = strong
            self.fb_counts['held_by_knob_side'] += 1
            return None
        return 'no knob candidate is predicted at max(2 %, 2 sigma_pair) or more with P >= 0.5'

    def _code(self, engine, inc, left, ask: bool = True) -> list:
        """llmopt's queue (rebased onto the incumbent and the chain; a patch measured already leaves it), then S1's ask
        when none is pending and one is due."""
        out = []
        for q in super()._code(engine, inc, left, ask=False):
            pid = q.hypothesis.split('|', 1)[1]
            rec = self.patches.get(pid) or {}
            ban = (self._fb_s6_banned(None, rec.get('files') or q.patch) if pid in self.v4_fix_authorized
                   else self._fb_s6_banned(rec.get('mechanism') or q.mechanism))   # (v4 r2: a fix call's revision)
            if engine._fresh(q) and ban is None:
                out.append(q)
                continue
            self.queue = [x for x in self.queue if x is not q]   # (measured already, or S6's no-retry: no ask held back)
            if self.ledger.get(q.hypothesis):
                self.ledger[q.hypothesis]['status'] = 'duplicate' if ban is None else 'dropped'
                if ban is not None:
                    self.ledger[q.hypothesis]['why'] = f'the mechanism of {ban["pid"]}, which ran out of memory (S6)'
                    self.fb_counts['s6_patches_dropped'] += 1
        if out and not (self._v4('v4_stack', engine) and not any(self._lowfi_ok(engine, q) for q in out)):
            return out                                   # (v4 r2: queued patches none of whose probes is affordable now
            #                                              stay queued and do not hold a due stack test back)
        if ask and not out and self._fb_fix_due(engine, left):  # v4: a refused / failed patch's fix call before anything else
            got = self._fb_fix_ask(engine, inc, left)
            if got:
                return got
        st = self._fb_stack_q(engine, inc) if self._v4('v4_stack', engine) else None
        if st is not None:                               # v4 F2: a due stack test before any ask (no call spent)
            return [st]
        if out or not ask:
            return out
        why = self._fb_ask_due(engine, inc, left)
        return self._fb_ask(engine, inc, left, why) if why else out

    def _fb_lift_review(self):
        """S1: the code review's no_patch reads REVIEW_VERDICT for this ask only (the proposer keeps the code and sends the
        patch request) - on a review already made, or right after the review made inside this ask's own call; returns
        what puts the review back as it answered."""
        prop, held = self.proposer, {}

        def lift(r):
            if isinstance(r, dict) and r.get('verdict') == 'no_patch':
                held['review'] = r
                prop.review = {**r, 'verdict': REVIEW_VERDICT}
        if getattr(prop, 'review', None) is not None:
            lift(prop.review)
        elif callable(getattr(prop, '_review', None)) and hasattr(prop, 'review'):
            real = prop._review

            def review_then_ask(engine, base, code, digest):
                out = real(engine, base, code, digest)
                lift(getattr(prop, 'review', None))
                return out
            prop._review = review_then_ask               # (an instance attribute over the method, for this call only)

        def restore():
            prop.__dict__.pop('_review', None)
            if 'review' in held:
                prop.review = held['review']
        return restore

    def _fb_ask(self, engine, inc, left, why: str, limit: int | None = None) -> list:
        """One patch call (S1): the extra fields, the review's veto lifted, at most PATCHES_PER_ASK patches queued (patches
        only, through wcs's proposal with its self-proof gate); recorded in `fb_asks`."""
        self.asks += 1
        self.last_ask_at = engine.full_screens()
        self.fb_runs_at_ask = self.fb_runs
        self._allowance()
        before = self._llm_used()
        flag: dict = {}
        restore_review = self._fb_lift_review() if self.fb_on['fb_review_lift'] else (lambda: None)
        restore_spy = XFer._spy_dispatch(self, flag)
        self._fb_asking, self._fb_shown = True, {}
        raised = False
        try:
            hyps = self.proposer.hypotheses(engine, inc, ledger=self.model_view())
            if not hyps and self._fb_slim_due(engine):   # v4 F5: the call hit its wall - once, the ask with a slim payload
                self._fb_slim, self._fb_shown = True, {}
                self.v4_counts['slim_reasks'] += 1
                hyps = self.proposer.hypotheses(engine, inc, ledger=self.model_view())
        except Exception as exc:                         # noqa: BLE001 - a failed call: nothing queued, not a dead arm
            (getattr(engine, 'log', None) or (lambda m: None))(f'[search] {self.name}: the patch call failed: {str(exc)[:200]}')
            hyps, raised = [], True
        finally:
            self._fb_asking = False
            slim, self._fb_slim = self._fb_slim, False
            restore_spy()
            restore_review()
        self.llm_calls = self._llm_used()
        sent = bool(flag.get('sent') or flag.get('direct') and not raised)
        shown = dict(self._fb_shown)
        if slim:
            shown['slim'] = 'v4 F5: asked again without the transfer field after the call hit its wall'
        if sent and shown.get('transfer'):
            self.v4_brief_sent = True                    # (v4 F10: later asks carry the transfer field without do-not-repeat)
        if sent and shown.get('fix'):
            for r in self.fb_inert_log:
                if r['pid'] == shown['fix']:
                    r['shown'] = True
        if not sent and self._llm_used() == before:
            self.fb_code_idle = True                     # (no model call made: the code track cannot use the reserve)
        queued, m = [], 0
        hyps = list(hyps or [])
        fix_pid = (self._fb_fixing or {}).get('pid')
        asked = rf'\s*\[fix\s+{re.escape(fix_pid)}\]' if fix_pid else None   # (the fix call's own tag)
        if asked is not None:                            # (v4 r2: the reply the fix call asked for first)
            hyps.sort(key=lambda h: not (isinstance(h, dict) and re.match(asked, str(h.get('mechanism') or ''), re.I)))
        for h in hyps:
            if not isinstance(h, dict) or not h.get('patch') or (h.get('change') or {}):
                continue                                 # patches only
            revision = asked is not None and bool(re.match(asked, str(h.get('mechanism') or ''), re.I))
            if self._fb_s6_banned(None if revision else h.get('mechanism'), h.get('patch')) is not None:
                self.fb_counts['s6_patches_dropped'] += 1   # (S6: a mechanism that ran out of memory, never retried -
                continue                                 #  v4 r2: a fix call's revision only when its files repeat)
            if m >= (limit or self.patches_per_ask):     # (v4: a fix call queues one)
                break
            m += 1
            q = self._patch_proposal(engine, inc, h)
            if q is None:
                continue
            self.queue.append(q)
            queued.append(q)
            pid = q.hypothesis.split('|', 1)[1]
            if revision:
                self.v4_fix_authorized.add(pid)
            fx = re.match(r'\s*\[fix\s+(p\d+)\]', str(h.get('mechanism') or ''), re.I)
            if fx and fx.group(1).lower() in self.patches and fx.group(1).lower() != pid:
                self.fb_fixes[pid] = fx.group(1).lower()
        self.fb_asks.append({'screens_done': engine.full_screens(), 'left': left, 'why': why, 'sent': sent,
                             'calls': [before, self._llm_used()], 'queued': [q.hypothesis.split('|', 1)[1] for q in queued],
                             **{k: v for k, v in shown.items() if v}})
        return queued

    def _fb_l1_open(self) -> bool:
        """(The knob-only study, trees pc_code_k*.) Whether the arm's open layers include patches (L1): a knob-only project
        (code.mutable empty) has the hooks' layers ('L0',); a proposer without hooks: both, as before."""
        return 'L1' in (getattr(getattr(self.proposer, 'hooks', None), 'layers', None) or ('L0', 'L1'))

    def _can_code(self) -> bool:
        """llmopt's gate - a model to ask - and patches open (_fb_l1_open): a knob-only arm has no code track, so S1 holds
        back no model calls for it and asks for no patch (and the ladder holds no allowance for its proofs)."""
        return self._fb_l1_open() and super()._can_code()

    def _fb_code_can_use_calls(self, engine) -> bool:
        """Whether the code track can still use model calls: on, with a model for it, no ask that could make no call,
        ASK_LEFT screens or more left, a patch probe affordable."""
        return (self.on['wcs_code'] and self._can_code() and not self.fb_code_idle and self._left(engine) >= ASK_LEFT
                and self._fb_code_probe_ok(engine))

    def _fb_knob_call_ok(self, engine) -> bool:
        """S1: a configuration call may start with only CODE_CALLS calls or fewer left only when the code track can no
        longer use them."""
        return self._calls_left() > CODE_CALLS or not self._fb_code_can_use_calls(engine)

    def _ask(self, engine, inc, purpose: str = 'round') -> list[dict]:
        """llmopt's configuration call, held back while the last CODE_CALLS calls are the code track's (S1) - and capped
        there: while the code track can use them, the call and every retry of it (BudgetDriver's attempts, the proposer's
        re-asks) see calls_max less CODE_CALLS, restored right after (SHARED_r1 P1-4)."""
        if not self._fb_knob_call_ok(engine):
            self.last_got = None
            self.fb_counts['knob_calls_held'] += 1
            return []
        if not self._fb_code_can_use_calls(engine):
            return super()._ask(engine, inc, purpose)
        full = self.calls_max
        self.calls_max = full - CODE_CALLS
        try:
            return super()._ask(engine, inc, purpose)
        finally:
            self.calls_max = full
            self._allowance()                            # (the code track's calls back in the proposer's allowance)

    # ------------------------------------------------------------ S1: the patch call's fields
    def context(self, engine, base) -> dict:
        """wcs's extra fields of the patch call and, during an ask, S1's: `measured_history`, `transfer`, `fix_inert`;
        v4.1 (`bounded_asks`): wcs's failed runs and inert patches folded (`_fb_fold_context`)."""
        out = super().context(engine, base)
        if self.bounded_asks:
            out = self._fb_fold_context(out)
        if not self._fb_asking:
            return out
        out = dict(out)
        rows = self._history(engine)
        if rows:
            out['measured_history'] = {'rule': HISTORY_RULE, 'rows': rows}
        view = None if self._fb_slim else self._fb_transfer_view(   # (v4 F5: the slim re-ask has no transfer field; F10:
            later=self.v4_brief_sent and self._v4('v4_brief_once', engine))   # do-not-repeat in the first ask only)
        if view:
            out['transfer'] = view
        fix = self._fb_fix_view(engine)
        if fix:
            out['fix_inert'] = fix
        refused = self._fb_fix_refused_view(engine, self._fb_fixing) if self._fb_fixing else None
        if refused:
            out['fix_refused'] = refused                 # (v4: the fix call's patch and finding)
        self._fb_shown = {'fix_refused': refused['id'] if refused else None, 'history_rows': len(rows),
                          'transfer': {'proven': [r['id'] for r in view['proven']],
                                       'do_not_repeat': len(view.get('do_not_repeat') or [])}
                          if view else None, 'fix': fix['id'] if fix else None}
        return out

    def _fb_fold_context(self, out: dict) -> dict:
        """v4.1: wcs's `failures` (its last ten) and `inert_patches` (its last six) become every failed run and every inert
        patch, the newest whole within their budgets, the older folded (search/fold.py)."""
        out = dict(out)
        if 'failures' in out:
            out['failures'] = fold_records(self.failures, chars=FAILURE_CHARS, line=failure_line, label=failure_label,
                                           what='failed runs')
        if 'inert_patches' in out:
            out['inert_patches'] = fold_records(self.inert, chars=INERT_CHARS, line=inert_line, label=inert_label,
                                                what='inert patches')
        return out

    def model_view(self) -> list[dict]:
        """wcs's ledger as the patch call sees it; v4.1 (`bounded_asks`): every entry, not the last 40 - the call folds what
        does not fit (the proposer's `bounded_ledger`, search/fold.py)."""
        return super().model_view(last=None if self.bounded_asks else 40)

    def _fb_transfer_view(self, later: bool = False) -> dict | None:
        """The transfer field: the proven entries by weight x gain (the similarity of their job's numbers to this one's),
        one per site (its `runs`: the entries of that site), at most PORT_MAX; the inert ones by weight - those of a shown
        site first -, one per site and reason, at most INERT_MAX, with what their proof lines printed; None when nothing
        weighs. `later` (v4 F10: an ask after the first that sent the field): the proven entries only."""
        if not self.fb_on['fb_transfer'] or not (self.fb_proven or self.fb_inert):
            return None
        def weight(e):
            return similarity(e['env'], self.fb_target)
        rows = sorted(((weight(e) * e['gain_pct'], weight(e), e) for e in self.fb_proven), key=lambda r: (-r[0], r[2]['id']))
        shown, groups = [], {}
        for _, w, e in rows:
            if w <= 0.0:
                continue
            key = _site_key(e)
            if key in groups:
                groups[key]['runs'] += 1
                continue
            row = {'id': e['id'], 'mechanism': e['mechanism'], 'site': e.get('site'), 'where': e.get('where'),
                   'gain_pct': round(e['gain_pct'], 2), 'weight': round(w, 3), 'proof': e.get('proof'),
                   'source': _shown_env(e['env']), 'runs': 1}
            shown.append({k: v for k, v in row.items() if v is not None})
            groups[key] = shown[-1]
        shown = shown[:PORT_MAX]
        sites = {_site_key(r) for r in shown}
        inert = sorted(((weight(e), e) for e in self.fb_inert), key=lambda r: (_site_key(r[1]) not in sites, -r[0], r[1]['id']))
        dnr, seen = [], set()
        for w, e in inert:
            key = (_site_key(e), str(e.get('reason') or '').split(':', 1)[0])
            if w <= 0.0 or key in seen:
                continue
            seen.add(key)
            dnr.append({k: v for k, v in {'mechanism': e['mechanism'], 'site': e.get('site'), 'reason': e.get('reason'),
                                          'proof': e.get('proof'), 'printed': e.get('printed'),
                                          'source': _shown_env(e['env'])}.items() if v is not None})
            if len(dnr) >= INERT_MAX:
                break
        if later:                                        # (v4 F10: the do-not-repeat list went in the first ask)
            return {'rule': TRANSFER_RULE, 'proven': shown} if shown else None
        if not shown and not dnr:
            return None
        return {'rule': TRANSFER_RULE, 'proven': shown, 'do_not_repeat': dnr}

    def _fb_added_lines(self, engine, pid: str, base: str | None) -> list[str]:
        files = (self.patches.get(pid) or {}).get('files') or {}
        try:
            old = files_of(engine, base) if base in engine.s.candidates else {}
        except Exception:                                # noqa: BLE001 - no base files: every line is new
            old = {}
        out = []
        for f, text in sorted(files.items()):
            have = set(str(old.get(f) or '').splitlines())
            for ln in str(text or '').splitlines():
                if ln.strip() and ln not in have:
                    out.append(f'{f}: {ln[:LINE_MAX]}')
                    if len(out) >= FIX_LINES:
                        return out
        return out

    def _fb_fix_view(self, engine) -> dict | None:
        """The last patch inert at its probe that no ask has shown yet: its self-proof record (proof.check's: reason,
        note, printed lines, what the scanner read) and the lines it added, with FIX_RULE."""
        rec = next((r for r in reversed(self.fb_inert_log) if not r['shown']), None)
        if rec is None:
            return None
        row = next((x for x in reversed(self.inert) if x.get('candidate') == rec['cid']), {})
        p = self.patches.get(rec['pid']) or {}
        out = {'rule': FIX_RULE, 'id': rec['pid'], 'mechanism': p.get('mechanism'), 'site': p.get('site'),
               'reason': row.get('reason'), 'note': row.get('note'), 'printed': list(row.get('printed') or [])[:6],
               'parsed': row.get('parsed'), 'added_lines': self._fb_added_lines(engine, rec['pid'], rec.get('base'))}
        return {k: v for k, v in out.items() if v not in (None, [], {})}

    def _fb_patch_result(self, pid: str) -> str | None:
        rec = self.patches.get(pid) or {}
        st = str(rec.get('status') or '')
        if st == 'proven':
            g = rec.get('gain_pct')
            return (f'proven: {g:+.2f} % over its base' if isinstance(g, (int, float)) else 'proven') + (
                ', adopted (every later configuration carries it)' if rec.get('adopted') else ', not adopted (within the noise)')
        if st.startswith('inert'):
            row = next((x for x in reversed(self.inert) if x.get('candidate') == self.fb_patch_cid.get(pid)), {})
            return 'did not prove at its probe that it took effect' + (f': {str(row["reason"])[:200]}' if row.get('reason') else '')
        return st or None

    def _history(self, engine) -> list[dict]:
        """llmopt's history with S1's additions: each measured configuration best first (its changes, the patches it
        carried, its peak, its speed or how it failed - an out-of-memory shortfall, thrash at a probe by its peak -, a
        tested patch's result), then the probes, the failures, the proposals not run; at most `history_max` rows
        (HISTORY_MAX at ten screens, F1's rate beyond). v4.1 (`bounded_asks`): the rows kept whole within
        fold.HISTORY_CHARS - the fastest, the newest -, the others and every earlier proposal not run in a last summary row
        (search/fold.py)."""
        ok, probes, fails = [], [], []
        thrash = {t['candidate']: t for t in self.fb_thrash}
        for n, (cid, r) in enumerate(self.runs.items()):
            if cid == 'c0':
                continue
            cfg = r.get('cfg') or {}
            row = {'change': self._words_of(cfg)}
            pids = [k[len('patch:'):] for k, v in cfg.items() if isinstance(k, str) and k.startswith('patch:') and v]
            if pids:
                row['code_patches'] = [str((self.patches.get(x) or {}).get('mechanism') or x)[:80] for x in pids]
            m = self.mem.runs.get(cid) or {}
            if isinstance(m.get('peak'), (int, float)):
                row['peak_gib'] = round(float(m['peak']), 1)
            tested = self.fb_cid_patch.get(cid)
            if tested:
                res = self._fb_patch_result(tested)
                if res:
                    row['patch_result'] = res
            kind, st = r.get('kind'), r.get('status')
            y = self.short_y.get(cid) if st == 'short' else r.get('y')
            if st in ('screened', 'short') and y is not None and kind != 'thrash':
                row['outcome'] = f'x{math.exp(y):.3f} the reference speed' + (' (a short run)' if st == 'short' else '')
                ok.append({'row': row, 'group': 'measured', 'order': n, 'y': y, 'label': 'measured'})
                continue
            if kind == 'oom':
                o = m.get('oom') if isinstance(m.get('oom'), dict) else {}
                short = o.get('short_gib')
                row['outcome'] = 'out of memory' + (f', short by {short:.1f} GiB' if isinstance(short, (int, float)) else '') + (
                    ' (at its probe)' if r.get('fidelity') == 'probe' else '')
                label = 'out of memory'
            elif kind == 'thrash':
                t = thrash.get(cid)
                row['outcome'] = (f'thrash at its probe: its measured peak {t["peak_gib"]:.1f} GiB is {100 * t["share"]:.0f} % '
                                  'of the card' if t else 'thrash at the memory edge (slow near the card)')
                label = 'thrash at the memory edge'
            elif st == 'aborted':
                row['outcome'] = 'aborted: slower than the best so far' + (
                    f' (about x{math.exp(y):.3f} the reference speed)' if y is not None and not r.get('bound') else '')
                label = 'aborted (slower than the best)'
            elif st == 'probed':
                row['outcome'] = 'its probe ran (it fits; the speed was not measured)'
                probes.append({'row': row, 'group': 'probe', 'order': n, 'y': None, 'label': 'probed (fits)'})
                continue
            elif kind in ('start_slow', 'start_cap', 'hang'):
                row['outcome'] = 'a slow start: stopped at its start cap'
                label = 'a slow start'
            elif kind == 'slow_probe':
                row['outcome'] = 'its first updates ran much slower than the reference: not run in full'
                label = 'a slow probe'
            elif kind in ('inert_at_start', 'patch_inert'):
                row['outcome'] = 'its patch did not take effect'
                label = 'patch did not take effect'
            else:
                row['outcome'] = label = f'failed: {kind or st}'
            fails.append({'row': row, 'group': 'failed', 'order': n, 'y': None, 'label': label})
        entries = sorted(ok, key=lambda e: -e['y']) + probes + fails + [
            {'row': {'change': x['change'], 'outcome': f'not run: {x["why"]}'}, 'group': 'not_run', 'order': i, 'y': None,
             'label': 'not run'} for i, x in enumerate(self.rejected if self.bounded_asks else self.rejected[-10:])]
        if self.bounded_asks:                            # (the proposals not run: the last ten whole, as before)
            return fold_history(entries, max_rows=self.history_max, limits={'not_run': 10})
        return [e['row'] for e in entries][:self.history_max]

    # ------------------------------------------------------------ S4: memory as a price
    def _levels(self, cfg_m: dict) -> list:
        """wcs's _levels with the memoised value key (the same list; a draw over a ~25k-row table calls it for every row,
        and wcs's own json key made one draw take about a minute: hard rule 6, decisions in seconds)."""
        return [(k, vk(v)) for k, v in cfg_m.items()
                if k.startswith('patch:') or (k in self.knobs and vk(v) != vk(self.ref.get(k)))]

    def _fb_forget(self) -> None:
        self._fb_iso_stamp = self._fb_combo_stamp = None

    def _fb_measured_names(self) -> set:
        """The ledger's values and patches some observation holds (`k=<value key>`, `patch:<id>`): measured."""
        led = self.mledger
        if led is None:
            return set()
        if self._fb_meas_stamp != len(led.obs):
            names = set()
            for c, *_ in led.obs:
                for k in led.knobs:
                    v = vk(c.get(k, led.ref.get(k)))
                    if v != vk(led.ref.get(k)):
                        names.add(f'{k}={v}')
                names |= {k for k, x in c.items() if isinstance(k, str) and k.startswith('patch:') and x}
            self._fb_measured, self._fb_meas_stamp = names, len(led.obs)
        return self._fb_measured

    def isolated_blame(self) -> dict:
        """S4's isolated_blame: value -> the runs that blame it alone. A value off the reference (a knob value, a carried
        patch as patch:<id>) is blamed alone only when, in every run that failed for memory (out of memory, thrash, at any
        rung) holding it, every other value off the reference appeared in some run that fitted; a failed run whose values
        a fitted run held all of is explained by what that run added (its savers) and blames nothing (hard rule 5's probe
        still covers its whole combination). Built once per observation, with each blamed value's contexts - the other
        values of every fitted run that held it (`_fb_iso_ctx`): a candidate holding one whole is not priced down for it
        (`_fb_blamed`)."""
        if self._fb_iso_stamp == self.fb_obs:
            return self._fb_iso
        fitted, wins, failed = set(), [], []
        for cid, r in self.runs.items():
            vals = set(self._levels(r.get('cfg') or {}))
            if r.get('kind') in FITTED:
                fitted |= vals
                wins.append(vals)
            elif r.get('kind') in MEM_FAIL:
                failed.append((cid, vals))
        failed = [(cid, vals) for cid, vals in failed if not any(vals <= w for w in wins)]
        out = {}
        for v in sorted({x for _, vals in failed for x in vals}):
            holders = [(cid, vals) for cid, vals in failed if v in vals]
            if all(vals - {v} <= fitted for _, vals in holders):
                out[v] = [cid for cid, _ in holders]
        self._fb_iso_ctx = {v: [frozenset(w - {v}) for w in wins if v in w] for v in out}
        self._fb_iso, self._fb_iso_stamp = out, self.fb_obs
        return out

    def _fb_blamed(self, held) -> list:
        """The values of a candidate (its levels, in order) that isolated_blame prices down: blamed alone, and the
        candidate holds no whole context in which the value fitted."""
        iso = self.isolated_blame()
        have = set(held)
        return [v for v in held if v in iso and not any(c <= have for c in self._fb_iso_ctx.get(v, ()))]

    def oom_combinations(self) -> list[tuple]:
        """[(the values off the reference of a run that failed for memory - its knob values and patches -, the run)], the
        smallest first; a configuration holding one whole is probed before any full run (S4)."""
        if self._fb_combo_stamp == self.fb_obs:
            return self._fb_combos
        seen, out = set(), []
        for cid, r in self.runs.items():
            if r.get('kind') in MEM_FAIL:
                c = frozenset(self._levels(r.get('cfg') or {}))
                if c and c not in seen:
                    seen.add(c)
                    out.append((c, cid))
        out.sort(key=lambda t: (len(t[0]), sorted(t[0]), t[1]))
        self._fb_combos, self._fb_combo_stamp = out, self.fb_obs
        return out

    def _fb_combo(self, cfg: dict, ids=()) -> tuple | None:
        """(the run, the values) of a combination measured to fail for memory that the configuration holds whole."""
        combos = self.oom_combinations()
        if not combos:
            return None
        held = set(self._levels(self._m_of({k: cfg.get(k) for k in self.knobs}, ids)))
        for c, cid in combos:
            if c <= held:
                return cid, c
        return None

    @staticmethod
    def _fb_words(values) -> str:
        return ', '.join(f'{k}={v}' for k, v in sorted(values))

    def _fb_floor(self, kcfg: dict, lids: tuple, p: float) -> tuple[float, str]:
        """S4: declared GiB never push P below P_FLOOR_DECLARED: the configuration without its declared values (and
        patches) that no observation holds - what the measurements alone say - caps how far they lower it."""
        led = self.mledger
        measured = self._fb_measured_names()
        knobs_back, patches_out = [], []
        for b in led.basis:
            if b['kind'] not in ('value', 'patch') or not b.get('mu') or b['name'] in measured:
                continue
            if b['kind'] == 'value' and vk(kcfg.get(b['knob'])) == b['vk']:
                knobs_back.append(b['knob'])
            elif b['kind'] == 'patch' and b['pid'] in lids:
                patches_out.append(b['pid'])
        if not knobs_back and not patches_out:
            return p, ''
        cfg2 = {**kcfg, **{k: self.ref.get(k) for k in knobs_back}}
        p2 = led.p_feasible(cfg2, ids=tuple(x for x in lids if x not in patches_out))
        q = max(p, min(p2, P_FLOOR_DECLARED))
        if q <= p:
            return p, ''
        self.fb_counts['floor_reads'] += 1
        return q, (f'declared memory of {len(knobs_back) + len(patches_out)} unmeasured value(s) alone: at least '
                   f'{P_FLOOR_DECLARED}')

    def _fb_starts(self, cfg: dict, ids, p: float, why: str) -> tuple[float, str]:
        """The pair table and the values that failed to start alone (the candidates' rule; not memory)."""
        m = self._m_of(cfg, ids)
        present = {(k, vk(v)) for k, v in m.items() if k.startswith('patch:') or vk(v) != vk(self.ref.get(k))}
        for pair, rec in self.pairs.items():
            if pair <= present and rec['p'] < 1.0:
                p *= rec['p']
                why = (why + '; ' if why else '') + f'{self._pair_words(pair)}: {rec["why"]}'
        for key, n in self.alone.items():
            if key in present and n:
                p *= P_ALONE_FAILED if n == 1 else 0.0
                why = (why + '; ' if why else '') + f'{key[0]}={key[1]} failed to start alone {n}x'
        return p, why

    def _fb_bad(self, cfg: dict, ids) -> str | None:
        """llmopt's rule: a configuration holding every change of a run that failed for a reason other than memory."""
        bad = self._bad_sets()
        if not bad:
            return None
        have = {(k, vk(cfg.get(k))) for k in self.knobs if vk(cfg.get(k)) != vk(self.ref.get(k))}
        have |= {(f'patch:{x}', 'true') for x in ids}
        for s, cid, kind in bad:
            if s <= have:
                return f'it holds every change of {cid}, which failed ({kind})'
        return None

    def _pf(self, cfg: dict, ids=()) -> tuple[float, str]:
        """S4's feasibility: the memory ledger alone (a run's own result for its whole configuration; else P(peak <= the
        card less 2 %), the declared-memory floor, isolated_blame), the pair table and llmopt's failed-change rule (start
        failures, not memory). A configuration holding a combination measured to fail for memory keeps the ledger's P (its
        probe is the plan's business, `_fb_probe_plan`) and says so; during a redraw outside the combinations it is 0.
        Without the ledger (a strategy's ablation): wcs's feasibility instead of the ledger's part."""
        kcfg = {k: cfg.get(k) for k in self.knobs}
        if self.mledger is None:
            p, why = WCS._p_feasible(self, cfg, ids)
        elif not self.on['wcs_feasibility']:
            p, why = 1.0, ''
        else:
            led = self.mledger
            lids = self._led_ids(ids)
            own = led.ran_status(kcfg, ids=lids)
            if own in MEM_FAIL:
                return 0.0, 'it ran out of memory' if own == 'oom' else 'it thrashed at the memory edge'
            if own == 'ok':
                return 1.0, 'it ran and fitted'
            p, why = led.p_feasible(kcfg, ids=lids), ''
            if led.cap:
                mu, sd = led.predict(kcfg, ids=lids)
                why = f'the ledger: {mu:.1f} +- {sd:.1f} GiB per GPU, the line {led.margin * led.cap:.1f} of {led.cap:.0f}'
            if p < P_FLOOR_DECLARED:
                p, note = self._fb_floor(kcfg, lids, p)
                if note:
                    why += f'; {note}'
            if self.fb_on['fb_isolated_blame']:
                blamed = self._fb_blamed(self._levels(self._m_of(kcfg, lids)))
                if blamed:
                    iso = self.isolated_blame()
                    p *= BLOCKED_VALUE ** len(blamed)
                    why += '; ' + ', '.join(f'{k}={v} failed alone in {", ".join(iso[(k, v)][:3])}' for k, v in blamed[:2])
        if p > 0.0 and self.mledger is not None and self.on['wcs_antagonism']:
            p, why = self._fb_starts(cfg, ids, p, why)
        if p <= 0.0:
            return p, why
        bad = self._fb_bad(cfg, ids)
        if bad:
            return 0.0, bad
        hit = self._fb_combo(kcfg, ids)
        if hit is not None:
            note = f'it holds {self._fb_words(hit[1])} as {hit[0]} did, which failed for memory: a probe before any full run'
            if self._fb_combo_out:
                return 0.0, note
            why = (why + '; ' if why else '') + note
        return p, why

    def _fb_probe_plan(self, engine, cfg: dict, ids, plan, *, count: bool = True) -> tuple | None:
        """The brief's hard rule 5: the plan of a configuration that holds a combination measured to fail for memory starts
        with a probe - the given one when it does, else the ladder's with the probe forced (within the allowance); None
        when the allowance has no probe for it: it is not run in full without one. Any other configuration: its plan."""
        plan = tuple(plan or ('full',))
        if self._fb_combo(cfg, ids) is None:
            return plan
        if plan[0] != 'probe':
            plan = self.ladder.plan(engine, 0.0, warmup=self._warmup(cfg) or None, probe=True, short=False)
        if plan[0] != 'probe':
            self.fb_refused = True
            self.fb_counts['combo_refused'] += 1
            return None
        if count:
            self.fb_counts['combo_probed'] += 1
        return plan

    def _p_feasible(self, cfg: dict, ids=()) -> tuple[float, str]:
        """wcs's own steps (a finalist that borrows one) read S4's feasibility too."""
        return self._pf(cfg, ids)

    def price_tags(self) -> dict:
        """S4's price list: (knob, value key) or (patch:<id>, 'true') -> {'gib': the ledger's fitted GiB (+ spends, - frees),
        'source': 'measured' (an observation holds it), 'declared' (its declaration alone) or 'unknown'}."""
        led = self.mledger
        if led is None:
            return {}
        try:
            theta = led.fit()
        except Exception:                                # noqa: BLE001 - a fit that fails: no prices
            return {}
        measured = self._fb_measured_names()
        out = {}
        for b, t in zip(led.basis, theta):
            if b['kind'] == 'value':
                key = (b['knob'], b['vk'])
            elif b['kind'] == 'patch':
                key = (f'patch:{b["pid"]}', 'true')
            else:
                continue
            out[key] = {'gib': round(t, 3), 'source': 'measured' if b['name'] in measured else 'declared' if b.get('mu')
                        else 'unknown'}
        return out

    def failed_material(self) -> list[dict]:
        """The runs that failed for memory, the latest first - what a completion, a rescue or a repair derives from (S3,
        S5): the run, its class and rung, its knob values, the patches it carried, the out-of-memory shortfall, its peak and
        the gap to close (out of memory: the shortfall plus the card's reserve; thrash: its peak - or the card - less the
        cliff's lower edge, THRASH_LO of the card)."""
        out = []
        cap = self.mem.cap
        probe_thrash = {t['candidate'] for t in self.fb_thrash}
        for cid in reversed(list(self.runs)):
            r = self.runs[cid]
            if r.get('kind') not in MEM_FAIL:
                continue
            mr = self.mem.runs.get(cid) or {}
            oom = mr.get('oom') if isinstance(mr.get('oom'), dict) else {}
            peak = mr.get('peak') if isinstance(mr.get('peak'), (int, float)) else None
            short = oom.get('short_gib') if isinstance(oom.get('short_gib'), (int, float)) else None
            if r['kind'] == 'oom':
                gap = short + CARD_RESERVE * cap if short is not None and cap else None
            else:
                gap = max(0.0, (peak if peak is not None else cap) - THRASH_LO * cap) if cap else None
            out.append({'run': cid, 'kind': r['kind'], 'fidelity': r.get('fidelity'), 'at_probe': r.get('fidelity') == 'probe',
                        'cfg': {k: (r.get('cfg') or {}).get(k, self.ref.get(k)) for k in self.knobs},
                        'pids': sorted(self.pids.get(cid) or ()), 'short_gib': short, 'peak_gib': peak, 'gap_gib': gap,
                        'origin': r.get('origin'), 'probe_peak': cid in probe_thrash})
        return out

    # ------------------------------------------------------------ S4: no full run of a combination member without a probe
    def _knob_q(self, engine, inc, cfg: dict, *, kind: str, plan: tuple, p: float | None = None, why: str = '',
                origin: str | None = None, s6_exempt: bool = False):
        """The candidates' proposal of a whole configuration; None when S6's gate refuses it (two knob-side runs in a row
        out of memory and its P(fits) below 0.9; `s6_exempt`: the uniform start), or when it holds a combination measured
        to fail for memory and the allowance has no probe for it (`_fb_probe_plan`: never a full run without one). v4: None
        when F1's floor refuses a held kind (the screen the code track's), or F8 holds a contrarian move with no probe."""
        if self._fb_floor_why(engine, inc, cfg, tuple(self.chain), p, kind):
            return None
        if not s6_exempt and not self._fb_s6_ok(cfg, tuple(self.chain)):
            return None
        plan = self._fb_probe_plan(engine, cfg, tuple(self.chain), plan)
        if plan is None:
            return None
        if kind == 'contrarian' and self._v4('v4_store', engine):
            plan = self._fb_store_plan(engine, inc, cfg, plan)
            if plan is None:
                return None
        q = super()._knob_q(engine, inc, cfg, kind=kind, plan=plan, p=p, why=why, origin=origin)
        if q is not None and tuple(q.fidelity)[0] == 'probe' and q.gate is None:
            q.gate = self._fb_gated(None)                # (a probe wcs built no gate for: S5's peak rule alone)
        return q

    def _run_item(self, engine, inc, item: dict, *, kind: str, plan: tuple | None = None, probe_ok: bool = True):
        """llmopt's proposal for an archive item, its plan by `_fb_probe_plan`; an item it refuses is closed and shown to
        the model as not run (None); one S6's gate holds back stays open for later (None)."""
        if self._fb_floor_why(engine, inc, item['cfg'], tuple(self.chain), item['p'], kind):
            return None                                  # (v4 F1: it stays open; the screen goes to the code track)
        if not self._fb_s6_ok(item['cfg'], tuple(self.chain)):
            return None
        if plan is None:
            plan = self.ladder.plan(engine, item['p'], warmup=self._warmup(item['cfg']) or None,
                                    probe=None if probe_ok else False, short=False)
        plan = self._fb_probe_plan(engine, item['cfg'], tuple(self.chain), plan, count=False)
        if plan is None:
            item['status'] = 'refused'
            self.rejected.append({'change': self._words_of(item['cfg']), 'source': item['source'],
                                  'why': 'it holds a combination measured to fail for memory and no probe is left'})
            return None
        if kind == 'contrarian' and self._v4('v4_store', engine):
            plan = self._fb_store_plan(engine, inc, item['cfg'], plan)
            if plan is None:
                return None                              # (v4 F8: no probe for it now - it stays open, eligible)
        return super()._run_item(engine, inc, item, kind=kind, plan=plan, probe_ok=probe_ok)

    def _tail_step(self, engine, inc, left):
        """llmopt's tail; a bundle refused by the combination rule ends the tail, one S6's gate holds back waits."""
        self.fb_refused = self.fb_s6_refused = False
        floored = self.fb_floor_refused
        self.fb_floor_refused = False
        q = super()._tail_step(engine, inc, left)
        if q is None and self.fb_floor_refused and self.tail_base is not None:
            self.tail_base = (self.tail_base[0], self.tail_base[1], 'ready')   # (v4 F1: offered again when the floor lifts)
        elif q is None and self.fb_s6_refused and self.tail_base is not None:
            self.tail_base = (self.tail_base[0], self.tail_base[1], 'ready')   # (tried again once the streak ends)
        elif q is None and self.fb_refused:
            self.tail.done = True
        self.fb_floor_refused = self.fb_floor_refused or floored   # (the step's flag: an earlier refusal kept)
        return q

    def _fallback(self, engine, inc, short: bool = False):
        """llmopt's draw (the same draw, rpp's weighted_draw, the same seed); a draw refused by the combination rule is
        drawn again outside the combinations. Under S6 only a configuration of P(fits) >= 0.9 is drawn; when the table
        has none, the code track's pending or newly asked patch takes the screen, else nothing is proposed (no knob
        candidate below 0.9 after two knob-side runs out of memory; SHARED_r1 P1-3)."""
        self.fb_refused = False
        q = self._fb_draw(engine, inc, short)
        if q is None and self.fb_refused:
            self._fb_combo_out = True
            try:
                q = self._fb_draw(engine, inc, short)
            finally:
                self._fb_combo_out = False
        if q is None and self.s6_active() and self.on['wcs_code']:
            q = self._code_step(engine, inc, self._left(engine))   # (the screen goes to the code track)
        return q

    def _fb_draw(self, engine, inc, short: bool):
        p_min = None if self.ladder.affordable(engine, 'probe') else self.ladder.probe_below
        s6 = self.s6_active()
        if s6:
            p_min = max(p_min or 0.0, S6_P)
        got = _llmopt.weighted_draw(self, engine, inc, a=1.0, T=0.02,
                                    salt=f'{self.name}|{len(self.rounds)}|{engine.full_screens()}', p_min=p_min)
        if got is None:
            return None
        cfg, ids, p, why, info = got
        if s6 and p < S6_P:                              # (the draw's p_min holds only when some row reaches it: none
            self.fb_s6_refused = True                    #  does - no eligible knob candidate)
            self.fb_counts['s6_no_knob'] += 1
            return None
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None, short=short)
        return self._knob_q(engine, inc, cfg, kind='draw', plan=plan, p=p, why=why)

    # ------------------------------------------------------------ S3: candidates derived from a failed run
    def derived_ids(self, cid: str) -> tuple:
        """S3: the patch ids a candidate derived from run `cid` carries - that run's own with the adopted chain."""
        return tuple(sorted(set(self.pids.get(cid) or ()) | set(self.chain)))

    def _fb_derived_files(self, engine, inc, cid: str, ids: tuple):
        """The files of a derived candidate: the chain's when the run carried no other patch; else the run's own files
        (its overlay over its parent) merged onto what is carried now (wcs's _rebase); _GONE when they no longer apply."""
        if not set(ids) - set(self.chain):
            return dict(self.overlay_best) if ids and self.overlay_best else None
        c = engine.s.candidates.get(cid)
        if c is None or not getattr(c, 'overlay', None):
            return _GONE
        base = c.parent if getattr(c, 'parent', None) in engine.s.candidates else 'c0'
        try:
            merged = self._rebase(engine, inc, dict(c.overlay), base)
        except Exception:                                # noqa: BLE001 - a merge that fails: the patches do not apply
            merged = None
        return merged if merged else _GONE

    def derived_q(self, engine, inc, cfg: dict, cid: str, *, kind: str, plan: tuple | None = None,
                  probe: bool | None = None, p: float | None = None, why: str = '', origin: str | None = None):
        """S3: a proposal of a whole configuration derived from the failed run `cid` (a completion, a rescue, a repair)
        that carries that run's patches with the adopted chain, their files rebased onto what is carried now; the ladder's
        plan (probe first below 0.7, `probe` forces or forbids it) unless given; a combination member only with a probe.
        None when it is illegal, ran already, is ruled out, its patches no longer apply or its plan lacks a probe it
        needs."""
        cfg = {k: cfg.get(k, self.ref.get(k)) for k in self.knobs}
        if cid not in self.runs or cid == 'c0':
            self.fb_counts['derived_dropped'] += 1       # (derived from no run of this search: nothing to carry)
            return None
        ids = self.derived_ids(cid)
        if not self._legal(engine, cfg) or self._ckey(cfg, ids) in self.tried:
            self.fb_counts['derived_dropped'] += 1
            return None
        if not self._fb_s6_ok(cfg, ids):                 # (S6: a knob-side candidate like any other)
            return None
        files = self._fb_derived_files(engine, inc, cid, ids)
        if files is _GONE:
            self.fb_counts['derived_dropped'] += 1
            self.fb_notes.append(f'a candidate derived from {cid} dropped: its patches no longer apply on {inc}')
            return None
        if p is None:
            p, w = self._pf(cfg, ids)
            why = (why + '; ' if why and w else why) + (w or '')
        if p <= 0.0:
            self.fb_counts['derived_dropped'] += 1
            return None
        unproven = [x for x in ids if x not in self.chain]   # (a carried patch not adopted: its proof is still owed here)
        code = bool(unproven) and self.ladder.code_probes_left() >= 1   # f1.1: that probe is a code proof's - the pool's
        if code:                                         # while it lasts; beyond it the knob side's, as in f1.0
            plan = tuple(plan) if plan is not None and tuple(plan)[0] == 'probe' else ('probe', 'full')
        elif plan is None:
            plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None, probe=True if unproven else probe,
                                    short=False)
        elif unproven and tuple(plan)[0] != 'probe':
            plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None, probe=True, short=False)
        if unproven and tuple(plan)[0] != 'probe':
            self.fb_counts['derived_dropped'] += 1       # (no affordable proof probe: never a full run of an unproven patch)
            return None
        plan = self._fb_probe_plan(engine, cfg, ids, plan)
        if plan is None:
            return None
        mu, sigma = self._mu(engine, inc, cfg, ids)
        q = self._proposal(engine, inc, cfg, ids, mu=mu, sigma=sigma, p=p, why=why, probe=tuple(plan)[0] == 'probe',
                           origin=origin or f'{self.name}-{kind}', kind=kind)
        q.patch = dict(files) if files else None
        q.fidelity = tuple(plan)
        if tuple(plan)[0] != 'probe':
            q.gate, q.probe_cap, q.probe_steps, q.keep_probe = None, None, None, False
        else:
            q.keep_probe = True
            if q.gate is None:
                q.gate = self._fb_gated(None)
            if unproven:                                 # SHARED_r1 P1-1: the patch's self-proof, after the peak and
                self._fb_proof_probe(engine, inc, q, files, unproven[0])   # speed gates, before any full run
        self.ledger[q.hypothesis].update(plan=list(plan), derived_from=cid, patches=list(ids),
                                         **({'proof_owed': unproven} if unproven else {}),
                                         **({'probe_pool': 'code'} if code else {}))
        if code:
            self.fb_code_hids.add(q.hypothesis)
        self.fb_counts['derived'] += 1
        return q

    def _fb_chain_carrier(self, engine) -> str:
        """The candidate whose run carried the adopted chain alone (its proof lines are what a new patch inherits): the
        run that adopted the last adopted patch (adopted_via: a completion that merged it with the chain), else that
        patch's own candidate, else the reference. (SHARED_r2: a patch's first candidate can be a run that carried it
        alone; the rest of the chain's proof lines would then look new to a later patch's self-proof.)"""
        rec = (self.patches.get(self.chain[-1]) or {}) if self.chain else {}
        cid = rec.get('adopted_via') or rec.get('candidate')
        return cid if cid in engine.s.candidates else 'c0'

    def _fb_proof_probe(self, engine, inc, q, files: dict | None, pid: str) -> None:
        """A derived candidate's probe proves the unadopted patch it carries (wcs's _patch_gate's self-proof, search/
        proof.check, against the chain's carrier: only the new patch's lines count): its gate is the peak and speed gates,
        then that check (failing it: `patch_inert`, no full run); its updates as long as the patch's declared first run."""
        rec = self.patches.get(pid) or {}
        h = {'patch': files, 'site': rec.get('site'), 'evidence_note': rec.get('evidence_note'),
             'active_from_update': rec.get('active_from_update'), 'mechanism': rec.get('mechanism')}
        a = self._supported(h, 'active_from_update')
        steps = min(PROBE_STEPS_MAX, max(GATED_STEPS, a or 0))
        if steps > max(int(q.probe_steps or 0), int(getattr(engine, 'probe_steps', 2) or 2)):
            q.probe_steps, q.probe_cap = steps, self._probe_cap(engine, inc, steps)
        self._fb_slow_start(engine, q, {**h, 'warmup_updates': rec.get('warmup_updates')})   # (v4 F7, r2: its warm-up too)
        q.gate = ProofGate(self, q.gate, self._fb_chain_carrier(engine), files, h)

    def _rescue(self, engine, inc):
        """wcs's edge rescue with S3's patches and S5's probe thrash: a run at the memory edge - thrash at a probe by its
        peak or with fast updates between stalls, or out of memory by a sliver (RESCUE_SHORT of the card) - plus the
        enabler the prior names for a value it holds, else the largest declared freeing value it lacks (a conflicting value
        back to the reference's, the coupled knobs completed), probed first, carrying the run's patches with the chain; once
        per (run, value), RESCUE_MAX per search."""
        if not self.on['wcs_feasibility'] or self.counts['rescue_probes'] >= RESCUE_MAX or not self.prior:
            return None
        cap = self.mem.cap or 0.0
        for m in self.failed_material():
            cid = m['run']
            if m['kind'] == 'thrash':
                if not (m['probe_peak'] or self._fast_updates(engine, cid, 'c0')):
                    continue
                edge = 'thrashed at the memory edge' + (' at its probe' if m['probe_peak'] else '')
            elif isinstance(m['short_gib'], (int, float)) and cap and m['short_gib'] <= RESCUE_SHORT * cap:
                edge = f'ran out of memory by {m["short_gib"]:.2f} GiB'
            else:
                continue
            cfg = dict(m['cfg'])
            held = set(self._levels(cfg))
            options = []
            for pr in self.prior.pairs:
                members = set(pr['members'])
                missing = members - held
                if pr['kind'] == 'enables' and len(missing) == 1 and members & held:
                    (k, key), = missing
                    options.append((2.0, k, key, 'the prior names it the enabler of a value it holds'))
            for (k, key), e in self.prior.levels.items():
                if e.get('mem_dir') == 'frees' and (k, key) not in held and vk(cfg.get(k)) != key:
                    options.append((1.0 + (e.get('mem_gib') or 0.0) / 1000.0, k, key, 'the largest declared freeing value it lacks'))
            options = sorted((o for o in options if (cid, o[1], o[2]) not in self.rescue_done), key=lambda t: -t[0])
            for _, k, key, note in options:
                v = next((x for x in self.knobs[k] if vk(x) == key), None)
                if v is None:
                    continue
                new = {**cfg, k: v}
                held2 = set(self._levels(new))
                for pair, rec in self.pairs.items():        # a held value that conflicts with the enabler goes back
                    if rec['p'] < 0.5 and pair <= held2:
                        for kk, kv in sorted(pair - {(k, key)}):
                            if (kk, kv) in held2 and not kk.startswith('patch:'):
                                new[kk] = self.ref.get(kk)
                                held2.discard((kk, kv))
                                break
                legal = self._complete(engine, new, {k})
                self.rescue_done.add((cid, k, key))
                if legal is None or vk(legal.get(k)) != key:
                    continue
                q = self.derived_q(engine, inc, legal, cid, kind='rescue', probe=True, why=f'{cid} {edge}; {k}={v}: {note}')
                if q is not None:
                    self.counts['rescue_probes'] += 1
                    return q
        return None

    # ------------------------------------------------------------ S5: thrash at a probe by its peak
    def _fb_peak_gate(self, engine, cid, row) -> tuple | None:
        """S5: the probe's measured peak (its detailed row's, else the candidate's probe result - where the session keeps
        it) at THRASH_PEAK of the card or more: (why, 'probe_thrash')."""
        cap = self.mem.cap
        peak = (row.get('memory_peak_measured_gib') or row.get('memory_peak_gib')) if isinstance(row, dict) else None
        if not isinstance(peak, (int, float)) or isinstance(peak, bool):
            peak = self._fb_probe_peak(engine, cid)
        if not cap or not isinstance(peak, (int, float)) or isinstance(peak, bool) or peak < THRASH_PEAK * cap:
            return None
        return (f'thrash at its probe: its measured peak {peak:.1f} GiB is {100 * peak / cap:.0f} % of the {cap:.0f} GiB '
                f'card ({100 * THRASH_PEAK:.0f} % or more): no full run', 'probe_thrash')

    def _fb_gated(self, gate):
        """A probe's gate with S5's peak rule first."""
        if not self.fb_on['fb_peak_thrash']:
            return gate
        return PeakGate(self, gate)

    def _proposal(self, engine, inc, cfg, ids, *, mu, sigma, p, why, probe, origin, kind):
        q = super()._proposal(engine, inc, cfg, ids, mu=mu, sigma=sigma, p=p, why=why, probe=probe, origin=origin,
                              kind=kind)
        if q.gate is not None:
            q.gate = self._fb_gated(q.gate)
        return q

    def _patch_proposal(self, engine, inc, h: dict):
        """wcs's patch proposal, its gate with S5's peak rule first, and its patch ids the base's as well (SHARED_r1 P1-2:
        written against the incumbent's files, it carries whatever patches they hold)."""
        q = super()._patch_proposal(engine, inc, h)
        if q is not None:
            q.gate = self._fb_gated(q.gate)
            pid = q.hypothesis.split('|', 1)[1]
            ids = frozenset(self.chain) | frozenset(self._pids(engine, inc)) | {pid}
            if ids != self.ideas.get(q.hypothesis):
                self.ideas[q.hypothesis] = ids
                self.ledger[q.hypothesis]['patches'] = sorted(ids)
            self.patches[pid]['warmup_updates'] = h.get('warmup_updates')   # (v4 r2: for a derived probe's F7)
            self._fb_slow_start(engine, q, h)            # (v4 F7)
        return q

    def _fb_adopt_carried(self, engine, out) -> None:
        """SHARED_r1 P1-2: a knob-side candidate that became the incumbent carrying patches the chain does not hold (a
        completion carrying a patch that ran out of memory) brings them into the chain, its overlay the carried files: the
        later patch proposals (written against its files) and configuration proposals (the chain and its overlay) carry
        the same patch ids as files, so hard rule 5 and the ledger see the true overlay."""
        extra = sorted(set(self.pids.get(out.cid) or ()) - set(self.chain))
        c = engine.s.candidates.get(out.cid)
        if not extra or c is None:
            return
        self.chain.extend(extra)
        self.overlay_best = dict(getattr(c, 'overlay', None) or {})
        for pid in extra:
            rec = self.patches.setdefault(pid, {})
            rec.update(adopted=True, adopted_via=out.cid)
            rec.setdefault('candidate', out.cid)
        self.fb_notes.append(f'{", ".join(extra)} adopted with {out.cid} (the incumbent carries them)')

    def _fb_probe_peak(self, engine, cid) -> float | None:
        c = engine.s.candidates.get(cid) if cid else None
        row = ((getattr(c, 'results', None) or {}).get('probe') or {}) if c is not None else {}
        peak = row.get('memory_peak_measured_gib') or row.get('memory_peak_gib')
        return float(peak) if isinstance(peak, (int, float)) and not isinstance(peak, bool) and peak > 0 else None

    def classify(self, engine, p, out) -> dict:
        """wcs's class of an outcome, with S5: a probe stopped by the peak rule, and a probe-only run whose measured peak
        reaches THRASH_PEAK of the card, are thrash."""
        cls = super().classify(engine, p, out)
        if not self.fb_on['fb_peak_thrash'] or not out.cid or not self.mem.cap:
            return cls
        stopped = out.status == 'infeasible' and 'probe_thrash' in (out.failed_checks or [])
        if stopped or out.status == 'probed' and out.fidelity == 'probe':
            peak = self._fb_probe_peak(engine, out.cid)
            if stopped or peak is not None and peak >= THRASH_PEAK * self.mem.cap:
                return {**{k: v for k, v in (cls or {}).items() if k not in ('kind', 'line')}, 'kind': 'thrash',
                        'at_start': True, 'probe_peak': True, **({'peak_gib': peak} if peak is not None else {}),
                        'line': f'thrash at its probe: measured peak {peak or 0:.1f} GiB of {self.mem.cap:.0f}'}
        return cls

    # ------------------------------------------------------------ S6: no out-of-memory streaks
    def s6_active(self) -> bool:
        """S6: the last S6_STREAK knob-side runs (probes and full runs) all ran out of memory."""
        return self.fb_on['fb_s6'] and self.fb_oom_streak >= S6_STREAK

    def _fb_s6_ok(self, cfg: dict, ids) -> bool:
        """S6's gate for a knob candidate: True unless the streak is on and its P(fits) (S4's, the ledger's) is below
        S6_P - then refused and counted (the screen is the code track's: `_fb_ask_due`)."""
        if not self.s6_active():
            return True
        if self._pf(cfg, ids)[0] >= S6_P:
            return True
        self.fb_s6_refused = True
        self.fb_counts['s6_refused'] += 1
        return False

    @staticmethod
    def _fb_mech_key(text) -> str:
        """A patch mechanism for S6's no-retry rule: its words, without the review / fix / port tags and digits."""
        t = str(text or '').lower()
        t = re.sub(r'^\s*review: [^:]{1,80}: ', '', t)
        t = re.sub(r'\[(?:fix|port)\s+[^\]]*\]', ' ', t)
        return ' '.join(re.findall(r'[a-z_]+', re.sub(r'\d+', ' ', t)))

    @staticmethod
    def _fb_files_key(files) -> str:
        return json.dumps(sorted((str(k), str(v)) for k, v in (files or {}).items()))

    def _fb_s6_banned(self, mechanism, files=None) -> dict | None:
        """The out-of-memory patch whose mechanism (or files) this one repeats (S6: never retried), else None."""
        if not self.fb_on['fb_s6'] or not self.fb_s6_bans:
            return None
        key, fk = self._fb_mech_key(mechanism), self._fb_files_key(files) if files else None
        return next((b for b in self.fb_s6_bans if key and key == b['key'] or fk and fk == b['files']), None)

    def uniform_start_q(self, engine, inc, cfg: dict, *, why: str = 'a uniform draw of the legal table (the control start)'):
        """S6.4: the uniform-draw start runs in full, as random's draws do - S6's gate does not hold it back; its result
        enters the ledger and the streak as any knob-side run's. (Hard rule 5 still probes a configuration that holds a
        combination measured to fail for memory.)"""
        return self._knob_q(engine, inc, cfg, kind='uniform', plan=('full',), p=None, why=why, s6_exempt=True)

    # ------------------------------------------------------------ outcomes
    def _fb_code_pool(self, engine, p, out) -> None:
        """f1.1: a code proof's probe that ran - a patch's (its self-proof or a fix_inert repair's), or a derived
        candidate's planned on the pool - noted with the GPU-hours of the probe rows its evaluation appended (every
        engine's rows; the session's carry no other mark of whose they are): the pool's while it has one left, else a
        patch's the code share's (`note_code_probe`)."""
        rows = engine.s.screens
        new, self._fb_rows_seen = rows[self._fb_rows_seen:], len(rows)
        hid = p.hypothesis or ''
        derived = hid in self.fb_code_hids
        if not out.cid or not (hid.startswith('patch|') or derived):
            return
        probes = [r for r in new if r.get('fidelity') == 'probe']
        if not probes:
            return
        if derived and self.ladder.code_probes_left() >= 1:
            self.ladder.code_cids.add(out.cid)           # (its runs are the code proofs', as a patch's are)
        self.ladder.note_code_probe(out.cid, hid, sum(r.get('gpu_hours') or 0.0 for r in probes),
                                    'derived' if derived else 'patch')

    def observe(self, engine, p, out):
        self._fb_code_pool(engine, p, out)
        self.fb_obs += 1
        self._fb_forget()
        cls = super().observe(engine, p, out)
        self.fb_obs += 1                                 # (the runs changed: the caches go again)
        self._fb_forget()
        hid = p.hypothesis or ''
        ran = bool(out.cid) and out.status not in ('duplicate', 'infra', 'rejected')
        kind = (cls or {}).get('kind')
        if ran:
            self.fb_runs += 1
        if hid.startswith('patch|'):                     # v4: a refused / failed patch waits for its fix call
            self._fb_fix_note(engine, p, out, cls)
        self._fb_events_seen = len(getattr(engine.s, 'events', None) or ())
        if hid.startswith('stack|'):                     # v4 F2: a stack test (not the knob side's: no S6 streak)
            self._fb_stack_seen(engine, p, out, kind)
            return cls
        if ran and not hid.startswith('patch|'):        # S6: the knob side's streak (a probe's result counts)
            self.fb_oom_streak = self.fb_oom_streak + 1 if kind == 'oom' else 0
            self.fb_oom_streak_max = max(self.fb_oom_streak_max, self.fb_oom_streak)
            if out.status == 'screened' and engine.incumbent() == out.cid:
                self._fb_adopt_carried(engine, out)
        if hid.startswith('patch|') and out.cid:
            pid = hid.split('|', 1)[1]
            self.fb_patch_cid[pid] = out.cid
            self.fb_cid_patch[out.cid] = pid
            if kind == 'inert_at_start':
                self.fb_inert_log.append({'pid': pid, 'cid': out.cid, 'base': p.base, 'shown': False,
                                          'screens_done': engine.full_screens()})
            if kind == 'oom' and ran:                    # S6: that mechanism is never retried as a patch
                rec = self.patches.get(pid) or {}
                self.fb_s6_bans.append({'pid': pid, 'key': self._fb_mech_key(rec.get('mechanism') or p.mechanism),
                                        'files': self._fb_files_key(rec.get('files') or p.patch),
                                        'mechanism': str(rec.get('mechanism') or p.mechanism)[:120]})
        if (cls or {}).get('probe_peak') and out.cid and not any(t['candidate'] == out.cid for t in self.fb_thrash):
            peak = cls.get('peak_gib')
            if self.ledger.get(hid):
                self.ledger[hid]['status'] = 'thrash at its probe'
            self.fb_thrash.append({'candidate': out.cid, 'id': hid, 'peak_gib': peak, 'cap_gib': self.mem.cap,
                                   'share': round(peak / self.mem.cap, 3) if peak and self.mem.cap else None,
                                   'patches': sorted(self.pids.get(out.cid) or ()), 'plan': list(p.fidelity or ())})
        return cls

    # ------------------------------------------------------------ S2: the write-back, the report
    def _fb_proof_of(self, cid: str | None) -> tuple[str | None, dict | None]:
        """(the self-proof's site name, {'found', 'set'}) of the site a patch's probe installed that its base's run did
        not, from the probe's detailed row; (None, None) without one."""
        eng = self.engine
        c = eng.s.candidates.get(cid) if eng is not None and cid else None
        row = ((getattr(c, 'results', None) or {}).get('probe') or {}) if c is not None else {}
        if not row:
            return None, None
        try:
            base = getattr(c, 'parent', None) or 'c0'
            got = parsed(proof_of(row, True), proof_of(events_of(eng, base), True))
        except Exception:                                # noqa: BLE001 - lines of an unexpected shape: no proof
            return None, None
        for site, r in got.items():
            if r.get('found') is not None and r.get('set') is not None:
                return site, {'found': one_line(r['found'], PROOF_MAX), 'set': one_line(r['set'], PROOF_MAX)}
        return None, None

    def patch_section(self) -> dict:
        """S2's write-back: this arm's patches as build_transfer_brief reads an arm's records - proven (a measured gain the
        search adopted), inert (its self-proof failed, it was not activated or deferred to the original, its full run
        measured no gain below 1 % or was slower, it failed at its run) - each with its mechanism, canonical site, the
        site's own words, its gain, the inert reason and the scanner's lines (the proof's found / set; for an inert one
        its printed lines and what the scanner read); never run, inconclusive (a larger gain not adopted) and memory
        failures (the card decides them) left out. {} when there is none."""
        proven, inert = [], []
        for pid in sorted(self.patches, key=lambda x: (len(x), x)):
            rec = self.patches[pid]
            st = str(rec.get('status') or '')
            if not st or st in UNTESTED or st in MEM_FAIL:
                continue
            cid = rec.get('candidate') or self.fb_patch_cid.get(pid)
            key, proof = self._fb_proof_of(cid)
            mech = rec.get('mechanism')
            base = {'mechanism': one_line(mech, MECH_MAX),
                    'site': site_of(key, rec.get('site'), mech, (proof or {}).get('found')),
                    'site_text': one_line(rec.get('site'), SITE_MAX) or None, 'proof': proof}
            g = _gain(rec.get('gain_pct'))
            if st == 'proven':
                if rec.get('adopted') and g is not None and g > 0:
                    proven.append({**base, 'gain_pct': g})
                elif g is None or g < INCONCLUSIVE:
                    inert.append({**base, 'gain_pct': g, 'reason': measured_reason(g)})
                continue
            if st.startswith('inert'):
                row = next((x for x in reversed(self.inert) if x.get('candidate') == cid and cid), None) or \
                    next((x for x in reversed(self.inert) if x.get('mechanism') == rec.get('mechanism')), {})
                inert.append({**base, 'gain_pct': None, 'reason': inert_reason(row.get('reason'), 'inert_at_start'),
                              'printed': [one_line(x, LINE_MAX) for x in row.get('printed') or []][:6] or None,
                              'parsed': row.get('parsed') or None})
                continue
            if st in ('slower', 'aborted'):
                inert.append({**base, 'gain_pct': g, 'reason': measured_reason(g, slower=True)})
                continue
            inert.append({**base, 'gain_pct': g, 'reason': inert_reason(None, st)})
        clean = lambda rows: [{k: v for k, v in r.items() if v is not None} for r in rows]   # noqa: E731
        return {k: v for k, v in (('proven', clean(proven)), ('inert', clean(inert))) if v}

    def finalist_view(self) -> dict:
        """The shared parts' account for a finalist's report: the asks, the reserve, the transfer, S4's blame and
        combinations, S5's probes, the derived candidates."""
        led = self.mledger
        lad = self.ladder
        r = lad.probe_ratio()[0]
        return {'base': FB_VERSION,
                'caps': {'rate_screens': self.fb_rate_screens, 'calls': self.calls_max, 'usd': self.usd_max,   # (F1)
                         'asks': self.code_asks, 'history_rows': self.history_max, 'code_probes': lad.code_probes_max},
                'probe_accounting': {'code_pool': lad.pool, 'code_probes_used': len(lad.code_log),
                                     'code_probes_max': lad.code_probes_max, 'code_share_probes': len(lad.code_share_log),
                                     'probe_below': lad.probe_below, 'probe_ratio': None if r is None else round(r, 3)},
                'parts_off': sorted(k for k, v in self.fb_on.items() if not v), 'asks': self.fb_asks[-24:],
                'asks_made': self.asks, 'protected_screens': sorted(self.protected_screens or ()),
                'held_back_by': self.fb_strong, 'code_idle': self.fb_code_idle, 'counts': dict(self.fb_counts),
                'ledger_line': None if led is None or not led.cap else round(led.margin * led.cap, 2),
                'transfer': {'proven': len(self.fb_proven), 'inert': len(self.fb_inert), 'from': dict(self.fb_sources),
                             'target_env': _shown_env(self.fb_target or {}) if self.fb_target else None,
                             'notes': self.fb_notes[-12:]},
                'isolated_blame': [{'value': f'{k}={v}', 'runs': runs} for (k, v), runs in self.isolated_blame().items()][:12],
                'oom_combinations': [{'values': self._fb_words(c), 'run': cid} for c, cid in self.oom_combinations()][:12],
                'probe_thrash': self.fb_thrash[-12:], 'fixes': dict(self.fb_fixes),
                'inert_shown': [r['pid'] for r in self.fb_inert_log if r['shown']],
                's6': {'oom_streak': self.fb_oom_streak, 'max_streak': self.fb_oom_streak_max, 'active': self.s6_active(),
                       'banned_patches': [b['mechanism'] for b in self.fb_s6_bans][:12]},
                'v4': self._v4_view()}

    # ------------------------------------------------------------ v4 (FABLE_CODE_DIAGNOSIS_20261102; module docstring)
    def _v4(self, part: str, engine=None) -> bool:
        """v4's fix `part` applies: on (V4_PARTS, `v4_off`), the code track open to this arm - wcs_code on, patches open,
        a model to ask (`_can_code()`; a knob-only arm runs v3's knob side byte for byte) - and the screen budget beyond
        ten screens (the open wave's arms; at ten screens or fewer every v3 suite runs as before, as F1's rates)."""
        if not (self.__dict__.get('v4_on') or {}).get(part):
            return False
        eng = engine if engine is not None else getattr(self, 'engine', None)
        return bool(self.on.get('wcs_code', True)) and self._can_code() and rate_screens(eng) > RATE_SCREENS

    def _v4_init(self, engine) -> None:
        """v4's set-up before the prior call: F4 - the session counts a patch's own lines (agent/search.py
        `patch_lines_own`, the whole overlay under OVERLAY_CEILING x the limit). (The spec's caps: `_fb_rates`.)"""
        s = getattr(engine, 's', None)
        if s is not None and self._v4('v4_lines', engine) and hasattr(type(s), 'patch_lines_own'):
            s.patch_lines_own, s.patch_overlay_ceiling = True, OVERLAY_CEILING
            self.v4_notes.append(f"F4: the per-patch line limit counts each patch's own lines; the overlay at most "
                                 f'{OVERLAY_CEILING} x the limit')

    def _fb_slim_due(self, engine) -> bool:
        """F5: the patch call that just ended hit its wall (the guard's last status), not already the slim re-ask, and a
        call and dollars are left: the ask once more with a slim payload."""
        d = getattr(self, '_bdriver', None)
        g = d.__dict__.get('_inner') if d is not None else None
        return (isinstance(g, CallGuard) and g.__dict__.get('last_status') == 'TIMEOUT' and not self._fb_slim
                and self._v4('v4_timeout', engine) and self._budget_ok(engine))

    # F1: the floor
    def _fb_floor_open(self, engine) -> bool:
        """F1: the floor holds while the code track can take the screen - v4 on, a full screen ran, ASK_LEFT screens left,
        a call and dollars left, a patch's probe affordable - and was not lifted for this step (`_fb_v4_step`)."""
        return (not self._fb_floor_lift and self._v4('v4_floor', engine) and engine.full_screens() >= 1
                and self._left(engine) >= ASK_LEFT and self._budget_ok(engine) and self._fb_code_probe_ok(engine))

    @staticmethod
    def _fb_floor_thr(engine) -> float:
        """The floor (ln): max(2 %, 2 sigma_pair) - S1's strong-candidate threshold."""
        return max(math.log1p(STRONG_MIN), 2 * engine.sigma_pair())

    def _fb_floor_why(self, engine, inc, cfg: dict, ids, p, kind: str) -> str | None:
        """F1: why the floor refuses a pick of a held kind (`fb_floor_kinds`) while it holds - its two-sigma upside (the
        time model's predicted gain over the incumbent + FLOOR_SD sd) below max(2 %, 2 sigma_pair), or its P below
        STRONG_P -, recorded (`v4_floor`; the step's flag `fb_floor_refused`); None when it passes or the floor does not
        hold."""
        if kind not in self.fb_floor_kinds or not self._fb_floor_open(engine):
            return None
        mu, sd = self._mu(engine, inc, cfg, tuple(ids))
        thr = self._fb_floor_thr(engine)
        if p is None:
            p = self._pf(cfg, ids)[0]
        if mu + FLOOR_SD * sd + 1e-12 < thr:
            why = (f'predicted {100 * math.expm1(mu):+.2f} % (sd {100 * sd:.2f} %): its two-sigma upside is below '
                   f'max(2 %, 2 sigma_pair) = {100 * math.expm1(thr):.2f} %')
        elif p < STRONG_P:
            why = f'P {p:.2f} below {STRONG_P}'
        else:
            return None
        self.fb_floor_refused = True
        self.v4_counts['floor_refused'] += 1
        self.v4_floor.append({'screens_done': engine.full_screens(), 'kind': kind, 'change': self._words_of(cfg),
                              'predicted_gain_pct': round(100 * math.expm1(mu), 2), 'sd_pct': round(100 * sd, 2),
                              'p_feasible': round(p, 3), 'why': why[:200]})
        del self.v4_floor[:-24]
        return why

    def _fb_v4_note(self, engine, q, why: str) -> None:
        """The strategy's record of a screen the code track took for the knob side (a finalist overrides it)."""

    def _fb_v4_step(self, engine, inc, left, knob_side):
        """F1 / F6: the knob side's step (`knob_side()`) under F1's floor; a step the floor emptied goes to the code track
        (`_fb_floor_code`), and when the code track has nothing to run either, the knob side's step again with the floor
        lifted; F6: a step with nothing on the knob side goes to the code track too before it gives up (the arm ends with
        screens left only when the calls or dollars are spent or nothing is legal)."""
        self._fb_floor_lift = self.fb_floor_refused = False
        mark = self.v4_counts['floor_refused']
        q = knob_side()
        if q is not None:
            return q
        floored = self.fb_floor_refused
        if not floored and not self._v4('v4_never_done', engine):
            return None
        why = "the knob side's picks are below the floor (F1)" if floored else 'the knob side has nothing to run (F6)'
        q = self._fb_floor_code(engine, inc, left, why)
        if q is not None:
            self.v4_counts['floor_to_code' if floored else 'empty_to_code'] += 1
            self._fb_v4_note(engine, q, why)
            return q
        if not floored:
            return None
        self._fb_floor_lift = True
        try:
            q = knob_side()
        finally:
            self._fb_floor_lift = False
        if q is not None:
            self.v4_counts['floor_lifted'] += 1
            for r in self.v4_floor[-(self.v4_counts['floor_refused'] - mark):]:
                r['lifted'] = 'the code track had nothing to run'
        return q

    def _fb_floor_code(self, engine, inc, left, why: str):
        """F1 / F6: the code track takes a screen the knob side gave up - a pending patch or a due stack test (F2), else an
        ask: S1's gates without the knob side's hold-back (`_fb_force_ask`); F6: with no run since the last ask when its
        patches all ended before a run (their refusals in the ledger the ask shows), asked again while an ask brings
        nothing to run and makes a call. None when the code track has nothing to run."""
        if not (self.on['wcs_code'] and self._can_code()):
            return None
        q = self._code_step(engine, inc, left, ask=False)
        asks = 0
        while q is None:
            n0, used0 = self.asks, self._llm_used()
            self._fb_force_ask, self._fb_force_why = True, why
            try:
                q = self._code_step(engine, inc, left, ask=True)
            finally:
                self._fb_force_ask = False
            if self.asks == n0:
                break                                    # (no ask was due: nothing for this screen)
            asks += 1
            if asks > 1:
                self.v4_counts['reasks'] += 1
            if q is not None or self._llm_used() == used0 or not self._v4('v4_never_done', engine):
                break                                    # (a patch to run; an ask with no call; F6 off: one ask)
        self.v4_code.append({'screens_done': engine.full_screens(), 'why': why[:120], 'asks': asks,
                             'got': q.hypothesis if q is not None else None})
        del self.v4_code[:-24]
        return q

    # the fix call (FABLE_V4_BUDGET_20261102 section 1)
    def _fb_fix_note(self, engine, p, out, cls) -> None:
        """A patch the guard refused (the session's newest `invalid_proposal` finding: its message with the lines it
        counted) or whose probe failed (FIX_KINDS: the run's reasons, its class line, the probe's peak) waits for one fix
        call (`_fb_fix_due`), each patch once; a full run's result is not fixed. Nothing with no fix calls set."""
        if self.v4_fix_max <= 0 or not self._v4('v4_fix', engine):
            return
        pid = (p.hypothesis or '').split('|', 1)[1]
        kind = (cls or {}).get('kind')
        if out.status == 'rejected':
            new = list(getattr(engine.s, 'events', None) or ())[self._fb_events_seen:]
            ev = next((e for e in reversed(new) if e.get('kind') == 'invalid_proposal'), None)
            finding = [str(x) for x in ((ev or {}).get('payload') or {}).get('findings') or []] or list(out.reasons or [])
            stage = 'guard'
        elif out.fidelity == 'probe' and kind in FIX_KINDS:
            peak, cap = self._fb_probe_peak(engine, out.cid), self.mem.cap
            finding = [f'{kind} at its probe' + (f': measured peak {peak:.1f} GiB of a {cap:.0f} GiB card'
                                                 if peak and cap else '')]
            finding += [str(x) for x in [(cls or {}).get('line')] + list(out.reasons or []) if x]
            stage = 'probe'
        else:
            return
        if any(r['pid'] == pid for r in self.v4_fix_pending + self.v4_fix_log):
            return
        self.v4_fix_pending.append({'pid': pid, 'stage': stage, 'kind': kind or out.status, 'base': p.base,
                                    'finding': [one_line(x, FIX_FINDING_MAX) for x in finding][:6]})

    def _fb_fix_due(self, engine, left) -> bool:
        """A fix call may be made now: one is waiting, v4's part on, the arm's fix calls (`v4_fix_max`) and this screen's
        (`v4_fix_per_screen`, none set: the arm's) not spent, the code track able to ask (a call and dollars, ASK_LEFT
        screens left, a patch probe affordable)."""
        if not self.v4_fix_pending or self.v4_fix_max <= 0 or not self._v4('v4_fix', engine):
            return False
        n = engine.full_screens()
        here = sum(1 for r in self.v4_fix_log if r['screens_done'] == n)
        return (len(self.v4_fix_log) < self.v4_fix_max and here < (self.v4_fix_per_screen or self.v4_fix_max)
                and left >= ASK_LEFT and self._can_code() and self._budget_ok(engine) and self._fb_code_probe_ok(engine))

    def _fb_fix_ask(self, engine, inc, left) -> list:
        """The fix call: one patch call (S1's ask) with the waiting patch's `fix_refused` field (FIX_REFUSED_RULE: its
        id, mechanism, site, the stage, the finding, the lines it added), at most one patch queued; recorded."""
        rec = self.v4_fix_pending.pop(0)
        self.v4_fix_log.append({'pid': rec['pid'], 'stage': rec['stage'], 'kind': rec['kind'],
                                'screens_done': engine.full_screens()})
        self._fb_fixing = rec
        try:
            queued = self._fb_ask(engine, inc, left, f'a fix call for {rec["pid"]} ({rec["stage"]}: {rec["kind"]})',
                                  limit=1)
        finally:
            self._fb_fixing = None
        self.v4_fix_log[-1]['queued'] = [q.hypothesis.split('|', 1)[1] for q in queued]
        del self.v4_fix_log[:-max(24, self.v4_fix_max)]
        self.v4_counts['fix_calls'] += 1
        return queued

    def _fb_fix_refused_view(self, engine, rec) -> dict | None:
        """The fix call's field: FIX_REFUSED_RULE, the patch (id, mechanism, site), the stage, the finding, its added
        lines."""
        p = self.patches.get(rec['pid']) or {}
        out = {'rule': FIX_REFUSED_RULE, 'id': rec['pid'], 'mechanism': p.get('mechanism'), 'site': p.get('site'),
               'stage': rec['stage'], 'finding': rec['finding'],
               'added_lines': self._fb_added_lines(engine, rec['pid'], rec.get('base'))}
        return {k: v for k, v in out.items() if v not in (None, [], {})}

    # F2: the stack
    def _observe_patch(self, engine, p, out, kind, y, base) -> None:
        """wcs's patch outcome (its adoption: gain > sigma_pair); v4 F2: a proven patch it did not adopt with a positive
        gain joins the pending stack."""
        super()._observe_patch(engine, p, out, kind, y, base)
        pid = (p.hypothesis or '').split('|', 1)[1]
        rec = self.patches.get(pid) or {}
        g = rec.get('gain_pct')
        if rec.get('status') == 'proven' and not rec.get('adopted') and isinstance(g, (int, float)) and g > 0 \
                and self._v4('v4_stack', engine) and pid not in self.v4_stack and pid not in self.v4_stacked:
            self.v4_stack.append(pid)
            rec['stack'] = 'pending'

    @staticmethod
    def _fb_line_limit(engine) -> int | None:
        """The per-patch line limit the session applies to a hook patch (the task's and the capability's, the smaller)."""
        s = engine.s
        lims = [getattr(getattr(s, 'hooks', None), 'patch_line_limit', None), getattr(s, 'patch_line_limit', None)]
        lims = [int(x) for x in lims if isinstance(x, int) and not isinstance(x, bool) and x > 0]
        return min(lims) if lims else None

    def _fb_stack_q(self, engine, inc):
        """F2: the pending stack's test - one full screen of the chain with the pending patches merged onto what is carried
        now (each patch's own edit against the files of the candidate it ran on, the best measured first, merged three-way
        with two insertions at one place both kept; their own lines within the per-patch limit together; one whose edit
        no longer merges leaves the stack) - when two or more of them merge and their summed measured gain (ln) reaches
        max(2 %, 2 sigma_pair); the same proposal until it is observed; None when no stack is due."""
        q = self._v4_stack_q
        if q is not None:
            return q if engine._fresh(q) else None
        members = [x for x in self.v4_stack if x not in self.chain and x in self.patches]
        self.v4_stack = members
        thr = self._fb_floor_thr(engine)

        def gain(x):
            return math.log1p(max(-0.99, float(self.patches[x].get('gain_pct') or 0.0) / 100.0))
        if len(members) < 2 or sum(gain(x) for x in members) + 1e-12 < thr or self._left(engine) < 1:
            return None
        carried = self._carried(engine, inc)
        limit = self._fb_line_limit(engine)
        merged, used, lines = {}, [], 0
        for pid in sorted(members, key=lambda x: (-gain(x), x)):
            c = engine.s.candidates.get(self.patches[pid].get('candidate'))
            if c is None:
                continue
            parent = c.parent if getattr(c, 'parent', None) in engine.s.candidates else 'c0'
            try:
                old, mine = files_of(engine, parent), files_of(engine, c.id)
            except Exception:                            # noqa: BLE001 - no files: it cannot be stacked now
                continue
            edit = {f: t for f, t in mine.items() if t != old.get(f)}
            own = sum(_own_count(old.get(f, ''), t) for f, t in edit.items())   # (r2: F4's count, proof lines out)
            if not edit or limit is not None and lines + own > limit:
                continue
            cur = {f: merged.get(f, carried.get(f, '')) for f in edit}
            got = merge_patches(old, [cur, edit], append=True)   # (both appending after one line: both kept, hls2 C7)
            if got is None or got == cur:
                self.patches[pid]['stack'] = 'its edit does not merge onto the chain' if got is None else 'nothing new'
                self.v4_stack.remove(pid)
                self.v4_counts['stack_dropped'] += 1
                continue
            merged.update(got)
            used.append(pid)
            lines += own
        if len(used) < 2 or sum(gain(x) for x in used) + 1e-12 < thr:
            return None
        files = {**(dict(self.overlay_best) if self.chain and self.overlay_best else {}), **merged}
        self._n += 1
        hid = f'stack|{self._n}'
        ids = frozenset(self.chain) | frozenset(used)
        self.ideas[hid] = ids
        mu = sum(gain(x) for x in used)
        p, _ = self._pf(self._knob_cfg(engine, inc), tuple(sorted(ids)))
        mech = (f'stack of {" + ".join(used)}: proven, each within the noise ('
                + ', '.join(f'{self.patches[x]["gain_pct"]:+.2f} %' for x in used) + ')')
        q = Proposal(change={}, patch=files, base=inc, mu=mu, sigma=max(0.005, engine.sigma_pair()),
                     p_feasible=min(0.99, max(0.01, p)), mechanism=mech[:200], hypothesis=hid, origin=f'{self.name}-stack',
                     fidelity=('full',), keep_probe=False)
        self.ledger[hid] = {'id': hid, 'kind': 'stack', 'mechanism': q.mechanism, 'change': {}, 'patches': sorted(ids),
                            'base': inc, 'predicted_gain_pct': round(100 * math.expm1(mu), 2), 'p_feasible': round(p, 3),
                            'plan': ['full'], 'status': 'open'}
        self.picks.append({'id': hid, 'kind': 'stack', 'screens_done': engine.full_screens(), 'mechanism': q.mechanism,
                           'p_feasible': round(p, 3), 'predicted_gain_pct': round(100 * math.expm1(mu), 2)})
        self.v4_stacks.append({'id': hid, 'members': list(used), 'summed_gain_pct': round(100 * math.expm1(mu), 2),
                               'own_lines': lines, 'base': inc, 'screens_done': engine.full_screens()})
        self.v4_stack = [x for x in self.v4_stack if x not in used]
        self.v4_stacked |= set(used)
        for x in used:
            self.patches[x]['stack'] = hid
        self.v4_counts['stack_tests'] += 1
        self._v4_stack_q = q
        return q

    def _fb_stack_seen(self, engine, p, out, kind) -> None:
        """F2: a stack test's outcome - its members join the chain (the run's overlay the carried files) when it measured
        more than sigma_pair over its base, or the engine made it the incumbent (later proposals are written against its
        files); else they stay proven, not adopted."""
        self._v4_stack_q = None
        rec = next((s for s in reversed(self.v4_stacks) if s['id'] == p.hypothesis), None)
        if rec is None:
            return
        y = (self.runs.get(out.cid) or {}).get('y') if out.cid else None
        rec.update(cid=out.cid, status=out.status, outcome=kind or out.status, adopted=False)
        if out.status != 'screened' or y is None:
            return
        gain = y - ((self.runs.get(p.base) or {}).get('y') or 0.0)
        rec['gain_pct'] = round(100 * math.expm1(gain), 2)
        incumbent = out.cid == engine.incumbent()
        if gain <= engine.sigma_pair() and not incumbent:
            return
        for pid in rec['members']:
            if pid not in self.chain:
                self.chain.append(pid)
            self.patches[pid].update(adopted=True, adopted_via=out.cid)
        c = engine.s.candidates.get(out.cid)
        if c is not None:
            self.overlay_best = dict(getattr(c, 'overlay', None) or {})
        rec['adopted'] = 'measured above sigma_pair' if gain > engine.sigma_pair() else 'the engine made it the incumbent'
        self.v4_counts['stack_adopted'] += 1

    # F7: a declared slow start
    def _fb_slow_start(self, engine, q, h) -> None:
        """F7: a patch probe whose patch declares warmup_updates or active_from_update of SLOW_START or more runs without a
        start cap (`probe_cap` None: the job's launch timeout bounds it) - a long first update is not a hang."""
        if q is None or tuple(q.fidelity or ())[:1] != ('probe',) or q.probe_cap is None or \
                not self._v4('v4_slow_start', engine):
            return
        if max(self._supported(h, 'warmup_updates') or 0, self._supported(h, 'active_from_update') or 0) >= SLOW_START:
            q.probe_cap = None
            self.v4_counts['slow_start_probes'] += 1

    def _probe_cap(self, engine, base, steps=None):
        """v4 (the user, 11-02 22:2x: the runs' time limit as Auto-Research's, whose every experiment runs under the job's
        launch timeout - autoresearch.py's 'Timeout: the job's launch timeout'): with v4 on, a probe - a patch's or the
        knob side's - has no start cap (`None`: the launch timeout bounds it; F7 did this for a declared slow start only).
        Otherwise the start cap of the strategy's base (WCS / HLS3: K_START x the base's time to its first update)."""
        if self._v4('v4_no_start_cap', engine):
            return None
        return super()._probe_cap(engine, base, steps)

    # F8: earlier arms' failed starts
    def _fb_store_sets(self) -> list:
        """F8: the value sets earlier arms failed to start together (wcs's `from_store`, mapped onto this table as
        `_start_from_store` maps them), two values or more each; read once after the start."""
        if self._v4_store is None:
            out = []
            for it in getattr(self, 'from_store', None) or []:
                vals = set()
                for m in it.get('members') or []:
                    for k, v in (m.items() if isinstance(m, dict) else ()):
                        if k in self.knobs:
                            w = _coerce(v, self.knobs[k])
                            if w is not None and vk(w) != vk(self.ref.get(k)):
                                vals.add((k, vk(w)))
                if len(vals) >= 2:
                    out.append(frozenset(vals))
            self._v4_store = out
        return self._v4_store

    def _fb_store_plan(self, engine, inc, cfg: dict, plan) -> tuple | None:
        """F8: a contrarian move that sets a value an earlier arm failed to start together with others is probed first
        (the ladder's forced probe) - never run in full first; None when no probe is affordable now (it stays eligible)."""
        plan = tuple(plan or ('full',))
        if plan[0] == 'probe':
            return plan
        inc_cfg = self._knob_cfg(engine, inc)
        new = {(k, vk(cfg.get(k))) for k in self.knobs if vk(cfg.get(k)) != vk(inc_cfg.get(k))}
        hit = next((s for s in self._fb_store_sets() if s & new), None)
        if hit is None:
            return plan
        forced = tuple(self.ladder.plan(engine, 0.0, warmup=self._warmup(cfg) or None, probe=True, short=False))
        if forced[0] != 'probe':
            self.v4_counts['store_held'] += 1
            return None
        self.v4_counts['store_probes'] += 1
        self.v4_notes.append(f'F8: a contrarian move setting {self._fb_words(sorted(hit & new))} probed first (an earlier '
                             f'arm failed to start {self._fb_words(sorted(hit))})')
        del self.v4_notes[:-24]
        return forced

    def _v4_view(self) -> dict:
        """v4's account for the report: whether it applies, its parts off, its counts, the floor's refusals, the screens the
        code track took, the stacks, the re-priced calls, its notes."""
        eng = getattr(self, 'engine', None)
        d = getattr(self, '_bdriver', None)
        g = d.__dict__.get('_inner') if d is not None else None
        return {'version': V4_VERSION, 'applies': bool(eng is not None and self.on.get('wcs_code', True) and self._can_code()
                                                       and rate_screens(eng) > RATE_SCREENS),
                'parts_off': sorted(k for k, v in self.v4_on.items() if not v), 'counts': dict(self.v4_counts),
                'floor': self.v4_floor[-12:], 'code_screens': self.v4_code[-12:],
                'stack': {'pending': list(self.v4_stack), 'tests': self.v4_stacks[-12:]},
                'repriced_calls': list(g.__dict__.get('repriced') or [])[-6:] if isinstance(g, CallGuard) else [],
                'spec_caps': dict(self.fb_spec_caps),
                'fix_calls': {'max': self.v4_fix_max, 'per_screen': self.v4_fix_per_screen,
                              'pending': [r['pid'] for r in self.v4_fix_pending], 'calls': self.v4_fix_log[-12:]},
                'notes': self.v4_notes[-12:]}

    def describe(self) -> dict:
        view, learned, prop = self.candidate_view()
        out = {'name': self.name, 'proposer': prop,
               self.name: {'name': self.name, **view, **self.llmopt_view(), 'finalist': self.finalist_view()}}
        if learned:
            out['learned'] = learned                     # D4: written back for a later arm
        pk = self.patch_section()
        if pk:
            out['patches'] = pk                          # S2: runs.write_back puts it on the arm's pack
        return out


# ---------------------------------------------------------------- registration on the Engine
def run_engine(ctx, cls, components: dict, **extra):
    """A finalist `cls` on the Engine: _engine_x6's set-up (the proposer with the strategy's features, the strategy with
    its switched-off components as `off`, the session, the engine). For search/__init__: STRATEGIES[name] = a function of
    ctx that imports its class and calls this."""
    from ..agent.capability import mode_for, policy_overrides, search_params
    from .engine import Engine
    from .proposers import LLMProposer
    o = ctx.options
    c = o.get('capability', 0.6)
    kp = search_params(c)
    exp = o.get('experience')                          # experience.store.plan for this job, or None
    off = sorted(x for x in set(o.get('ablate') or ()) if x in components)
    facts = getattr(ctx.hooks, 'memory_facts', lambda: None)() or {}
    acc = o.get('accounting')
    prop = LLMProposer(ctx.driver, ctx.hooks, max_hypotheses=kp['max_hypotheses'], menu=kp['menu'],
                       sigma_scale=kp['sigma_scale'], prior=o.get('prior_results'), experience=(exp or {}).get('payload'),
                       search_calibrates=True, accounting=acc, features=cls.proposer_features(off))
    strategy = cls(prop, off=off, facts=facts, accounting=acc, experience=exp, kappa0=kp['kappa0'], seed=ctx.seed, **extra)
    session = ctx.session(o.get('arm') or cls.name, stop_after_same_failures=SAME_FAILURE_STOP, mode=mode_for(c),
                          patch_line_limit=policy_overrides(c)['patch_line_limit'],
                          layers=tuple(getattr(ctx.hooks, 'layers', None) or ('L0', 'L1')),
                          count_rescores=bool(o.get('count_rescores')))
    if ctx.start_receipt is None:                      # no reference run before the search: the start is its first
        session.screen('c0')                           # counted screen (as _engine)
    engine = Engine(session, full_steps=o['full_steps'], short_steps=o.get('short_steps'),
                    probe_steps=o.get('probe_steps', 2), warmup=o.get('warmup', 3), job_gpu_hours=o['job_gpu_hours'],
                    multi_fidelity=o.get('multi_fidelity', True), run_noise=o.get('run_noise', 0.015),
                    noise_pairs=o.get('noise_pairs'), log=o.get('log'), checkpoint=o.get('checkpoint'), accounting=acc)
    if o.get('resume') is not None:                    # `infrarsi run --resume` (project/resume.py): the saved search onto
        o['resume'](session, engine, strategy)         # the objects built above
    return engine.run(strategy)
