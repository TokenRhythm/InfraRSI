"""hls3: hls2 with the general search strategies of FABLE_TAKEAWAYS_20261030 section 4 (Fable 5.1, 2026-10-30), each a
component that `search.ablate` / `run --ablate` switches off by its name; every one off is hls2 (tested).

    hls3_feasibility    S-1   memory as a budget: each run's measured peak, an out-of-memory run's failed allocation and
                              the memory free then, and each move's direction (frees / spends / neutral: the runs, the
                              other values of an ordered knob, the proposer's declaration) give a dominance model - a
                              candidate that needs at least what an out-of-memory run needed by measured moves is not
                              screened (its entry is deferred); one that only the declarations put there, or that an
                              out-of-memory run nearly bounds, is probed first; and a probe is kept while the screens are
                              the scarcer budget (a failed probe spends no screen; the engine otherwise drops a probe that
                              costs half a full run: G4U-code-env-opus-h2 screened five OOMs)
    hls3_memory_schema  S-1   the proposer declares each move's memory direction and GiB: they order the freeing moves,
                              build the bundles and may ask for a probe, never rule a configuration out (off: the runs
                              and the knob order only)
    hls3_bundles        S-2   a spender predicted not to fit, or out of memory on the incumbent, is proposed again with the
                              best freeing moves the knob table, the declarations, the code ideas and the runs offer, as
                              one candidate; its composition is recorded for attribution
    hls3_failures       S-3   every failed run classified - out of memory (allocation, free memory), compiler, collective,
                              killed at the launch timeout, hang, crash, numerics, and thrash: valid or aborted but at least
                              1.5x slower than its base with a measured peak near the card or an out-of-memory run that
                              bounds it by measured moves, the edge of what fits rather than a slow move (a slowdown
                              without that evidence is at most a possible thrash, noted) - and a summary for the proposer
    hls3_antagonism     S-3   changes that failed together (crash, compiler, collective, launch timeout, hang) are retried
                              separately before they are combined again
    hls3_stall_probes   S-4   at a stall, cheap feasibility probes of the blocked spender with each candidate enabler
                              (freeing ones and moves whose memory effect nobody knows, alternating), not a 2^k screen
    hls3_rebase         S-5   after the incumbent changes, the strongest moves rejected, aborted or out of memory on an
                              earlier one are queued on it again (reverts allowed), at most two per change
    hls3_explore        S-6   at a stall, one whole-configuration step: the experience start composite, else the proposer's
                              strongest untested moves, made coherent (antagonism, memory) - not a single move
    hls3_phases         S-7   after an accepted change, the incumbent's open moves are re-ranked by its own phase split when
                              its run measured one (no extra profile run)
    hls3_sites          S-8   the code prompts ask for the site first and the expected gain per changed line
    hls3_variance       S-9   the search's own decisions on medians: the incumbent, the early abort, the measured gains, the
                              robust split halves (Engine robust mode); a move declared to remove stalls is judged by the
                              mean and proposed early when the reference shows outlier updates; the verifier is unchanged
    hls3_prune          S-10  a configuration whose training state alone (the shared accounting of the whole configuration,
                              while it agrees with every run that fitted) is above the card is not proposed; a value whose
                              declared spend puts the sharded-state floor (parameters x 16 B over every GPU) above the card
                              is probed first; the proposer's accounting gets the states sharded (the shared accounting is
                              unchanged)

FABLE_HLS3_FIXES_R1 (the first hls3 arms, 2026-10-30), each its own component:

    hls3_start_cap      F1a/c a probe stopped at K_START x its base's time to the first update (start_slow: not a start
                              failure for C5; its entry gets one full retry at p <= 0.3, its candidate reopened); a probe
                              kept for an unsure, memory-spending or start-risky move (its feasibility below the probe rule:
                              a full screen has no start cap, D1), priced at its cap, while the GPU-hours and the wall left
                              pay every screen left - a start-risky one also when (1 - p_feasible) x a full run pays for the
                              probe and keeping it costs no screen the budget would admit
    hls3_probe_gate     F1b   no full run after a probe whose updates ran GATE x slower than its base's same updates on a
                              configuration that spends memory - thrash (memory evidence) only with a measured peak near the
                              card or an out-of-memory bound, else a slow probe that records nothing about memory
    hls3_coverage       F2    with two or more open moves below 2 sigma_pair, the compatible start-safe knob moves among them
                              that are predicted to help and spend no memory as one bundle, probed first, its members not
                              screened singly while it is proposed (back to their single screens when it does not move)
    hls3_need           F3    a spender's next bundle must free more than every bundle of it that ran out of memory
                              (need_lb, when GiB are declared); out-of-memory moves re-based only after a freeing move as
                              big; no unknown-direction enabler in a bundle and once per spender at a stall; after
                              SPENDER_PROBES tries a proposal holding the spender waits for a measured freeing move (not the
                              full run of a passed probe, nor a configuration a fitting run bounds by measured moves)
    hls3_complete       F4    a change to part of a workload invariant's knobs completed with the others (the validator)
    hls3_pairs          F5    every pair of a set of at most PAIR_SET changes that failed to start is suspect (by
                              configuration): probed first after one such failure, deferred after two; a value whose every
                              failed start had a partner is probed alone (an invariant's knobs together) before C5 rules it
                              out; a proposal blocked by a
                              pair with an incumbent member is tried as the incumbent without that member (a swap)
    hls3_remainder      F6    the part of a multi-knob entry the new incumbent did not take is carried to it
    hls3_warn           F7    a memory warning of the store or the estimate caps feasibility only where hls3's runs bound it
    hls3_dead_patch     F8    the same fatal start line (an exception named) twice rules out the new elements both runs
                              shared - knob values, patch idea, site, the code lines both patches added; an inert patch is
                              reported and its site's next patch without a self-check line waits

FABLE_HLS3_FIXES_R2_Q9H_M35H2 (the Q9H and M35H2 arms, 2026-10-30), each its own component:

    hls3_self_proof     F10   every patch is probed and proves there that it took effect: its `[infrarsi-patch] <site>:
                              found X -> set Y` line with X != Y, its `active` line (the patched code ran), and the job-log
                              note it targets (a note naming a code name of its claim: site, evidence note, the names its
                              added code sets) gone from the probe's notes - else `inert_at_start`: no full run, the reason, the note,
                              its lines and its code back to the proposer; post-hoc inert below max(sigma_pair, kappa mu / 4)
    hls3_fear_prior     F11   a declared memory spender that no run bounds (the memory model has no verdict) is ranked with
                              p = max(this run's declarations of the move, (fitted + 1) / (tried + 2) over this run's runs
                              of feared moves - declared spenders below the probe rule - once one of them ran); its probe
                              first as before; a measured bound overrides; the ledger and the report keep the proposer's p
    hls3_gate_low       F12   the probe speed gate at GATE_LOW (its calibration: no valid probe of the recorded arms above
                              x1.31, both thrashes above x1.8), the proposal's own patch spend counted, a change of unknown
                              memory direction counted as a possible spend

FABLE_HLS3_FIXES_R3 (the G4U and Q9H -h5 arms, 2026-10-30), each its own component:

    hls3_scan_whole     F13   F10 reads a patch's proof lines whole (project.signals keeps them uncut as `patch_proof_whole`
                              when that says more than the 300-character display lines; a site with '::' keeps its name);
                              the inert feedback carries the found / set pair read per site, and the length of a line too
                              long to display
    hls3_parse          F14   the hypothesis answer read robustly (LLMProposer): text blocks split over the CLI's resume
                              turns joined, each hypothesis validated alone (a bad one dropped and reported back, non-schema
                              `edits` rewritten from their text or the `patch` text), retries that name the exact errors
    hls3_seed_mu        F15   an experience seed with no prior that ran out of memory or thrashed is a spender at the most
                              this run predicted for its move (the transfer judgment's, the proposer's), else at 2 sigma_pair
                              - spender identification and the value of its pair probes only; the ledger keeps the zero
    hls3_pair_probes    F16   a memory-blocked spender no S-2 bundle covers is probed (start cap, F3's count) with each other
                              member of a composite declared to run that died at its start without a memory verdict (F16-A),
                              else with up to PAIR_PARTNERS single-knob moves of no measured or declared memory effect that
                              are judged '+' or predicted to help - a move slower alone included (F16-B); each pair once
    hls3_cov_unresolved F17   the members of a coverage bundle measured within 2 sigma_pair are resolved by it (unresolved,
                              not screened one by one); a new prediction of one above 2 sigma_pair runs it alone
    hls3_carry_patches  F18   an open patch entry follows a knob-only change of the incumbent (the same code: HLS._carry_open)
    hls3_pair_set       F19   F5a's pairs are suspect in a failed start of up to the transfer judgment's start composite's
                              size (at most PAIR_SET_MAX); a pair probe is exempt from the pair it tests
    hls3_deferred       F21   a patch's third proof line, `deferred (<why>)` - its replacement handed a call back to the
                              original - and `active` printed only after the new path ran: an install whose calls all
                              deferred is inert with that reason; a note still reported after `active` names the deferral
    hls3_jitter_thrash  F23   a run slower than its base (beyond 2 sigma_pair) whose update intervals scatter - MAD / median
                              at max(JITTER_MIN, JITTER_X x its base's) - on a configuration that spends memory (declared,
                              measured or unknown) is thrash: the memory edge without a measured peak (a spender; its
                              supersets bounded as by an out-of-memory run); scatter without a spend is a note

No knowledge of particular knobs, models or environments: what a move does to memory comes from the runs, the manifest
(the card, the GPUs, the parameter count, the knob table and its invariants, the shared accounting) and the proposer's
declarations - which order and build, and at most ask for a probe: hard decisions (a deferral, a pruned configuration, a
memory boundary) rest on out-of-memory runs, measured peaks or the configuration's own state arithmetic.
"""
from __future__ import annotations

import dataclasses
import difflib
import functools
import itertools
import json
import math
import re
import statistics

from ..project.signals import COLLECTIVE_RE, intervals
from . import HLS3_COMPONENTS
from . import library
# the patch self-proof (F10 / F13 / F21) is shared by any strategy that opts in (search/proof.py); its names stay
# importable from here
from .proof import (CODE_NAME, PROBE_STEPS_MAX, QUOTE_MIN, SET_NAME, _note_key, _quote_key,  # noqa: F401
                    claimed_notes, code_names, events_of, files_of, new_installs, note_reason, notes_of, overlong,
                    parsed, patch_claim, proof_of, proof_reason, proof_verdict, set_names, targeted_notes)
from .engine import Outcome, Proposal
from .strategies import HLS2, _mkey

GIB = {'B': 2.0 ** -30, 'KiB': 2.0 ** -20, 'MiB': 2.0 ** -10, 'GiB': 1.0, 'TiB': 1024.0, 'KB': 1e3 / 2 ** 30,
       'MB': 1e6 / 2 ** 30, 'GB': 1e9 / 2 ** 30}
_AMT = r'(\d+(?:\.\d+)?)\s*(B|KiB|MiB|GiB|TiB|KB|MB|GB)\b'
# PyTorch's out-of-memory message: "Tried to allocate 8.00 GiB. GPU 1 has a total capacity of 139.81 GiB of which 3.47 GiB
# is free. Including non-PyTorch memory, this process has 136.33 GiB memory in use. Of the allocated memory 134.37 GiB is
# allocated by PyTorch, and 436.24 MiB is reserved by PyTorch but unallocated."
OOM_FIELDS = (('tried_gib', rf'Tried to allocate {_AMT}'), ('capacity_gib', rf'total capacity of {_AMT}'),
              ('free_gib', rf'of which {_AMT} is free'), ('in_use_gib', rf'this process has {_AMT} memory in use'),
              ('allocated_gib', rf'memory {_AMT} is allocated by PyTorch'),
              ('reserved_gib', rf'{_AMT} is reserved by PyTorch but unallocated'))
OOM_TEXT = re.compile(r'out of memory|OutOfMemoryError(?!: out of resource)|out-of-memory line|CUBLAS_STATUS_ALLOC_FAILED|'
                      r'cudaErrorMemoryAllocation|RESOURCE_EXHAUSTED', re.I)    # not Inductor's autotune "out of resource"
COMPILER_RE = re.compile(r'_inductor|InductorError|_dynamo|BackendCompilerFailed|LoweringException|triton\.compiler|'
                         r'CompilationError|_functorch|aot_autograd', re.I)
TIMEOUT_RE = re.compile(r'killed at the (?:[\d.]+ s )?timeout after (\d+)/')
NUMERICS = {'loss_within_tolerance', 'grad_norm_within_tolerance'}
TOGETHER = ('compiler', 'crash', 'start_cap', 'start_slow', 'collective', 'hang')   # failures a combination can cause
THRASH = math.log(1.5)          # S-3: valid but at least this much slower (log) than its base, near the memory cap
NEAR = 0.9                      # ... "near": a measured peak at 90 % of the card or more
DIRS = ('spends', 'frees', 'neutral')
FLIP = {'spends': 'frees', 'frees': 'spends', 'neutral': 'neutral'}
BYTES_PER_PARAM = 16            # S-10: fp32 master + Adam moments + bf16 weights and gradients (full fine-tuning)
K_START = 4                     # F1a: a probe's cap, x its base's time to the first update (no recorded start that reached
#                                 an update took longer; the five that hung took 8-25x)
GATE = 2.0                      # F1b: a probe this many times slower than its base's same updates does not go on in full
GATE_LOW = 1.6                  # F12: ... with hls3_gate_low (the recorded arms: valid probes <= x1.31, thrashes x1.80 / 3.25)
WARMUP = 2                      # F12: the updates a run may spend warming up (compiling; recompiling in the second): not
#                                 compared - a speed-gated probe runs one more at least (GATED_STEPS) - Codex HLS4R2 r1
GATED_STEPS = WARMUP + 1
WARMUP_MAX = 6                  # F12: a declared longer warm-up is honoured up to this many updates (beyond: no speed veto)
# (PROBE_STEPS_MAX, QUOTE_MIN, CODE_NAME, SET_NAME: search/proof.py - F10's self-proof lines, read by project.signals)
SPENDER_PROBES = 3              # F3: bundles and probes of one spender in a search (more after a measured freeing move)
PAIR_SET = 3                    # F5a: pairs are suspect only in a failed set of at most this many changes (a bigger one -
#                                 a coverage bundle, a whole step - would make every pair of its members suspect)
PAIR_SET_MAX = 5                # F19: ... up to the transfer judgment's start composite's size, at most this many
DIED_AT_START = ('start_slow', 'start_cap', 'compiler', 'hang')   # F16-A: a composite's death that says nothing of memory
PAIR_PARTNERS = 2               # F16-B: undeclared partners probed with one spender
JITTER_MIN, JITTER_X = 0.02, 5.0   # F23: a run's update-interval scatter (MAD / median) at max(2 %, 5x its base's)
SELF_CHECK = re.compile(r'\bprint\(|\blog(?:ger|ging)?\.\w+\(|warnings\.warn\(')   # F8b: a line that says the hook ran
SITE_ID = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+')   # F8b: the code names a site gives (Experts.forward)


def site_keys(site) -> set:
    """F8b: what identifies a code site: the dotted code names its text gives, else the text itself."""
    s = ' '.join(str(site or '').lower().split())
    return set(SITE_ID.findall(s)) or ({s} if s else set())


def parse_oom(text: str | None) -> dict | None:
    """The numbers of an out-of-memory message in GiB: the allocation that failed (`tried_gib`), the card's capacity, the
    memory free then, what this process held; `short_gib` = tried - free, the least the run was short by, and `need_gib`
    = in use + tried, the least it needed. None without an allocation size."""
    if not text:
        return None
    out = {}
    for key, pat in OOM_FIELDS:
        m = re.search(pat, text)
        if m:
            out[key] = round(float(m.group(1)) * GIB[m.group(2)], 3)
    if 'tried_gib' not in out:
        return None
    if 'free_gib' in out:
        out['short_gib'] = round(max(0.0, out['tried_gib'] - out['free_gib']), 3)
    if 'in_use_gib' in out:
        out['need_gib'] = round(out['in_use_gib'] + out['tried_gib'], 3)
    return out


def _v(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def _ordinal(raw: str):
    """A knob value's place on its knob's scale: a number, a bool as 0 / 1, or the one number a string holds
    ('{"every_n_layers": 3}'); None otherwise (an unordered choice)."""
    try:
        x = json.loads(raw)
    except ValueError:
        return None
    if isinstance(x, (bool, int, float)):
        return float(x)
    if isinstance(x, str):
        nums = re.findall(r'-?\d+(?:\.\d+)?', x)
        return float(nums[0]) if len(nums) == 1 else None
    return None


def flat(cfg: dict, native: bool) -> dict:
    """A config as knob -> value (a native env dict: each variable as env.NAME)."""
    if not native:
        return dict(cfg)
    out = {k: v for k, v in cfg.items() if k != 'env'}
    out.update({f'env.{n}': v for n, v in (cfg.get('env') or {}).items()})
    return out


def after(cfg: dict, change: dict, native: bool) -> dict:
    """`cfg` with `change` applied as the session applies it (a native env change sets or, None, removes variables)."""
    out = dict(cfg)
    for k, v in (change or {}).items():
        if native and k == 'env' and isinstance(v, dict):
            env = dict(out.get('env') or {})
            for n, val in v.items():
                env.pop(n, None) if val is None else env.__setitem__(n, val)
            out['env'] = env
        else:
            out[k] = v
    return out


def sig(frm: dict, to: dict) -> frozenset:
    """A move as a comparable key: the (knob, from, to) triples of the knobs it changes."""
    return frozenset((k, _v(frm.get(k)), _v(to.get(k))) for k in set(frm) | set(to) if _v(frm.get(k)) != _v(to.get(k)))


def sharded_floor(facts: dict | None) -> dict | None:
    """S-10: the least memory per GPU the training state needs whatever frees activations - parameters x 16 B (full fine-
    tuning with Adam) sharded over every GPU, the most favourable sharding, so that the bound never prunes a value that
    some sharding would fit; None for a job the arithmetic does not describe (no parameter count or card, a LoRA run)."""
    f = facts or {}
    p, world, cap = f.get('params_b'), f.get('world'), f.get('cap_gib')
    if not (isinstance(p, (int, float)) and p > 0 and isinstance(world, int) and world > 0 and cap):
        return None
    if 'lora' in str(f.get('trainable') or '').lower():
        return None
    whole = p * 1e9 * BYTES_PER_PARAM / 2 ** 30
    return {'floor_gib': round(whole / world, 2), 'whole_gib': round(whole, 2), 'shard': world,
            'basis': f'{p:g}B parameters x {BYTES_PER_PARAM} B (fp32 master, Adam moments, bf16 weights and gradients) '
                     f'sharded over all {world} GPUs'}


def sharded_accounting(view, facts: dict | None):
    """S-10: the proposer's accounting (experience.runs.accounting_for: (config, step time) -> view) with the training
    state sharded over every GPU where the closed form counted it whole and so put the reference above the card it ran
    on ("510 GiB per GPU of 77.3"); the shared accounting itself (doctor, report, hls / hls2) is not changed."""
    fl = sharded_floor(facts)
    if view is None or fl is None:
        return view

    def wrapped(config: dict, step_time=None):
        a = view(config, step_time)
        if not a or not isinstance(a.get('terms_gib'), dict) or not a.get('capacity_gib'):
            return a
        old = a['terms_gib'].get('weights_grads_optimizer')
        if not isinstance(old, (int, float)) or a.get('memory_gib', 0) <= a['capacity_gib'] or old <= fl['floor_gib']:
            return a
        mem = round(a['memory_gib'] - old + fl['floor_gib'], 2)
        return {**a, 'memory_gib': mem, 'margin_gib': round(a['capacity_gib'] - mem, 2),
                'terms_gib': {**a['terms_gib'], 'weights_grads_optimizer': fl['floor_gib']},
                'notes': list(a.get('notes') or []) + [
                    f'hls3: the weights, gradients and optimizer states are taken as sharded ({fl["basis"]}); the closed '
                    f'form counted them whole ({old:.0f} GiB per GPU) and put the reference above the card it ran on']}
    return wrapped


def proposer_flags(off=()) -> dict:
    """What the LLMProposer of an hls3 run asks and explains (search/__init__._engine), by the components on."""
    off = set(off or ())
    on = {k: k not in off for k in HLS3_COMPONENTS}
    return {'memory': on['hls3_memory_schema'], 'memory_view': on['hls3_feasibility'] or on['hls3_prune'],
            'failures': on['hls3_failures'], 'together': on['hls3_antagonism'], 'variance': on['hls3_variance'],
            'sites': on['hls3_sites'], 'rebase': on['hls3_rebase'],
            'inert': on['hls3_dead_patch'], 'self_proof': on['hls3_self_proof'],
            'warmup': on['hls3_gate_low'] and on['hls3_probe_gate'],
            'coverage': on['hls3_coverage'], 'parse': on['hls3_parse'],
            'deferred': on['hls3_deferred'] and on['hls3_self_proof'],
            'library_source': on['hls3_library_source']}


# ---------------------------------------------------------------- S-1: the memory model
class MemoryModel:
    """Feasibility from dominance: a configuration that needs at least the memory an out-of-memory (or thrashing) run
    needed does not fit; one that needs no more than a run that fitted does. "Needs at least" = every knob change from
    the one to the other spends memory or is neutral - by a move the proposer declared or the runs measured (a multi-knob
    move matched whole first), or along an ordered knob's scale once one of its moves has a direction (monotone: if
    every_n_layers 1 -> 4 spends, 4 -> 3 frees and 3 -> 4 spends). A configuration that an out-of-memory run bounds but
    for at most `slack` freeing or unknown changes is unsure (an enabler added: probe first). Patches are pseudo-knobs
    (patch:<idea> absent / true) with the direction their idea declared."""

    def __init__(self, cap_gib: float | None = None, *, slack: int = 2):
        self.cap, self.slack = cap_gib, slack
        self.runs: dict[str, dict] = {}                     # cid -> {'cfg', 'status': ok | oom | thrash, 'peak', 'oom'}
        self.measured: dict[frozenset, str] = {}            # move -> direction the runs showed
        self.gib: dict[frozenset, float] = {}               # D3: move -> GiB it freed, from two measured peaks
        self.floor: dict | None = None                      # S-10: sharded_floor(facts)

    def record(self, cid: str, cfg: dict, status: str, *, peak=None, oom: dict | None = None) -> None:
        self.runs[cid] = {'cfg': dict(cfg), 'status': status, 'peak': peak, 'oom': oom}
        cap = (oom or {}).get('capacity_gib')
        if isinstance(cap, (int, float)) and cap > 0:       # what the card really offers (139.81 GiB of a "141")
            self.cap = min(self.cap, cap) if self.cap else cap

    def relate(self, frm: dict, to: dict, direction: str) -> None:
        s = sig(frm, to)
        if s and direction in DIRS:
            self.measured[s] = direction

    def _knob(self, k: str, a: str, b: str, known: dict) -> str | None:
        s = frozenset({(k, a, b)})
        if s in known:
            return known[s]
        r = frozenset({(k, b, a)})
        if r in known:
            return FLIP[known[r]]
        oa, ob = _ordinal(a), _ordinal(b)
        if oa is None or ob is None or oa == ob:
            return None
        trend = set()                                       # an ordered knob: the direction of its scale
        for s2, d in known.items():
            if len(s2) != 1 or d == 'neutral':
                continue
            (k2, x, y), = s2
            ox, oy = (_ordinal(x), _ordinal(y)) if k2 == k else (None, None)
            if ox is not None and oy is not None and ox != oy:
                trend.add((oy > ox) == (d == 'spends'))     # True: memory grows with the value
        if len(trend) != 1:
            return None
        return 'spends' if (ob > oa) == trend.pop() else 'frees'

    def split(self, frm: dict, to: dict, known: dict) -> dict:
        """The changes from `frm` to `to` by memory direction: {'spends' | 'frees' | 'neutral' | 'unknown': [moves]}."""
        left = {k: (_v(frm.get(k)), _v(to.get(k))) for k in set(frm) | set(to) if _v(frm.get(k)) != _v(to.get(k))}
        out = {'spends': [], 'frees': [], 'neutral': [], 'unknown': []}
        for s, d in sorted(((s, d) for s, d in known.items() if len(s) > 1), key=lambda x: -len(x[0])):
            for move, dd in ((s, d), (frozenset((k, b, a) for k, a, b in s), FLIP[d])):
                if all(left.get(k) == (a, b) for k, a, b in move):
                    out[dd].append(move)
                    for k, _, _ in move:
                        left.pop(k)
                    break
        for k, (a, b) in sorted(left.items()):
            out[self._knob(k, a, b, known) or 'unknown'].append(frozenset({(k, a, b)}))
        return out

    def dominates(self, x: dict, y: dict, known: dict) -> bool:
        """x needs at least the memory y needs."""
        sp = self.split(y, x, known)
        return not sp['frees'] and not sp['unknown']

    def predict(self, cfg: dict, known: dict, ref: dict | None = None, *, hard: dict | None = None
                ) -> tuple[str | None, str, str | None]:
        """('infeasible' | 'unsure' | 'feasible' | None, why, the run it is read from). Infeasible only through `hard`,
        the moves the runs measured (default: all of `known`): the configuration holds what an out-of-memory run held
        plus changes that spend or are neutral by them. One that the proposer's declarations alone put there is unsure
        (probed first): a declaration never rules a configuration out. Unsure also: it holds what an out-of-memory run
        changed, plus at most `slack` changes that free memory or whose effect is unknown - a change back to the
        reference's value (`ref`, which fitted) means it does not hold that run's spender. A run that fitted with this
        configuration, or one that needs at least as much by measured moves, makes it feasible whatever is declared."""
        hard = known if hard is None else hard
        same = next((cid for cid, r in self.runs.items() if r['status'] == 'ok' and not sig(r['cfg'], cfg)), None)
        if same:
            return 'feasible', f'{same} ran this configuration', same
        ok = [(cid, r) for cid, r in self.runs.items() if r['status'] == 'ok']
        fits_hard = next((cid for cid, r in ok if self.dominates(r['cfg'], cfg, hard)), None)
        fits = fits_hard or next((cid for cid, r in ok if self.dominates(r['cfg'], cfg, known)), None)
        bad, said, near = None, None, None
        for cid, r in self.runs.items():
            if r['status'] not in ('oom', 'thrash'):
                continue
            if self.dominates(cfg, r['cfg'], hard):
                bad = bad or cid
                continue
            sp = self.split(r['cfg'], cfg, known)
            enab = sp['frees'] + sp['unknown']
            if not enab:
                said = said or cid
            elif near is None and len(enab) <= self.slack and (sp['spends'] or sp['frees']) and not any(
                    ref is not None and all(_v(ref.get(k)) == b for k, _, b in m) for m in enab):
                near = cid
        if bad and fits_hard:                               # a run that fitted needs more than one that did not: the two
            return 'unsure', f'{fits_hard} fitted and {bad} did not; it needs as much as both', bad   # disagree (a cliff)
        if bad:
            return 'infeasible', f'it needs at least the memory of {bad} ({self.runs[bad]["status"]})', bad
        if fits_hard:
            return 'feasible', f'{fits_hard} fitted and needs at least as much', fits_hard
        if said:
            return 'unsure', (f'by the declared moves it needs at least the memory of {said} '
                              f'({self.runs[said]["status"]}); a declaration does not rule it out'), said
        if fits:
            return 'feasible', f'{fits} fitted and needs at least as much', fits
        if near:
            return 'unsure', f'{near} ran out of memory; this differs from it by changes that may free enough', near
        return None, '', None

    def short(self, cid: str | None) -> float | None:
        """The least GiB the run was short by (its failed allocation minus the memory free then)."""
        r = self.runs.get(cid or '') or {}
        return (r.get('oom') or {}).get('short_gib')


# ---------------------------------------------------------------- the strategy
class HLS3(HLS2):
    """hls2 + the components of the module docstring. `off`: the components switched off (their ablate names); `facts`:
    the manifest's memory facts (ProjectHooks.memory_facts: card, GPUs, parameters, coupled knobs); `accounting`: the
    shared closed-form accounting (experience.runs.accounting_for: (config, step time) -> view), S-10's state arithmetic."""
    name = 'hls3'

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        """The proposer's features by the components on (proposer_flags): the shared ones (parse F14, self_proof F10,
        deferred F21, library_source D7) and hls3's own fields."""
        return proposer_flags(off)

    def __init__(self, proposer, *, off=(), facts: dict | None = None, accounting=None, stall_after: int = 3,
                 stall_steps: int = 2, stall_probes: int = 3, rebase_k: int = 2, coverage_when: str = 'any', **kw):
        super().__init__(proposer, **kw)
        off = set(off or ())
        self.on = {k: k not in off for k in HLS3_COMPONENTS}
        self.robust_stats = self.on['hls3_variance']        # Engine.run reads it (S-9)
        self.facts = dict(facts or {})
        self.accounting = accounting if self.on['hls3_prune'] else None
        self._states: dict[str, float | None] = {}          # S-10: configuration -> its training state (GiB per GPU)
        self.mem = MemoryModel(self.facts.get('cap_gib'))
        self.mem.floor = sharded_floor(self.facts) if self.on['hls3_prune'] else None
        self.stall_after, self.stall_steps, self.stall_probes, self.rebase_k = stall_after, stall_steps, stall_probes, rebase_k
        self.coverage_when = coverage_when                  # F2: 'budget' (more weak moves than runs left), 'weak'
        #                                                     (nothing strong left) or 'any' (two weak moves)
        self.failures: list[dict] = []                      # S-3: each failed run, classified
        self.together: list[dict] = []                      # S-3: changes that failed together; which ran separately since
        self.mem_bundles: list[dict] = []                   # S-2 / S-4: each memory bundle's composition and fate
        self.pids: dict[str, frozenset] = {'c0': frozenset()}   # candidate -> the patch ideas it carries
        self.idle, self.last_inc = 0, 'c0'                  # S-4 / S-6: ideas tested since the incumbent last changed
        self._kn: dict | None = None                        # the move directions of the current proposal round
        self.stalls: list[dict] = []                        # S-4 / S-6: what each stall step queued
        self.queue: list = []                               # S-4 / S-6: the stall step's proposals, before anything else
        self.held: dict[str, object] = {}                   # S-4: bundles waiting for their feasibility probe
        self.rebased: list[dict] = []                       # S-5
        self.rebased_moves: set = set()
        self.pruned: dict[str, str] = {}                    # S-10: 'knob=value' -> why
        self.variance: dict = {}                            # S-9: the reference's outlier updates, the variance moves
        self.phase_ranked: list[dict] = []                  # S-7
        self.counts = {'deferred': 0, 'kept_probes': 0, 'unsure': 0, 'feasible': 0, 'possible_thrash': 0}
        # FABLE_HLS3_FIXES_R1
        self.slow_starts: list[dict] = []                   # F1a: probes stopped at their start cap
        self.slow_moves: set = set()                        # F1a: moves (and patch ideas) whose probe was stopped
        self.gated: list[dict] = []                         # F1b: probes whose updates were GATE x slower than the base's
        self.coverage: list[dict] = []                      # F2: each coverage bundle and its fate
        self.need: dict[str, dict] = {}                     # F3: spender move -> need_lb, tries, unknown tried, last base
        self.completed: set = set()                         # F4: proposals completed with an invariant's other knobs
        self.suspects: dict[frozenset, dict] = {}           # F5: suspect pair -> the run that failed to start with it
        self.alone_fails: set = set()                       # F5: (knob, value) that failed to start alone
        self.ambiguous: dict[tuple, dict] = {}              # F5: (knob, value) kept runnable by the veto, its alone probe
        self.swaps: list[dict] = []                         # F5: swaps considered
        self.remainders: list[dict] = []                    # F6: the rests of partly taken entries carried
        self.warn_kept = 0                                  # F7: warned moves whose feasibility hls3's runs left alone
        self.fatal: list[dict] = []                         # F8: start crashes: fatal line, the new elements
        self.dead_elements: dict[tuple, str] = {}           # F8: element -> why it is ruled out
        self.inert: list[dict] = []                         # F8: patches with no measurable effect
        self.inert_sites: dict[str, str] = {}               # F8: code site -> the inert run
        self.started_lines: set = set()                     # F8: code lines of patches whose runs started
        self.continued: set = set()                         # F3: full continuations of passed feasibility probes
        # FABLE_HLS3_FIXES_R2
        self.gate_ratio = GATE_LOW if self.on['hls3_gate_low'] else GATE   # F12
        self.fear = {'tried': 0, 'fitted': 0, 'seen': set()}  # F11: this run's runs of declared spenders
        self.ranked: dict[str, float] = {}                  # F11: entry -> the p it was ranked with
        self.proof_fail: dict[str, dict] = {}               # F10: candidate -> why its probe did not prove the patch
        self.note_feedback: list[dict] = []                 # F10: notes still reported that no veto rests on
        # FABLE_HLS3_FIXES_R3
        self.seed_mu: dict[str, float] = {}                 # F15: seed entry -> the mu it counted as a spender at
        self.composites: list[dict] = []                    # F16-A: composites declared to run that died at their start
        self.pair_probes: list[dict] = []                   # F16: each pair probe (spender, partner, base, kind, fate)
        self.pair_seen: set = set()                         # F16: (spender move, partner move, base) queued
        self.pair_ran: set = set()                          # F16: (spender move, partner move) that ran
        self.captured: list[dict] = []                      # F17: coverage members resolved by their bundle
        self.jitter: list[dict] = []                        # F23: slower runs read by their update-interval scatter
        self.oom_runs: list[dict] = []                      # F22: each out-of-memory (or thrash) run: base, changes, need
        self.shared: dict[str, dict] = {}                   # F22: the element several of them shared -> its account
        self.carry_patches = self.on['hls3_carry_patches']  # F18 (HLS._carry_open reads it)
        self.from_store: list[dict] = []                    # F9b: earlier arms' start failures this one starts with
        self.lib_cache: dict[tuple, list | None] = {}       # D7: names -> the library excerpts read for them
        if self.on['hls3_writeback']:
            self._start_from_store()
        if hasattr(proposer, 'ask_proof'):                  # F10: the simulated model prints the two lines when asked
            proposer.ask_proof = self.on['hls3_self_proof']
        if hasattr(proposer, 'ask_defer'):                  # F21: ... and its third line
            proposer.ask_defer = self.on['hls3_self_proof'] and self.on['hls3_deferred']
        if hasattr(proposer, 'context') and any(self.on.values()):
            proposer.context = self.context                 # the proposer's extra input (failures, memory, variance)

    # ------------------------------------------------------------ report
    def describe(self) -> dict:
        out = super().describe()
        if not any(self.on.values()):
            return out
        off = sorted(k for k, v in self.on.items() if not v)
        out['hls3'] = {
            'off': off, 'memory': {'capacity_gib': self.mem.cap, 'floor': self.mem.floor, 'runs': {
                cid: {k: r[k] for k in ('status', 'peak') if r.get(k) is not None} | (
                    {'short_gib': r['oom'].get('short_gib'), 'need_gib': r['oom'].get('need_gib')} if r.get('oom') else {})
                for cid, r in self.mem.runs.items()},
                'measured_moves': [{'move': self._words(s), 'direction': d} for s, d in self.mem.measured.items()]},
            'counts': dict(self.counts), 'failures': self.failures, 'failed_together': [
                {**{k: t[k] for k in ('candidate', 'kind')}, 'members': [self._words(m) for m in t['members']],
                 'separately': {self._words(m): v for m, v in t['separately'].items()}} for t in self.together],
            'bundles': self.mem_bundles, 'stalls': self.stalls, 'rebased': self.rebased, 'pruned': dict(self.pruned),
            'variance': self.variance, 'phase_ranked': self.phase_ranked}
        r1 = {'start_slow': self.slow_starts if self.on['hls3_start_cap'] else None,
              'gated': self.gated if self.on['hls3_probe_gate'] else None,
              'coverage': self.coverage if self.on['hls3_coverage'] else None,
              'need': {k: {kk: v[kk] for kk in ('need_lb', 'probes', 'unknown')} for k, v in self.need.items()}
              if self.on['hls3_need'] else None,
              'completed': sorted(self.completed) if self.on['hls3_complete'] else None,
              'suspect_pairs': [{'pair': [self._words(frozenset((k, None, v) for k, v in m)) for m in sorted(pr, key=sorted)],
                                 **v} for pr, v in self.suspects.items()] if self.on['hls3_pairs'] else None,
              'ambiguous': [{'knob': a['knob'], 'value': a['value'], 'probe': a['probe']} for a in self.ambiguous.values()]
              if self.on['hls3_pairs'] else None,
              'swaps': self.swaps if self.on['hls3_pairs'] else None,
              'remainders': self.remainders if self.on['hls3_remainder'] else None,
              'warn_kept': self.warn_kept if self.on['hls3_warn'] else None,
              'dead_elements': {' '.join(str(x) for x in e): w for e, w in self.dead_elements.items()}
              if self.on['hls3_dead_patch'] else None,
              'inert': self.inert if self.on['hls3_dead_patch'] else None}
        r1 = {k: v for k, v in r1.items() if v is not None}
        if r1:
            out['hls3']['r1'] = r1
        ignored = self._ignored_declarations()
        r2 = {'inert_at_start': [x for x in self.inert if x.get('at') == 'probe'] if self.on['hls3_self_proof'] else None,
              'declarations_ignored': ignored or None,
              'note_feedback': self.note_feedback[-10:] if self.on['hls3_self_proof'] and self.note_feedback else None,
              'fear_prior': {'tried': self.fear['tried'], 'fitted': self.fear['fitted'], 'pi': round(self._pi(), 3),
                             'ranked': dict(self.ranked)} if self.on['hls3_fear_prior'] else None,
              'gate': self.gate_ratio if self.on['hls3_gate_low'] else None}
        r2 = {k: v for k, v in r2.items() if v is not None}
        if r2:
            out['hls3']['r2'] = r2
        r3 = {'pair_set': self._pair_set() if self.on['hls3_pair_set'] and self._pair_set() != PAIR_SET else None,
              'seed_mu': dict(self.seed_mu) if self.on['hls3_seed_mu'] and self.seed_mu else None,
              'composites': [{'candidate': c['candidate'], 'kind': c['kind'], 'origin': c['origin'],
                              'members': [self._words(frozenset((k, None, v) for k, v in m)) for m in c['atoms']]}
                             for c in self.composites] if self.on['hls3_pair_probes'] and self.composites else None,
              'pair_probes': self.pair_probes if self.on['hls3_pair_probes'] and self.pair_probes else None,
              'captured': self.captured if self.on['hls3_cov_unresolved'] and self.captured else None,
              'carried_patches': self.carried_patches if self.on['hls3_carry_patches'] and self.carried_patches else None,
              'jitter': self.jitter if self.on['hls3_jitter_thrash'] and self.jitter else None,
              'dropped_hypotheses': list(getattr(self.proposer, 'dropped', None) or []) or None
              if self.on['hls3_parse'] else None,
              'shared_spenders': {k: {kk: v[kk] for kk in ('need_lb', 'probes', 'runs')} for k, v in self.shared.items()}
              if self.on['hls3_shared_spender'] and self.shared else None}
        r3 = {k: v for k, v in r3.items() if v is not None}
        if self.on['hls3_writeback'] and self.from_store:
            r3['from_store'] = self.from_store
        if r3:
            out['hls3']['r3'] = r3
        learned = self._learned() if self.on['hls3_writeback'] else None
        if learned:
            out['learned'] = learned                        # F9b (D4): runs.write_back writes it as constraints
        return out

    @staticmethod
    def _words(move) -> str:
        return ', '.join(f'{k}={json.loads(b) if b != "null" else None}' if not str(k).startswith('patch:') else k
                         for k, *_, b in sorted(move))

    def _row(self, h: dict, *, trace: bool) -> dict:
        r = super()._row(h, trace=trace)
        for k in ('deferred', 'failure', 'memory', 'site', 'rebased_from', 'separately_of', 'patch_lines',
                  'gain_pct_per_line'):
            if h.get(k) and (trace or k in ('deferred', 'failure', 'memory')):
                r[k] = h[k]
        if h.get('mem_dir') and self.on['hls3_memory_schema']:
            r['declared_memory'] = {'direction': h['mem_dir'], **({'gib': h['mem_gib']} if h.get('mem_gib') is not None else {})}
        return r

    def _view_status(self, h: dict, st: str) -> str:
        return 'deferred' if st == 'queued' and h.get('deferred') else st

    # ------------------------------------------------------------ the proposer's input
    def context(self, engine, base) -> dict:
        """The extra fields of the hypothesis prompt (LLMProposer.context): failures, the antagonism table, the memory
        view, the reference's outlier updates; empty when their components are off (the incumbent's own phases reach the
        model through its receipt digest already)."""
        out = {}
        if self.on['hls3_failures'] and self.failures:
            out['failures'] = [{k: f[k] for k in ('candidate', 'base', 'change', 'class', 'detail') if f.get(k) is not None}
                               for f in self.failures[-10:]]
        if self.on['hls3_antagonism'] and self.together:
            out['failed_together'] = [{'changes': [self._words(m) for m in t['members']], 'class': t['kind'],
                                       'run': t['candidate'], 'separately': {self._words(m): v for m, v in t['separately'].items()}}
                                      for t in self.together[-6:]]
        if self.on['hls3_feasibility'] or self.on['hls3_prune']:
            view = self._memory_view(engine, base)
            if view:
                out['memory_view'] = view
        v = self.variance.get('reference') or {}
        if self.on['hls3_variance'] and v.get('outliers'):
            out['variance'] = {k: v[k] for k in ('mean_over_median_pct', 'stalls', 'updates', 'stall_factor')}
        if (self.on['hls3_dead_patch'] or self.on['hls3_self_proof']) and self.inert:
            out['inert_patches'] = self.inert[-6:]
        if self.on['hls3_self_proof'] and self.note_feedback:
            out['patch_notes'] = self.note_feedback[-6:]      # F10: notes still reported after a patch (no veto)
        ignored = self._ignored_declarations()
        if ignored:
            out['declarations_ignored'] = ignored[-6:]        # F10 / F12: unsupported values, not honoured
        if self.on['hls3_library_source']:                    # D7: read-only library text (the code prompts only;
            out['library_source'] = self._library_source(engine, base)   # None: none - the proposer reads no other)
        # (F14: the hypotheses of its answers not used, and why, the proposer adds itself - its `parse` feature)
        if self.on['hls3_coverage'] and any(r.get('status') for r in self.coverage):
            out['coverage'] = [{k: r[k] for k in ('members', 'status', 'candidate', 'gain_pct') if r.get(k) is not None}
                               for r in self.coverage[-4:] if r.get('status')]
        return out

    def _memory_view(self, engine, base) -> dict:
        ref = self._mcfg(engine, 'c0')
        runs = []
        for cid, r in list(self.mem.runs.items())[-10:]:
            row = {'run': cid, 'status': r['status'], 'change_vs_reference': {
                k: (None if b == 'null' else json.loads(b)) for k, _, b in sorted(sig(ref, r['cfg']))}}
            if r.get('peak') is not None:
                row['peak_gib'] = r['peak']
            if r.get('oom'):
                row.update({k: r['oom'][k] for k in ('tried_gib', 'free_gib', 'short_gib', 'need_gib') if k in r['oom']})
            runs.append(row)
        out = {}
        if self.mem.cap:
            out['capacity_gib'] = self.mem.cap
        if self.mem.floor:
            out['sharded_state_floor_gib'] = self.mem.floor['floor_gib']
            out['floor_basis'] = self.mem.floor['basis']
        if runs:
            out['runs'] = runs
        deferred = [{'change': h.get('change'), 'why': h['deferred']} for h in self.ledger.values()
                    if h['base'] == base and h['status'] == 'open' and h.get('deferred')]
        if deferred:
            out['deferred'] = deferred[:10]
        if self.pruned:
            out['pruned'] = [f'{k}: {v}' for k, v in list(self.pruned.items())[:8]]
        return out

    # ------------------------------------------------------------ configurations as the memory model reads them
    def _native(self, engine) -> bool:
        return engine.s.hooks is None

    def _pids(self, engine, cid) -> frozenset:
        """The patch ideas a candidate carries: its own record, else its parent's (a config change keeps the code)."""
        seen = set()
        while cid is not None and cid not in self.pids and cid not in seen:
            seen.add(cid)
            c = engine.s.candidates.get(cid)
            cid = getattr(c, 'parent', None) if c is not None else None
        return self.pids.get(cid, frozenset()) if cid is not None else frozenset()

    def _mcfg(self, engine, cid, change=None, pids=()) -> dict:
        c = engine.s.candidates[cid]
        native = self._native(engine)
        out = flat(after(c.config, change, native) if change else c.config, native)
        out.update({f'patch:{p}': True for p in sorted(self._pids(engine, cid) | set(pids))})
        return out

    def _patch_ideas(self, p) -> set:
        """The ideas whose code a proposal carries: a bundle's members with a patch, else its own entry's."""
        if not p.patch:
            return set()
        keys = self.members.get(p.hypothesis or '', []) if str(p.hypothesis or '').startswith('bundle|') else [p.hypothesis]
        return {self.ledger[k]['mechanism'] for k in keys if k in self.ledger and self.ledger[k].get('patch')} or {
            p.mechanism or 'patch'}

    def _known(self, engine) -> dict:
        """Move -> direction: the proposer's declarations (each entry's change from its base; a patch idea as its pseudo-
        knob), the runs' over them (computed once per proposal round)."""
        if self._kn is not None:
            return self._kn
        out = {}
        if self.on['hls3_memory_schema']:
            for h in self.ledger.values():
                d = h.get('mem_dir')
                if d not in DIRS or h['base'] not in engine.s.candidates:
                    continue
                frm = self._mcfg(engine, h['base'])
                to = self._mcfg(engine, h['base'], h.get('change'), {h['mechanism']} if h.get('patch') else ())
                s = sig(frm, to)
                if s:
                    out[s] = d
            if self.on['hls3_need']:
                out.update({s: d for s, d in self._derived(out).items() if s not in out})
        out.update(self.mem.measured)
        return out

    def _suspect_spend(self, triple, known: dict) -> bool:
        """F3c: a change of unknown effect that a declared spending combination holds while no other change of it spends
        by the declarations: possibly that combination's spender - not taken as an enabler of unknown effect."""
        for s, d in known.items():
            if d == 'spends' and len(s) > 1 and triple in s and not any(
                    self.mem._knob(k, a, b2, known) == 'spends' for k, a, b2 in s if (k, a, b2) != triple):
                return True
        return False

    def _derived(self, known: dict) -> dict:
        """F3c: single-knob directions the declarations imply: a declared combination that spends (or frees) whose every
        other change frees (spends) or is neutral by the declarations makes its one remaining change spend (free) - the
        same knob move however the entries grouped it (a seed's direction from the proposer's combined entry)."""
        out = {}
        for s, d in known.items():
            if len(s) < 2 or d == 'neutral':
                continue
            unk = []
            for k, a, b2 in s:
                dd = self.mem._knob(k, a, b2, known)
                if dd is None:
                    unk.append((k, a, b2))
                elif dd == d:
                    break                                   # another part goes the same way: nothing follows
            else:
                if len(unk) == 1:
                    out.setdefault(frozenset(unk), d)
        return out

    def _predict(self, engine, cfg: dict) -> tuple:
        """The memory model's verdict on a configuration (MemoryModel.predict), the reference as the start that fitted:
        the declarations count, but only the measured moves can rule it out."""
        return self.mem.predict(cfg, self._known(engine), self._mcfg(engine, 'c0'), hard=self.mem.measured)

    def _declared(self, k: str, v) -> tuple[str | None, float | None]:
        """The latest declaration of the single-knob move k = v (direction, GiB), its GiB raised to what a run showed it
        needs at least (`mem_gib_bound`, an out-of-memory run on a base whose peak was measured)."""
        d = g = None
        for h in self.ledger.values():
            ch = h.get('change') or {}
            if not h.get('patch') and h.get('mem_dir') in DIRS and len(ch) == 1 and k in ch and _v(ch[k]) == _v(v):
                d, g = h['mem_dir'], h.get('mem_gib')
                if d == 'spends' and isinstance(h.get('mem_gib_bound'), (int, float)):
                    g = max(g or 0.0, h['mem_gib_bound'])
        return d, g

    # ------------------------------------------------------------ proposals
    def _screens_scarce(self, engine, cap: float | None = None, steps=None) -> bool:
        """The screens are the scarcer budget: the GPU-hours left, less this probe, would still pay every screen left -
        a full run each, the last one admitted as the session admits it (its screen cost x (1 + scoring_reserve_screens),
        SearchLedger.admit_screen) - so a probe that saves a screen takes nothing a screen needs (without a GPU-hour cap:
        always). (A comparison of the shares left turned the kept probe off from the first probe until screens caught
        up: on the memory toy the out-of-memory screens hls3 spent came from probes it had stopped keeping; without the
        scoring reserve a kept probe could cost the last admissible screen, Codex HLS3 r2.) With the start cap (F1c) the
        probe is priced at its cap (GPUs x cap), not at the mean probe, and the wall left must pay the same."""
        b, led = engine.s.budget, engine.s.ledger
        u = led.used
        sb, gb = getattr(b, 'screens', None), getattr(b, 'gpu_hours', None)
        if not sb:
            return False
        if not gb:
            return True
        left = sb - u.get('screens', 0)
        if left <= 0:
            return False
        c = engine.costs()
        mean = getattr(led, 'mean_screen', None)
        est = (mean() if callable(mean) else None) or getattr(led, 'prior', None)
        # The screen cost admission prices (GPU-hours): the mean of every counted screen, cheap failed ones included, so
        # the full runs still to come raise it towards the full cost; the larger of the two bounds it (Codex HLS3 r3).
        adm = max(est[0] if est else c['full'], c['full'])
        reserve = getattr(b, 'scoring_reserve_screens', 0.0) or 0.0
        probe = c['probe'] + self._extra_probe(engine, steps)[0]
        world = self._world(engine) if cap and self.on['hls3_start_cap'] else None
        if world:
            probe = max(probe, world * cap / 3600.0)
        if gb - u.get('gpu_hours', 0.0) - probe < (left - 1) * c['full'] + adm * (1 + reserve):
            return False
        wb, wall = getattr(b, 'wall_seconds', None), getattr(led, 'wall', None)
        full = self._wall_full(engine) if self.on['hls3_start_cap'] and wb and callable(wall) else None
        if not full:
            return True
        w_adm = max(est[1] if est and len(est) > 1 and est[1] else full, full)
        return wb - wall() - (cap or 0.0) >= (left - 1) * full + w_adm * (1 + reserve)

    def _probe_pays(self, engine, q, cap: float | None = None, steps=None) -> bool:
        """F1c: a start-risky move's probe pays for itself - the full run it saves when the move does not start,
        (1 - p_feasible) x a full run's GPU-hours, is at least the probe's (at its cap when it has one). It keeps a probe
        only where keeping it costs no admissible screen (_costs_no_screen: affordability is a veto)."""
        if not self.on['hls3_start_cap'] or q.p_feasible >= self.probe_below:
            return False
        c = engine.costs()
        probe = c['probe'] + self._extra_probe(engine, steps)[0]
        world = self._world(engine) if cap else None
        if world:
            probe = max(probe, world * cap / 3600.0)
        return (1.0 - q.p_feasible) * c['full'] >= probe

    def _costs_no_screen(self, engine, cap: float | None = None, steps=None) -> bool:
        """F1c (Codex HLS4 r1 P1-4): keeping this probe never costs a screen the budget would admit - the full screens
        the session admits after it (SearchLedger.admit_screen: each needs the mean screen cost x (1 + the scoring
        reserve) left, in GPU-hours and wall; each adds a full run's cost to that mean) are as many as without it."""
        c = engine.costs()
        xg, xw = self._extra_probe(engine, steps)
        probe_g, probe_w = c['probe'] + xg, None
        world = self._world(engine) if cap else None
        if world:
            probe_g, probe_w = max(probe_g, world * cap / 3600.0), cap
        if probe_w is None:
            full_w = self._wall_full(engine)
            probe_w = (full_w * c['probe'] / c['full'] if full_w and c['full'] else 0.0) + xw
        led = engine.s.ledger
        g0 = led.used.get('gpu_hours', 0.0)
        w0 = led.wall() if callable(getattr(led, 'wall', None)) else 0.0
        return self._admits(engine, g0 + probe_g, w0 + probe_w) >= self._admits(engine, g0, w0)

    def _admits(self, engine, gpu0: float, wall0: float) -> int:
        """How many more full screens the session would admit from these GPU-hours and wall spent (SearchLedger.
        admit_screen), the screens left at most."""
        b, led = engine.s.budget, engine.s.ledger
        costs = list(getattr(led, 'screen_costs', None) or [])
        prior = getattr(led, 'prior', None)
        left = int((getattr(b, 'screens', None) or 0) - led.used.get('screens', 0))
        full_g, full_w = engine.costs()['full'], self._wall_full(engine) or 0.0
        reserve = getattr(b, 'scoring_reserve_screens', 0.0) or 0.0
        gb, wb = getattr(b, 'gpu_hours', None), getattr(b, 'wall_seconds', None)
        g, w, n = gpu0, wall0, 0
        while n < min(left, 1000):
            mg, mw = ((sum(x[0] for x in costs) / len(costs), sum(x[1] for x in costs) / len(costs)) if costs else
                      prior if prior else (full_g, full_w))
            if gb is not None and g + mg * (1 + reserve) > gb or wb is not None and w + mw * (1 + reserve) > wb:
                break
            g, w, n = g + full_g, w + full_w, n + 1
            costs.append((full_g, full_w))
        return n

    def _adjust(self, engine, inc, props: list) -> list:
        if not any(self.on.values()):
            return props
        if self.on['hls3_pairs'] and hasattr(engine.s, 'start_failure_limit'):
            engine.s.start_failure_veto = self._veto        # F5b: C5 asks before it rules a value out
        self.queue = [q for q in self.queue if q.base == inc or str(q.hypothesis or '').startswith('alone|')]
        if self.on['hls3_pairs']:
            self.queue = self._alone_probes(engine) + self.queue
        if self.queue:                                      # a stall step runs before anything else (S-4 / S-6)
            return list(self.queue)
        self._kn = None
        self._kn = self._known(engine)
        try:
            out = [q for q in (self._screen(engine, inc, p) for p in props) if q is not None]
            if self.on['hls3_bundles']:
                out += [q for q in (self._screen(engine, inc, b) for b in self._spender_bundles(engine, inc))
                        if q is not None]
            if self.on['hls3_coverage']:                    # F2: one bundle, not a screen each
                cov = next((x for x in out if str(x.origin) == 'hls3-coverage'), None)
                if cov is None and not any(str(x.origin) == 'hls3-coverage' for x in props):
                    cov = self._coverage_bundle(engine, inc)
                    cov = self._screen(engine, inc, cov) if cov is not None else None
                    if cov is not None:
                        out.append(cov)
                inside = set(self.members.get(cov.hypothesis) or ()) if cov is not None else set()
                kept = [x for x in out if x.hypothesis not in inside]
                value, cost, stops = (getattr(engine, n, None) for n in ('value', 'expected_cost', 'stops_at'))
                if kept and len(kept) < len(out) and value and cost and stops and \
                        stops(max(value(x) / cost(x) for x in kept)) and not stops(max(value(x) / cost(x) for x in out)):
                    kept = out                              # withheld, they would end the search: offered too (C3c's rule)
                out = kept
            if self.idle >= self.stall_after and len(self.stalls) < self.stall_steps:
                steps = self._stall(engine, inc)
                value, cost, stops = (getattr(engine, n, None) for n in ('value', 'expected_cost', 'stops_at'))
                if steps and value and cost and stops and stops(max(value(p) / cost(p) for p in steps)):
                    return out + steps                  # alone they would end the search (the stop rule): they compete
                if steps:
                    self.queue = steps
                    return list(steps)
            return out
        finally:
            self._kn = None

    def _screen(self, engine, inc, p):
        """S-3 antagonism, S-10 pruning, S-1 feasibility, the kept probe and S-9's statistic for one proposal; None
        when it is not proposed now."""
        h = self.ledger.get(p.hypothesis or '')
        base = p.base or inc
        if self.on['hls3_complete'] and p.change:           # F4: an invariant's other knobs completed
            ch = self._complete(engine, base, p.change)
            if ch is None:
                if h is not None and h['status'] == 'open':
                    h.update(status='invalid', why='no value of the knobs a workload invariant ties to it keeps the '
                                                   'invariant (the validator)')
                return None
            if ch != p.change:
                dup = next((k for k, e in self.ledger.items() if h is not None and k != h['key'] and e['base'] == base
                            and e['status'] in ('open', 'unresolved') and not e.get('patch')
                            and _mkey(e.get('completed') or e.get('change')) == _mkey(ch)), None)
                if dup is not None and not str(p.hypothesis).startswith('bundle|'):
                    if h['status'] == 'open':               # (completed, it is another entry's move: that one runs)
                        h.update(status='superseded', why=f'completed by the workload invariant, it is the move of {dup}')
                    return None
                self.completed.add(p.hypothesis or p.mechanism)
                p = dataclasses.replace(p, change=ch)
                if h is not None and not str(p.hypothesis).startswith('bundle|'):
                    h['completed'] = ch
        if self.on['hls3_need'] and p.change and not str(p.origin).startswith(('hls3-swap', 'hls3-alone')) \
                and p.hypothesis not in self.continued:     # (Codex HLS4 r1 P1-3: a passed probe's full run)
            why = self._capped(engine, base, p.change)      # F3: a spender tried SPENDER_PROBES times waits
            if why:
                self._defer(h, why)
                return None
        if self.on['hls3_dead_patch']:                      # F8a: a new element two start crashes shared
            why = self._element_dead(engine, base, h if h is not None else {'change': p.change}, patch=p.patch)
            if why:
                if h is not None and h['status'] == 'open':
                    h.update(status='invalid', why=f'unrunnable: {why}')
                return None
        cfg = self._mcfg(engine, base, p.change, self._patch_ideas(p))
        if self.on['hls3_antagonism']:
            t = self._blocked(cfg)
            if t is not None:
                self._defer(h, 'failed together with ' + '; '.join(self._words(m) for m in t['members'])
                            + f' in {t["candidate"]}: each runs separately first')
                self._separately(engine, inc, t)
                return None
        pair_soft = None
        if self.on['hls3_pairs']:                           # F5a: a pair that failed to start together (after S-3's separate runs)
            pb = self._pair_block(cfg)
            if pb is not None:
                words = ' and '.join(self._words(frozenset((k, None, v) for k, v in m)) for m in pb['members'])
                if not str(p.origin).startswith('hls3-swap'):
                    q = self._swap(engine, inc, p, h, pb)
                    if q is not None:
                        self._defer(h, f'it holds {words}, which failed to start together in {pb["candidate"]}: tried as a '
                                       'swap')
                        return self._screen(engine, inc, q)
                if pb['strikes'] >= 2:                      # a second failed start that held it: deferred
                    self._defer(h, f'it holds {words}, which failed to start together {pb["strikes"]} times (first in '
                                   f'{pb["candidate"]}) and never started together')
                    return None
                pair_soft = f'it holds {words}, which failed to start together in {pb["candidate"]}: probed first'
        soft = [pair_soft] if pair_soft else []             # why a probe first: estimates, not evidence that it cannot fit
        if self.on['hls3_prune']:
            why = self._prune_why(engine, base, p.change)
            if why:
                if h is not None and h['status'] == 'open':
                    h.update(status='invalid', why=f'unrunnable on this card: {why}')
                return None
            why = self._floor_flag(engine, base, p.change)
            if why:
                soft.append(why)
        q, verdict, why = p, None, ''
        if self.on['hls3_feasibility']:
            verdict, why, src = self._predict(engine, cfg)
            if verdict == 'infeasible':
                self._defer(h, f'predicted out of memory: {why}', short=self.mem.short(src))
                self.counts['deferred'] += 1
                return None
            if h is not None:
                h.pop('deferred', None)
            if verdict == 'feasible':                       # a run that fitted needs as much: no estimate overrides it
                soft = []
                self.counts['feasible'] += 1
            elif verdict == 'unsure':
                soft.append(why)
        if h is not None and (soft or verdict):
            h['memory'] = 'unsure: ' + '; '.join(soft) if soft else f'{verdict}: {why}'
        if soft and tuple(q.fidelity) != ('probe',):
            self.counts['unsure'] += 1
            q = dataclasses.replace(q, p_feasible=min(q.p_feasible, 0.4), fidelity=('probe', 'full'))
        cap, slow = None, False
        sp = self.mem.split(self._mcfg(engine, base), cfg, self._known(engine))
        ps = self._probe_steps(engine, h, bool(p.patch), bool(sp['spends'] or sp['unknown']))   # F12 / F10: a gated
        #                                                   probe's updates (None: the engine's)
        if self.on['hls3_start_cap'] and 'probe' in q.fidelity:
            if self._slow_move(engine, base, p, h):         # its probe was stopped at the start cap: in full, rarely
                slow = True
                q = dataclasses.replace(q, fidelity=('full',), p_feasible=min(q.p_feasible, 0.3), keep_probe=False,
                                        retry=bool(h is not None and h.get('retry_full')))
            else:
                cap = self._probe_cap(engine, base, ps)
                if cap:
                    q = dataclasses.replace(q, probe_cap=cap)
        if self.on['hls3_feasibility'] and 'probe' in q.fidelity and tuple(q.fidelity) != ('probe',) \
                and not q.keep_probe and (self._screens_scarce(engine, cap, ps) or self._probe_pays(engine, q, cap, ps)
                                          and self._costs_no_screen(engine, cap, ps)):
            if not self.on['hls3_start_cap'] or soft or self._spends(engine, base, cfg) or p.p_feasible < self.probe_below:
                q = dataclasses.replace(q, keep_probe=True)      # (F1c: an unsure, memory-spending or start-risky move;
                self.counts['kept_probes'] += 1                  # the start cap bounds probes only, D1)
        if self.on['hls3_self_proof'] and p.patch and not slow and not q.keep_probe:
            pc = cap if 'probe' in q.fidelity else self._probe_cap(engine, base, ps) if self.on['hls3_start_cap'] else None
            if self._screens_scarce(engine, pc, ps) or self._costs_no_screen(engine, pc, ps):   # F10: a patch's proof is read
                if tuple(q.fidelity) == ('full',):          # at its probe - one that costs no admissible screen
                    q = dataclasses.replace(q, fidelity=('probe', 'full'), probe_cap=pc or None)
                q = dataclasses.replace(q, keep_probe=True)
        if 'probe' in q.fidelity and tuple(q.fidelity) != ('probe',) and (
                self.on['hls3_probe_gate'] or self.on['hls3_self_proof'] and p.patch):
            if self.on['hls3_gate_low'] or self.on['hls3_self_proof']:
                q = dataclasses.replace(q, probe_steps=ps, gate=functools.partial(
                    self._gate, ideas=frozenset(self._patch_ideas(p)) if self.on['hls3_gate_low'] else frozenset(),
                    claim=self._claim(engine, base, p, h) if self.on['hls3_self_proof'] and p.patch else None,
                    h=h if p.patch else None, warmup=self._supported(h, 'warmup_updates')))
            else:
                q = dataclasses.replace(q, gate=self._gate)
        inert = next((self.inert_sites[k] for k in sorted(site_keys(h.get('site'))) if k in self.inert_sites), None) \
            if self.on['hls3_dead_patch'] and p.patch and h is not None and h.get('site') else None
        if inert is not None and not self._self_check(p.patch):
            q = dataclasses.replace(q, mu=min(q.mu, -engine.sigma_pair()))   # F8b: after the other moves
            h['inert_site'] = inert
        if self.on['hls3_variance'] and h is not None and h.get('variance'):
            q = dataclasses.replace(q, stat='mean')
        if self.on['hls3_fear_prior'] and self.on['hls3_feasibility'] and h is not None and not p.patch \
                and h.get('mem_dir') == 'spends' and not soft and not slow and verdict is None \
                and h['p_feasible'] < self.probe_below:
            pr = self._p_rank(engine, base, h)              # F11: no run bounds it: this run's evidence ranks it
            if pr > q.p_feasible:
                self.ranked[h['key']] = round(pr, 3)
                q = dataclasses.replace(q, p_feasible=pr)   # (the ranking only: the entry keeps the proposer's p)
        return q

    # ------------------------------------------------------------ F10 / F11
    def _claim(self, engine, base, p, h) -> str:
        """F10a: what a patch says it fixes (proof.patch_claim)."""
        return patch_claim(self._files(engine, base), p.patch, h)

    @staticmethod
    def _events(engine, cid) -> dict:
        return events_of(engine, cid)

    def _notes(self, engine, cid) -> list:
        """The job-log notes of a candidate's run (its receipt), else the reference's."""
        return notes_of(engine, cid)

    def _supported(self, h, key: str) -> int | None:
        """A declaration hls3 honours - `warmup_updates` 1..WARMUP_MAX, `active_from_update` 1..PROBE_STEPS_MAX - else
        None: a value out of range is absent, never an exemption from a gate (Codex HLS4R2 r2)."""
        hi = {'warmup_updates': WARMUP_MAX, 'active_from_update': PROBE_STEPS_MAX}[key]
        v = (h or {}).get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= hi else None

    def _ignored_declarations(self) -> list:
        """F10 / F12: the proposer's declarations hls3 did not honour (out of their range), for its feedback."""
        if not (self.on['hls3_self_proof'] or self.on['hls3_gate_low']):
            return []
        return [{'mechanism': h['mechanism'], 'ignored': list(h['rejected'])} for h in self.ledger.values()
                if h.get('rejected')]

    @staticmethod
    def _proof_of(ev: dict, whole: bool = False) -> dict:
        """A run's per-site proof evidence (proof.proof_of; F13 `whole`: its lines read whole)."""
        return proof_of(ev, whole)

    def _proof_reason(self, engine, base, row, claim: str | None, h=None, feedback=None
                      ) -> tuple[str | None, str | None]:
        """F10: (why the patch did not prove at its probe that it took effect, the note it targets) - proof.proof_reason,
        F13 (`whole`) and F21 (`deferred`) by their components."""
        return proof_reason(row, self._events(engine, base), self._notes(engine, base), claim, h, feedback,
                            whole=self.on['hls3_scan_whole'], deferred=self.on['hls3_deferred'])

    def _note_reason(self, cur: dict, old: dict, n: str) -> str:
        """F10 / F21: why a note still reported after the install means inert (proof.note_reason)."""
        return note_reason(cur, old, n, deferred=self.on['hls3_deferred'])

    @staticmethod
    def _overlong(proof: dict) -> str:
        return overlong(proof)

    def _parsed(self, engine, base, row) -> dict:
        """F13: what the scanner read of a probe's proof lines (proof.parsed)."""
        return parsed(proof_of(row, True), proof_of(self._events(engine, base), True))

    def _pi(self) -> float:
        """F11: the share of this run's feared moves that fitted, Beta(1, 1): (fitted + 1) / (tried + 2)."""
        return (self.fear['fitted'] + 1) / (self.fear['tried'] + 2)

    def _p_rank(self, engine, base, h) -> float:
        """F11: max(every feasibility this run declared for this move - its entries, the transfer judgment's for the same
        change (not one an invariant's knobs completed into it: that was said of another configuration) -, pi); pi only
        once a feared move of this run has run, fitted or failed (before, its Beta(1, 1) mean is the prior alone)."""
        m = _mkey(h.get('change'))
        decl = [float(e['p_feasible']) for e in self.ledger.values() if not e.get('patch') and
                _mkey(e.get('change')) == m and isinstance(e.get('p_feasible'), (int, float))]
        decl += [float(pj) for jm, (_, pj) in self._judgments().items() if jm == m and isinstance(pj, (int, float))]
        return max(decl + ([self._pi()] if self.fear['tried'] else []))

    def _fear_count(self, engine, p, out, kind, base) -> None:
        """F11: a run of a feared move - declared to spend memory at a feasibility below the probe rule, the moves F11
        lifts (its entry, or a member of its bundle) - once per candidate: fitted when it ran (a passed probe, a screen, an
        abort after updates), not when it ran out of memory or thrashed; other failures say nothing about memory. (A
        confident spender's fit says nothing about the fear: counted, it lifted the feared moves of the memory toy, which
        ran out of memory.)"""
        if out.cid is None or out.cid in self.fear['seen'] or kind not in ('ok', 'slower', 'numerics', 'oom', 'thrash'):
            return
        hyp = str(p.hypothesis or '')
        hyp = hyp[len('probe|'):] if hyp.startswith('probe|') else hyp     # (a stall probe: its bundle's members)
        keys = self.members.get(hyp, []) if hyp.startswith('bundle|') else [hyp]
        if not any(self._feared(self.ledger.get(k)) for k in keys):
            return
        self.fear['seen'].add(out.cid)
        self.fear['tried'] += 1
        self.fear['fitted'] += kind not in ('oom', 'thrash')

    def _feared(self, e: dict | None) -> bool:
        """F11: an entry declared to spend memory at a feasibility below the probe rule (as the proposer declared it)."""
        return bool(e) and e.get('mem_dir') == 'spends' and isinstance(e.get('p_feasible'), (int, float)) \
            and e['p_feasible'] < self.probe_below

    def _defer(self, h: dict | None, why: str, short: float | None = None) -> None:
        if h is not None and h['status'] == 'open':
            h['deferred'] = why
            if short is not None:
                h['short_gib'] = short

    # ------------------------------------------------------------ S-10
    def _prune_why(self, engine, base, change) -> str | None:
        """S-10's hard rule, for one whole configuration: its training state alone - weights, gradients, optimizer
        states and the parameters it keeps between forward and backward, by the shared accounting of that configuration
        (the manifest's sharding arithmetic) - is above the card, so nothing that frees activations makes it fit. Only
        while that accounting agrees with every run that fitted (a run whose state term alone is above the card ran: the
        estimate is wrong for this job, as the closed form's 510 GiB per GPU of a 77.3 GiB card was); a declaration never
        prunes, and nothing is pruned for good: each configuration is judged on its own."""
        cap = self.mem.cap
        if self.accounting is None or not cap:
            return None
        native = self._native(engine)
        cur = engine.s.candidates[base].config
        cfg = after(cur, change, native)
        st = self._state_gib(cfg)
        if st is None or st <= 1.02 * cap or self._accounting_contradicted(engine, cap):
            return None
        key = self._words(sig(flat(cur, native), flat(cfg, native))) or base
        self.pruned[key] = (f'its training state alone is {st:.0f} GiB per GPU by the shared accounting of this '
                            f'configuration (which every run that fitted agrees with): above the {cap:.0f} GiB card, '
                            'whatever frees activations')
        return f'{key}: {self.pruned[key]}'

    def _state_gib(self, cfg: dict) -> float | None:
        """The shared accounting's training state of a configuration (GiB per GPU), None without one."""
        key = _v(cfg)
        if key not in self._states:
            try:
                a = self.accounting(cfg, None) if self.accounting is not None else None
            except Exception:                               # noqa: BLE001 - an estimate never fails the search
                a = None
            v = ((a or {}).get('terms_gib') or {}).get('weights_grads_optimizer') if isinstance(a, dict) else None
            self._states[key] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        return self._states[key]

    def _accounting_contradicted(self, engine, cap: float) -> str | None:
        """A run that fitted although the accounting puts its training state alone above the card (its id), else None."""
        for cid in self._fitted(engine):
            st = self._state_gib(engine.s.candidates[cid].config)
            if st is not None and st > 1.02 * cap:
                return cid
        return None

    def _fitted(self, engine) -> list[str]:
        """The candidates whose run fitted: the reference, every valid run, the runs the memory model holds as fitting."""
        out = ['c0'] + [cid for cid, c in engine.s.candidates.items() if cid != 'c0' and (c.verdict or {}).get('valid')]
        return out + [cid for cid, r in self.mem.runs.items() if r['status'] == 'ok' and cid not in out
                      and cid in engine.s.candidates]

    def _floor_flag(self, engine, base, change) -> str | None:
        """S-10's estimate: a changed knob value whose declared spend (raised to what a run showed it needs) puts the
        sharded-state floor above the card is probed first - never ruled out (a declaration is relative to its base and
        may be wrong); no flag once a run that fitted had that value."""
        fl, cap = self.mem.floor, self.mem.cap
        if not fl or not cap:
            return None
        native = self._native(engine)
        cur = flat(engine.s.candidates[base].config, native)
        new = flat(after(engine.s.candidates[base].config, change, native), native)
        fitted = [flat(engine.s.candidates[c].config, native) for c in self._fitted(engine)]
        for k in sorted(new):
            if _v(new.get(k)) == _v(cur.get(k)) or any(_v(f.get(k)) == _v(new[k]) for f in fitted):
                continue
            d, g = self._declared(k, new[k])
            if d == 'spends' and isinstance(g, (int, float)) and fl['floor_gib'] + g > 1.02 * cap:
                return (f'{k}={new[k]}: the sharded state alone is {fl["floor_gib"]:.0f} GiB per GPU ({fl["basis"]}) and '
                        f'the proposer declared about {g:.0f} GiB more for it, above the {cap:.0f} GiB card')
        return None

    def _neighbourhood(self, engine, inc):
        props = super()._neighbourhood(engine, inc)
        if not self.on['hls3_prune'] or self.accounting is None:
            return props
        keep = []
        for p in props:
            why = self._prune_why(engine, inc, p.change)
            if why:
                self.ledger[p.hypothesis].update(status='invalid', why=f'unrunnable on this card: {why}')
            else:
                keep.append(p)
        return keep

    # ------------------------------------------------------------ S-3 antagonism
    def _coupled(self) -> list:
        return [set(g) for g in self.facts.get('coupled') or [] if len(g) > 1]

    def _members(self, engine, p) -> list[frozenset]:
        """The changes a failed proposal combined: a bundle's members, else its knobs and its patch idea; each as
        {(knob, value)}, the knobs a workload invariant ties together in one member either way (micro-batch and
        accumulation run only together: two members would each need a separate run that the invariant refuses)."""
        base = p.base or engine.incumbent()
        frm = self._mcfg(engine, base)
        if str(p.hypothesis or '').startswith('bundle|') and self.members.get(p.hypothesis):
            out = []
            for k in self.members[p.hypothesis]:
                h = self.ledger.get(k)
                if h is None:
                    continue
                to = self._mcfg(engine, base, h.get('change'), {h['mechanism']} if h.get('patch') else ())
                m = frozenset((kk, b) for kk, _, b in sig(frm, to))
                if m:
                    out.append(m)
            return self._merge_coupled(out)
        to = self._mcfg(engine, base, p.change, self._patch_ideas(p))
        return self._merge_coupled([frozenset({(k, b)}) for k, _, b in sorted(sig(frm, to))])

    def _atoms(self, engine, p) -> list[frozenset]:
        """F5a: the single changes a run combined - each knob value and each patch idea on its own (the knobs of one
        workload invariant together), however the entries grouped them (an entry "compile together with Liger" is two)."""
        base = p.base or engine.incumbent()
        frm = self._mcfg(engine, base)
        to = self._mcfg(engine, base, p.change, self._patch_ideas(p))
        return self._merge_coupled([frozenset({(k, b)}) for k, _, b in sorted(sig(frm, to))])

    def _merge_coupled(self, members: list[frozenset]) -> list[frozenset]:
        """`members` with every two that touch the knobs of one workload invariant merged into one."""
        groups, out = self._coupled(), [set(m) for m in members]
        i = 0
        while i < len(out):
            j = next((j for j in range(i + 1, len(out)) if any(
                g & {k for k, _ in out[i]} and g & {k for k, _ in out[j]} for g in groups)), None)
            if j is None:
                i += 1
            else:
                out[i] |= out.pop(j)                        # (and look again: the merged one may touch another)
        uniq = []
        for m in map(frozenset, out):
            if m not in uniq:
                uniq.append(m)
        return uniq

    @staticmethod
    def _has(cfg: dict, m: frozenset) -> bool:
        return all(_v(cfg.get(k)) == v for k, v in m)

    def _touches(self, engine, h: dict, m: frozenset) -> bool:
        """The entry sets part of member `m` (a knob value of it, or its patch idea)."""
        ch = flat(h.get('change') or {}, self._native(engine))
        return any((k, _v(v)) in m for k, v in ch.items()) or bool(h.get('patch')) and (f'patch:{h["mechanism"]}',
                                                                                         'true') in m

    def _blocked(self, cfg: dict) -> dict | None:
        """The failed-together set this configuration combines again before every member ran separately."""
        for t in self.together:
            if all(self._has(cfg, m) for m in t['members']) and len(t['separately']) < len(t['members']):
                return t
        return None

    def _note_separately(self, cfg: dict, kind: str) -> None:
        for t in self.together:
            for m in t['members']:
                if m not in t['separately'] and self._has(cfg, m) and not any(
                        self._has(cfg, o) for o in t['members'] if o != m):
                    t['separately'][m] = kind

    def _separately(self, engine, inc, t: dict) -> None:
        """Entries that run each knob member of a failed-together set alone on the incumbent (the others of the set
        reverted to the reference's values where the incumbent has them: reverts allowed); once per set and incumbent."""
        if inc in t['queued_on']:
            return
        t['queued_on'].add(inc)
        ref, cur = self._mcfg(engine, 'c0'), self._mcfg(engine, inc)
        for m in t['members']:
            if m in t['separately'] or any(str(k).startswith('patch:') for k, _ in m):
                continue
            change = {k: json.loads(v) for k, v in m}
            for o in t['members']:
                if o != m and self._has(cur, o):
                    change.update({k: ref.get(k) for k, _ in o if not str(k).startswith('patch:')})
            if self._native(engine) or self._has(cur, m) and all(not self._has(cur, o) for o in t['members'] if o != m):
                continue
            mech = f'separately: {self._words(m)} (failed together in {t["candidate"]})'
            key = f'{inc}|{mech}'
            if key in self.ledger:
                continue
            prior = max((h['mu'] for h in self.ledger.values() if not h.get('patch') and h.get('change')
                         and all(_v(h['change'].get(k)) == v for k, v in m)), default=0.0)
            self.ledger[key] = {'mechanism': mech, 'change': change, 'patch': None, 'mu': max(0.0, prior),
                                'sigma': max(2 * engine.sigma_pair(), float(self.prior_sigma or 0.0)), 'p_feasible': 0.7,
                                'base': inc, 'status': 'open', 'key': key, 'separately_of': t['candidate']}

    # ------------------------------------------------------------ S-2 bundles
    def _spenders(self, engine, inc) -> list[dict]:
        sp2 = 2 * engine.sigma_pair()
        kap = self.kappa()
        out = []
        for h in self.ledger.values():
            if h['base'] != inc or h.get('patch') or not h.get('change') or h.get('explore') and not h.get('deferred'):
                continue
            mem = h['status'] == 'open' and str(h.get('deferred') or '').startswith('predicted out of memory') or \
                h['status'] == 'invalid' and h.get('failure') in ('oom', 'thrash')
            mu = self._spender_mu(engine, h) if mem else h['mu']   # (F15: a seed's zero prior replaced)
            if mem and self._kappa_for(h, kap) * mu >= sp2:
                out.append((mu, h))
        return [h for _, h in sorted(out, key=lambda x: -x[0])]

    def _enablers(self, engine, inc, spender: dict, *, unknown: bool = False) -> list[tuple[dict, float | None]]:
        """The incumbent's untested entries (the proposer's moves and code ideas, its single-knob neighbours, experience
        seeds) that free memory - declared, measured, or along an ordered knob's scale - then, with `unknown`, single-knob
        moves whose memory effect nobody declared or measured (S-4's probes); never one of the spender's knobs, one that
        spends, a pruned value, a move judged '-', or a pairing that failed together. [(entry, declared GiB freed or
        None)], best first: freeing before unknown, more GiB, judged '+', the larger prediction, the likelier to run."""
        native = self._native(engine)
        cur = self._mcfg(engine, inc)
        known = self._known(engine)
        own = set(flat(spender.get('change') or {}, native))
        judged, kap = self._judgments(), self.kappa()
        out = []
        for h in self.ledger.values():
            if h['base'] != inc or h['key'] == spender['key'] or h['status'] not in ('open', 'unresolved'):
                continue
            ch = flat(h.get('change') or {}, native)
            if set(ch) & own or h.get('deferred') or str(h.get('why') or '').startswith('unrunnable') or any(
                    0 < len(g & set(ch)) < len(g) for g in self._coupled()):
                continue                                    # (half of an invariant's knobs: refused before it runs)
            sp = self.mem.split(cur, self._mcfg(engine, inc, h.get('change'), {h['mechanism']} if h.get('patch') else ()),
                                known)
            if sp['spends']:
                continue
            if sp['frees']:
                rank, gib = 0, h.get('mem_gib') if h.get('mem_dir') == 'frees' else None
                if gib is None and self.on['hls3_peaks'] and len(sp['frees']) == 1:   # D3: measured, when two peaks did
                    gib = self.mem.gib.get(sp['frees'][0])
            elif unknown and not h.get('patch') and len(ch) == 1 and sp['unknown'] and not (
                    self.on['hls3_need'] and any(self._suspect_spend(x, known) for m in sp['unknown'] for x in m)):
                rank, gib = 1, None
            else:
                continue
            both = {**(spender.get('change') or {}), **(h.get('change') or {})}
            cfg = self._mcfg(engine, inc, both)
            if self.on['hls3_antagonism'] and self._blocked(cfg) or self.on['hls3_prune'] and self._prune_why(
                    engine, inc, h.get('change')):
                continue
            pb = self._pair_block(cfg) if self.on['hls3_pairs'] else None
            if pb is not None and not self._swappable(engine, inc, both, pb):
                continue                                    # (F5: a suspect pair; one with an incumbent member: a swap)
            sign = judged.get(_mkey(h.get('change')), (None, None))[0]
            if sign == '-':
                continue
            out.append((rank, -(gib or 0.0), 0 if sign == '+' else 1, -self._kappa_for(h, kap) * h['mu'],
                        -h['p_feasible'], h['key'], h, gib))
        return [(x[6], x[7]) for x in sorted(out, key=lambda x: x[:6])]

    def _scaled_mu(self, keys) -> float:
        kap = self.kappa()
        return sum(self.ledger[k]['measured'] if self.ledger[k].get('measured') is not None else
                   self._kappa_for(self.ledger[k], kap) * self.ledger[k]['mu'] for k in keys)

    def _spender_bundles(self, engine, inc) -> list:
        """S-2: each memory-blocked spender of the incumbent with the freeing moves that cover the shortfall the runs
        showed (else the best one; after an out-of-memory bundle, a larger one: one enabler more), probed first. Only
        enablers that go together: not another value of a knob already in the bundle, a patch that merges, no pair that
        failed together - an entry that does not is passed over for the next one."""
        out = []
        for h in self._spenders(engine, inc)[:2]:
            choice = self._s2_choice(engine, inc, h)
            if choice is None:
                continue
            keys, gibs, need = choice
            q = self._mem_bundle(engine, inc, h, keys, origin='hls3-bundle', plan=('probe', 'full'), need=need, gibs=gibs)
            if q is not None:
                out.append(q)
        return out

    def _s2_choice(self, engine, inc, h: dict) -> tuple | None:
        """S-2's enablers for one spender on the incumbent - (entry keys, declared GiB each, the shortfall) - or None when
        it has no bundle to try now (_spender_bundles; F16 asks whether S-2 covers a spender)."""
        tries = [b for b in self.mem_bundles if b['spender'] == h['key'] and b['base'] == inc]
        if any(b.get('status') in (None, 'probe passed') for b in tries) or len(tries) >= 2:
            return None
        if tries and tries[-1].get('status') not in ('oom', 'thrash'):
            return None                                     # it ran, or failed for another reason: no larger one
        n = self._need(h) if self.on['hls3_need'] else None
        if n is not None and n['probes'] >= SPENDER_PROBES and not self._unlocked(engine, h, inc):
            return None                                     # F3: tried enough; a measured freeing move reopens it
        en = self._enablers(engine, inc, h)
        if not en:
            return None
        need, keys, gibs = h.get('short_gib'), [], {}
        lb = n['need_lb'] if n is not None else 0.0         # F3: more than every bundle of it that ran out of memory
        if not any(g is not None for _, g in en):
            lb = 0.0                                        # (no enabler declares GiB: nothing to compare, r4's rule)
        first = list(tries[0].get('enabler_keys') or []) if tries else None
        for e, gib in en:
            if not self._compatible(engine, inc, [h['key']] + keys + [e['key']], swap_ok=self.on['hls3_pairs']):
                continue
            keys.append(e['key'])
            gibs[e['key']] = gib
            freed = sum(g or 0.0 for g in gibs.values())
            if lb and (freed > lb or len(keys) >= 3):
                break
            if not lb and first is None and (len(keys) >= 2 or need is None or freed >= need):
                break
            if not lb and first is not None and (len(keys) > len(first) or len(keys) >= 3):
                break
        if lb and sum(g or 0.0 for g in gibs.values()) <= lb:
            return None                                     # F3: what they free is no more than what failed already
        if not keys or not lb and first is not None and (len(keys) <= len(first) or set(keys) == set(first)):
            return None                                     # no compatible enabler, or no larger bundle to try
        return keys, gibs, need

    def _compatible(self, engine, inc, keys, swap_ok: bool = False) -> bool:
        """The entries `keys` make one bundle as _bundle_proposal builds it, none left out: no knob set to two values,
        the patches merge, no set that failed together, no suspect pair (F5; with `swap_ok` a pair whose other member
        only the incumbent holds is left to the swap)."""
        change, patch = {}, None
        c = engine.s.candidates[inc]
        for k in keys:
            h = self.ledger[k]
            ch = h.get('change') or {}
            if any(n in change and change[n] != v for n, v in ch.items()):
                return False
            if h.get('patch'):
                files = engine.s.hooks.candidate_files(c.overlay) if engine.s.hooks is not None else dict(c.overlay or {})
                patch = self.merge(files, [patch, h['patch']] if patch else [h['patch']])
                if patch is None:
                    return False
            change.update(ch)
        cfg = self._mcfg(engine, inc, change)
        if self.on['hls3_pairs'] and self._pair_block(cfg) is not None and not (
                swap_ok and self._swappable(engine, inc, change, self._pair_block(cfg))):
            return False
        return not (self.on['hls3_antagonism'] and self._blocked(cfg))

    def _swappable(self, engine, inc, change: dict, pb: dict) -> bool:
        """F5c: exactly one member of the pair is the incumbent's and the change does not set it."""
        own = {(k, _v(v)) for k, v in flat(change or {}, self._native(engine)).items()}
        cur = self._mcfg(engine, inc)
        return sum(1 for m in pb['members'] if self._has(cur, m) and not (m & own)) == 1

    def _mem_bundle(self, engine, inc, spender, enabler_keys, *, origin, plan, need=None, gibs=None):
        """The bundle of `spender` and `enabler_keys` on `inc` (held for its probe, or pending); its row records the
        members the materialized bundle kept and the GiB they were declared to free."""
        keys = [spender['key']] + list(enabler_keys)
        if frozenset(keys) in self.evaluated or frozenset(keys) in self.bundle_ids:
            return None
        bid = f'bundle|mem|{len(self.mem_bundles)}'
        q = self._bundle_proposal(engine, inc, keys, p_feasible=0.5, plan=plan, bid=bid, origin=origin)
        if q is None or spender['key'] not in self.members[q.hypothesis]:
            return None
        kept = [k for k in self.members[q.hypothesis] if k != spender['key']]
        freed = sum((gibs or {}).get(k) or 0.0 for k in kept)
        q = dataclasses.replace(q, mu=self._scaled_mu(self.members[q.hypothesis]))
        self.mem_bundles.append({'bundle': q.hypothesis, 'base': inc, 'spender': spender['key'],
                                 'spender_mechanism': spender['mechanism'], 'enabler_keys': kept,
                                 'enablers': [self.ledger[k]['mechanism'] for k in kept], 'origin': origin,
                                 'short_gib': need, 'freed_gib_declared': freed or None, 'status': None,
                                 'why': spender.get('deferred') or f'{spender.get("failure")} on {inc}'})
        if 'probe' == plan[-1]:
            self.held[q.hypothesis] = q
        else:
            self.pending[q.hypothesis] = q
        return q

    # ------------------------------------------------------------ S-4 / S-6 at a stall
    def _stall(self, engine, inc) -> list:
        step = {'after_screens': engine.full_screens(), 'incumbent': inc, 'probes': [], 'whole': None}
        out = []
        if self.on['hls3_stall_probes']:
            for h in self._spenders(engine, inc)[:1]:
                n = self._need(h) if self.on['hls3_need'] else None
                if n is not None and n['probes'] >= SPENDER_PROBES and not self._unlocked(engine, h, inc):
                    continue                                # F3: tried enough
                freeing = self._enablers(engine, inc, h)
                keys = {e['key'] for e, _ in freeing}
                unknown = [x for x in self._enablers(engine, inc, h, unknown=True) if x[0]['key'] not in keys]
                if n is not None:                           # F3: freeing enough, an unknown one once per spender
                    freeing = [x for x in freeing if not n['need_lb'] or x[1] is None or x[1] > n['need_lb']]
                    unknown = [] if n['unknown'] else unknown[:1]
                mixed = [x for pair in zip(freeing, unknown) for x in pair] + freeing[len(unknown):] + unknown[len(freeing):]
                budget = self.stall_probes if n is None else max(0, min(self.stall_probes, SPENDER_PROBES - n['probes']))
                for e, gib in mixed[:budget]:                # alternating: a proposer's blind spot gets its turn
                    q = self._mem_bundle(engine, inc, h, [e['key']], origin='hls3-stall-probe', plan=('probe',),
                                         gibs={e['key']: gib})
                    if q is None:
                        continue
                    if n is not None and e['key'] in {x[0]['key'] for x in unknown}:
                        n['unknown'] = True
                    if self.on['hls3_complete']:            # F4: completed before it is queued
                        ch = self._complete(engine, inc, q.change)
                        if ch is None:
                            continue
                        q = dataclasses.replace(q, change=ch)
                    pb = self._pair_block(self._mcfg(engine, inc, q.change)) if self.on['hls3_pairs'] else None
                    if pb is not None:                      # F5c: blocked only by an incumbent member: the swap
                        s = self._swap(engine, inc, q, None, pb)
                        if s is not None:
                            out.append(s)
                            step['probes'].append(f'swap: {s.mechanism}')
                        continue
                    out.append(dataclasses.replace(q, fidelity=('probe',), keep_probe=True, abortable=False,
                                                   hypothesis='probe|' + q.hypothesis,
                                                   probe_cap=self._probe_cap(engine, inc) if self.on['hls3_start_cap']
                                                   else None))
                    step['probes'].append(self.mem_bundles[-1]['enablers'][0])
        if self.on['hls3_explore']:
            q = self._whole(engine, inc)
            if q is not None:
                out.append(q)
                step['whole'] = q.mechanism
        self.idle = 0
        if out:
            self.stalls.append(step)
        return out

    def _whole(self, engine, inc):
        """S-6: the experience start composite on the incumbent, else the proposer's strongest untested moves (one per
        knob, at most four), each set made coherent: a member combined with one it failed together with is left out
        (the weaker), and spenders go while the whole is predicted out of memory. Probed first."""
        exp = self.pending.get('bundle|exp')
        keys = list(self.members.get('bundle|exp') or []) if exp is not None and exp.base == inc else []
        origin = 'hls3-whole-experience'
        if len(keys) < 2 or frozenset(keys) in self.evaluated:
            kap, sp2, seen = self.kappa(), 2 * engine.sigma_pair(), set()
            keys = []
            for h in sorted((h for h in self.ledger.values() if h['base'] == inc and h['status'] == 'open'
                             and not h.get('patch') and not h.get('explore') and h.get('change')),
                            key=lambda h: -self._kappa_for(h, kap) * h['mu']):
                knobs = set(h['change'])
                if self._kappa_for(h, kap) * h['mu'] < sp2 or knobs & seen:
                    continue
                keys.append(h['key'])
                seen |= knobs
                if len(keys) == 4:
                    break
            origin = 'hls3-whole-proposer'
        keys = [k for k in keys if k in self.ledger and not (self.on['hls3_prune'] and self._prune_why(
            engine, inc, self.ledger[k].get('change')))]                   # (a value impossible on this card leaves)
        known = self._known(engine)
        while len(keys) >= 2:
            change = {}
            for k in keys:
                change.update(self.ledger[k].get('change') or {})
            cfg = self._mcfg(engine, inc, change)
            t = self._blocked(cfg) if self.on['hls3_antagonism'] else None
            if t is None and self.on['hls3_pairs']:         # F5a: a suspect pair both of whose members it sets
                pb = self._pair_block(cfg)
                t = pb if pb is not None and not self._swappable(engine, inc, change, pb) else None
            if t is not None:
                inside = [k for k in keys if any(self._touches(engine, self.ledger[k], m) for m in t['members'])]
                keys.remove(min(inside or keys, key=lambda k: _gain_of(self.ledger[k])))
                continue
            if self.on['hls3_feasibility'] and self._predict(engine, cfg)[0] == 'infeasible':
                cur = self._mcfg(engine, inc)
                spend = [k for k in keys if self.mem.split(cur, self._mcfg(engine, inc, self.ledger[k].get('change')),
                                                           known)['spends']]
                if not spend:
                    return None
                keys.remove(max(spend, key=lambda k: self.ledger[k].get('mem_gib') or 0.0))
                continue
            break
        if len(keys) < 2 or frozenset(keys) in self.evaluated:
            return None
        bid = self.bundle_ids.get(frozenset(keys)) or f'bundle|whole|{len(self.stalls)}'
        q = self._bundle_proposal(engine, inc, keys, p_feasible=0.5, plan=('probe', 'full'), bid=bid, origin=origin)
        if q is None:
            return None
        q = dataclasses.replace(q, mu=max(self._scaled_mu(self.members[q.hypothesis]), 2 * engine.sigma_pair()),
                                keep_probe=True)
        if self.on['hls3_complete']:                        # F4: validated and completed before it takes a stall step
            ch = self._complete(engine, inc, q.change)
            if ch is None:
                return None
            q = dataclasses.replace(q, change=ch)
        if self.on['hls3_pairs']:                           # F5c: a pair with an incumbent member: the swap instead
            pb = self._pair_block(self._mcfg(engine, inc, q.change))
            if pb is not None:
                s = self._swap(engine, inc, q, None, pb)
                if s is None:
                    return None
                q = dataclasses.replace(s, keep_probe=True)
        if self.on['hls3_start_cap']:
            q = dataclasses.replace(q, probe_cap=self._probe_cap(engine, inc))
        if not str(q.origin).startswith('hls3-swap'):
            self.pending[q.hypothesis] = q
        return q

    # ------------------------------------------------------------ refresh: S-5, S-7, S-9
    def _refresh(self, engine, inc):
        old = self.refreshed_for
        if old is None and self.on['hls3_variance']:
            self._reference_variance(engine)
        super()._refresh(engine, inc)
        if old is not None and old != inc:
            if self.on['hls3_rebase']:
                self._rebase(engine, inc)
            if self.on['hls3_phases']:
                self._phase_rank(engine, inc)

    def _ask(self, engine, inc) -> int:
        before = set(self.ledger)
        new = super()._ask(engine, inc)
        floor = (self.variance.get('reference') or {}).get('prior')
        for k in set(self.ledger) - before:
            h = self.ledger[k]
            if self.on['hls3_cov_unresolved'] and h['status'] == 'open' and not h.get('patch') and h.get('change'):
                cap = next((c for c in self.captured if c['base'] == inc and c['move'] == _mkey(h['change'])
                            and not c.get('released')), None)
                if cap is not None and self._kappa_for(h, self.kappa()) * h['mu'] < 2 * engine.sigma_pair():
                    h.update(status='unresolved', captured_by=cap['candidate'], why=cap['why'])   # F17
            if self.on['hls3_variance'] and h.get('variance') and floor:
                h['mu'] = max(h['mu'], floor)               # S-9: early, at half the stalls' share of the mean
            if self.on['hls3_sites'] and h.get('patch'):    # S-8: the patch's size and the gain it promises per line
                n = changed_lines(self._files(engine, inc), h['patch'])
                h['patch_lines'] = n
                h['gain_pct_per_line'] = round(100 * h['mu'] / n, 4) if n else None
        return new

    @staticmethod
    def _files(engine, cid) -> dict:
        return files_of(engine, cid)

    def _rebase(self, engine, inc) -> None:
        """S-5: the strongest moves refuted, aborted or out of memory on an earlier incumbent, queued on this one at half
        their prediction (a move's sign can flip with its base), at most rebase_k per change, each move once."""
        cfg = engine.s.candidates[inc].config
        native = self._native(engine)
        cands = []
        for h in self.ledger.values():
            if h['base'] == inc or h.get('patch') or not h.get('change') or h.get('explore') or h.get('rebased_from'):
                continue
            if not (h['status'] == 'refuted' or h['status'] == 'invalid' and h.get('failure') in ('slower', 'thrash', 'oom')):
                continue
            m = _mkey(h['change'])
            if m in self.rebased_moves or all(_v(after(cfg, h['change'], native).get(k)) == _v(cfg.get(k))
                                               for k in h['change']) or self._queued(inc, h['change']):
                continue
            if self.start_failures and self._dead_in(engine, cfg, h['change']):
                continue
            if self.on['hls3_need'] and h.get('failure') in ('oom', 'thrash') and not self._room(engine, h, inc):
                continue                                    # F3b: the new incumbent freed nothing that big since
            cands.append(h)
        for h in sorted(cands, key=lambda h: -h['mu'])[:self.rebase_k]:
            if h['mu'] <= 0:
                break
            key = f'{inc}|rebase:{h["mechanism"]}'
            if key in self.ledger:
                continue
            self.rebased_moves.add(_mkey(h['change']))
            self.ledger[key] = {**{k: h[k] for k in ('change', 'change_class', 'mem_dir', 'mem_gib') if k in h},
                                'mechanism': f'{h["mechanism"]} (again on {inc})', 'patch': None, 'mu': 0.5 * h['mu'],
                                'sigma': max(h['sigma'], 2 * engine.sigma_pair()), 'p_feasible': min(h['p_feasible'], 0.7),
                                'base': inc, 'status': 'open', 'key': key, 'rebased_from': h['key']}
            self.rebased.append({'move': h['mechanism'], 'from': h['base'], 'to': inc, 'was': h['status'],
                                 'failure': h.get('failure')})

    def _room(self, engine, h: dict, inc) -> bool:
        """F3b: the new incumbent holds, against the base the move failed on, freeing moves (measured or declared) whose
        declared GiB reach the move's need_lb (any freeing move when nothing bounds it)."""
        if h['base'] not in engine.s.candidates:
            return False
        sp = self.mem.split(self._mcfg(engine, h['base']), self._mcfg(engine, inc), self._known(engine))
        if not sp['frees']:
            return False
        lb = self._need(h)['need_lb']
        if not lb:
            return True
        freed = 0.0
        for move in sp['frees']:
            for e in self.ledger.values():
                if e.get('mem_dir') == 'frees' and isinstance(e.get('mem_gib'), (int, float)) and \
                        {(k, _v(v)) for k, v in (e.get('change') or {}).items()} == {(k, b) for k, _, b in move}:
                    freed += e['mem_gib']
                    break
        return freed >= lb

    def _slow_move(self, engine, base, p, h) -> bool:
        """F1a: the move (or patch idea) of a probe stopped at its start cap."""
        return _mkey(p.change) in self.slow_moves or bool(p.patch) and h is not None and \
            ('idea', h['mechanism']) in self.slow_moves

    def _phase_rank(self, engine, inc) -> None:
        """S-7: the incumbent's open knob moves capped at what its own phase split leaves them (the reference's
        profile does not describe a later incumbent; nothing without the incumbent's own phases)."""
        c = engine.s.candidates[inc]
        if not ((c.receipt or {}).get('evidence') or {}).get('phases'):
            return
        shares = self._class_shares(engine, inc)
        if not shares:
            return
        n = 0
        for h in self.ledger.values():
            if h['base'] == inc and h['status'] == 'open' and not h.get('patch') and h.get('mu_cap') is None:
                cap = self._cap(engine, h, shares)
                if cap is not None:
                    h.update(mu_raw=h['mu'], mu=cap, mu_cap=cap)
                    n += 1
        self.phase_ranked.append({'incumbent': inc, 'shares': shares, 'capped': n})

    def _reference_variance(self, engine) -> None:
        """S-9: the reference's outlier updates (stalls: an update at least `stall_factor` x the median): the mean's
        excess over the median and their count; a move declared to remove them gets half that excess as its prior."""
        stats = update_stats((engine.s.candidates['c0'].receipt or {}).get('evidence'))
        if not stats:
            return
        excess = math.log(stats['mean'] / stats['median']) if stats['median'] else 0.0
        self.variance['reference'] = {'mean_over_median_pct': round(100 * math.expm1(excess), 2),
                                      'stalls': stats['stalls'], 'updates': stats['n'], 'stall_factor': STALL,
                                      'outliers': stats['stalls'] >= 2 or excess >= 0.02,
                                      'prior': round(0.5 * excess, 5) if excess > 0 else None}

    # ------------------------------------------------------------ outcomes
    def classify(self, engine, p, out) -> dict:
        """S-3: the class of an outcome: ok, slower (aborted), thrash, oom, start_cap, hang, compiler, collective,
        numerics, crash, refused (no run), infra; with the out-of-memory numbers and the measured peak."""
        if out.status in ('infra', 'duplicate', 'rejected'):
            return {'kind': 'infra' if out.status == 'infra' else 'refused'}
        c = engine.s.candidates.get(out.cid) if out.cid else None
        rec = (c.receipt if c is not None else None) or {}
        ev = rec.get('evidence') if isinstance(rec.get('evidence'), dict) else {}
        events = ev.get('events') or {}
        texts = [str(x) for x in (out.reasons or [])] + [str(events.get(k) or '') for k in ('error', 'fatal')] + \
            [str(x) for x in events.get('samples') or []]
        text = '\n'.join(texts)
        res = ((getattr(c, 'results', None) or {}).get('probe') or {}) if c is not None else {}
        peak = (((c.verdict or {}).get('peak_reserved_gib') if c is not None else None) or ev.get('memory_peak_gib')
                or res.get('memory_peak_gib'))     # a passed probe keeps its peak there, before any full receipt (HLS4 r2)
        measured = None
        if peak is None and self.on['hls3_peaks']:          # D3 / F9a: the harness's measured peak (every process)
            peak = measured = ev.get('memory_peak_measured_gib') or res.get('memory_peak_measured_gib')
        info = {'peak_gib': peak} if isinstance(peak, (int, float)) else {}
        if isinstance(measured, (int, float)):
            info['peak_source'] = 'measured'
        if out.status == 'infeasible' and res.get('start_slow'):
            return {'kind': 'start_slow', 'line': f'stopped at its start cap after {res.get("wall_s") or "?"} s, '
                                                  'before its updates'}
        if out.status == 'infeasible' and 'patch_inert' in (out.failed_checks or []):     # F10
            return {'kind': 'inert_at_start', 'line': (out.reasons or [''])[0][:300]}
        if out.status == 'infeasible' and 'probe_speed' in (out.failed_checks or []):
            g = next((x for x in reversed(self.gated) if x['candidate'] == out.cid), {})
            if self._near_cap(engine, p, peak):         # measured: a peak near the card, or an out-of-memory bound
                return {'kind': 'thrash', 'slowdown': g.get('ratio'), 'at_start': True, **info}
            return {'kind': 'slow_probe', 'slowdown': g.get('ratio'), **info,   # (Codex HLS4 r1 P1-1: a declaration and
                    'line': f'its first updates ran x{g.get("ratio")} slower than its base\'s: no full run; no memory '
                            'evidence'}                     # a probe's timing are no memory boundary)
        if out.status in ('screened', 'aborted', 'probed', 'short'):
            if out.status in ('probed', 'short'):
                return {'kind': 'ok', **info}
            g = out.gain
            if out.status == 'aborted' and out.cid:
                tp, tb = engine.t_partial(out.cid), engine.t(p.base or engine.incumbent())
                g = math.log(tb / tp) if tp and tb else None
            if g is not None and g <= -THRASH and self._near_cap(engine, p, peak):
                return {'kind': 'thrash', 'slowdown': round(math.exp(-g), 3), **info}
            jit = self._jitter(engine, p, out, measured) if self.on['hls3_jitter_thrash'] and g is not None and \
                g < -2 * engine.sigma_pair() else None
            if jit is not None and jit['thrash']:             # F23: its updates scatter on a spending configuration
                return {'kind': 'thrash', 'slowdown': round(math.exp(-g), 3), 'jitter': jit, **info}
            if g is not None and g <= -THRASH:
                info['possible_thrash'] = round(math.exp(-g), 3)   # slow, but nothing measured puts it at the edge
            if jit is not None and jit['scattered']:
                info['jitter'] = jit                            # (scattered, spending nothing: noted only)
            return {'kind': 'ok' if out.status == 'screened' else 'slower', **info}
        oom = parse_oom(text)
        failed = set(out.failed_checks or [])
        m = TIMEOUT_RE.search(text)
        if oom or int(events.get('oom') or 0) > 0 or OOM_TEXT.search(text):
            return {'kind': 'oom', **({'oom': oom} if oom else {}), **info}
        if ev.get('timed_out') or m:
            done = ev.get('steps_completed') if ev.get('timed_out') else int(m.group(1))
            return {'kind': 'start_cap' if not done else 'hang', 'line': text.splitlines()[0][:200] if text else ''}
        if COMPILER_RE.search(text):
            return {'kind': 'compiler', 'line': self._line(text, COMPILER_RE)}
        if int(events.get('collective') or 0) > 0 and 'no_errors' in failed or COLLECTIVE_RE.search(text):
            return {'kind': 'collective', 'line': self._line(text, COLLECTIVE_RE)}
        if failed and failed <= NUMERICS | {'no_errors'} and failed & NUMERICS:
            return {'kind': 'numerics', 'checks': sorted(failed)}
        return {'kind': 'crash', 'line': (str(events.get('error') or '') or (texts[0] if texts else ''))[:200]}

    @staticmethod
    def _line(text: str, rx) -> str:
        return next((ln for ln in text.splitlines() if rx.search(ln)), text.splitlines()[0] if text else '')[:200]

    def _near_cap(self, engine, p, peak) -> bool:
        """Evidence that a slow run is at the memory edge: a measured peak at NEAR of the card or more; without a peak,
        an out-of-memory run that bounds this configuration through measured moves alone (the memory model's unsure or
        infeasible from the runs). A declared or inferred spend is no evidence: without it a slowdown stays a slowdown."""
        cap = self.mem.cap
        if isinstance(peak, (int, float)) and cap:
            return peak >= NEAR * cap
        base = p.base or engine.incumbent()
        cfg = self._mcfg(engine, base, p.change, self._patch_ideas(p))
        return self.mem.predict(cfg, self.mem.measured, self._mcfg(engine, 'c0'))[0] in ('unsure', 'infeasible')

    def observe(self, engine, p, out: Outcome):
        if not any(self.on.values()):
            return super().observe(engine, p, out)
        cls = self.classify(engine, p, out)
        kind = cls['kind']
        base = p.base or engine.incumbent()
        if out.cid and kind not in ('refused', 'infra') and base in engine.s.candidates:
            self._record(engine, p, out, cls, base)
        if self.on['hls3_fear_prior']:                      # F11: every run of a feared move, a stall probe's too (Codex
            self._fear_count(engine, p, out, kind, base)    # HLS4R2 r1: before the stall probe's early return)
        if out.fidelity in ('full', 'probe') and kind not in ('refused', 'infra') and not str(p.hypothesis or '').startswith('probe|'):
            inc = engine.incumbent()                        # S-4 / S-6: ideas tested since the incumbent last changed
            self.idle, self.last_inc = (0, inc) if inc != self.last_inc else (self.idle + 1, inc)
        self.queue = [q for q in self.queue if q.hypothesis != p.hypothesis]
        ev = ((getattr(engine.s.candidates.get(out.cid), 'receipt', None) or {}).get('evidence') or {}) if out.cid else {}
        at_start = kind in TOGETHER and (kind == 'start_slow' or self._updates_done(out, ev if isinstance(ev, dict)
                                                                                   else {}) == 0)
        if kind in TOGETHER and self.on['hls3_antagonism']:
            members = self._members(engine, p)
            if len(members) >= 2 and not any(set(t['members']) == set(members) for t in self.together):
                self.together.append({'members': members, 'kind': kind, 'candidate': out.cid, 'separately': {},
                                      'queued_on': set()})
        if at_start and self.on['hls3_pairs']:              # F5a: every pair of a (small) set that failed to start
            members = self._atoms(engine, p)
            for a, b2 in itertools.combinations(members if len(members) <= self._pair_set() else [], 2):
                self.suspects.setdefault(frozenset({a, b2}), {'run': out.cid, 'kind': kind, 'strikes': 0})
            ran = self._mcfg(engine, out.cid) if out.cid in engine.s.candidates else \
                self._mcfg(engine, base, p.change, self._patch_ideas(p))
            for pair, info in self.suspects.items():        # (each failed start that held a pair: a strike)
                if all(self._has(ran, m) for m in pair):
                    info['strikes'] = info.get('strikes', 0) + 1
            if len(members) == 1 and len(members[0]) == 1:
                (k1, v1), = members[0]
                cfg0 = flat(engine.s.candidates[base].config, self._native(engine)) if base in engine.s.candidates else {}
                cfg1 = {**cfg0, k1: json.loads(v1)} if not str(k1).startswith('patch:') else cfg0
                if not str(k1).startswith('patch:') and self._alone(cfg1, cfg0, not p.patch, k1, json.loads(v1)):
                    self.alone_fails.add((k1, v1))
        if at_start and self.on['hls3_pair_probes'] and kind in DIED_AT_START and self._declared_composite(p):
            atoms = self._atoms(engine, p)                  # F16-A: a composite declared to run died at its start
            if len(atoms) >= 2:
                self.composites.append({'atoms': atoms, 'candidate': out.cid, 'kind': kind, 'origin': str(p.origin)})
        if str(p.hypothesis or '').startswith('alone|'):    # F5b: an ambiguous value (or invariant group) probed alone
            for key, a in self.ambiguous.items():
                if a.get('hyp') == p.hypothesis:
                    a['probe'] = 'started' if out.status == 'probed' else kind
                    if at_start:                            # it failed to start alone: C5 rules it out next time
                        self.alone_fails.add(key)
            return
        hh = self.ledger.get(p.hypothesis or '')
        if at_start and self.on['hls3_dead_patch'] and kind in ('crash', 'compiler'):
            self._fatal(engine, p, out, cls, base, hh)      # F8a
        self._need_account(engine, p, out, cls)            # F3: what the spender needs at least
        if self.on['hls3_shared_spender'] and kind in ('oom', 'thrash') and out.cid:
            self._shared_spender(engine, p, out, cls, base)   # F22: what several such runs shared
        if str(p.hypothesis or '').startswith('probe|'):    # S-4: a stall probe ran
            self._probe_done(engine, p, out, cls)
            return
        if p.hypothesis in {b['bundle'] for b in self.mem_bundles}:
            row = next(b for b in self.mem_bundles if b['bundle'] == p.hypothesis)
            row.update(status=kind if kind not in ('ok',) else out.status, candidate=out.cid,
                       gain=None if out.gain is None else round(out.gain, 5),
                       moved=out.cid is not None and engine.incumbent() == out.cid)
        if kind == 'thrash' and self.on['hls3_failures']:   # the edge of what fits, not a slow move: no calibration pair
            j = cls.get('jitter')
            out = dataclasses.replace(out, status='invalid', valid=False, reasons=[
                f'thrash: x{cls["slowdown"]} slower than its base near the memory cap' if not j else
                f'thrash: x{cls["slowdown"]} slower than its base, its updates irregular (MAD / median {j["scatter_pct"]} % '
                f'against {_pct(j["base_pct"])} for its base) on a configuration that spends memory: the memory edge'])
        h = self.ledger.get(p.hypothesis or '')
        if out.status == 'screened' and h is not None and out.gain is not None and self.on['hls3_variance']:
            self._variance_account(engine, p, out, h)
        cov = next((r for r in self.coverage if r['bundle'] == p.hypothesis), None)
        before = set(self.pending)
        super().observe(engine, p, out)
        if p.retry and h is not None:                       # F1a: its one full retry ran
            h['retry_full'] = False
        if kind == 'start_slow' and h is not None:          # F1a: not a start failure: one full retry, at p <= 0.3
            first = not h.get('start_slow')
            h['start_slow'] = True
            if first and h['status'] == 'invalid':          # (Codex HLS4 r1 P1-2: hls2 retired it as infeasible)
                h.update(status='open', retry_full=True, p_feasible=min(h['p_feasible'], 0.3))
            h['why'] = (f'start slower than {K_START}x the base\'s: its probe was stopped at its start cap; one full run '
                        'at a lower feasibility, its candidate reopened')
            self.slow_moves.add(_mkey(p.change) if not p.patch else ('idea', h['mechanism']))
            self.slow_starts.append({'candidate': out.cid, 'move': h['mechanism'], 'base': base,
                                     'cap_s': p.probe_cap})
        if cov is not None:                                 # F2: the coverage bundle's fate
            moved = out.cid is not None and engine.incumbent() == out.cid
            cov.update(status=out.status, candidate=out.cid, moved=moved,
                       gain_pct=None if out.gain is None else round(100 * math.expm1(out.gain), 2))
            if not moved:
                inside = set(self.members.get(p.hypothesis) or ())
                for k in [k for k in set(self.pending) - before if set(self.members.get(k) or ()) <= inside]:
                    del self.pending[k]                     # no attribution halves (hls's group testing)
                captured = self.on['hls3_cov_unresolved'] and out.status == 'screened' and out.gain is not None \
                    and -2 * engine.sigma_pair() < out.gain <= 0.0 \
                    and not self._explained(engine, p.base or base, p.hypothesis)   # F17: together, no gain
                gp = None if out.gain is None else round(100 * math.expm1(out.gain), 2)
                for k in self.members.get(p.hypothesis, []):
                    m = self.ledger.get(k)                  # (back to their single screens: the toybench lost 0.11
                    if m is not None and m['status'] == 'open':   # of frac when a rejected bundle retired them)
                        m['coverage'] = {'candidate': out.cid, 'status': out.status, 'gain_pct': gp}
                        if captured:                        # F17: resolved by the bundle, not screened one by one
                            why = (f'bundled, measured together: {gp} % in {out.cid} (a coverage bundle within the noise); '
                                   'predicted again above the noise, it runs alone')
                            m.update(status='unresolved', captured_by=out.cid, why=why)
                            self.captured.append({'base': m['base'], 'move': _mkey(m.get('change')), 'candidate': out.cid,
                                                  'mechanism': m['mechanism'], 'gain_pct': gp, 'why': why, 'key': k,
                                                  'bundle': p.hypothesis})
        if self.on['hls3_cov_unresolved'] and h is not None and h['status'] == 'refuted':
            self._release(engine, h)                        # F17: a part measured slower alone explains a null bundle
        if kind == 'inert_at_start' and h is not None:      # F10: proven inert at its probe
            self._inert_at_start(engine, p, out, h, base)
        elif h is not None and (self.on['hls3_dead_patch'] or self.on['hls3_self_proof']) and h.get('patch'):
            self._inert(engine, p, out, h)                  # F8b (F10c: its floor, the full run's own proof)
        if cls.get('possible_thrash'):                      # noted only: an ordinary slowdown until a run shows the edge
            self.counts['possible_thrash'] += 1
            if h is not None:
                h['memory'] = (f'possible thrash: x{cls["possible_thrash"]} slower than its base, but no measured peak '
                               'near the card and no out-of-memory run bounds it: not a memory boundary')
        if h is not None and kind not in ('ok', 'refused', 'infra'):
            h['failure'] = kind
            short = (cls.get('oom') or {}).get('short_gib')
            if short is not None:
                h['short_gib'] = short                      # the least a bundle must free (S-2)
                peak = (self.mem.runs.get(base) or {}).get('peak')
                if h.get('mem_dir') == 'spends' and isinstance(peak, (int, float)) and self.mem.cap:
                    h['mem_gib_bound'] = round(self.mem.cap - peak + short, 2)   # the declaration corrected (S-1)
        for r in self.pair_probes:                          # F16: a swap it queued ran
            if r.get('swap_key') and r['swap_key'] == p.hypothesis and r['status'] == 'swap':
                r['status'] = f'swap {kind if kind not in ("ok",) else out.status}'
                self.pair_ran.add(r['moves'])
        if self.on['hls3_pair_probes'] and h is not None and kind in ('oom', 'thrash') and not h.get('patch') \
                and h['base'] == engine.incumbent():
            self._pair_probes(engine, h['base'], h)         # F16: a memory-blocked spender's undeclared partners

    def _record(self, engine, p, out, cls, base) -> None:
        """The run in the memory model (S-1), the failure list (S-3), the separate runs of failed-together sets."""
        kind = cls['kind']
        pids = self._pids(engine, base) | self._patch_ideas(p)
        self.pids[out.cid] = frozenset(pids)
        cfg = self._mcfg(engine, out.cid) if out.cid in engine.s.candidates else \
            self._mcfg(engine, base, p.change, self._patch_ideas(p))     # what ran (the validated config)
        status = {'oom': 'oom', 'thrash': 'thrash', 'ok': 'ok', 'slower': 'ok', 'numerics': 'ok'}.get(kind)
        if status is not None and self.on['hls3_feasibility']:
            self.mem.record(out.cid, cfg, status, peak=cls.get('peak_gib'), oom=cls.get('oom'))
            b = self.mem.runs.get(base)
            if b is None and base == 'c0':                  # the reference ran (the search starts from it)
                b = {'cfg': self._mcfg(engine, 'c0'), 'status': 'ok',
                     'peak': (engine.s.candidates['c0'].verdict or {}).get('peak_reserved_gib') or (
                         self._ev(engine.s.candidates['c0']).get('memory_peak_measured_gib') if self.on['hls3_peaks']
                         else None)}                    # (D3: the reference's measured peak)
                self.mem.runs['c0'] = b
            if b is not None and b['status'] == 'ok':
                if status in ('oom', 'thrash'):
                    self.mem.relate(b['cfg'], cfg, 'spends')
                elif isinstance(b.get('peak'), (int, float)) and isinstance(cls.get('peak_gib'), (int, float)):
                    d = cls['peak_gib'] - b['peak']
                    tol = max(1.0, 0.02 * (self.mem.cap or 0.0))
                    self.mem.relate(b['cfg'], cfg, 'spends' if d > tol else 'frees' if d < -tol else 'neutral')
                    if cls.get('peak_source') == 'measured' and d < -tol:
                        self.mem.gib[sig(b['cfg'], cfg)] = round(-d, 2)   # D3: what it freed, measured
        if kind not in ('ok', 'slower'):
            c = engine.s.candidates[base].config
            new = after(c, p.change, self._native(engine))
            row = {'candidate': out.cid, 'base': base, 'class': kind,
                   'change': {k: v for k, v in flat(new, self._native(engine)).items()
                              if _v(v) != _v(flat(c, self._native(engine)).get(k))} or None}
            if p.patch:
                row['patch'] = sorted(self._patch_ideas(p))
            o = cls.get('oom')
            if o:
                row['detail'] = (f'tried to allocate {o["tried_gib"]:.2f} GiB' + (f' with {o["free_gib"]:.2f} GiB free'
                                 if 'free_gib' in o else '') + (f' of {o["capacity_gib"]:.2f} GiB' if 'capacity_gib' in o
                                                                else '') + (f': short by at least {o["short_gib"]:.2f} GiB'
                                                                            if 'short_gib' in o else ''))
            elif kind == 'thrash':
                j = cls.get('jitter')
                row['detail'] = f'x{cls["slowdown"]} slower than its base' + (
                    f', peak {cls["peak_gib"]:.1f} GiB' if cls.get('peak_gib') is not None else '') + (
                    f', its update intervals scattered (MAD / median {j["scatter_pct"]} %, its base {_pct(j["base_pct"])})'
                    if j else '')
            elif cls.get('line'):
                row['detail'] = cls['line']
            self.failures.append(row)
        if self.on['hls3_antagonism'] and self.together:
            self._note_separately(cfg, kind)
        if kind in ('ok', 'slower', 'thrash', 'numerics', 'oom') and (self.on['hls3_pairs'] or self.on['hls3_dead_patch']):
            ev = (getattr(engine.s.candidates.get(out.cid), 'receipt', None) or {}).get('evidence') or {}
            done = self._updates_done(out, ev if isinstance(ev, dict) else {})
            if kind != 'oom' or (done or 0) >= 1:           # it reached an update: what it holds started
                for pair in [pr for pr in self.suspects if all(self._has(cfg, m) for m in pr)]:
                    del self.suspects[pair]                 # (F5a)
                if self.on['hls3_dead_patch'] and p.patch:  # (F8a: its code lines are not what kills a start)
                    self.started_lines |= self._added(engine, base, p.patch)

    def _need_account(self, engine, p, out, cls) -> None:
        """F3: a bundle or probe of a spender that ran out of memory (or thrashed) at its start: need_lb at least what its
        enablers were declared to free plus the shortfall; each try counted."""
        if not self.on['hls3_need']:
            return
        bid = str(p.hypothesis or '')
        bid = bid[len('probe|'):] if bid.startswith('probe|') else bid
        row = next((b for b in self.mem_bundles if b['bundle'] == bid), None)
        h = self.ledger.get(row['spender']) if row is not None else None
        if h is None:
            h = self.ledger.get(p.hypothesis or '')
            if h is None or cls['kind'] not in ('oom', 'thrash') or h.get('patch'):
                return
            n = self._need(h)                               # the spender's own run: a try, its shortfall a bound
            n['need_lb'] = max(n['need_lb'], float((cls.get('oom') or {}).get('short_gib') or 0.0))
            n['probes'] += 1
            n['last'] = p.base or engine.incumbent()
            return
        n = self._need(h)
        n['probes'] += 1
        n['last'] = row['base']
        if cls['kind'] in ('oom', 'thrash'):
            n['need_lb'] = max(n['need_lb'], float(row.get('freed_gib_declared') or 0.0)
                               + float((cls.get('oom') or {}).get('short_gib') or 0.0))

    def _probe_done(self, engine, p, out, cls) -> None:
        bid = p.hypothesis[len('probe|'):]
        q = self.held.pop(bid, None)
        row = next((b for b in self.mem_bundles if b['bundle'] == bid), None)
        ok = out.status == 'probed'
        if row is not None:
            row.update(status='probe passed' if ok else cls['kind'], candidate=out.cid)
        if q is not None and ok:                            # it fits: a full screen (no second probe)
            self.pending[bid] = dataclasses.replace(q, fidelity=('full',), p_feasible=0.95, keep_probe=False)
            self.continued.add(bid)                         # (F3: the same try, not another one)
        elif q is not None:
            self.evaluated.add(frozenset(self.members.get(bid) or ()))
        rec = next((r for r in self.pair_probes if r.get('bundle') == bid), None)
        if rec is not None:                                 # F16: the pair ran
            rec['status'] = 'probe passed' if ok else cls['kind']
            self.pair_ran.add(rec['moves'])
            h = self.ledger.get(row['spender']) if row is not None else None
            if not ok and h is not None and h['base'] == engine.incumbent() and not any(
                    r['status'] == 'queued' and r['spender'] == rec['spender'] and r['base'] == rec['base']
                    for r in self.pair_probes):
                self._pair_probes(engine, h['base'], h)     # (its composite's pairs failed: the undeclared ones next)

    def _variance_account(self, engine, p, out, h) -> None:
        """S-9: a variance move's stalls before and after, with both gains (the mean's and the median's)."""
        if not h.get('variance'):
            return
        c, b = engine.s.candidates.get(out.cid), engine.s.candidates.get(p.base or 'c0')
        sa = update_stats(((c.receipt if c else None) or {}).get('evidence'))
        sb = update_stats(((b.receipt if b else None) or {}).get('evidence'))
        self.variance.setdefault('moves', []).append({
            'move': h['mechanism'], 'candidate': out.cid, 'gain_mean': None if out.gain_mean is None else round(out.gain_mean, 5),
            'gain_used': round(out.gain, 5), 'stalls_before': (sb or {}).get('stalls'), 'stalls_after': (sa or {}).get('stalls')})

    # ------------------------------------------------------------ FABLE_HLS3_FIXES_R1: F1 the probe's start and speed
    def _t_first(self, engine, cid) -> tuple[float, float] | None:
        """(time from the launch to the end of the first update, update time) of a candidate's run from its receipt: the
        run's wall less the span of its stamped updates (one clock: no skew between hosts), else the wall less updates x
        the step time; the reference's when the candidate has no timed run; None without either."""
        for x in (cid, 'c0'):
            c = engine.s.candidates.get(x)
            rec = getattr(c, 'receipt', None) or {}
            ev = rec.get('evidence') if isinstance(rec.get('evidence'), dict) else {}
            wall = ev.get('wall_s')
            st = ev.get('step_time') or (engine.t(x) if c is not None else None)
            if not (isinstance(wall, (int, float)) and wall > 0 and isinstance(st, (int, float)) and st > 0):
                continue
            ps = [r for r in ev.get('per_step') or [] if isinstance(r, dict) and isinstance(r.get('t'), (int, float))]
            if len(ps) >= 2:
                tf = wall - (ps[-1]['t'] - ps[0]['t'])
            elif isinstance(ev.get('steps_completed'), int):
                tf = wall - ev['steps_completed'] * st
            else:
                continue
            return max(tf, 0.0), float(st)
        return None

    def _probe_cap(self, engine, base, steps=None) -> float | None:
        """F1a: a probe's own wall cap - K_START x its base's time to the first update, at least that time plus the
        probe's updates (`steps`: a gated probe's, else the engine's) and a minute (every run of the recorded arms that
        reached an update started within it)."""
        tf = self._t_first(engine, base)
        if tf is None:
            return None
        steps = int(steps or getattr(engine, 'probe_steps', 2) or 2)
        return round(max(K_START * tf[0], tf[0] + steps * tf[1] + 60.0), 1)

    def _world(self, engine) -> float | None:
        w = self.facts.get('world')
        if isinstance(w, (int, float)) and w > 0:
            return float(w)
        rec = getattr(engine.s.candidates.get('c0'), 'receipt', None) or {}
        cost = rec.get('cost') if isinstance(rec.get('cost'), dict) else {}
        g, s = cost.get('gpu_seconds'), cost.get('wall_seconds')
        return g / s if isinstance(g, (int, float)) and isinstance(s, (int, float)) and s > 0 else None

    def _wall_full(self, engine) -> float | None:
        """The wall of a full run: the mean of the valid full screens', else the reference's."""
        walls = [r['wall_seconds'] for r in engine.s.screens if r.get('fidelity', 'full') == 'full' and r.get('valid')
                 and isinstance(r.get('wall_seconds'), (int, float))]
        if walls:
            return sum(walls) / len(walls)
        rec = getattr(engine.s.candidates.get('c0'), 'receipt', None) or {}
        cost = rec.get('cost') if isinstance(rec.get('cost'), dict) else {}
        w = cost.get('wall_seconds') or ((rec.get('evidence') or {}).get('wall_s') if isinstance(rec.get('evidence'), dict)
                                         else None)
        return float(w) if isinstance(w, (int, float)) and w > 0 else None

    def _base_intervals(self, engine, base) -> list[float]:
        c = engine.s.candidates.get(base)
        rec = getattr(c, 'receipt', None) or {}
        ev = rec.get('evidence') if isinstance(rec.get('evidence'), dict) else {}
        ps = [r for r in ev.get('per_step') or [] if isinstance(r, dict) and isinstance(r.get('t'), (int, float))]
        return [b['t'] - a['t'] for a, b in zip(ps, ps[1:])][:8]

    def _gate(self, engine, cid, base, row, ideas=frozenset(), claim=None, h=None, warmup=None):
        """The probe's gate: F1b's speed (at self.gate_ratio; with hls3_gate_low past the warm-up, the proposal's own
        patch ideas counting towards its spend), then F10's proof for a patch (`claim` set): (why, 'patch_inert') when it
        did not prove it; notes no veto rests on go to the proposer as feedback."""
        why = self._speed_gate(engine, cid, base, row, ideas, warmup) if self.on['hls3_probe_gate'] else None
        if why or claim is None or not self.on['hls3_self_proof']:
            return why
        fb: list = []
        reason, note = self._proof_reason(engine, base, row, claim, h, fb)
        self.note_feedback += [{'candidate': cid, **x} for x in fb]
        if reason:
            self.proof_fail[cid] = {'reason': reason, 'note': note, 'lines': list(row.get('patch_lines') or [])[:6],
                                    **({'parsed': self._parsed(engine, base, row)} if self.on['hls3_scan_whole'] else {})}
            return reason, 'patch_inert'
        return None

    def _probe_steps(self, engine, h, patch: bool, vetoable: bool = True) -> int | None:
        """F12 / F10: a gated probe's updates - past the warm-up (WARMUP, or a declared longer one up to WARMUP_MAX) by one
        when the speed gate can stop the move (`vetoable`: it spends memory or changes something of unknown memory
        effect), up to a patch's declared first run - at most PROBE_STEPS_MAX; None when the engine's probe has them."""
        n0 = int(getattr(engine, 'probe_steps', 2) or 2)
        need = n0
        if self.on['hls3_gate_low'] and self.on['hls3_probe_gate'] and vetoable:
            w = max(WARMUP, self._supported(h, 'warmup_updates') or WARMUP)   # (unsupported: the default warm-up)
            need = max(need, GATED_STEPS, w + 1)
        a = self._supported(h, 'active_from_update')
        if self.on['hls3_self_proof'] and patch and a:
            need = max(need, a)
        need = min(need, max(PROBE_STEPS_MAX, n0))
        return need if need > n0 else None

    def _extra_probe(self, engine, steps) -> tuple[float, float]:
        """(GPU-hours, wall seconds) a probe of `steps` updates costs above the engine's probe: its extra updates at a full
        run's per-update cost."""
        n0 = int(getattr(engine, 'probe_steps', 2) or 2)
        if not steps or steps <= n0:
            return 0.0, 0.0
        full = max(1, int(getattr(engine, 'full_steps', 30) or 30))
        w = self._wall_full(engine) or 0.0
        return (steps - n0) * engine.costs()['full'] / full, (steps - n0) * w / full

    def _speed_gate(self, engine, cid, base, row, ideas=frozenset(), warmup=None) -> str | None:
        """F1b: a probe's updates against its base's same updates (each run's intervals between consecutive stamped
        updates: both include the warm-up, so they compare like for like); the median ratio at the gate (GATE; F12:
        GATE_LOW, the proposal's own patch ideas in its configuration) or more on a
        configuration that spends memory against its base (declared or measured; F12: or changes something of unknown
        memory effect) stops the full run; otherwise it is only noted (a compiled run's first updates are slow and legal). The stop is `thrash` - memory evidence - only
        when a measured peak near the card or an out-of-memory run bounds the configuration (_near_cap); else a
        `slow_probe`, which records nothing in the memory model."""
        probe = [x for x in row.get('intervals') or [] if isinstance(x, (int, float))]
        ref = self._base_intervals(engine, base)
        if self.on['hls3_gate_low']:                        # F12's warm-up guard (Codex HLS4R2 r1): the intervals after
            if row.get('recompiles'):                       # the warm-up only - a run that recompiled: no speed reading
                return None
            w = warmup if isinstance(warmup, int) and not isinstance(warmup, bool) and WARMUP < warmup <= WARMUP_MAX \
                else WARMUP                                 # (out of range: the default warm-up, never no reading)
            probe, ref = probe[w - 1:], ref[w - 1:]         # interval i runs from update i + 1 to update i + 2
        pairs = [a / b for a, b in zip(probe, ref) if b > 0]
        if not pairs:
            return None
        ratio = statistics.median(pairs)
        if ratio < self.gate_ratio:
            return None
        c = engine.s.candidates.get(cid)
        native = self._native(engine)
        cfg = flat(c.config, native) if c is not None else {}
        cfg.update({f'patch:{x}': True for x in sorted(self._pids(engine, base) | set(ideas))})
        sp = self.mem.split(self._mcfg(engine, base), cfg, self._known(engine))
        spends = bool(sp['spends'])
        maybe = not spends and self.on['hls3_gate_low'] and bool(sp['unknown'])   # F12: an unknown direction may spend
        self.gated.append({'candidate': cid, 'base': base, 'ratio': round(ratio, 2), 'spends': spends,
                           **({'unknown': True} if maybe else {}), 'full_run': not (spends or maybe)})
        if not (spends or maybe):
            return None
        return (f'slow start: its first updates ran x{ratio:.2f} slower than {base}\'s, on a configuration that '
                + ('spends memory' if spends else 'may spend memory (a change of unknown memory effect)') + ': no full run')

    def _spends(self, engine, base, cfg) -> bool:
        return bool(self.mem.split(self._mcfg(engine, base), cfg, self._known(engine))['spends'])

    # ------------------------------------------------------------ F4 coupled knobs completed
    def _complete(self, engine, base, change: dict) -> dict | None:
        """F4: `change` with the knobs a workload invariant ties to those it sets: the first values of the knob table that
        keep the configuration valid for the session's validator (the invariants, the constraints); None when none does;
        the change itself when it touches no group partly, or without a validator."""
        groups = self._coupled()
        validate = getattr(engine.s.hooks, 'validate_config', None) if engine.s.hooks is not None else None
        if not groups or not change or validate is None:
            return change
        from .knob_space import choices
        opts = choices(engine.s.task)
        cfg = engine.s.candidates[base].config
        out = dict(change)
        for g in groups:
            touched = g & set(out)
            if not touched or touched == g:
                continue
            rest = sorted(g - touched)
            for combo in itertools.product(*[opts.get(k) or [cfg.get(k)] for k in rest]):
                cand = {**out, **dict(zip(rest, combo))}
                _, errs, _ = validate({**cfg, **cand})
                if not errs:
                    out = {k: v for k, v in cand.items() if k in change or _v(v) != _v(cfg.get(k))}
                    break
            else:
                return None
        return out

    # ------------------------------------------------------------ F2 one coverage bundle of the moves below the noise
    def _coverage_bundle(self, engine, inc):
        """F2: the incumbent's open moves below 2 sigma_pair (one run cannot tell any of them from nothing) that are start-
        safe knob moves predicted to help - the proposer's, and the neighbours and seeds with a zero prior that the
        transfer judgment calls '+'; no patch, no spender, none below the probe rule, none that failed: one crash takes
        the bundle down - as one bundle when two or more are open (`coverage_when` 'any', FABLE_HLS3_FIXES_R1's trigger;
        'weak': only once the strongest open move is below 2 sigma_pair; 'budget': once they outnumber the full runs the
        budget has left, hls2's C6 principle), one value per knob, the invariant's knobs completed, no set that failed
        together, no suspect pair, not ruled out by the memory model; mu = max(sum of kappa x mu, 2 sigma_pair) and
        sigma sqrt(n) x sigma_pair (hls2's bundle: the members' own spreads would make it a lottery ticket), probed first
        (the probe kept, as S-6's whole step); it competes with the strong moves, and its members are not screened one by
        one while it is proposed. Accepted, its members are bundled; not accepted, they go back to their single screens.
        Not carried to a new incumbent (C1): rebuilt there from the moves still open."""
        sp2, kap, judged = 2 * engine.sigma_pair(), self.kappa(), self._judgments()
        opens = [h for h in self.ledger.values() if h['base'] == inc and h['status'] == 'open' and not h.get('deferred')
                 and (h.get('change') or h.get('patch')) and not h.get('start_slow') and not h.get('retry_of')
                 and not h.get('separately_of') and not h.get('swap_of') and not str(h.get('why') or '').startswith('unrunnable')]
        if not opens:
            return None
        score = {h['key']: self._kappa_for(h, kap) * h['mu'] for h in opens}
        if max(score.values()) >= sp2 and self.coverage_when == 'weak':
            return None                                     # a move one run can tell from nothing is still open
        known, cur = self._known(engine), self._mcfg(engine, inc)
        sub = [h for h in opens if score[h['key']] < sp2 and (score[h['key']] > 0 or judged.get(
            _mkey(h.get('change')), (None, None))[0] == '+')   # predicted to help, or a zero prior judged '+'
               and h.get('change') and not h.get('patch') and not h.get('coverage') and not h.get('failure')
               and h['p_feasible'] >= self.probe_below            # start-safe knob moves: one crash takes the bundle
               and not self.mem.split(cur, self._mcfg(engine, inc, h['change']), known)['spends']]
        if len(sub) < 2:
            return None
        left = self._runs_left(engine)
        if self.coverage_when == 'budget' and (left is None or len(sub) <= left):
            return None                                     # the budget can still test them one by one
        native = self._native(engine)
        keys, knobs = [], set()
        for h in sorted(sub, key=lambda h: (-score[h['key']], -h['p_feasible'], h['key'])):
            ch = set(flat(h.get('change') or {}, native))
            if ch & knobs or self._element_dead(engine, inc, h) or not self._compatible(engine, inc, keys + [h['key']]) \
                    or self.on['hls3_need'] and h.get('change') and self._capped(engine, inc, h['change']):
                continue
            keys.append(h['key'])
            knobs |= ch
        while len(keys) >= 2 and self.on['hls3_feasibility']:
            change = {}
            for k in keys:
                change.update(self.ledger[k].get('change') or {})
            if self._predict(engine, self._mcfg(engine, inc, change))[0] != 'infeasible':
                break
            spend = [k for k in keys if self.mem.split(cur, self._mcfg(engine, inc, self.ledger[k].get('change')),
                                                       known)['spends']]
            keys.remove(max(spend, key=lambda k: self.ledger[k].get('mem_gib') or 0.0) if spend else keys[-1])
        if len(keys) < 2 or frozenset(keys) in self.evaluated or frozenset(keys) in self.bundle_ids:
            return None
        ps = [self.ledger[k]['p_feasible'] for k in keys]
        bid = f'bundle|coverage|{len(self.coverage)}'
        q = self._bundle_proposal(engine, inc, keys, p_feasible=math.prod(ps) ** (1 / len(ps)), plan=('probe', 'full'),
                                  bid=bid, origin='hls3-coverage')
        if q is None:
            return None
        kept = self.members[q.hypothesis]                   # probed first, kept: a member pair may not start together
        q = dataclasses.replace(q, mu=max(sum(max(0.0, score.get(k, 0.0)) for k in kept), sp2), keep_probe=True)
        self.pending[q.hypothesis] = q
        self.coverage.append({'bundle': q.hypothesis, 'base': inc, 'members': [self.ledger[k]['mechanism'] for k in kept],
                              'mu': round(q.mu, 5), 'status': None})
        return q

    def _runs_left(self, engine) -> float | None:
        """F2: the full runs the budget has left - the screens left and, with a GPU-hour cap, the GPU-hours left over a
        full run's (None without either)."""
        b, led = engine.s.budget, engine.s.ledger
        out = []
        if getattr(b, 'screens', None):
            out.append(b.screens - led.used.get('screens', 0))
        full = engine.costs().get('full') if getattr(b, 'gpu_hours', None) else None
        if full:
            out.append((b.gpu_hours - led.used.get('gpu_hours', 0.0)) / full)
        return min(out) if out else None

    def _carry_pending(self, engine, old, inc) -> None:
        if self.on['hls3_coverage']:                        # F2: rebuilt on the new incumbent, not carried whole (C1)
            for bid in [b for b, q in self.pending.items() if str(q.origin) == 'hls3-coverage' and q.base == old]:
                del self.pending[bid]
        super()._carry_pending(engine, old, inc)

    # ------------------------------------------------------------ F3 what a spender needs
    def _need(self, h: dict) -> dict:
        return self.need.setdefault(_mkey(h.get('change')), {'need_lb': 0.0, 'probes': 0, 'unknown': False, 'last': None,
                                                            'change': dict(h.get('change') or {})})

    def _unlocked(self, engine, h: dict, inc) -> bool:
        return self._unlocked_n(engine, self._need(h), inc)

    def _unlocked_n(self, engine, n: dict, inc) -> bool:
        """F3: a spender past its tries may be tried again once the incumbent took a move the runs measured to free
        memory since its last try."""
        if n['last'] is None or n['last'] == inc or n['last'] not in engine.s.candidates:
            return False
        return bool(self.mem.split(self._mcfg(engine, n['last']), self._mcfg(engine, inc), self.mem.measured)['frees'])

    def _capped(self, engine, base, change: dict) -> str | None:
        """F3: a proposal that holds a spender which ran out of memory SPENDER_PROBES times (alone, bundled or probed)
        waits until the incumbent takes a move the runs measured to free memory."""
        if any(n['probes'] >= SPENDER_PROBES for n in self.need.values()) and self.mem.predict(
                self._mcfg(engine, base, change), self.mem.measured, self._mcfg(engine, 'c0'))[0] == 'feasible':
            return None                                     # a run that fitted needs at least as much (measured): no try
        for n in self.need.values():
            sc = n.get('change') or {}
            if sc and n['probes'] >= SPENDER_PROBES and all(_v(change.get(k)) == _v(v) for k, v in sc.items()) \
                    and not self._unlocked_n(engine, n, base):
                return (f'its memory spender was tried {n["probes"]} times (it needs at least {n["need_lb"]:.2f} GiB more '
                        'than its failed bundles freed): it waits for a move the runs measure to free memory')
        return None

    # ------------------------------------------------------------ F5 suspect pairs, the ambiguous death, the swap
    def _pair_block(self, cfg: dict, exempt=None) -> dict | None:
        """F5: a suspect pair this configuration holds (both members) that no run reaching an update held since (its
        strikes: the failed starts that held it). `exempt` (F16): the changes a pair probe tests - a pair within them is
        what it tests, not a block."""
        for pair, info in self.suspects.items():
            if exempt and pair <= exempt:
                continue
            if all(self._has(cfg, m) for m in pair):
                return {'members': sorted(pair, key=sorted), 'candidate': info['run'], 'kind': info['kind'], 'separately': {},
                        'queued_on': set(), 'pair': True, 'strikes': info.get('strikes', 1)}
        return None

    def _veto(self, k, v, c, p) -> bool:
        """F5b (the session's C5 asks it at the limit): True keeps knob = value runnable when it never failed to start
        alone - every failed start that set it changed something else too, or held a suspect partner - and queues one
        probe of it alone on the reference (the start cap applies); a failure there is alone, and C5 rules it out."""
        if not self.on['hls3_pairs']:
            return False
        key = (k, _v(v))
        if key in self.alone_fails:
            return False
        if self._alone(c.config, p.config, c.overlay == p.overlay, k, v):
            self.alone_fails.add(key)
            return False
        a = self.ambiguous.setdefault(key, {'knob': k, 'value': v, 'probe': None, 'held_by': c.id})
        a['held_by'] = c.id
        return True

    def _alone(self, cfg: dict, parent: dict, same_code: bool, k, v) -> bool:
        """A failed start that changed only knob = value (and the knobs a workload invariant ties to it; no other knob, no
        code) and held no suspect partner of it."""
        g = next((g for g in self._coupled() if k in g), {k})
        if not same_code or any(kk not in g and kk != 'env' and cfg.get(kk) != parent.get(kk) for kk in set(cfg) | set(parent)):
            return False
        mine = (k, _v(v))
        flat_cfg = {kk: _v(vv) for kk, vv in cfg.items()}
        for pair in self.suspects:
            if any(mine in m for m in pair):
                other = [m for m in pair if mine not in m]
                if other and all(flat_cfg.get(kk) == vv for kk, vv in other[0]):
                    return False
        return True

    def _alone_probes(self, engine) -> list:
        """F5b: the queued probes of ambiguous values, each alone on the reference (probe only, start cap, kept) - the
        knobs of one workload invariant as one member: the group's ambiguous values together, completed with the
        group's other knobs, one probe per resulting change (Codex HLS4 r1 P2-5: {mbs: 2} and {ga: 1} alone are both
        invalid under mbs x ga fixed)."""
        out, seen = [], set()
        groups = self._coupled()
        for key, a in list(self.ambiguous.items()):
            if a['probe'] is not None:
                continue
            g = next((g for g in groups if a['knob'] in g), {a['knob']})
            members = [x for x in self.ambiguous.values() if x['probe'] is None and x['knob'] in g]
            change = self._complete(engine, 'c0', {x['knob']: x['value'] for x in members})
            for x in members:
                x['probe'] = 'queued'
            if change is None:                              # no value of the group's other knobs keeps the invariant
                for x in members:
                    x['probe'] = 'invalid (the workload invariant)'
                continue
            hyp = 'alone|' + ','.join(f'{k}={_v(v)}' for k, v in sorted(change.items()))
            for x in members:
                x['hyp'] = hyp
            if hyp in seen:
                continue
            seen.add(hyp)
            cap = self._probe_cap(engine, 'c0') if self.on['hls3_start_cap'] else None
            words = ', '.join(f'{k}={v}' for k, v in sorted(change.items()))
            out.append(Proposal(change=change, base='c0', mu=0.0, sigma=2 * engine.sigma_pair(), p_feasible=0.5,
                                mechanism=f'alone: {words} (every failed start of it had a partner)', hypothesis=hyp,
                                origin='hls3-alone-probe', fidelity=('probe',), abortable=False, keep_probe=True,
                                probe_cap=cap))
        return out

    def _swap(self, engine, inc, p, h, block) -> object:
        """F5c: a proposal blocked only by a suspect pair whose other member w the incumbent holds (the proposal does not
        set it): the incumbent without w (its knobs back to the reference's values) plus the proposal, mu = the
        proposal's less w's measured gain (half a bundle's gain when w came in a bundle), only at 2 sigma_pair or more;
        probed first. An entry `swap_of` w on the incumbent."""
        native = self._native(engine)
        base = p.base or inc
        cur = self._mcfg(engine, base)
        own = {(kk, _v(vv)) for kk, vv in flat(p.change or {}, native).items()}
        held = [m for m in block['members'] if self._has(cur, m) and not (m & own)]
        if len(held) != 1 or any(str(kk).startswith('patch:') for kk, _ in held[0]):
            return None
        w = held[0]
        ref = self._mcfg(engine, 'c0')
        change = dict(p.change or {})
        change.update({kk: (None if ref.get(kk) is None else json.loads(_v(ref.get(kk)))) for kk, _ in w})
        gain = self._member_gain(engine, base, w)
        kap = self.kappa()
        mu = (self._kappa_for(h, kap) * h['mu'] if h is not None else p.mu) - gain
        words = self._words(frozenset((kk, None, vv) for kk, vv in w))
        if mu < 2 * engine.sigma_pair():
            self.swaps.append({'base': base, 'without': words, 'for': p.mechanism, 'mu': round(mu, 5), 'proposed': False})
            return None
        key = f'{base}|swap:{words}:{p.mechanism}'[:240]
        if key in self.ledger:
            return None
        self.ledger[key] = {'mechanism': f'{p.mechanism} without {words} (they failed to start together)', 'change': change,
                            'patch': p.patch, 'mu': mu, 'sigma': max(p.sigma, 2 * engine.sigma_pair()),
                            'p_feasible': min(p.p_feasible, 0.5), 'base': base, 'status': 'open', 'key': key,
                            'swap_of': words, 'swap_for': p.hypothesis}
        self.swaps.append({'base': base, 'without': words, 'for': p.mechanism, 'mu': round(mu, 5), 'proposed': True,
                           'entry': key})
        return Proposal(change=change, patch=p.patch, base=base, mu=mu, sigma=max(p.sigma, 2 * engine.sigma_pair()),
                        p_feasible=min(p.p_feasible, 0.5), mechanism=self.ledger[key]['mechanism'], hypothesis=key,
                        origin='hls3-swap', fidelity=('probe', 'full'), keep_probe=True)

    def _member_gain(self, engine, inc, w: frozenset) -> float:
        """The measured gain of the move that brought member w into the incumbent's line: an entry confirmed or unresolved
        that set w alone (its gain) or with other changes (half), else a bundle with it (half its gain); 0 when none."""
        for h in self.ledger.values():
            ch = {(kk, _v(vv)) for kk, vv in flat(h.get('change') or {}, self._native(engine)).items()}
            if not w <= ch:
                continue
            if h['status'] in ('confirmed', 'unresolved') and h.get('measured') is not None:
                g = max(0.0, float(h['measured']))
                return g if w == ch else 0.5 * g            # (with other changes in one entry: half, as a bundle's)
            if h['status'] == 'bundled' and h.get('bundle_gain') is not None:
                return max(0.0, 0.5 * float(h['bundle_gain']))
        return 0.0

    def _updates_done(self, out, ev: dict) -> int | None:
        if isinstance(ev.get('steps_completed'), int):
            return ev['steps_completed']
        for r in out.reasons or []:
            m = re.search(r'(\d+)/\d+ updates', str(r))
            if m:
                return int(m.group(1))
        return None

    # ------------------------------------------------------------ F8 dead elements and inert patches
    def _elements(self, engine, base, change, h, patch=None) -> set:
        """The new elements a run brought: its knob values, its patch idea, that idea's code site and the code lines the
        patch adds (substantive ones: no import, comment or short line, none a patch that started also added)."""
        native = self._native(engine)
        cfg = flat(engine.s.candidates[base].config, native)
        out = {('knob', kk, _v(vv)) for kk, vv in flat(after(engine.s.candidates[base].config, change or {}, native),
                                                       native).items() if _v(vv) != _v(cfg.get(kk))}
        if h is not None and h.get('patch'):
            out.add(('idea', h['mechanism']))
            if h.get('site'):
                out.add(('site', str(h['site']).strip().lower()))
        code = patch or (h or {}).get('patch')
        if code:
            out |= {('line', x) for x in self._added(engine, base, code) if x not in self.started_lines}
        return out

    def _added(self, engine, base, patch: dict) -> set:
        """The substantive lines a patch adds to its base's files, whitespace collapsed."""
        files, out = self._files(engine, base), set()
        for f, text in (patch or {}).items():
            have = {' '.join(ln.split()) for ln in str(files.get(f) or '').splitlines()}
            for ln in str(text or '').splitlines():
                x = ' '.join(ln.split())
                if len(x) >= 20 and x not in have and not x.startswith('#') and not re.match(r'(import|from)\s', x):
                    out.add(x)
        return out

    def _element_dead(self, engine, base, h, patch=None) -> str | None:
        if not self.on['hls3_dead_patch'] or not self.dead_elements or h is None:
            return None
        for e in self._elements(engine, base, h.get('change'), h, patch=patch):
            if e in self.dead_elements:
                return self.dead_elements[e]
        return None

    @staticmethod
    def _fatal_sig(line: str) -> str:
        """An exception type and the start of its message, numbers and quoted names out: what two crashes share."""
        t = re.sub(r"'[^']*'|\"[^\"]*\"", "'_'", str(line or '').strip())
        t = re.sub(r'0x[0-9a-fA-F]+|\d+', 'N', t)
        return t[:120]

    def _fatal(self, engine, p, out, cls, base, h) -> None:
        """F8a: a start crash with the same fatal line as an earlier one rules out the new elements the two runs share."""
        line = cls.get('line') or ''
        m = re.search(r'the log: (.+)', line)
        line = m.group(1) if m else line                    # the job's own words, not the verifier's prefix
        if not re.search(r'\b\w*(?:Error|Exception)\b', line):
            return                                          # no exception named: nothing two crashes can share
        sig_ = self._fatal_sig(line)
        mine = self._elements(engine, base, p.change, h, patch=p.patch)
        for f in self.fatal:
            if f['sig'] == sig_:
                for e in f['elements'] & mine:
                    self.dead_elements.setdefault(e, f'the same fatal line in {f["candidate"]} and {out.cid}: {line[:120]}')
        self.fatal.append({'sig': sig_, 'elements': mine, 'candidate': out.cid})

    def _inert(self, engine, p, out, h) -> None:
        """F8b: a patch predicted at 2 sigma_pair or more that measured within sigma_pair of nothing - F10c: within
        max(sigma_pair, kappa x mu / 4), with the reason its run's own lines give: its hook may not have run. Reported to
        the proposer; the next patch at its site without a self-check line waits."""
        if not h.get('patch') or out.gain is None or out.status != 'screened':
            return
        sp = engine.sigma_pair()
        floor = max(sp, self._kappa_for(h, self.kappa()) * h['mu'] / 4) if self.on['hls3_self_proof'] else sp   # F10c
        why = None
        if h['mu'] >= 2 * sp and abs(out.gain) < floor and self.on['hls3_self_proof']:   # (what its run's lines say)
            base = p.base or engine.incumbent()
            ev = dict(self._events(engine, out.cid))
            ev.setdefault('steps', getattr(engine, 'full_steps', None))
            why = self._proof_reason(engine, base, ev, self._claim(engine, base, p, h), h)[0]
        if h['mu'] >= 2 * sp and abs(out.gain) < floor:
            h['inert'] = True
            row = {'candidate': out.cid, 'mechanism': h['mechanism'], 'site': h.get('site'),
                   'predicted_pct': round(100 * math.expm1(h['mu']), 2), 'measured_pct': round(100 * math.expm1(out.gain), 2),
                   **({'reason': why} if why else {})}
            self.inert.append(row)
            if h.get('site'):
                self.inert_sites.update({k: out.cid for k in site_keys(h['site'])})

    def _inert_at_start(self, engine, p, out, h, base) -> None:
        """F10: a patch its probe did not prove: its entry is done (hls2 marked it invalid), the proposer gets the reason,
        the note it targets, the lines it printed and its code."""
        f = self.proof_fail.get(out.cid) or {'reason': (out.reasons or [''])[0]}
        files, code = self._files(engine, base), []
        for fn, text in (p.patch or {}).items():
            have = set(str(files.get(fn) or '').splitlines())
            code += [ln for ln in str(text or '').splitlines() if ln not in have]
        h.update(inert=True, why=f'inert at its probe: {f["reason"]}')
        row = {'candidate': out.cid, 'mechanism': h['mechanism'], 'site': h.get('site'), 'at': 'probe',
               'reason': f['reason'], **({'note': f['note']} if f.get('note') else {}),
               'printed': f.get('lines') or [], **({'parsed': f['parsed']} if f.get('parsed') else {}),
               'patch': [ln[:160] for ln in code[:12]]}
        self.inert.append(row)
        if h.get('site'):
            self.inert_sites.update({k: out.cid for k in site_keys(h['site'])})

    @staticmethod
    def _self_check(patch: dict | None) -> bool:
        return any(SELF_CHECK.search(ln) for text in (patch or {}).values() for ln in str(text or '').splitlines())

    # ------------------------------------------------------------ F6 the rest of a partly adopted entry; F7 warnings
    def _remainder(self, engine, h: dict, old, inc) -> str | None:
        """F6: a multi-knob entry part of which the new incumbent took (the same values): the knobs it did not take,
        carried as an entry on the incumbent - mu less the measured gain of what was taken (the incumbent's own gain when
        its change is that part, else half), memory declared per its knob; none when nothing is left, or it would undo
        a change the incumbent carries."""
        if not self.on['hls3_remainder'] or h.get('patch'):
            return None
        native = self._native(engine)
        c, o = engine.s.candidates[inc], engine.s.candidates[old]
        cfg, ocfg = flat(c.config, native), flat(o.config, native)
        ch = flat(h.get('change') or {}, native)
        taken = {k for k, v in ch.items() if _v(cfg.get(k)) == _v(v) and _v(ocfg.get(k)) != _v(v)}
        rest = {k: v for k, v in (h.get('change') or {}).items() if k not in taken}
        if not taken or not rest or any(_v(cfg.get(k)) != _v(ocfg.get(k)) for k in flat(rest, native)) or \
                self._undoes(engine, inc, rest):
            return None
        moved = {k for k in set(cfg) | set(ocfg) if _v(cfg.get(k)) != _v(ocfg.get(k))}
        t_old, t_inc = engine.t(old), engine.t(inc)
        if moved <= taken and t_old and t_inc:
            mu = h['mu'] - math.log(t_old / t_inc)
        else:
            mu = 0.5 * h['mu']
        key = f'{inc}|rest:{h["mechanism"]}'[:240]
        if key in self.ledger or self._queued(inc, rest) or mu <= 0:
            return None
        mem = {}
        if len(rest) == 1:
            d, g = self._declared(*next(iter(rest.items())))
            if d:
                mem = {'mem_dir': d, **({'mem_gib': g} if g is not None else {})}
        self.ledger[key] = {**{k: h[k] for k in ('change_class', 'rationale') if k in h}, **mem,
                            'mechanism': f'{h["mechanism"]} (the rest on {inc})', 'change': rest, 'patch': None, 'mu': mu,
                            'sigma': h['sigma'], 'p_feasible': min(h['p_feasible'], 0.9), 'base': inc, 'status': 'open',
                            'key': key, 'remainder_of': h['key'], 'carried_from': old, 'rebased': True}
        self.remainders.append({'from': h['mechanism'], 'on': inc, 'taken': sorted(taken), 'rest': rest,
                                'mu_pct': round(100 * math.expm1(mu), 2)})
        return key

    def _warn_p(self, engine, cfg: dict, change: dict, p: float, warn: list) -> float:
        """F7: a memory warning from the experience store or the closed-form estimate does not cap the feasibility of a
        move hls3's own runs leave unconstrained (the memory model sees it fit, or no out-of-memory run bounds it): the
        warning stays for the model to read; a constraint warning, or memory with evidence, caps it at 0.2 as before."""
        if not (self.on['hls3_warn'] and self.on['hls3_feasibility']) or \
                any(not str(w).startswith('memory') for w in warn or []):
            return super()._warn_p(engine, cfg, change, p, warn)
        native = self._native(engine)
        verdict = self._predict(engine, flat(after(cfg, change or {}, native), native))[0]
        if verdict in (None, 'feasible'):
            self.warn_kept += 1
            return p
        return super()._warn_p(engine, cfg, change, p, warn)

    # ------------------------------------------------------------ FABLE_HLS3_FIXES_R3: F15, F16, F19, F23
    def _release(self, engine, h: dict) -> None:
        """F17: a single change measured slower alone on a base where a coverage bundle that held it measured nothing -
        that part explains the bundle's null result: its members go back to their single screens (released)."""
        native = self._native(engine)
        ch = h.get('change') or {}
        if h.get('patch') or len(ch) != 1 or (h.get('measured') or 0.0) >= 0:
            return
        kv = {(k, _v(v)) for k, v in flat(ch, native).items()}
        for bid in dict.fromkeys(c['bundle'] for c in self.captured if c['base'] == h['base'] and not c.get('released')):
            held = {(k, _v(v)) for m in self.members.get(bid) or [] if m in self.ledger
                    for k, v in flat(self.ledger[m].get('change') or {}, native).items()}
            if not kv & held:
                continue
            for c in [c for c in self.captured if c['bundle'] == bid]:
                e = self.ledger.get(c.get('key') or '')
                if e is not None and e['status'] == 'unresolved' and e.get('captured_by'):
                    e.update(status='open', why=f'released: {h["mechanism"][:80]} measured slower alone explains the '
                                                f'bundle {e["captured_by"]}')
                    e.pop('captured_by', None)
                c['released'] = h['mechanism'][:120]

    def _explained(self, engine, base, bid) -> bool:
        """F17: a member of the bundle holds a change measured slower alone on its base - that part explains the bundle's
        null result, so it says nothing of the others (they go back to their single screens)."""
        native = self._native(engine)
        slow = {(k, _v(v)) for e in self.ledger.values() if e['base'] == base and e['status'] == 'refuted'
                and not e.get('patch') and len(e.get('change') or {}) == 1 and (e.get('measured') or 0.0) < 0
                for k, v in flat(e['change'], native).items()}
        return any((k, _v(v)) in slow for m in self.members.get(bid) or [] if m in self.ledger
                   for k, v in flat(self.ledger[m].get('change') or {}, native).items())

    # ------------------------------------------------------------ D7: read-only library source
    def _library_sites(self) -> list[str]:
        """D7: the patch sites the proposer gave and the sites inert patches printed (search/library.names reads the
        last 12 with the job-log notes)."""
        sites = [str(h['site']) for h in self.ledger.values() if h.get('patch') and h.get('site')]
        return sites + [str(r.get('site') or '') for r in self.inert] + [s for r in self.inert for s in (r.get('parsed') or {})]

    def _library_names(self, engine, base) -> list[str]:
        """D7: the code names the job-log notes and the patch sites give (search/library.names)."""
        return library.names(engine, base, self._library_sites())

    def _library_source(self, engine, base) -> dict | None:
        """D7: read-only excerpts of the library source the job imported, at those names (search/library.source)."""
        return library.source(engine, base, self._library_sites(), self.lib_cache)

    # ------------------------------------------------------------ F9b (D4): what a later arm starts from
    def _learned(self) -> dict:
        """F9b: what this run learned of its starts, in knob values - the sets that failed to start together (what each
        member did alone), the pairs two or more failed starts held, the values that failed to start alone (code ideas
        and what came from the store left out)."""
        val = lambda m: {k: json.loads(v) for k, v in sorted(m)}                  # noqa: E731
        knobs_only = lambda ms: not any(str(k).startswith('patch:') for m in ms for k, _ in m)   # noqa: E731
        sets = [{'members': [val(m) for m in t['members']], 'kind': t['kind'], 'run': t['candidate'],
                 'separately': {self._words(frozenset((k, None, v) for k, v in m)): s for m, s in t['separately'].items()}}
                for t in self.together if knobs_only(t['members'])]
        pairs = [{'members': [val(m) for m in sorted(pr, key=sorted)], 'kind': info['kind'], 'run': info['run'],
                  'strikes': info.get('strikes', 1)} for pr, info in self.suspects.items()
                 if not info.get('from_store') and info.get('strikes', 1) >= 2 and knobs_only(pr)]
        alone = [{'members': [{k: json.loads(v)}], 'kind': 'start'} for k, v in sorted(self.alone_fails)
                 if not str(k).startswith('patch:')]
        from ..experience.learned import section   # (the strategy-neutral `learned` section: runs.write_back)
        return section(failed_together=sets, pairs=pairs, alone=alone)

    def _start_from_store(self) -> None:
        """F9b: the start failures earlier arms wrote back (the plan's `start_failures`, from a shared store): every pair
        of each set suspect (F5a: probed first, deferred after a failure here) and each set a composite that died at its
        start (F16-A's partners)."""
        for it in (self.exp or {}).get('start_failures') or []:
            atoms = self._merge_coupled([frozenset({(k, _v(v))}) for m in it.get('members') or [] for k, v in m.items()])
            if len(atoms) < 2 or len(atoms) > PAIR_SET_MAX:
                continue
            for a, b2 in itertools.combinations(atoms, 2):
                self.suspects.setdefault(frozenset({a, b2}), {'run': f'store:{it.get("id")}', 'kind': 'store', 'strikes': 1,
                                                              'from_store': True})
            self.composites.append({'atoms': atoms, 'candidate': f'store:{it.get("id")}', 'kind': 'store',
                                    'origin': 'experience'})
            self.from_store.append({'members': it.get('members'), 'id': it.get('id'), 'message': it.get('message')})

    def _pair_set(self) -> int:
        """F5a's largest failed start whose pairs are suspect - F19: up to the transfer judgment's start composite's size
        (the set a judgment declares to run together), at most PAIR_SET_MAX."""
        if not self.on['hls3_pair_set']:
            return PAIR_SET
        n = len((self.exp.get('judgment') or {}).get('start_config') or {})
        return max(PAIR_SET, min(PAIR_SET_MAX, n))

    def _judged_gain(self, change) -> float | None:
        """The transfer judgment's expected gain (log) for exactly this change; None without one."""
        m = _mkey(change)
        g = [float(a['expected_gain_pct']) for a in (self.exp.get('judgment') or {}).get('assessment') or []
             if isinstance(a, dict) and isinstance(a.get('change'), dict) and _mkey(a['change']) == m
             and isinstance(a.get('expected_gain_pct'), (int, float)) and not isinstance(a.get('expected_gain_pct'), bool)]
        return math.log1p(max(-0.9, max(g) / 100)) if g else None

    def _spender_mu(self, engine, h: dict) -> float:
        """F15: the mu a memory-blocked entry is a spender at (S-2, S-4 and F16 identify and order spenders by it; its pair
        probes are valued at it): its own - but an experience seed with no prior that ran out of memory or thrashed takes
        the most this run predicted for its move (the transfer judgment's expected gain for the same change, the
        proposer's predictions of the same move on any base), else just enough to count (2 sigma_pair at kappa). The
        ledger keeps its zero prior."""
        if not (self.on['hls3_seed_mu'] and h.get('source') == 'experience' and h['mu'] <= 0
                and h['status'] == 'invalid' and h.get('failure') in ('oom', 'thrash')):
            return h['mu']
        m = _mkey(h.get('change'))
        said = [g for g in (self._judged_gain(h.get('change')),) if g is not None]
        said += [float(e['mu']) for e in self.ledger.values() if e.get('source') != 'experience' and not e.get('patch')
                 and not e.get('explore') and not e.get('pair_partner') and _mkey(e.get('change')) == m
                 and isinstance(e.get('mu'), (int, float))]
        said += [float(e['llm_mu']) for e in self.ledger.values() if e.get('source') == 'experience'   # (the model's
                 and _mkey(e.get('change')) == m and isinstance(e.get('llm_mu'), (int, float))]          # twin of a seed)
        best = max(said, default=None)
        if best is None:                                    # (nothing said: at the spender threshold itself)
            best = 2 * engine.sigma_pair() / max(self._kappa_for(h, self.kappa()), 1e-6) * (1 + 1e-6)
        self.seed_mu[h['key']] = round(best, 5)
        return best

    @staticmethod
    def _ev(c) -> dict:
        rec = getattr(c, 'receipt', None) or {}
        return rec.get('evidence') if isinstance(rec.get('evidence'), dict) else {}

    def _jitter(self, engine, p, out, peak=None) -> dict | None:
        """F23: a slower run's update-interval scatter (`scatter`: MAD / median over its window) against its base's, and
        whether its configuration spends memory against the base - declared, measured, or a change of unknown effect
        (F12's basis). Scattered - at max(JITTER_MIN, JITTER_X x the base's) - on a spending configuration: thrash. With a
        measured peak (D3) the peak decides - at NEAR of the card or more: thrash - and the scatter is the fallback."""
        base = p.base or engine.incumbent()
        x = scatter(self._ev(engine.s.candidates.get(out.cid)))
        if isinstance(peak, (int, float)) and self.mem.cap:
            row = {'candidate': out.cid, 'base': base, 'peak_gib': peak, 'near_gib': round(NEAR * self.mem.cap, 2),
                   'scatter_pct': None if x is None else round(100 * x, 2), 'base_pct': None, 'spends': None,
                   'scattered': None, 'thrash': peak >= NEAR * self.mem.cap}
            self.jitter.append(row)
            return row
        if x is None:
            return None
        b = scatter(self._ev(engine.s.candidates.get(base)))
        thr = max(JITTER_MIN, JITTER_X * b) if b is not None else JITTER_MIN
        cfg = self._mcfg(engine, out.cid) if out.cid in engine.s.candidates else \
            self._mcfg(engine, base, p.change, self._patch_ideas(p))
        sp = self.mem.split(self._mcfg(engine, base), cfg, self._known(engine))
        spends = bool(sp['spends'] or sp['unknown'])
        row = {'candidate': out.cid, 'base': base, 'scatter_pct': round(100 * x, 2),
               'base_pct': None if b is None else round(100 * b, 2), 'threshold_pct': round(100 * thr, 2),
               'spends': spends, 'scattered': x >= thr, 'thrash': x >= thr and spends}
        self.jitter.append(row)
        return row

    def _declared_composite(self, p) -> bool:
        """F16-A: a configuration a source declared to run as a whole - the transfer judgment's or the experience's start
        composite (its bundle, S-6's whole step from it), or one hypothesis of two or more knobs at p_feasible 0.5 or
        more (not one hls3 composed: coverage, memory and pair bundles, swaps, retries)."""
        o, hyp = str(p.origin or ''), str(p.hypothesis or '')
        if hyp == 'bundle|exp' or o in ('hls3-whole-experience', 'experience-bundle'):
            return True
        h = self.ledger.get(hyp)
        return h is not None and not hyp.startswith(('bundle|', 'probe|', 'alone|')) and not h.get('patch') \
            and len(h.get('change') or {}) >= 2 and float(h.get('p_feasible') or 0.0) >= 0.5 \
            and not any(h.get(k) for k in ('swap_of', 'retry_of', 'separately_of', 'explore', 'pair_partner'))

    def _bundle_covers(self, engine, inc, h: dict) -> bool:
        """F16: S-2 has a bundle for this spender on the incumbent - one still to run, or its next choice (_s2_choice) -
        whose enablers may free what the spender needs: F3's rule on the GiB declared for each enabler's move (an entry
        without its own declaration takes its move's, _freed_gib: S-2 counts it unknown); an enabler of unknown GiB may."""
        if not self.on['hls3_bundles']:
            return False
        n = self._need(h) if self.on['hls3_need'] else None
        lb = n['need_lb'] if n is not None else 0.0

        def enough(keys) -> bool:
            g = [self._freed_gib(engine, k) for k in keys]
            return not lb or any(x is None for x in g) or sum(g) > lb
        pending = [b for b in self.mem_bundles if b['spender'] == h['key'] and b['base'] == inc
                   and b['origin'] == 'hls3-bundle' and b.get('status') in (None, 'probe passed')]
        if pending:                                         # (one of its S-2 bundles is still to run)
            return any(enough(b['enabler_keys']) for b in pending)
        choice = self._s2_choice(engine, inc, h)
        if choice is None:
            return False
        keys = frozenset([h['key']] + list(choice[0]))
        return keys not in self.evaluated and keys not in self.bundle_ids and enough(choice[0])

    def _freed_gib(self, engine, key) -> float | None:
        """F16: the GiB an enabler entry frees - its own declaration, else what two measured peaks showed for its move
        (D3), else the latest declaration of its single-knob move (a neighbour of the incumbent carries none of its own) -
        None when nobody declared or measured it."""
        e = self.ledger.get(key) or {}
        if e.get('mem_dir') == 'frees' and isinstance(e.get('mem_gib'), (int, float)):
            return float(e['mem_gib'])
        ch = e.get('change') or {}
        if self.on['hls3_peaks'] and self.mem.gib and e.get('base') in engine.s.candidates and ch and not e.get('patch'):
            g = self.mem.gib.get(sig(self._mcfg(engine, e['base']), self._mcfg(engine, e['base'], ch)))
            if g is not None:
                return float(g)
        if len(ch) == 1 and not e.get('patch'):
            d, g = self._declared(*next(iter(ch.items())))
            if d == 'frees' and isinstance(g, (int, float)):
                return float(g)
        return None

    def _line_ids(self, engine, inc) -> set:
        """The incumbent and its ancestors (the reference included)."""
        out, cid = set(), inc
        while cid is not None and cid not in out and cid in engine.s.candidates:
            out.add(cid)
            cid = getattr(engine.s.candidates[cid], 'parent', None)
        return out | {'c0'}

    def _partner_rank(self, change: dict) -> tuple:
        """F16: a partner's order - judged '+' first, then the most predicted for it (the judgment's, the ledger's)."""
        m = _mkey(change)
        sign = self._judgments().get(m, (None, None))[0]
        said = [g for g in (self._judged_gain(change),) if g is not None]
        said += [float(e['mu']) for e in self.ledger.values() if not e.get('patch') and not e.get('pair_partner')
                 and _mkey(e.get('change')) == m and isinstance(e.get('mu'), (int, float))]
        return 0 if sign == '+' else 1 if sign is None else 2, -max(said, default=0.0), m

    def _composite_partners(self, engine, inc, h: dict) -> list[dict]:
        """F16-A: the other knob members of each composite declared to run that died at its start without a memory verdict
        and held this spender (its source says they fit together; alone the spender does not, so an enabler of it is
        among them, whatever was declared): each a change the incumbent does not hold, best first."""
        cur = self._mcfg(engine, inc)
        own = self._merge_coupled([frozenset({(k, b)}) for k, _, b in sorted(sig(cur, self._mcfg(engine, inc, h.get('change'))))])
        out = []
        for c in self.composites:
            if not own or not all(a in c['atoms'] for a in own):
                continue
            for m in c['atoms']:
                if m in own or any(str(k).startswith('patch:') for k, _ in m) or self._has(cur, m):
                    continue
                ch = {k: json.loads(v) for k, v in sorted(m)}
                if ch not in out:
                    out.append(ch)
        return sorted(out, key=self._partner_rank)

    def _undeclared_partners(self, engine, inc, h: dict) -> list[dict]:
        """F16-B: single-knob moves on the incumbent's line whose memory effect no run measured and no declaration names
        (declared neutral, of unknown effect or not declared) that the transfer judgment calls '+' or someone predicted
        to help - a move slower alone included (alone it says nothing of what it enables) - none the spender sets, none
        declared to spend or free (a declared enabler is S-2's), none judged '-', ruled out or held by the incumbent; best
        first."""
        native = self._native(engine)
        cur, cfg0 = self._mcfg(engine, inc), engine.s.candidates[inc].config
        own = set(flat(h.get('change') or {}, native))
        line, judged = self._line_ids(engine, inc), self._judgments()
        said = {_mkey(e.get('change')) for e in self.ledger.values() if e.get('mem_dir') in ('spends', 'frees')
                and not e.get('patch')}
        out = {}
        for e in self.ledger.values():
            ch = e.get('change') or {}
            if e.get('patch') or len(ch) != 1 or e['base'] not in line or e.get('pair_partner') or set(flat(ch, native)) & own:
                continue
            (k, v), = ch.items()
            m = _mkey(ch)
            if m in out or m in said or _v(cur.get(k)) == _v(v):
                continue
            sign = judged.get(m, (None, None))[0]
            if sign == '-' or not (sign == '+' or float(e.get('mu') or 0.0) > 0):
                continue
            if self.mem._knob(k, _v(cur.get(k)), _v(v), self.mem.measured) is not None:
                continue                                    # its memory effect was measured
            if self.start_failures and self._dead_in(engine, cfg0, ch) or (k, _v(v)) in self.alone_fails or \
                    self.on['hls3_prune'] and self._prune_why(engine, inc, ch) or \
                    self.on['hls3_dead_patch'] and self._element_dead(engine, inc, {'change': ch}):
                continue
            out[m] = ch
        return sorted(out.values(), key=self._partner_rank)

    def _pair_probes(self, engine, inc, h: dict) -> list:
        """F16: a memory-blocked spender (out of memory or thrash alone, or deferred by the memory model) that no S-2 bundle
        covers - no declared freeing move, or not enough declared - is probed with its partners (F16-A: the members of a
        composite declared to run that died at its start; else F16-B: up to PAIR_PARTNERS undeclared moves), each pair once,
        probed first with the start cap, counted in F3's tries; queued before anything else. Returns the queued steps."""
        if self._native(engine) or not any(x is h for x in self._spenders(engine, inc)):
            return []
        n = self._need(h) if self.on['hls3_need'] else None
        if n is not None and n['probes'] >= SPENDER_PROBES and not self._unlocked(engine, h, inc):
            return []                                       # F3: tried enough
        if self._bundle_covers(engine, inc, h):
            return []                                       # S-2 bundles it with declared freeing moves
        sm = _mkey(h.get('change'))
        fresh = lambda ch: (sm, _mkey(ch)) not in self.pair_ran and (sm, _mkey(ch), inc) not in self.pair_seen   # noqa: E731
        kind, partners = 'composite', [ch for ch in self._composite_partners(engine, inc, h) if fresh(ch)]
        if not partners:
            kind = 'undeclared'
            done = sum(1 for r in self.pair_probes if r['moves'][0] == sm and r['kind'] == 'undeclared')
            partners = [ch for ch in self._undeclared_partners(engine, inc, h) if fresh(ch)][:max(0, PAIR_PARTNERS - done)]
        out = [q for q in (self._pair_probe(engine, inc, h, ch, kind) for ch in partners) if q is not None]
        if out:
            self.queue = [q for q in self.queue if q.base == inc] + out
        return out

    def _pair_probe(self, engine, inc, h: dict, ch: dict, kind: str):
        """F16: spender h with partner change `ch` on the incumbent as a memory bundle, probed first (a passed probe goes
        on to its full run, valued at the spender's mu); blocked by a suspect pair other than the one it tests whose other
        member only the incumbent holds: the swap (F5c) instead. None when it cannot be made."""
        sm, pm = _mkey(h.get('change')), _mkey(ch)
        words = ', '.join(f'{k}={v}' for k, v in sorted(ch.items()))
        key = next((k for k, e in self.ledger.items() if e['base'] == inc and not e.get('patch') and e.get('change') == ch),
                   None)
        if key is None:                                     # (a partner no entry of the incumbent makes: one of its own)
            key = f'{inc}|pair partner: {words}'[:240]
            self.ledger.setdefault(key, {'mechanism': f'pair partner: {words}', 'change': dict(ch), 'patch': None,
                                         'mu': 0.0, 'sigma': 2 * engine.sigma_pair(), 'p_feasible': 0.7, 'base': inc,
                                         'status': 'superseded', 'key': key, 'pair_partner': True,
                                         'why': f'probed with {h["mechanism"][:120]} (a pair probe): not a move of its own'})
        fs = frozenset([h['key'], key])
        old = self.bundle_ids.get(fs)
        if old is not None and fs not in self.evaluated and old in self.pending:
            del self.pending[old], self.bundle_ids[fs]      # (a pending half of a failed bundle, at its members' zero
        #                                                     priors: the pair probe takes it over, priced and first)
        q = self._mem_bundle(engine, inc, h, [key], origin='hls3-pair-probe', plan=('probe',), gibs={key: None})
        if q is None:
            return None
        self.pair_seen.add((sm, pm, inc))
        self.mem_bundles[-1]['pair'] = kind
        if self.on['hls3_complete']:                        # F4: completed before it is queued
            c2 = self._complete(engine, inc, q.change)
            if c2 is None:
                self.held.pop(q.hypothesis, None)
                self.mem_bundles[-1]['status'] = 'invalid (a workload invariant)'
                return None
            q = dataclasses.replace(q, change=c2)
        q = dataclasses.replace(q, mu=max(q.mu, self._kappa_for(h, self.kappa()) * self._spender_mu(engine, h)))
        self.held[q.hypothesis] = q
        rec = {'spender': h['mechanism'][:160], 'partner': words, 'base': inc, 'kind': kind, 'bundle': q.hypothesis,
               'moves': (sm, pm), 'mu': round(q.mu, 5), 'status': 'queued'}
        self.pair_probes.append(rec)
        cur = self._mcfg(engine, inc)
        tested = frozenset(self._merge_coupled([frozenset({(k, b)}) for k, _, b in
                                                sorted(sig(cur, self._mcfg(engine, inc, q.change)))]))
        blocks = self._pair_blocks(self._mcfg(engine, inc, q.change), tested) if self.on['hls3_pairs'] else []
        if blocks:                                          # F5c: other suspect pairs, a member of each the incumbent's
            s = self._swap_all(engine, inc, q, blocks)
            self.held.pop(q.hypothesis, None)
            if s is None:
                rec['status'] = 'blocked by a suspect pair'
                return None
            rec.update(status='swap', swap=s.mechanism[:200], swap_key=s.hypothesis, mu=round(s.mu, 5))
            return s
        return dataclasses.replace(q, fidelity=('probe',), keep_probe=True, abortable=False,
                                   hypothesis='probe|' + q.hypothesis,
                                   probe_cap=self._probe_cap(engine, inc) if self.on['hls3_start_cap'] else None)

    def _shared_spender(self, engine, p, out, cls, base) -> None:
        """F22: the changes every out-of-memory (or thrashing) run on one base shared - at least two runs, none of them a
        change a run that fitted holds - is a spender of its own however the entries grouped it: its account (F3) takes
        the most any of those runs needed (a bundle's declared freeing plus its shortfall) and their count as its tries,
        so a proposal that holds it waits as F3's spenders wait (SPENDER_PROBES tries, until a measured freeing move)."""
        bid = str(p.hypothesis or '')
        bid = bid[len('probe|'):] if bid.startswith('probe|') else bid
        row = next((b for b in self.mem_bundles if b['bundle'] == bid), None)
        need = float((row or {}).get('freed_gib_declared') or 0.0) + float((cls.get('oom') or {}).get('short_gib') or 0.0)
        atoms = frozenset(self._atoms(engine, p))
        self.oom_runs.append({'base': base, 'atoms': atoms, 'need': round(need, 3), 'cid': out.cid})
        runs = [r for r in self.oom_runs if r['base'] == base]
        if len(runs) < 2:
            return
        common = frozenset.intersection(*[r['atoms'] for r in runs])
        fitted = [r['cfg'] for r in self.mem.runs.values() if r['status'] == 'ok']
        for a in sorted(common, key=sorted):
            if any(str(k).startswith('patch:') for k, _ in a) or any(self._has(cfg, a) for cfg in fitted):
                continue                                    # (a code idea, or a change a run that fitted holds)
            ch = {k: json.loads(v) for k, v in sorted(a)}
            n = self.need.setdefault(_mkey(ch), {'need_lb': 0.0, 'probes': 0, 'unknown': False, 'last': None,
                                                 'change': dict(ch)})
            n['need_lb'] = max(n['need_lb'], max(r['need'] for r in runs))
            n['probes'] = max(n['probes'], sum(1 for r in self.oom_runs if a in r['atoms']))
            n['last'] = base
            self.shared[self._words(frozenset((k, None, v) for k, v in a))] = {
                'need_lb': round(n['need_lb'], 3), 'probes': n['probes'], 'runs': [r['cid'] for r in runs]}

    def _pair_blocks(self, cfg: dict, tested: frozenset) -> list:
        """F16: every suspect pair other than the tested one that the configuration holds."""
        return [pr for pr in self.suspects if not pr <= tested and all(self._has(cfg, m) for m in pr)]

    def _swap_all(self, engine, inc, q, blocks: list):
        """F16 (F5c for a pair probe): the pair probe without every incumbent member of the suspect pairs it would hold
        (their knobs back to the reference's values) - mu less each member's measured gain, at 2 sigma_pair or more; probed
        first. None when a pair has no incumbent member to drop (both are what it tests) or too little is left."""
        native, cur = self._native(engine), self._mcfg(engine, inc)
        own = {(kk, _v(vv)) for kk, vv in flat(q.change or {}, native).items()}
        drop = []
        for pr in blocks:
            held = [m for m in pr if self._has(cur, m) and not (m & own)]
            if len(held) != 1 or any(str(kk).startswith('patch:') for kk, _ in held[0]):
                return None
            if held[0] not in drop:
                drop.append(held[0])
        ref = self._mcfg(engine, 'c0')
        change = dict(q.change or {})
        for w in drop:
            change.update({kk: (None if ref.get(kk) is None else json.loads(_v(ref.get(kk)))) for kk, _ in w})
        mu = q.mu - sum(self._member_gain(engine, inc, w) for w in drop)
        words = '; '.join(self._words(frozenset((kk, None, vv) for kk, vv in w)) for w in drop)
        self.swaps.append({'base': inc, 'without': words, 'for': q.mechanism, 'mu': round(mu, 5),
                           'proposed': mu >= 2 * engine.sigma_pair()})
        if mu < 2 * engine.sigma_pair():
            return None
        key = f'{inc}|swap:{words}:{q.mechanism}'[:240]
        if key in self.ledger:
            return None
        sigma = max(q.sigma, 2 * engine.sigma_pair())
        self.ledger[key] = {'mechanism': f'{q.mechanism} without {words} (they failed to start together)', 'change': change,
                            'patch': q.patch, 'mu': mu, 'sigma': sigma, 'p_feasible': min(q.p_feasible, 0.5), 'base': inc,
                            'status': 'open', 'key': key, 'swap_of': words, 'swap_for': q.hypothesis}
        self.swaps[-1]['entry'] = key
        return Proposal(change=change, patch=q.patch, base=inc, mu=mu, sigma=sigma, p_feasible=min(q.p_feasible, 0.5),
                        mechanism=self.ledger[key]['mechanism'], hypothesis=key, origin='hls3-swap',
                        fidelity=('probe', 'full'), keep_probe=True,
                        probe_cap=self._probe_cap(engine, inc) if self.on['hls3_start_cap'] else None)

    def residual(self, engine) -> dict:
        out = super().residual(engine)
        if any(self.on.values()):
            out['deferred'] = sum(1 for h in self.ledger.values() if h['status'] == 'open' and h.get('deferred'))
        return out


STALL = 1.5                     # S-9: an update this many times the median is a stall


def update_stats(ev) -> dict | None:
    """The scoring window's per-update intervals: mean, median and the stalls (updates >= STALL x the median)."""
    if not isinstance(ev, dict):
        return None
    w = ev.get('window')
    if not (isinstance(w, (list, tuple)) and len(w) == 2):
        return None
    gaps = intervals(ev.get('per_step') or [], list(w))[1]
    if len(gaps) < 3:
        return None
    med = statistics.median(gaps)
    return {'n': len(gaps), 'mean': statistics.fmean(gaps), 'median': med,
            'stalls': sum(1 for x in gaps if med > 0 and x >= STALL * med)}


def scatter(ev) -> float | None:
    """F23: the scatter of a run's adjacent update intervals - their median absolute deviation over their median - in its
    window: from the 11th update when it ran 20 or more, else from the 3rd (an aborted run's updates past the warm-up);
    None with fewer than 5 intervals."""
    if not isinstance(ev, dict):
        return None
    ps = sorted((r for r in ev.get('per_step') or [] if isinstance(r, dict) and isinstance(r.get('t'), (int, float))
                 and isinstance(r.get('step'), int)), key=lambda r: r['step'])
    first = 11 if len(ps) >= 20 else 3
    ts = [r['t'] for r in ps if r['step'] >= first]
    d = [b - a for a, b in zip(ts, ts[1:])]
    if len(d) < 5:
        return None
    med = statistics.median(d)
    if med <= 0:
        return None
    return statistics.median([abs(x - med) for x in d]) / med


def _pct(x) -> str:
    return 'no reading' if x is None else f'{x} %'


def _gain_of(h: dict) -> float:
    return h['measured'] if h.get('measured') is not None else h['mu']


def changed_lines(files: dict, patch: dict) -> int:
    """The lines a patch adds or removes against the files it is applied to (S-8: the gain per changed line)."""
    n = 0
    for f, text in (patch or {}).items():
        a, b = str(files.get(f) or '').splitlines(), str(text or '').splitlines()
        n += sum(1 for ln in difflib.unified_diff(a, b, lineterm='', n=0)
                 if ln[:1] in '+-' and not ln.startswith(('+++', '---')))
    return n
