"""wcs: the whole-configuration search (FABLE_ARCH_REVIEW_20261031 section 4) on the Engine.

The knob side of the search is a model over the whole knob table, not a ledger of moves on an incumbent:

    prior      one model call before any run - for every knob value its expected gain, its memory direction and GiB,
               a warm-up, the pairs that enable or conflict, up to four whole configurations (component wcs_prior);
               the experience store's measured moves replace the model's guess where they cover a value
    feasible   P(a configuration fits) = the sharded-state arithmetic and the declared memory against the measured peak
               (never 0 by declarations alone) x dominance by the runs (hls3's MemoryModel: a configuration that needs at
               least what an out-of-memory or thrashing run needed does not fit; one that needs no more than a run that
               fitted does) x the pair table (values that failed to start together) (wcs_feasibility, wcs_antagonism)
    surrogate  the log step time gain over the reference as a Bayesian linear model over the knob values (one feature per
               value other than the reference's; a proven patch is a pseudo-knob; a declared enabling pair a term), the
               prior's means and spreads, every valid run an observation: the measurements dominate the prior
    acquire    one joint draw of the posterior (wcs_thompson) scores every legal configuration of the enumerated table
               (`knob_space.choices` filtered by the session's own check; a table above the cap is sampled); the draw's
               best improvement over the incumbent, weighted by feasibility and cost, is proposed; the first three screens
               differ in at least two big knobs (wcs_diversity); a configuration below 0.7 feasibility is probed first
               with hls3's start cap and speed gate (wcs_probe_first); a spender the runs block gets one probe with each
               knob of unknown memory direction (wcs_partner_probe); the last screen is a coverage bundle (wcs_coverage)
    code       the model asked for patches only - at most three per ask, three asks, when two or more screens are left and
               none is pending - each proved at its probe (hls3's self-proof); a proven patch rides on every later
               configuration as a pseudo-knob with its own measured effect (wcs_code)

What stays as it is: the Engine (fidelities, abort, noise, incumbent, value / cost), the session (verifier, admission,
budgets, patch cap), the promotion, random, hls / hls2 / hls3. The strategy returns at most a few proposals per step and
lets Engine.decide order them by expected improvement per cost (a patch competes with the next configuration on the
same scale), so the engine's stop rule applies as for every other strategy. Nothing here names a model, a library, an
environment or a knob: every fact comes from the manifest, the prior call, the experience store and the runs.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import re
import statistics

from . import WCS_COMPONENTS
from .engine import BoundGate, Outcome, Proposal, Strategy
from .hls3 import DIRS, GATE_LOW, GATED_STEPS, TOGETHER, WARMUP, WARMUP_MAX, HLS3, MemoryModel, after, flat, sharded_floor, sig
from .knob_space import choices
from .proof import PROBE_STEPS_MAX, files_of
from .proof import check as proof_check
from .strategies import merge_patches

TABLE_CAP = 50000               # the knob product enumerated at most (the recorded tables: 2,048-49,152 with the invariant's
#                                 illegal combinations); a larger one is sampled (as random draws it)
PAIR_SET_MAX = 12               # a start failure that added this many values at most marks the pairs among them
SD_UNKNOWN = 0.05               # prior spread of a value the model does not know (log gain)
SD_PAIR = 0.03                  # prior spread of a declared enabling pair's own term
PAIR_ENABLES_MU = 0.03          # an enabling pair's own gain when the model gives none (it pays together: positive)
SD_CONFIG = 0.05                # a whole configuration's stated gain enters as an observation with at least this spread
SD_FLOOR = 0.02                 # no prior spread below this: the model's confidence never pins a value
P_UNSURE = 0.4                  # feasibility of a configuration the memory model calls unsure
P_DECLARED_MIN = 0.15           # feasibility by declarations alone never drops below this (a declaration asks for a probe)
P_FAR = 0.03                    # the store's closed-form estimate at 1.25x the card or more: nearly ruled out
PEAK_LINE = 0.95                # with the reference's reserved peak measured, the line is the card less this reserve (the
                                # reserved peak already holds the allocator's slack; NEAR_CAP applies to the floor estimate)
SPEND_STEP, SPEND_MAX = 0.25, 2.0   # a failure the arithmetic rated likely to fit: declared spends scaled up by this step
RESCUE_MAX, RESCUE_SHORT = 3, 0.05  # edge-rescue probes per search; an OOM short by at most this share of the card is 'at the edge'
DECLARED_ERR = 0.5              # the relative error of a declared GiB (the recorded declarations: a third or more off; the
#                                 memory toy's simulated model: 30 %) - a wide one: a full run lost to an out-of-memory
#                                 configuration costs a screen, a probe a third of one
NEAR_CAP = 0.9                  # the allocator thrashes from about this share of the card (FABLE_TAKEAWAYS T3: 92-97 %)
UNKNOWN_SPEND, UNKNOWN_SD = 0.12, 0.15   # a move of unknown memory direction: its expected spend and spread (of the card)
P_PAIR_DECLARED = 0.3           # a pair the prior calls a conflict, before any run
P_PAIR_FAILED = 0.05            # a pair that failed to start together in a run with few new values
P_ALONE_FAILED = 0.2            # a value that failed to start alone once (twice: 0)
PROBE_BELOW = 0.7               # a configuration below this feasibility is probed first
DIVERSITY_SCREENS = 3           # the first full screens differ from every run in at least two big knobs
BIG = 0.02                      # a knob is big when some value of it has a prior mean gain of this much (log)
JITTER = 0.02                   # the cliff: MAD / median of the update intervals at least this (and 5x the base's)
JITTER_X = 5.0
CODE_ASKS, PATCHES_PER_ASK = 3, 3
LLM_CALLS_MAX = 6               # model calls a search may make in all: the prior, the review, the asks and their re-asks
PARTNER_MAX = 4                 # partner probes in one search at most
BLOCKED_AFTER, BLOCKED_VALUE = 3, 0.25   # a value held by this many failed runs and no fitted one: its feasibility x this
ABORT_SD = 2.0                  # an aborted run's partial time is an observation with this many times the noise
ABORT_MIN, ABORT_DEPTH, ABORT_BOUND_SD = 0.02, 0.05, 0.08   # ... and without a partial time: the incumbent's gain less
                                         # max(2 sigma_pair, ABORT_MIN) less ABORT_DEPTH (an aborted run is usually well
                                         # below the rule's margin), spread at least ABORT_BOUND_SD: a bound, not a measurement


def vk(v) -> str:
    """A knob value as a key (json; None as 'null')."""
    return json.dumps(v, sort_keys=True, default=str)


def _coerce(v, values):
    """A value the model wrote matched to one of a knob's values (json text, numbers as strings, case), else None."""
    keys = {vk(x): x for x in values}
    if vk(v) in keys:
        return keys[vk(v)]
    if isinstance(v, str):
        try:
            w = json.loads(v)
            if vk(w) in keys:
                return keys[vk(w)]
        except ValueError:
            pass
        low = {str(x).lower(): x for x in values}
        if v.strip().lower() in low:
            return low[v.strip().lower()]
    return None


def _num(x, lo=None, hi=None):
    if isinstance(x, str):                               # a number the model wrote as text
        try:
            x = float(x.strip().rstrip('%'))
        except ValueError:
            return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    x = float(x)
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def proposer_flags(off=()) -> dict:
    """What the LLMProposer of a wcs run asks of the model (proposers.FEATURES and hls3's own fields): robust parsing
    (F14) always; for the code track its patches' site, memory, self-proof lines (F10 / F13), `deferred` line (F21),
    warm-up, the inert-patch feedback and the read-only library source (D7); the memory view and the failures as context."""
    off = set(off or ())
    code = 'wcs_code' not in off
    return {'parse': True, 'self_proof': code, 'deferred': code, 'library_source': code,
            'memory': code, 'sites': code, 'warmup': code, 'inert': code,
            'memory_view': 'wcs_feasibility' not in off, 'failures': True, 'together': 'wcs_antagonism' not in off}


# ---------------------------------------------------------------- the prior
class Prior:
    """The model's prior over the knob table, parsed robustly: per value (knob, vk) -> {'mu', 'sd', 'mem_dir', 'mem_gib',
    'warmup', 'p_runs', 'why'} (mu None: unknown), `pairs` [{'members': [(knob, vk)], 'kind', 'why'}], `configs`
    [{'name', 'change', 'why'}], `notes`. `flat(knobs, ref)`: no prior (every value unknown)."""

    def __init__(self):
        self.levels: dict[tuple, dict] = {}
        self.pairs: list[dict] = []
        self.configs: list[dict] = []
        self.notes = ''
        self.source = 'none'
        self.dropped = 0                                # entries of the answer that named no knob value of the table

    @classmethod
    def flat(cls, knobs: dict, ref: dict) -> 'Prior':
        p = cls()
        for k, vals in knobs.items():
            for v in vals:
                if vk(v) != vk(ref.get(k)):
                    p.levels[(k, vk(v))] = {'mu': None, 'sd': None, 'mem_dir': None, 'mem_gib': None, 'warmup': None,
                                            'p_runs': None, 'why': ''}
        p.source = 'flat'
        return p

    @classmethod
    def from_answer(cls, answer, knobs: dict, ref: dict) -> 'Prior':
        p = cls.flat(knobs, ref)
        p.source = 'model'
        a = answer if isinstance(answer, dict) else {}
        for row in (a.get('levels') or [])[:400]:
            if not isinstance(row, dict) or not isinstance(row.get('knob'), str) or row['knob'] not in knobs:
                p.dropped += 1                           # (a knob name that is not a string: schema-valid, unusable)
                continue
            k = row['knob']
            v = _coerce(row.get('value'), knobs[k])
            if v is None or vk(v) == vk(ref.get(k)):
                p.dropped += 1
                continue
            e = p.levels[(k, vk(v))]
            g = _num(row.get('gain_pct'), -90.0, 400.0)
            e['mu'] = None if g is None else math.log1p(g / 100.0)
            s = _num(row.get('sd_pct'), 0.0, 200.0)
            e['sd'] = None if s is None else math.log1p(s / 100.0)
            m = row.get('memory') if isinstance(row.get('memory'), dict) else {}
            d = str(m.get('direction') or '').strip().lower()
            e['mem_dir'] = d if d in DIRS else None
            gib = _num(m.get('gib'), 0.0, 10000.0)
            e['mem_gib'] = gib if d in ('spends', 'frees') and gib else None
            w = row.get('warmup_updates')
            e['warmup'] = w if isinstance(w, int) and not isinstance(w, bool) and 1 <= w <= WARMUP_MAX else None
            e['p_runs'] = _num(row.get('p_runs'), 0.0, 1.0)
            e['why'] = str(row.get('why') or '')[:160]
        for row in (a.get('interactions') or [])[:60]:
            if not isinstance(row, dict) or not isinstance(row.get('members'), list):
                continue
            members = []
            for m in row['members'][:4]:
                if isinstance(m, dict) and isinstance(m.get('knob'), str) and m['knob'] in knobs:
                    v = _coerce(m.get('value'), knobs[m['knob']])
                    if v is not None and vk(v) != vk(ref.get(m['knob'])):
                        members.append((m['knob'], vk(v)))
            kind = str(row.get('kind') or '').strip().lower()
            if len(members) >= 2 and kind in ('enables', 'conflicts', 'redundant') and len({m[0] for m in members}) == len(members):
                g, sd = _num(row.get('gain_pct'), -90.0, 400.0), _num(row.get('sd_pct'), 0.0, 400.0)
                p.pairs.append({'members': members, 'kind': kind, 'why': str(row.get('why') or '')[:160],
                                'mu': None if g is None else math.log1p(g / 100.0),      # the members' gain together
                                'sd': None if sd is None else math.log1p(sd / 100.0)})   # beyond their own
        for row in (a.get('configs') or [])[:4]:
            if not isinstance(row, dict) or not isinstance(row.get('change'), dict):
                continue
            change = {}
            for k, v in row['change'].items():
                if k in knobs:
                    w = _coerce(v, knobs[k])
                    if w is not None:
                        change[k] = w
            if change:
                g, sd = _num(row.get('gain_pct'), -90.0, 400.0), _num(row.get('sd_pct'), 0.0, 400.0)
                p.configs.append({'name': str(row.get('name') or f'config {len(p.configs) + 1}')[:60], 'change': change,
                                  'why': str(row.get('why') or '')[:160],
                                  'mu': None if g is None else math.log1p(g / 100.0),    # its gain over the reference
                                  'sd': None if sd is None else math.log1p(sd / 100.0)})
        p.notes = str(a.get('notes') or '')[:400]
        return p

    def absorb(self, rows, knobs: dict, ref: dict) -> int:
        """Experience seeds (store.plan 'seeds': {'change', 'gain', 'level'}): a covered single-knob move's measured gain
        replaces the model's guess (sd half of it, at least SD_FLOOR); returns how many."""
        n = 0
        for row in rows or []:
            ch = row.get('change') if isinstance(row, dict) else None
            if not isinstance(ch, dict) or len(ch) != 1 or row.get('level') != 'covered':
                continue
            (k, v), = ch.items()
            if k not in knobs or vk(v) == vk(ref.get(k)) or (k, vk(v)) not in self.levels:
                continue
            g = _num(row.get('gain'), -3.0, 3.0)
            if g is None:
                continue
            e = self.levels[(k, vk(v))]
            e.update(mu=g, sd=max(SD_FLOOR, 0.5 * abs(g)), why=(e.get('why') or '') + ' [experience]')
            n += 1
        return n

    def declared(self, ref: dict) -> dict:
        """Move -> memory direction for the memory model (each value's change from the reference)."""
        out = {}
        for (k, key), e in self.levels.items():
            if e.get('mem_dir') in DIRS:
                out[frozenset({(k, vk(ref.get(k)), key)})] = e['mem_dir']
        return out

    def describe(self) -> dict:
        known = [e for e in self.levels.values() if e.get('mu') is not None]
        return {'source': self.source, 'values': len(self.levels), 'known': len(known),
                'declared_memory': sum(1 for e in self.levels.values() if e.get('mem_dir')),
                'pairs': [{'members': [f'{k}={m}' for k, m in p['members']], 'kind': p['kind'],
                           **({'gain_pct': round(100 * math.expm1(p['mu']), 2)} if p.get('mu') is not None else {})}
                          for p in self.pairs],
                'configs': [{'name': c['name'], 'change': c['change'],
                             **({'gain_pct': round(100 * math.expm1(c['mu']), 2)} if c.get('mu') is not None else {})}
                            for c in self.configs], 'dropped': self.dropped,
                'top': sorted(({'value': f'{k}={m}', 'gain_pct': round(100 * math.expm1(e['mu']), 2),
                                'memory': e.get('mem_dir')} for (k, m), e in self.levels.items() if e.get('mu') is not None),
                              key=lambda r: -r['gain_pct'])[:8], 'notes': self.notes}


# ---------------------------------------------------------------- the surrogate
class Surrogate:
    """Bayesian linear regression of the log gain over the reference on indicator features (a knob value other than the
    reference's, a proven patch, a declared enabling pair), independent Gaussian priors per feature, Gaussian noise per
    observation. Pure Python (at most a few dozen features): the posterior precision A = diag(1/sd0^2) + sum x x^T / s^2
    is solved by Cholesky; a draw is mu + L^-T z."""

    def __init__(self):
        self.names: list = []
        self.mu0: list[float] = []
        self.sd0: list[float] = []
        self.obs: list[tuple] = []                       # (features: dict name -> 1, y, sd)
        self._post = None

    def ensure(self, name, mu0: float = 0.0, sd0: float = SD_UNKNOWN) -> int:
        if name in self.names:
            return self.names.index(name)
        self.names.append(name)
        self.mu0.append(float(mu0))
        self.sd0.append(max(SD_FLOOR, float(sd0)))
        self._post = None
        return len(self.names) - 1

    def observe(self, feats: dict, y: float, sd: float, tag=None) -> None:
        for f in feats:
            self.ensure(f)
        self.obs.append((dict(feats), float(y), max(1e-3, float(sd)), tag))
        self._post = None

    def retire(self, pred) -> int:
        """Drops the tagged observations `pred(tag, feats)` accepts (a suggestion a measurement supersedes)."""
        keep = [o for o in self.obs if o[3] is None or not pred(o[3], o[0])]
        n = len(self.obs) - len(keep)
        if n:
            self.obs, self._post = keep, None
        return n

    def _solve(self):
        n = len(self.names)
        idx = {f: i for i, f in enumerate(self.names)}
        A = [[0.0] * n for _ in range(n)]
        b = [0.0] * n
        for i in range(n):
            A[i][i] = 1.0 / self.sd0[i] ** 2
            b[i] = self.mu0[i] / self.sd0[i] ** 2
        for feats, y, sd, _ in self.obs:
            w = 1.0 / sd ** 2
            ids = [idx[f] for f in feats if f in idx]
            for i in ids:
                b[i] += w * y
                for j in ids:
                    A[i][j] += w
        L = [[0.0] * n for _ in range(n)]                 # Cholesky A = L L^T
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = math.sqrt(max(s, 1e-12)) if i == j else s / L[j][j]
        z = [0.0] * n                                     # L z = b; L^T mu = z
        for i in range(n):
            z[i] = (b[i] - sum(L[i][k] * z[k] for k in range(i))) / L[i][i]
        mu = [0.0] * n
        for i in reversed(range(n)):
            mu[i] = (z[i] - sum(L[k][i] * mu[k] for k in range(i + 1, n))) / L[i][i]
        self._post = (mu, L)
        return self._post

    def posterior(self) -> tuple[list[float], list[list[float]]]:
        return self._post or self._solve()

    def mean(self) -> dict:
        mu, _ = self.posterior()
        return dict(zip(self.names, mu))

    def sd(self) -> dict:
        """The marginal posterior sd per feature (the diagonal of (L L^T)^-1)."""
        mu, L = self.posterior()
        n = len(self.names)
        out = {}
        for j in range(n):                               # column j of L^-1 via forward substitution; diag of L^-T L^-1
            e = [0.0] * n
            for i in range(n):
                s = (1.0 if i == j else 0.0) - sum(L[i][k] * e[k] for k in range(j, i))
                e[i] = s / L[i][i] if i >= j else 0.0
            out[self.names[j]] = math.sqrt(sum(x * x for x in e))
        return out

    def sample(self, rng: random.Random) -> dict:
        mu, L = self.posterior()
        n = len(self.names)
        z = [rng.gauss(0.0, 1.0) for _ in range(n)]
        v = [0.0] * n                                     # L^T v = z (back substitution): v ~ N(0, A^-1)
        for i in reversed(range(n)):
            v[i] = (z[i] - sum(L[k][i] * v[k] for k in range(i + 1, n))) / L[i][i]
        return {f: mu[i] + v[i] for i, f in enumerate(self.names)}

    @staticmethod
    def value(feats: dict, beta: dict) -> float:
        return sum(beta.get(f, 0.0) for f in feats)

    def var(self, feats: dict) -> float:
        """The posterior variance of a configuration's predicted gain: x^T A^-1 x = |L^-1 x|^2."""
        _, L = self.posterior()
        idx = {f: i for i, f in enumerate(self.names)}
        n = len(self.names)
        x = [0.0] * n
        for f in feats:
            if f in idx:
                x[idx[f]] = 1.0
        y = [0.0] * n
        for i in range(n):
            y[i] = (x[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
        return sum(t * t for t in y)


# ---------------------------------------------------------------- the strategy
class WCS(Strategy):
    """The whole-configuration search (module docstring). `off`: components switched off (their ablate names); `facts`:
    ProjectHooks.memory_facts (card, GPUs, parameters); `experience`: experience.store.plan (its seeds are data for the
    prior; the transfer judgment is not called - the prior call replaces it); `kappa0`: the shrinkage of the model's
    patch predictions (the knob prior is shrunk by its own spread, not by kappa)."""
    name = 'wcs'
    uses_engine_stop = True
    # search-state checkpoints (agent/state.py): the proposer (a part: its model handles are the resuming run's) and the
    # accounting view (rebuilt from the project) are live
    CHECKPOINT_LIVE = ('proposer', 'accounting')

    def checkpoint_parts(self) -> dict:
        return {'proposer': self.proposer}

    def state_dict(self) -> dict:
        out = super().state_dict()
        if isinstance(out.get('exp'), dict):             # the plan's `transfer` hook is HLS's (once, at its start; a wcs
            out['exp'] = {k: v for k, v in out['exp'].items() if not callable(v)}   # strategy never calls it): not state
        return out

    # hls3's methods used as they are (an explicit borrowing, no inheritance): the failure classifier and its evidence
    # for the memory edge, the probe's start cap and the budget rules of a kept probe, the self-proof of a patch
    classify, _near_cap, _line = HLS3.classify, HLS3._near_cap, staticmethod(HLS3._line)
    _mcfg, _pids, _native = HLS3._mcfg, HLS3._pids, HLS3._native
    _t_first, _probe_cap, _world, _wall_full, _base_intervals = (HLS3._t_first, HLS3._probe_cap, HLS3._world, HLS3._wall_full,
                                                                 HLS3._base_intervals)
    _screens_scarce, _probe_pays, _costs_no_screen, _admits, _extra_probe = (HLS3._screens_scarce, HLS3._probe_pays,
                                                                             HLS3._costs_no_screen, HLS3._admits, HLS3._extra_probe)

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        """What this strategy asks of its LLMProposer (proposers.FEATURES and the patch fields): proposer_flags."""
        return proposer_flags(off)

    @staticmethod
    def _supported(h, key: str) -> int | None:
        """A declaration honoured - `warmup_updates` 1..WARMUP_MAX, `active_from_update` 1..PROBE_STEPS_MAX - else None."""
        hi = {'warmup_updates': WARMUP_MAX, 'active_from_update': PROBE_STEPS_MAX}[key]
        v = (h or {}).get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= hi else None

    def __init__(self, proposer, *, off=(), facts: dict | None = None, accounting=None, experience: dict | None = None,
                 kappa0: float = 0.5, probe_below: float = PROBE_BELOW, seed: int = 42, code_asks: int = CODE_ASKS,
                 patches_per_ask: int = PATCHES_PER_ASK, table_cap: int = TABLE_CAP, diversity_screens: int = DIVERSITY_SCREENS):
        self.proposer = proposer
        off = set(off or ())
        self.on = {k: k not in off for k in WCS_COMPONENTS}
        # hls3's borrowed code reads these switches: its probe rules, its measured-peak reading (D3), its own jitter
        # rule (off: wcs has its own, _jitter)
        self.on['hls3_start_cap'] = self.on['wcs_probe_first']
        self.on['hls3_peaks'] = self.on['wcs_feasibility']
        self.on['hls3_jitter_thrash'] = False
        self.facts = dict(facts or {})
        self.accounting = accounting
        self.exp = experience or {}
        self.kappa0, self.probe_below = float(kappa0), float(probe_below)
        self.rng = random.Random(seed)
        self.code_asks, self.patches_per_ask, self.table_cap = int(code_asks), int(patches_per_ask), int(table_cap)
        self.diversity_screens = int(diversity_screens)
        # hls3's borrowed methods read these
        self.mem = MemoryModel(self.facts.get('cap_gib'))
        self.mem.floor = sharded_floor(self.facts) if self.on['wcs_feasibility'] else None
        self.gated: list[dict] = []
        self.pids: dict[str, frozenset] = {'c0': frozenset()}
        self.gate_ratio = GATE_LOW
        self.proof_fail: dict[str, dict] = {}
        self.note_feedback: list[dict] = []
        self.from_store: list[dict] = []                 # D4: the start failures earlier arms wrote back, read at start
        self.started: list[dict] = []                    # D4: this run's failed starts (sets of values), for write-back
        # the search's own state
        self.prior: Prior | None = None
        self.surr = Surrogate()
        self.knobs: dict = {}
        self.ref: dict = {}
        self.table: list[dict] | None = None
        self.table_whole = True
        self.declared: dict = {}                         # move -> direction the prior declared (the memory model's known)
        self.pairs: dict[frozenset, dict] = {}           # pair of (knob, vk) -> {'p', 'why'} (the antagonism table)
        self.alone: dict[tuple, int] = {}                # (knob, vk) -> start failures alone
        self.runs: dict[str, dict] = {}                  # cid -> {'cfg', 'kind', 'y', 'spends', 'origin', ...}
        self.tried: set = set()                          # configuration keys (knob values + patch ids) run or proposed
        self.ideas: dict[str, frozenset] = {}            # proposal id -> the patch ids it carries
        self.ledger: dict[str, dict] = {}                # proposal id -> its row (the trace)
        self.picks: list[dict] = []                      # each step's choice and why
        self.partner_done: set = set()
        self.partner_skip: set = set()                   # (blocking run, knob) with no legal, fresh form: not a chance
        self.patches: dict[str, dict] = {}               # patch id -> {'files', 'status', 'mu', 'mechanism', 'site', ...}
        self.chain: list[str] = []                       # the proven patches every later configuration carries
        self.overlay_best: dict | None = None            # ... their files (the overlay of the last proven candidate)
        self.queue: list[Proposal] = []                  # patch proposals waiting for their turn
        self.asks, self.last_ask_at = 0, -1
        self.inert: list[dict] = []
        self.failures: list[dict] = []
        self.counts = {'prior_config': 0, 'rescue_probes': 0, 'calibrations': 0, 'infeasible': 0, 'unsure': 0, 'probe_first': 0, 'partner_probes': 0, 'thompson': 0, 'greedy': 0,
                       'coverage': 0, 'patches': 0, 'diversity_filtered': 0, 'jitter_thrash': 0, 'unprobed_skipped': 0,
                       'blocked_values': 0}
        self.llm_calls = 0
        self._n = 0
        self._fcache: dict[tuple, dict] = {}
        self._probe_ratio = 0.35                         # a probe's cost over a full run's (the engine's, once measured)
        self.spend_scale = 1.0                           # declared spends x this (raised by a failure the arithmetic did not expect)
        self.rescue_done: set = set()                    # (run, knob, value) edge rescues made
        self._pcache: dict[tuple, tuple] = {}            # feasibility by configuration, cleared by every observation
        self._ocache: dict[str, object] = {}             # the known moves and the blocked values, likewise
        if hasattr(proposer, 'ask_proof'):               # the simulated model prints the two proof lines when asked
            proposer.ask_proof = self.on['wcs_code']
        if hasattr(proposer, 'context'):
            proposer.context = self.context
        if hasattr(proposer, 'patch_only'):
            proposer.patch_only = True

    # ------------------------------------------------------------ set-up
    def _init(self, engine) -> None:
        s = engine.s
        self.knobs = choices(s.task)
        c0 = s.candidates['c0']
        self.ref = flat(c0.config, self._native(engine))
        self.ref = {k: self.ref.get(k) for k in self.knobs}
        prior = None
        if self.on['wcs_prior'] and hasattr(self.proposer, 'knob_prior') and self._llm_used() < LLM_CALLS_MAX:
            self._allowance()
            try:
                answer = self.proposer.knob_prior(engine, 'c0')
            except Exception as exc:                     # noqa: BLE001 - a failed call is a flat prior, not a dead search
                answer, exc_text = None, str(exc)
                (getattr(engine, 'log', None) or (lambda m: None))(f'[search] wcs: the prior call failed: {exc_text[:200]}')
            if isinstance(answer, dict):
                try:                                     # the parsing is inside the same boundary: an answer the
                    prior = Prior.from_answer(answer, self.knobs, self.ref)   # parser cannot use is a flat prior too
                except Exception as exc:                 # noqa: BLE001
                    prior = None
                    (getattr(engine, 'log', None) or (lambda m: None))(f'[search] wcs: the prior answer is unusable: {str(exc)[:200]}')
        self.llm_calls = self._llm_used()
        self.prior = prior or Prior.flat(self.knobs, self.ref)
        self.prior.absorbed = self.prior.absorb(self.exp.get('seeds'), self.knobs, self.ref) if self.exp else 0
        for (k, key), e in self.prior.levels.items():
            mu = e['mu'] if e.get('mu') is not None else 0.0
            sd = e['sd'] if e.get('sd') is not None else SD_UNKNOWN
            if e.get('mu') is not None:
                sd = max(sd, 0.5 * abs(mu))              # the model's confidence never pins a value: half its own size
            self.surr.ensure((k, key), mu, sd)
        for p in self.prior.pairs:
            if p['kind'] == 'enables':                   # its own term: the stated gain, else the enabling default
                mu = p['mu'] if p.get('mu') is not None else PAIR_ENABLES_MU
                self.surr.ensure(('pair', frozenset(p['members'])), mu, max(p.get('sd') or 0.0, 0.5 * abs(mu), SD_PAIR))
            elif p['kind'] == 'conflicts' and self.on['wcs_antagonism']:
                for a, b in itertools.combinations(p['members'], 2):
                    self.pairs[frozenset((a, b))] = {'p': P_PAIR_DECLARED, 'why': 'declared a conflict by the prior'}
        seen = set()
        for c in self.prior.configs:                   # a whole configuration's stated gain: a wide observation, once
            if c.get('mu') is not None:
                cfg = {**{k: self.ref.get(k) for k in self.knobs}, **{k: v for k, v in c['change'].items() if k in self.knobs}}
                key = self._ckey(cfg, ())
                if key in seen:
                    continue
                seen.add(key)
                self.surr.observe(self._feats(cfg), c['mu'], max(c.get('sd') or 0.0, SD_CONFIG), tag=('config', key))
        self.declared = self.prior.declared(self.ref) if self.on['wcs_feasibility'] else {}
        ev = (c0.receipt or {}).get('evidence') if isinstance((c0.receipt or {}).get('evidence'), dict) else {}
        peak = ev.get('memory_peak_measured_gib')       # D3: the harness's measured peak (every process of every node)
        if not isinstance(peak, (int, float)):
            peak = (c0.verdict or {}).get('peak_reserved_gib')
        if not isinstance(peak, (int, float)):
            peak = ev.get('memory_peak_gib')
        self.mem.record('c0', self._mcfg(engine, 'c0'), 'ok', peak=peak if isinstance(peak, (int, float)) else None)
        self._start_from_store()
        self.runs['c0'] = {'cfg': self._mcfg(engine, 'c0'), 'kind': 'ok', 'y': 0.0, 'origin': 'reference'}
        self.tried.add(self._ckey(self.ref, ()))
        self._table(engine)

    def _table(self, engine) -> list[dict]:
        """The legal configurations of the knob table (the session's own check: schema, constraints, invariants), the
        reference first; above the cap a uniform sample of the product (the cap's worth), the prior's configurations
        added when legal."""
        if self.table is not None:
            return self.table
        s = engine.s
        c0 = s.candidates['c0']
        names = list(self.knobs)
        vals = [list(self.knobs[k]) for k in names]
        n = math.prod(len(v) for v in vals) if vals else 0
        base = dict(c0.config)
        out, seen = [], set()

        check = getattr(s, 'check_config', None)

        def legal(cfg):
            key = self._ckey(cfg, ())
            if key in seen:
                return
            seen.add(key)
            full = {**base, **cfg}
            if check is None:                            # (a session without the check: the engine refuses at the run)
                out.append(cfg)
                return
            try:
                ok = check(dict(full), c0)[0] is not None
            except Exception:                            # noqa: BLE001 - a check that fails rejects the configuration
                ok = False
            if ok:
                out.append(cfg)

        legal(dict(self.ref))
        if n <= self.table_cap:
            for combo in itertools.product(*vals):
                legal(dict(zip(names, combo)))
            self.table_whole = True
        else:
            self.table_whole = False
            rng = random.Random(self.rng.random())
            for _ in range(self.table_cap):
                i = rng.randrange(n)
                cfg = {}
                for k, v in zip(names, vals):
                    i, j = divmod(i, len(v))
                    cfg[k] = v[j]
                legal(cfg)
        for c in (self.prior.configs if self.prior else []):
            legal({**self.ref, **c['change']})
        self.table = out
        return out

    # ------------------------------------------------------------ configurations
    def _ckey(self, cfg: dict, ids) -> tuple:
        return tuple(sorted((k, vk(cfg.get(k))) for k in self.knobs)) + tuple(sorted(f'patch:{x}' for x in ids))

    def _feats(self, cfg: dict, ids=()) -> dict:
        key = self._ckey(cfg, ())
        base = self._fcache.get(key)
        if base is None:
            base = {(k, vk(cfg.get(k))): 1 for k in self.knobs if vk(cfg.get(k)) != vk(self.ref.get(k))}
            self._fcache[key] = base
        out = dict(base)
        for x in ids:
            out[(f'patch:{x}', 'true')] = 1
        for f in self.surr.names:
            if isinstance(f, tuple) and f[0] == 'pair' and all(m in out for m in f[1]):
                out[f] = 1
        return out

    def _words(self, cfg: dict, ids=()) -> str:
        parts = [f'{k}={cfg.get(k)}' for k in self.knobs if vk(cfg.get(k)) != vk(self.ref.get(k))]
        parts += [f'patch:{x}' for x in ids]
        return ' + '.join(parts) if parts else 'the reference'

    def _patch_ideas(self, p) -> set:
        """hls3's reading of a proposal's patch ideas: the ids a proposal carries (knob picks: the chain; a patch: its id)."""
        return set(self.ideas.get(p.hypothesis or '', frozenset()))

    def _m_of(self, cfg: dict, ids=()) -> dict:
        """A configuration as the memory model reads it (knob values plus patch pseudo-knobs)."""
        out = dict(cfg)
        out.update({f'patch:{x}': True for x in sorted(ids)})
        return out

    def _known(self, engine=None) -> dict:
        out = self._ocache.get('known')
        if out is None:
            out = dict(self.declared)
            for pid, rec in self.patches.items():
                d = rec.get('mem_dir')
                if d in DIRS:
                    out[frozenset({(f'patch:{pid}', 'null', 'true')})] = d
            out.update(self.mem.measured)
            self._ocache['known'] = out
        return out

    def _forget(self) -> None:
        """Every observation changes the feasibility of the table: the caches go."""
        self._pcache.clear()
        self._ocache.clear()

    # ------------------------------------------------------------ feasibility
    def _p_feasible(self, cfg: dict, ids=()) -> tuple[float, str]:
        """P(the configuration fits and starts) with the reason; cached between observations (every run changes it)."""
        key = self._ckey(cfg, ids)
        out = self._pcache.get(key)
        if out is None:
            out = self._pcache[key] = self._p_feasible_now(cfg, ids)
        return out

    def _p_feasible_now(self, cfg: dict, ids=()) -> tuple[float, str]:
        if not self.on['wcs_feasibility'] and not self.on['wcs_antagonism']:
            return 1.0, ''
        m = self._m_of(cfg, ids)
        p, why = 1.0, ''
        if self.on['wcs_feasibility']:
            known = self._known()
            verdict, w, _ = self.mem.predict(m, known, self.runs['c0']['cfg'], hard=self.mem.measured)
            if verdict == 'infeasible':
                return 0.0, w
            if verdict == 'feasible':
                p, why = 1.0, w
            else:
                p, why = self._p_arith(m)
                near = self._near_failed(m, known)      # a failed run it differs from only by spends, neutral moves and
                if near:                                 # one or two moves of unknown direction: a probe, not a full run
                    p, why = min(p, near[1]), f'{near[0]} did not fit; this differs from it by {near[2]} move(s) of unknown memory direction'
                if verdict == 'unsure':                  # dominance cannot tell (unknown moves): at most P_UNSURE, less
                    p, why = min(p, P_UNSURE), (w if p >= P_UNSURE else (w + '; ' + why if why else w))   # by arithmetic
        if self.on['wcs_feasibility'] and p > P_UNSURE and verdict is None:   # (a run's own evidence outranks the store)
            warn = self._warned({k: v for k, v in m.items() if k in self.knobs})
            if warn:
                p, why = min(p, self._warn_level(warn)), (why + '; ' if why else '') + f'the store warns: {str(warn[0])[:120]}'
        if self.on['wcs_feasibility'] and p > 0.0:
            held = self._blocked_values()
            bad = [kv for kv in self._levels(m) if kv in held]
            if bad:
                p *= BLOCKED_VALUE ** len(bad)
                self.counts['blocked_values'] += 1
                why = (why + '; ' if why else '') + ', '.join(f'{k}={json.loads(v)} failed in {held[(k, v)]} runs and fitted in none'
                                                              for k, v in bad[:2])
        if self.on['wcs_antagonism']:
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

    def _blocked_values(self) -> dict:
        """Values (knob, vk) that BLOCKED_AFTER or more runs that did not fit held and no run that fitted: a spender whose
        enabler the search has not found (the partner probe looks for it); their configurations are held back."""
        out = self._ocache.get('blocked')
        if out is None:
            fails, fits = {}, set()
            for r in self.mem.runs.values():
                for kv in self._levels(r['cfg']):
                    if r['status'] == 'ok':
                        fits.add(kv)
                    else:
                        fails[kv] = fails.get(kv, 0) + 1
            out = self._ocache['blocked'] = {kv: n for kv, n in fails.items() if n >= BLOCKED_AFTER and kv not in fits}
        return out

    def _near_failed(self, m: dict, known: dict) -> tuple | None:
        """(run, feasibility, n) when an out-of-memory or thrashing run differs from `m` by no freeing move and at most
        two moves of unknown memory direction (the rest spend or are neutral): most unknown moves are neutral, so the
        configuration is probed first - and after two such runs the moves are measured neutral (_record_memory)."""
        best = None
        for cid, r in self.mem.runs.items():
            if r['status'] not in ('oom', 'thrash'):
                continue
            sp = self.mem.split(r['cfg'], m, known)
            if sp['frees'] or len(sp['unknown']) > 2 or not sp['unknown']:
                continue
            p = 0.3 if len(sp['unknown']) == 1 else 0.2
            if best is None or p < best[1]:
                best = (cid, p, len(sp['unknown']))
        return best

    def _p_arith(self, m: dict) -> tuple[float, str]:
        """Feasibility by arithmetic: the reference's measured peak plus the declared GiB of the configuration's changes
        against the card; without a peak, the sharded-state floor plus the declared spends. Never below P_DECLARED_MIN:
        a declaration asks for a probe, it rules nothing out."""
        cap = self.mem.cap
        if not cap:
            return 0.9, ''
        gib, total, unknown = 0.0, 0.0, 0
        for k, v in m.items():
            if k.startswith('patch:'):
                rec = self.patches.get(k[6:]) or {}
                d, g = rec.get('mem_dir'), rec.get('mem_gib')
            else:
                if vk(v) == vk(self.ref.get(k)):
                    continue
                e = (self.prior.levels.get((k, vk(v))) if self.prior else None) or {}
                d, g = e.get('mem_dir'), e.get('mem_gib')
            if d == 'spends' and g:
                gib += g * self.spend_scale
                total += g * self.spend_scale
            elif d == 'frees' and g:
                gib -= g
                total += g
            elif d is None:
                unknown += 1
        ref_peak = (self.mem.runs.get('c0') or {}).get('peak')
        measured = isinstance(ref_peak, (int, float)) and ref_peak > 0
        if measured:
            est, line = ref_peak + gib, PEAK_LINE * cap  # (a reserved peak holds the allocator's slack: the card is the line)
            why = f'about {est:.0f} GiB by the declared memory (reference peak {ref_peak:.0f}) of {cap:.0f}'
        elif self.mem.floor:
            est, line = self.mem.floor['floor_gib'] + max(0.0, gib), NEAR_CAP * cap
            why = f'at least {est:.0f} GiB (sharded state {self.mem.floor["floor_gib"]:.0f} + declared spends) of {cap:.0f}'
        else:
            return 0.9, ''
        # the declared sizes are guesses (the recorded declarations were off by a third or more): the estimate's error
        # grows with what was declared; a move of unknown direction may spend (a share of the card, with a wide spread:
        # two such moves ask for a probe); the allocator thrashes from about 90 % of the card (the cliff)
        est += UNKNOWN_SPEND * cap * unknown
        sd = DECLARED_ERR * total + 0.02 * cap + UNKNOWN_SD * cap * unknown
        p = 0.5 * (1.0 + math.erf((line - est) / (sd * math.sqrt(2))))
        if unknown:
            why += f'; {unknown} move(s) of unknown memory direction'
        if self.spend_scale > 1.0:
            why += f'; declared spends x{self.spend_scale:.2f} (a failure the arithmetic did not expect)'
        # the floor: a declaration asks for a probe, it rules nothing out (the declared sizes err both ways: the toy's
        # over-declared spenders would never be probed without it; the smokes' under-declared ones are met by the
        # calibration above and by the dominance once a run failed)
        return max(P_DECLARED_MIN, min(1.0, p)), why

    def _start_from_store(self) -> None:
        """D4: the start failures earlier arms wrote back (the experience plan's `start_failures`): a set's pairs enter
        the pair table suspect (P_PAIR_FAILED for a set of at most three values, else half), a single value as one that
        failed alone; the OOM boundaries travel as the store's constraints, which the plan's check warns about."""
        if not self.on['wcs_antagonism']:
            return
        for it in (self.exp or {}).get('start_failures') or []:
            members = []
            for m in it.get('members') or []:
                for k, v in (m or {}).items():
                    if k in self.knobs:
                        w = _coerce(v, self.knobs[k])
                        if w is not None and vk(w) != vk(self.ref.get(k)):
                            members.append((k, vk(w)))
            members = sorted(set(members))
            if not members:
                continue
            why = f'an earlier arm: {str(it.get("message") or it.get("id") or "")[:120]}'
            if len(members) == 1:
                self.alone[members[0]] = max(self.alone.get(members[0], 0), 1)
            else:
                f = P_PAIR_FAILED if len(members) <= 3 else 0.5
                for a, b in itertools.combinations(members, 2):
                    rec = self.pairs.setdefault(frozenset((a, b)), {'p': 1.0, 'why': ''})
                    if rec['p'] > f:
                        rec.update(p=f, why=why)
            self.from_store.append({'members': it.get('members'), 'id': it.get('id'), 'message': it.get('message')})

    @staticmethod
    def _warn_level(warn: list) -> float:
        """What the store's warnings make of the feasibility: a closed-form estimate beyond the card by a quarter or more
        nearly rules out (P_FAR), beyond it 0.2; a measured neighbour that ran out of memory 0.1; else P_UNSURE."""
        level = P_UNSURE
        for w in warn:
            t = str(w)
            m = re.search(r'about (\d+(?:\.\d+)?) GiB of (\d+(?:\.\d+)?)', t)
            if m and float(m.group(2)) > 0:
                ratio = float(m.group(1)) / float(m.group(2))
                level = min(level, P_FAR if ratio >= 1.25 else 0.2 if ratio > 1.0 else P_UNSURE)
            if t.startswith('memory:') and ('ran out of memory' in t or 'peaks at' in t):
                level = min(level, 0.1)
        return level

    def _warned(self, cfg: dict) -> list:
        """The experience plan's warnings for a configuration (its constraints in scope, a memory estimate): a warned
        configuration is probed first (HLS's rule for a warned move), never ruled out."""
        check = (self.exp or {}).get('check')
        if not callable(check):
            return []
        try:
            return list(check({**dict(self.ref), **cfg}) or [])
        except Exception:                                # noqa: BLE001 - a check that fails warns of nothing
            return []

    @staticmethod
    def _pair_words(pair) -> str:
        return ' + '.join(f'{k}={json.loads(m) if m != "null" else None}' for k, m in sorted(pair))

    def _spends(self, base_m: dict, m: dict) -> tuple[bool, bool]:
        """(spends, unknown): whether a configuration spends memory against its base by the declared or measured
        moves, and whether it changes something of unknown memory direction."""
        sp = self.mem.split(base_m, m, self._known())
        return bool(sp['spends']), bool(sp['unknown'])

    # ------------------------------------------------------------ proposals
    def propose(self, engine, k):
        if self.prior is None:
            self._init(engine)
        s = engine.s
        left = int(getattr(s.budget, 'screens', 0) or 0) - engine.full_screens()
        if left <= 0:
            return []
        inc = engine.incumbent()
        q = self._prior_pick(engine, inc)                # the first screen: the prior's own best configuration, alone
        if q is not None and engine._fresh(q):
            return [q]
        q = self._partner(engine, inc) if self.on['wcs_partner_probe'] and left > 1 else None
        if q is not None and engine._fresh(q):
            return [q]                                   # guaranteed: proposed alone, it does not compete with the draw
        q = self._rescue(engine, inc) if self.on['wcs_partner_probe'] and left > 1 else None
        if q is not None and engine._fresh(q):
            return [q]
        if left == 1 and self.on['wcs_coverage']:        # the last screen is the bundle's alone when one exists
            q = self._coverage(engine, inc)              # (fresh, legal, feasible); else the ordinary acquisition
            if q is not None and engine._fresh(q):
                return [q]
        props = []                                       # (the engine takes the best value per cost of what it is given)
        if self.on['wcs_code']:
            props += self._code(engine, inc, left)
        q = self._acquire(engine, inc, left)
        if q is not None:
            props.append(q)
        return [p for p in props if engine._fresh(p)]

    def _prior_pick(self, engine, inc) -> Proposal | None:
        """The prior's own best whole configuration runs first (the portfolio both reviews asked for; the Q9H smoke lost
        the model's 'aggressive' suggestion - the winning class - because its benefit sat in unquantified pairs and the
        configurations only entered the table): on the first screen, the highest of them by its stated gain, else by
        the posterior, that is legal, untried and not ruled out by the runs; below the threshold it is probed first."""
        if not self.on['wcs_prior'] or self.prior is None or not self.prior.configs or engine.full_screens() > 0 \
                or self.counts['prior_config']:
            return None
        mean = self.surr.mean()
        y_inc = Surrogate.value(self._incumbent_feats(engine, inc), mean)
        ranked = []
        for c in self.prior.configs:
            cfg = {**{k: self.ref.get(k) for k in self.knobs}, **{k: v for k, v in c['change'].items() if k in self.knobs}}
            if not self._legal(engine, cfg) or self._ckey(cfg, ()) in self.tried:
                continue
            x = self._feats(cfg)
            mu = Surrogate.value(x, mean) - y_inc
            ranked.append((c['mu'] if c.get('mu') is not None else mu, mu, cfg, x, c['name']))
        for _, mu, cfg, x, name in sorted(ranked, key=lambda t: -t[0]):
            p, why = self._p_feasible(cfg, ())
            if p <= 0.0:
                continue
            probe = p < self.probe_below and self.on['wcs_probe_first']
            self.counts['prior_config'] += 1
            return self._proposal(engine, inc, cfg, (), mu=mu, sigma=math.sqrt(max(1e-6, self.surr.var(x))), p=p,
                                  why=f"the prior's '{name}' configuration" + (f'; {why}' if why else ''), probe=probe,
                                  origin='wcs-prior', kind='prior')
        return None

    def _incumbent_feats(self, engine, inc) -> dict:
        c = engine.s.candidates[inc]
        cfg = flat(c.config, self._native(engine))
        return self._feats(cfg, self._pids(engine, inc))

    def _candidates(self, engine, inc) -> list[tuple[dict, tuple]]:
        """(configuration, patch ids) pairs the acquisition scores: the table with the proven chain, less what ran."""
        ids = tuple(self.chain)
        out = []
        for cfg in self._table(engine):
            key = self._ckey(cfg, ids)
            if key in self.tried:
                continue
            out.append((cfg, ids))
        return out

    def _big_knobs(self) -> set:
        mean = self.surr.mean()
        out = set()
        for f, mu in mean.items():
            if isinstance(f, tuple) and f[0] in self.knobs and abs(mu) >= BIG:
                out.add(f[0])
            elif isinstance(f, tuple) and f[0] == 'pair' and abs(mu) >= BIG:
                out.update(k for k, _ in f[1])          # (a value whose gain sits in its enabling pair)
        return out

    def _diverse(self, cfg: dict, mode: str = 'strong') -> bool:
        """'strong' (the first screens): at least two knobs the prior calls big differ from every run; 'weak' (after):
        at least one - a screen that moves no big knob measures noise knobs one by one (the review's R-B), which the
        coverage screen does in one bundle. Nothing qualifies: the rule is dropped for that pick."""
        big = self._big_knobs()
        if not big:
            return True
        need = 2 if mode == 'strong' else 1
        for r in self.runs.values():
            rc = r['cfg']
            d = sum(1 for k in big if vk(cfg.get(k)) != vk(rc.get(k)))
            if d < need:
                return False
        return True

    def _acquire(self, engine, inc, left) -> Proposal | None:
        """The next configuration: one draw of the posterior (or its mean) scores every candidate of the table; the best
        improvement over the incumbent weighted by feasibility and by the cost of a probe is proposed."""
        cands = self._candidates(engine, inc)
        if not cands:
            return None
        thompson = self.on['wcs_thompson']
        beta = self.surr.sample(self.rng) if thompson else self.surr.mean()
        mean = self.surr.mean() if thompson else beta
        x_inc = self._incumbent_feats(engine, inc)
        y_inc = Surrogate.value(x_inc, beta)
        scored = []
        for cfg, ids in cands:
            x = self._feats(cfg, ids)
            scored.append((Surrogate.value(x, beta) - y_inc, cfg, ids, x))
        scored.sort(key=lambda t: -t[0])
        mode = None
        if self.on['wcs_diversity']:
            mode = 'strong' if engine.full_screens() < self.diversity_screens else 'weak'
        probes_ok = self._probes_affordable(engine)
        self._price_context(engine, inc)
        best = self._best(scored, mode, probes_ok) or (self._best(scored, None, probes_ok) if mode else None)
        kind = 'thompson' if thompson else 'greedy'
        if best is None:                                 # nothing improves under this draw (or everything that would is
            y_m = Surrogate.value(x_inc, mean)           # ruled out): the most promising by mean + spread that fits,
            ucb = sorted(((Surrogate.value(x, mean) - y_m + math.sqrt(max(0.0, self.surr.var(x))), cfg, ids, x)
                          for _, cfg, ids, x in scored), key=lambda t: -t[0])
            fair = [row for row in ucb if Surrogate.value(row[3], mean) - y_m >= -1e-9]   # a negative mean is no bet
            best = (self._best(fair, None, probes_ok, any_sign=True)
                    # the probes are spent and every such candidate is doubtful: one runs in full (its permitted plan)
                    # before any candidate with a negative mean is considered (WCS_r2 5)
                    or self._best(fair, None, probes_ok, any_sign=True, doubtful_ok=True)
                    or self._best(ucb, None, probes_ok, any_sign=True)
                    or self._best(ucb, None, probes_ok, any_sign=True, doubtful_ok=True))
            kind = 'explore'
        if best is None:
            return None
        cfg, ids, x, p, why, probe = best
        mu = Surrogate.value(x, mean) - Surrogate.value(x_inc, mean)
        sigma = math.sqrt(max(1e-6, self.surr.var(x)))
        self.counts['thompson' if thompson else 'greedy'] += 1
        return self._proposal(engine, inc, cfg, ids, mu=mu, sigma=sigma, p=p, why=why, probe=probe, origin='wcs', kind=kind)

    def _price_context(self, engine, inc) -> None:
        """What a probe costs here: the engine's measured probe / full ratio, its nominal probe length and the full
        run's updates (a longer probe adds updates at the full run's rate: Engine.expected_cost), the incumbent (the
        start cap and the keep rule read it)."""
        try:
            c = engine.costs()
            self._probe_ratio = max(0.01, c['probe'] / max(c['full'], 1e-9))
        except Exception:                                # noqa: BLE001
            pass
        self._price_ctx = {'engine': engine, 'inc': inc, 'probe_steps': int(getattr(engine, 'probe_steps', 2) or 2),
                           'full_steps': max(1, int(getattr(engine, 'full_steps', 30) or 30))}

    def _probe_steps_of(self, cfg: dict) -> int:
        """A candidate's probe length: the largest warm-up its values declare, plus one, within the gate's bounds."""
        warm = max([e.get('warmup') or 0 for (k, key), e in (self.prior.levels.items() if self.prior else [])
                    if vk(cfg.get(k)) == key] + [0])
        return min(PROBE_STEPS_MAX, max(GATED_STEPS, warm + 1))

    def _price(self, cfg: dict, probe: bool, p: float | None = None) -> float:
        """The candidate's effective plan priced in full runs (Engine.expected_cost): a probe-first plan costs the
        probe (the nominal share plus its extra updates at the full run's rate) and then a full run with probability
        P(fit); when the engine would drop the probe (costlier than half a full run and not kept) or the plan is full
        only, one full run. p / (r + p) <= 1, so the scan's bound holds."""
        if not probe:
            return 1.0
        ctx = getattr(self, '_price_ctx', None) or {}
        if p is None:
            p = self._p_feasible(cfg)[0]
        steps = self._probe_steps_of(cfg)
        extra = max(0, steps - ctx.get('probe_steps', 2)) / ctx.get('full_steps', 30)
        r = self._probe_ratio + extra
        if self._probe_ratio >= 0.5 and ctx.get('engine') is not None:   # Engine.plan's test reads the NOMINAL probe:
            cap = self._probe_cap(ctx['engine'], ctx['inc'], steps)        # such a probe stays only while screens are
            if not self._screens_scarce(ctx['engine'], cap, steps):        # scarce (keep_probe); a longer declared
                return 1.0                                                 # warm-up adds updates to a kept probe
        return r + p

    def _best(self, scored, diverse: str | None, probes_ok: bool, any_sign: bool = False, doubtful_ok: bool = False):
        """The best of the scored (improvement, cfg, ids, x) rows, in descending order, by improvement x feasibility /
        the plan's expected cost (a probe-first plan: the probe's share plus P(fit) of a full run; p / (r + p) <= 1, so
        the bound below holds), among those the diversity rule (`diverse`: its mode, None for no rule) admits and that
        fit at all: the scan stops when no later row
        can beat the best found (feasibility is at most 1, and it is cached between observations). A doubtful
        configuration (below the probe threshold) runs only as a probe - once the probes are spent it is skipped unless
        `doubtful_ok` (the last resort); None when none qualifies.
        `any_sign`: the fallback (rows ordered by mean + spread): a non-positive score still ranks by feasibility."""
        best, best_v = None, 0.0
        floor = 0.01 if any_sign else 0.0
        for impr, cfg, ids, x in scored:
            u = max(impr, floor)
            if u <= best_v:
                break
            if diverse and not self._diverse(cfg, diverse):
                self.counts['diversity_filtered'] += 1
                continue
            p, why = self._p_feasible(cfg, ids)
            if p <= 0.0:
                self.counts['infeasible'] += 1
                continue
            probe = p < self.probe_below and self.on['wcs_probe_first'] and probes_ok
            if p < self.probe_below and not probe and self.on['wcs_probe_first'] and not doubtful_ok:
                self.counts['unprobed_skipped'] += 1   # the probes are spent: a doubtful configuration is not run in full
                continue
            v = u * p / self._price(cfg, probe, p)      # the effective plan's expected cost over a full run's:
            if v > best_v:                               # C_probe (its updates) + P(fit) x C_full, or C_full (D7)
                best, best_v = (cfg, ids, x, p, why, probe), v
        return best

    def _proposal(self, engine, inc, cfg, ids, *, mu, sigma, p, why, probe, origin, kind) -> Proposal:
        self._n += 1
        hid = f'{kind}|{self._n}'
        self.ideas[hid] = frozenset(ids)
        plan = ('probe', 'full') if probe and self.on['wcs_probe_first'] else ('full',)
        base_m = self._mcfg(engine, inc)
        m = self._m_of(cfg, ids)
        spends, unknown = self._spends(base_m, m) if self.on['wcs_feasibility'] else (False, False)
        warm = max([e.get('warmup') or 0 for (k, key), e in (self.prior.levels.items() if self.prior else [])
                    if vk(cfg.get(k)) == key] + [0])
        steps = max(GATED_STEPS, warm + 1) if plan[0] == 'probe' else None
        steps = min(steps, PROBE_STEPS_MAX) if steps else None
        cap = self._probe_cap(engine, inc, steps) if plan[0] == 'probe' else None
        keep = plan[0] == 'probe' and self._screens_scarce(engine, cap, steps)
        if plan[0] == 'probe':
            self.counts['probe_first'] += 1
        full = {**dict(engine.s.candidates['c0'].config), **cfg}      # the complete configuration (keys beyond the knobs kept)
        q = Proposal(change=full, patch=dict(self.overlay_best) if ids and self.overlay_best else None, base=None,
                     mu=mu, sigma=max(0.005, sigma), p_feasible=min(0.99, max(0.01, p)), mechanism=self._words(cfg, ids)[:200],
                     hypothesis=hid, origin=origin, fidelity=plan, absolute=True, keep_probe=keep, probe_cap=cap,
                     probe_steps=steps if steps and steps > int(getattr(engine, 'probe_steps', 2) or 2) else None,
                     gate=BoundGate(self, '_speed_gate', spends, unknown, warm, base='c0') if plan[0] == 'probe' else None)
        self.ledger[hid] = {'id': hid, 'kind': kind, 'mechanism': q.mechanism, 'change': {k: v for k, v in cfg.items()
                                                                                    if vk(v) != vk(self.ref.get(k))},
                            'patches': list(ids), 'base': inc, 'predicted_gain_pct': round(100 * math.expm1(mu), 2),
                            'sigma_pct': round(100 * sigma, 2), 'p_feasible': round(p, 3), 'why_feasible': why[:200],
                            'plan': list(plan), 'status': 'open'}
        self.picks.append({'id': hid, 'kind': kind, 'screens_done': engine.full_screens(), 'mechanism': q.mechanism,
                           'p_feasible': round(p, 3), 'predicted_gain_pct': round(100 * math.expm1(mu), 2)})
        return q

    def _probes_affordable(self, engine) -> bool:
        """Probes of knob picks stay at about one per full screen over the search: no more probe-first picks once the
        probes made reach the screen budget (patches always probe; the engine's GPU-hour and wall budgets bound them too).
        Early probes are the valuable ones (the first stacks are the riskiest), so the allowance is the whole budget, not
        the screens made so far."""
        if not self.on['wcs_probe_first']:
            return False
        probes = sum(1 for r in engine.s.screens if r.get('fidelity') == 'probe')
        budget = int(getattr(engine.s.budget, 'screens', 0) or 0)
        return probes < max(budget, engine.full_screens() + 2)

    def _speed_gate(self, engine, cid, base, row, spends: bool, unknown: bool, warm: int) -> str | None:
        """hls3's speed gate (F1b, F12 with the compile warm-up guard): a probe's updates after the warm-up against its
        base's; at GATE_LOW or more on a configuration that spends memory or changes something of unknown direction,
        no full run. The classifier reads `gated` for the ratio and decides thrash or slow_probe."""
        if row.get('recompiles'):
            return None
        probe = [x for x in row.get('intervals') or [] if isinstance(x, (int, float))]
        ref = self._base_intervals(engine, base)
        w = warm if isinstance(warm, int) and WARMUP < warm <= WARMUP_MAX else WARMUP
        probe, ref = probe[w - 1:], ref[w - 1:]
        pairs = [a / b for a, b in zip(probe, ref) if b > 0]
        if not pairs:
            return None
        ratio = statistics.median(pairs)
        if ratio < self.gate_ratio:
            return None
        self.gated.append({'candidate': cid, 'base': base, 'ratio': round(ratio, 2), 'spends': spends,
                           **({'unknown': True} if unknown and not spends else {}), 'full_run': not (spends or unknown)})
        if not (spends or unknown):
            return None
        return (f'slow start: its first updates ran x{ratio:.2f} slower than {base}\'s, on a configuration that '
                + ('spends memory' if spends else 'may spend memory (a change of unknown memory effect)') + ': no full run')

    # ------------------------------------------------------------ the partner probe (an undeclared enabler)
    def _legal(self, engine, cfg: dict) -> bool:
        """The session's own check (schema, constraints, invariants), as the table's rows passed it."""
        check = getattr(engine.s, 'check_config', None)
        if check is None:
            return True
        c0 = engine.s.candidates['c0']
        try:
            return check({**dict(c0.config), **cfg}, c0)[0] is not None
        except Exception:                                # noqa: BLE001
            return False

    def _complete(self, engine, cfg: dict, keep: set) -> dict | None:
        """`cfg` when legal, else the legal row of the table nearest to it (fewest knobs differing) that holds cfg's
        values of the knobs in `keep` - a coupled knob (a global batch's accumulation) completed, not broken (WCS_r1 D4)."""
        if self._legal(engine, cfg):
            return cfg
        best, bd = None, None
        for row in self._table(engine):
            if any(vk(row.get(k)) != vk(cfg.get(k)) for k in keep):
                continue
            d = sum(1 for k in self.knobs if vk(row.get(k)) != vk(cfg.get(k)))
            if bd is None or d < bd:
                best, bd = {k: row.get(k) for k in self.knobs}, d
        if best is None and keep and self._legal_any(engine):
            return self._complete(engine, cfg, set())    # nothing legal holds them all: the nearest row, then
        return best

    def _legal_any(self, engine) -> bool:
        return bool(self._table(engine))

    def _rescue(self, engine, inc) -> Proposal | None:
        """A run at the memory edge - thrash (the cliff: fast updates between stalls), or out of memory by a sliver (short
        by at most RESCUE_SHORT of the card) - is a class with one memory enabler missing (the G4U smoke: the compiler
        with recompute off thrashed at 138 of 140 GiB with 2.7 s updates among 4-5 s ones; random's winner was the
        same with the allocator's enabler). One probe of that configuration plus the enabler the prior names for a
        value it holds (an `enables` set's one missing member), else the largest declared freeing value it lacks; a
        conflicting value swapped back; coupled knobs completed; once per (run, value), RESCUE_MAX per search."""
        if not self.on['wcs_feasibility'] or self.counts['rescue_probes'] >= RESCUE_MAX or not self.prior:
            return None
        cap = self.mem.cap or 0.0
        mean = self.surr.mean()
        y_inc = Surrogate.value(self._incumbent_feats(engine, inc), mean)
        for cid in reversed(list(self.mem.runs)):        # the latest edge first
            r = self.mem.runs[cid]
            short = (r.get('oom') or {}).get('short_gib') if isinstance(r.get('oom'), dict) else None
            if r['status'] == 'thrash':
                if not self._fast_updates(engine, cid, 'c0'):
                    continue                             # no fast updates among the stalls: no potential to rescue
                edge = 'thrashed at the memory edge with fast updates between the stalls'
            elif r['status'] == 'oom' and isinstance(short, (int, float)) and cap and short <= RESCUE_SHORT * cap:
                edge = f'ran out of memory by {short:.2f} GiB'
            else:
                continue
            cfg = {k: r['cfg'].get(k) for k in self.knobs}
            held = set(self._levels(cfg))
            options = []
            for p in self.prior.pairs:                   # the enabler the prior names for a value the run holds
                if p['kind'] != 'enables':
                    continue
                members = set(p['members'])
                missing = members - held
                if len(missing) == 1 and (members & held):
                    (k, key), = missing
                    options.append((2.0, k, key, 'the prior names it the enabler of a value it holds'))
            for (k, key), e in self.prior.levels.items():   # else the largest declared freeing value it lacks
                if e.get('mem_dir') == 'frees' and (k, key) not in held and vk(cfg.get(k)) != key:
                    options.append((1.0 + (e.get('mem_gib') or 0.0) / 1000.0, k, key, 'the largest declared freeing value it lacks'))
            options = [o for o in options if (cid, o[1], o[2]) not in self.rescue_done]
            options.sort(key=lambda t: -t[0])
            ids = tuple(self.chain)
            for _, k, key, note in options:
                v = next((x for x in self.knobs[k] if vk(x) == key), None)
                if v is None:
                    continue
                new = dict(cfg)
                new[k] = v
                held2 = set(self._levels(new))
                for pr, rec in self.pairs.items():       # a held value that conflicts with the enabler goes back
                    if rec['p'] < 0.5 and pr <= held2:
                        for kk, kv in sorted(pr - {(k, key)}):
                            if (kk, kv) in held2 and not kk.startswith('patch:'):
                                new[kk] = self.ref.get(kk)
                                held2.discard((kk, kv))
                                break
                legal = self._complete(engine, new, {k})
                self.rescue_done.add((cid, k, key))
                if legal is None or vk(legal.get(k)) != key or self._ckey(legal, ids) in self.tried:
                    continue
                pf, why = self._p_feasible(legal, ids)
                if pf <= 0.0:
                    continue
                x = self._feats(legal, ids)
                self.counts['rescue_probes'] += 1
                q = self._proposal(engine, inc, legal, ids, mu=Surrogate.value(x, mean) - y_inc,
                                   sigma=math.sqrt(max(1e-6, self.surr.var(x))), p=max(pf, P_UNSURE),
                                   why=f'{cid} {edge}; {k}={v}: {note}', probe=True, origin='wcs-rescue', kind='rescue')
                q.keep_probe = True
                return q
        return None

    @staticmethod
    def _fast_updates(engine, cid, base) -> bool:
        """The cliff's signature (the G4U smoke: 2.7 s updates among 4-5 s stalls against the reference's steady 3.5 s):
        the run's fastest post-warm-up update is faster than nine tenths of its base's typical one. Without the
        timestamps: no."""
        def gaps(c):
            rec = getattr(engine.s.candidates.get(c), 'receipt', None) or {}
            ev = rec.get('evidence') if isinstance(rec.get('evidence'), dict) else {}
            ps = [r['t'] for r in ev.get('per_step') or [] if isinstance(r, dict) and isinstance(r.get('t'), (int, float))]
            return [b - a for a, b in zip(ps, ps[1:])][2:]
        g, gb = gaps(cid), gaps(base)
        if len(g) < 5 or len(gb) < 5:
            return False
        med = statistics.median(gb)
        return med > 0 and min(g) < 0.9 * med

    def _strip_unknown(self, m: dict, known: dict) -> dict:
        """`m` with every move from the reference of unknown memory direction undone: such a move (a loader knob, an
        undeclared flag) says nothing about the memory a configuration needs, so it must not hide a blocking run."""
        ref = self.runs['c0']['cfg']
        out = dict(m)
        for move in self.mem.split(ref, m, known)['unknown']:
            for k, _, _ in move:
                out[k] = ref.get(k)
        return out

    def _partner(self, engine, inc) -> Proposal | None:
        """When the posterior's best configuration does not fit by the runs (a spender the runs block), one probe of
        the blocking run's configuration with each knob of unknown memory direction the prior does not call harmful,
        the most uncertain first; once per (blocking run, knob)."""
        if not self.on['wcs_feasibility'] or not self.mem.runs or self.counts['partner_probes'] >= PARTNER_MAX:
            return None
        mean = self.surr.mean()
        ids = tuple(self.chain)
        cands = [(cfg, ids) for cfg in self._table(engine)]   # the whole table: the best may be a run that did not fit
        if not cands:
            return None
        x_inc = self._incumbent_feats(engine, inc)
        y_inc = Surrogate.value(x_inc, mean)
        best = max(cands, key=lambda t: (round(Surrogate.value(self._feats(t[0], t[1]), mean), 9), -len(self._feats(t[0], t[1]))))
        if Surrogate.value(self._feats(best[0], best[1]), mean) - y_inc <= 2 * engine.sigma_pair():
            return None                                  # (ties: the fewest changes, so an unknown knob is not dragged in)
        known = self._known()                           # the strict reading (declared and measured moves): every failed
        m_best = self._strip_unknown(self._m_of(best[0], best[1]), known)   # run the best configuration needs at least
        blockers = [cid for cid, r in self.mem.runs.items() if r['status'] in ('oom', 'thrash')   # as much as, the
                    and self.mem.dominates(m_best, self._strip_unknown(r['cfg'], known), known)]   # fewest moves first
        blockers.sort(key=lambda cid: len(sig(self.runs['c0']['cfg'], self.mem.runs[cid]['cfg'])))
        if not blockers:
            return None
        sds = self.surr.sd()
        c_inc = engine.s.candidates[inc]
        inc_cfg = {kk: vv for kk, vv in flat(c_inc.config, self._native(engine)).items() if kk in self.knobs}
        ids = tuple(self.chain)                          # every later configuration carries the chain (the probe too)
        for block in blockers:                           # a probe is all it costs
            bcfg = self.mem.runs[block]['cfg']
            options = []
            for (k, key), e in (self.prior.levels.items() if self.prior else []):
                if e.get('mem_dir') in DIRS or vk(bcfg.get(k)) == key or (block, k) in self.partner_done \
                        or (block, k) in self.partner_skip:
                    continue
                if vk(bcfg.get(k)) != vk(self.ref.get(k)):
                    continue                             # the blocking run moved this knob: not a partner for it
                if mean.get((k, key), 0.0) < -0.1:
                    continue                             # the prior calls it harmful (an enabler may cost a little alone)
                options.append((sds.get((k, key), SD_UNKNOWN), k, key))
            options.sort(key=lambda t: -t[0])
            for _, k, key in options:                    # the most uncertain first; one without a legal, fresh form is
                v = next((x for x in self.knobs[k] if vk(x) == key), None)   # skipped, the next tried (WCS_r2 2)
                if v is None:
                    self.partner_skip.add((block, k))
                    continue
                # the probe: the incumbent, the blocking run's largest move by the posterior (the spender) and the
                # partner - not the blocking run's whole configuration, which may hold other spenders that fail alone
                cfg = dict(inc_cfg)
                sp = self.mem.split(cfg, bcfg, known)    # the spender: among the blocking run's moves that spend (or
                spenders = {kk for move in sp['spends'] + sp['unknown'] for kk, _, _ in move if kk != k and kk in self.knobs}
                moved = [(kk, vk(bcfg.get(kk))) for kk in spenders if vk(bcfg.get(kk)) != vk(cfg.get(kk))]   # may), the largest
                sk = None
                if moved:
                    sk, skey = max(moved, key=lambda t: mean.get(t, 0.0))
                    cfg[sk] = next(x for x in self.knobs[sk] if vk(x) == skey)
                cfg[k] = v
                held = set(self._levels(cfg))
                swapped = []
                for pr, rec in self.pairs.items():       # a value held that conflicts with the partner or the spender
                    if rec['p'] < 0.5 and pr <= held:    # goes back to the reference's value (hls3 F5c's swap)
                        keep = {(k, key), (sk, vk(cfg.get(sk)))} if sk else {(k, key)}
                        for kk, kv in sorted(pr - keep) or sorted(pr - {(k, key)}):
                            if (kk, kv) in held and not kk.startswith('patch:'):
                                cfg[kk] = self.ref.get(kk)
                                held.discard((kk, kv))
                                swapped.append(f'{kk}={json.loads(kv) if kv != "null" else None}')
                                break
                legal = self._complete(engine, cfg, {k, sk} if sk else {k})   # (the coupled knobs completed)
                if legal is None or vk(legal.get(k)) != vk(cfg.get(k)) or (sk and vk(legal.get(sk)) != vk(cfg.get(sk))):
                    self.partner_skip.add((block, k))    # no legal form holds the spender and the partner: not one
                    continue
                cfg = legal
                if self._ckey(cfg, ids) in self.tried:
                    self.partner_skip.add((block, k))
                    continue
                self.partner_done.add((block, k))        # consumed only now: a probe is offered
                self.counts['partner_probes'] += 1
                x = self._feats(cfg, ids)
                mu = Surrogate.value(x, mean) - y_inc
                q = self._proposal(engine, inc, cfg, ids, mu=mu, sigma=math.sqrt(max(1e-6, self.surr.var(x))), p=P_UNSURE,
                                   why=f'{block} did not fit; {k}={v} has no known memory direction'
                                       + (f'; {", ".join(swapped)} back to the reference (it conflicts with {k}={v})' if swapped else ''),
                                   probe=True, origin='wcs-partner', kind='partner')
                q.keep_probe = True
                return q
        return None

    # ------------------------------------------------------------ the coverage screen
    def _coverage(self, engine, inc) -> Proposal | None:
        c = engine.s.candidates[inc]
        cfg = {k: v for k, v in flat(c.config, self._native(engine)).items() if k in self.knobs}
        mean, sds = self.surr.mean(), self.surr.sd()
        changed = False
        for k in self.knobs:
            best, bv = None, 0.0
            for v in self.knobs[k]:
                f = (k, vk(v))
                if f in mean and mean[f] > 0 and mean[f] >= 0.5 * sds.get(f, 0.0) and mean[f] > bv:
                    best, bv = v, mean[f]
            if best is not None and vk(best) != vk(cfg.get(k)):
                cfg[k] = best
                changed = True
        ids = tuple(self.chain)
        if not changed:
            return None
        inc_cfg = {k: v for k, v in flat(c.config, self._native(engine)).items() if k in self.knobs}
        cfg = self._complete(engine, cfg, {k for k in self.knobs if vk(cfg.get(k)) != vk(inc_cfg.get(k))})   # (coupled
        # knobs completed from the table, the bundle's values kept when a legal row holds them all, WCS_r1 D4)
        if cfg is None or all(vk(cfg.get(k)) == vk(inc_cfg.get(k)) for k in self.knobs) or self._ckey(cfg, ids) in self.tried:
            return None
        p, why = self._p_feasible(cfg, ids)
        if p < self.probe_below:
            return None
        x = self._feats(cfg, ids)
        mu = Surrogate.value(x, mean) - Surrogate.value(self._incumbent_feats(engine, inc), mean)
        self.counts['coverage'] += 1
        return self._proposal(engine, inc, cfg, ids, mu=mu, sigma=math.sqrt(max(1e-6, self.surr.var(x))), p=p, why=why,
                              probe=False, origin='wcs-coverage', kind='coverage')

    # ------------------------------------------------------------ the code track
    def _code(self, engine, inc, left) -> list[Proposal]:
        """The queued patch proposals, rebased on the incumbent when it changed; a new ask when none is pending, two or
        more screens are left, a screen ran since the last ask and the asks are not spent."""
        out = []
        for q in list(self.queue):
            pid = q.hypothesis.split('|', 1)[1]
            want = frozenset(self.chain) | {pid}
            if q.base != inc or self.ideas.get(q.hypothesis) != want:   # the incumbent or the carried chain changed:
                patch = self._rebase(engine, inc, q.patch, q.base)    # rebased onto what is carried now (WCS_r1 D3)
                if patch is None:
                    self.queue.remove(q)
                    self.ledger[q.hypothesis]['status'] = 'dropped'
                    self.ledger[q.hypothesis]['why'] = f'its edit does not apply on {inc} with {sorted(self.chain)}'
                    continue
                q.patch, q.base = patch, inc
                self.ideas[q.hypothesis] = want
                self.ledger[q.hypothesis]['patches'] = sorted(want)
                self.patches[pid]['files'] = dict(patch)
            out.append(q)
        if out or left < 2 or self.asks >= self.code_asks or engine.full_screens() < 1 or self.last_ask_at == engine.full_screens():
            return out
        if not hasattr(self.proposer, 'hypotheses') or getattr(self.proposer, 'driver', True) is None:
            return out                                   # (no model: no code track)
        if self._llm_used() >= LLM_CALLS_MAX:
            return out                                   # the allowance is spent: the knob search goes on alone (D6)
        self.asks += 1
        self.last_ask_at = engine.full_screens()
        self._allowance()
        try:
            hyps = self.proposer.hypotheses(engine, inc, ledger=self.model_view())
        except Exception as exc:                         # noqa: BLE001
            (getattr(engine, 'log', None) or (lambda m: None))(f'[search] wcs: the patch call failed: {str(exc)[:200]}')
            hyps = []
        self.llm_calls = self._llm_used()
        n = 0
        for h in hyps or []:
            if not isinstance(h, dict) or not h.get('patch') or (h.get('change') or {}):
                continue                                 # patches only
            if n >= self.patches_per_ask:
                break
            n += 1
            q = self._patch_proposal(engine, inc, h)
            if q is not None:
                self.queue.append(q)
                out.append(q)
        return out

    def _llm_used(self) -> int:
        """The model calls made so far, every call counted (`driver_calls`: the prior, the review, the asks and their
        re-asks; a proposer without it: its asks)."""
        n = getattr(self.proposer, 'driver_calls', None)
        return int(n) if n is not None else int(getattr(self.proposer, 'calls', 0) or 0)

    def _allowance(self) -> None:
        """What the proposer may still spend of LLM_CALLS_MAX (its re-asks included)."""
        if hasattr(self.proposer, 'calls_left'):
            self.proposer.calls_left = max(0, LLM_CALLS_MAX - self._llm_used())

    def _carried(self, engine, inc) -> dict:
        """The files a configuration runs with now: the incumbent's under the adopted overlay (a patch adopted without
        moving the engine's incumbent lives in overlay_best, not in the incumbent's files)."""
        files = files_of(engine, inc)
        return {**files, **self.overlay_best} if self.chain and self.overlay_best else files

    def _rebase(self, engine, inc, patch: dict, base: str) -> dict | None:
        """A patch written against `base`'s files, merged (three-way) onto what is carried now; None when its edit no
        longer applies or adds nothing."""
        old = files_of(engine, base) if base in engine.s.candidates else {}
        carried = self._carried(engine, inc)
        cur = {f: carried.get(f, '') for f in patch}
        merged = merge_patches(old, [cur, patch])
        if merged is None or merged == cur:
            return None
        if self.chain and self.overlay_best:             # every adopted file rides along, the edited ones merged
            return {**self.overlay_best, **merged}       # (WCS_r2 1: a patch on another file kept only its own)
        return merged

    def _patch_proposal(self, engine, inc, h: dict) -> Proposal | None:
        patch = dict(h['patch'])
        if self.chain and self.overlay_best:             # written against the incumbent's files: onto the chain (D3)
            patch = self._rebase(engine, inc, patch, inc)
            if patch is None:
                return None
        h = {**h, 'patch': patch}
        self._n += 1
        pid = f'p{self._n}'
        hid = f'patch|{pid}'
        mech = str(h.get('mechanism') or f'patch {pid}')[:200]
        mu = self.kappa0 * float(h.get('mu') or 0.0)
        p = min(0.95, max(0.05, float(h.get('p_feasible') or 0.8)))
        ids = frozenset(self.chain) | {pid}
        self.ideas[hid] = ids
        self.patches[pid] = {'files': dict(h['patch']), 'status': 'queued', 'mu': mu, 'mechanism': mech,
                             'site': h.get('site'), 'mem_dir': h.get('mem_dir'), 'mem_gib': h.get('mem_gib'),
                             'evidence_note': h.get('evidence_note'), 'active_from_update': h.get('active_from_update')}
        a = self._supported(h, 'active_from_update')
        steps = min(PROBE_STEPS_MAX, max(GATED_STEPS, a or 0))
        cap = self._probe_cap(engine, inc, steps)
        keep = True                                      # the self-proof needs the probe: a full run of an inert patch
        spends = h.get('mem_dir') == 'spends'            # would waste a screen (hls3 F10: the same)
        unknown = h.get('mem_dir') not in DIRS
        q = Proposal(change={}, patch=dict(h['patch']), base=inc, mu=mu, sigma=max(0.005, float(h.get('sigma') or 0.02)),
                     p_feasible=p, mechanism=mech, hypothesis=hid, origin='wcs-patch', fidelity=('probe', 'full'),
                     keep_probe=keep, probe_cap=cap, change_class=h.get('change_class'),
                     probe_steps=steps if steps > int(getattr(engine, 'probe_steps', 2) or 2) else None,
                     gate=BoundGate(self, '_patch_gate', h, spends, unknown))
        self.counts['patches'] += 1
        self.ledger[hid] = {'id': hid, 'kind': 'patch', 'mechanism': mech, 'change': {}, 'patches': sorted(ids),
                            'base': inc, 'predicted_gain_pct': round(100 * math.expm1(mu), 2), 'p_feasible': round(p, 3),
                            'plan': ['probe', 'full'], 'status': 'open', 'site': h.get('site')}
        self.picks.append({'id': hid, 'kind': 'patch', 'screens_done': engine.full_screens(), 'mechanism': mech,
                           'p_feasible': round(p, 3), 'predicted_gain_pct': round(100 * math.expm1(mu), 2)})
        return q

    def _patch_gate(self, engine, cid, base, row, h, spends, unknown):
        """A patch's probe gate: the speed gate, then the shared self-proof (search/proof.check: the patch printed its
        installation and active lines read whole, F13; the note it names is gone; a `deferred` line, F21); a patch that
        did not prove it took effect gets no full run, and the proposer is told why."""
        why = self._speed_gate(engine, cid, base, row, spends, unknown, self._supported(h, 'warmup_updates') or 0)
        if why:
            return why
        fb: list = []
        rec = proof_check(engine, base, row, h.get('patch'), h, whole=True, deferred=True, feedback=fb)
        self.note_feedback += [{'candidate': cid, **x} for x in fb]
        if rec:
            self.proof_fail[cid] = rec
            return rec['reason'], 'patch_inert'
        return None

    def model_view(self, last: int | None = 40) -> list[dict]:
        """The ledger as the patch call sees it: every proposal with its outcome, the inert patches with their reasons; the
        `last` entries (None: every one - v4.1's finalists, whose patch call folds what does not fit)."""
        rows = list(self.ledger.values())
        return [{k: r[k] for k in ('mechanism', 'change', 'patches', 'base', 'status', 'measured_gain_pct', 'why', 'site')
                 if r.get(k) is not None} for r in (rows[-last:] if last else rows)]

    def context(self, engine, base) -> dict:
        """The extra fields of the model's patch call: the failed runs, the inert patches, the pair table."""
        out = {}
        if self.failures:
            out['failures'] = self.failures[-10:]
        if self.inert:
            out['inert_patches'] = self.inert[-6:]
        if self.note_feedback:
            out['patch_notes'] = self.note_feedback[-6:]
        pairs = [{'pair': self._pair_words(pr), 'p': rec['p'], 'why': rec['why']} for pr, rec in self.pairs.items() if rec['p'] < 1.0]
        if pairs:
            out['failed_together'] = pairs[:8]
        if self.mem.runs:
            out['memory_view'] = {'capacity_gib': self.mem.cap, 'runs': [
                {'run': cid, 'status': r['status'], **({'peak_gib': r['peak']} if r.get('peak') is not None else {}),
                 **({k: r['oom'][k] for k in ('tried_gib', 'free_gib', 'short_gib') if k in r['oom']} if r.get('oom') else {})}
                for cid, r in list(self.mem.runs.items())[-10:]]}
        return out

    # ------------------------------------------------------------ outcomes
    def observe(self, engine, p, out: Outcome):
        if out.status == 'infra':
            return
        self._forget()
        hid = p.hypothesis or ''
        row = self.ledger.get(hid)
        ids = self._patch_ideas(p)
        if p.absolute:
            self.tried.add(self._ckey({**self.ref, **(p.change or {})}, ids))
        if out.status == 'duplicate':
            if row:
                row['status'] = 'duplicate'
            self.queue = [q for q in self.queue if q.hypothesis != hid]
            return
        cls = self.classify(engine, p, out)
        kind = cls['kind']
        base = 'c0' if p.absolute else (p.base or engine.incumbent())
        if out.cid:
            self.pids[out.cid] = frozenset(ids)
        cfg_m = self._mcfg(engine, out.cid) if out.cid in engine.s.candidates else \
            self._m_of({**self.ref, **(p.change or {})} if p.absolute else after(engine.s.candidates[base].config, p.change,
                                                                                 self._native(engine)), ids)
        self.tried.add(self._ckey({k: cfg_m.get(k) for k in self.knobs}, ids))
        kind = self._jitter(engine, p, out, cfg_m, base, kind, cls)
        t0 = engine.t('c0')
        y, bound = None, False
        if out.status == 'screened' and out.cid and engine.t(out.cid) and t0:
            y = math.log(t0 / engine.t(out.cid))
        elif out.status == 'aborted' and out.cid and t0 and kind != 'thrash':
            tp = engine.t_partial(out.cid)
            if tp:
                y = math.log(t0 / tp)
            else:                                        # no timestamps: the abort rule itself is the evidence - slower
                inc = engine.incumbent()                 # than the incumbent by at least its margin (a wide observation,
                ti = engine.t(inc) if inc in engine.s.candidates else None   # so that the stack is not proposed again)
                if ti:
                    y, bound = math.log(t0 / ti) - max(2.0 * engine.sigma_pair(), ABORT_MIN) - ABORT_DEPTH, True
        if out.cid:
            self.runs[out.cid] = {'cfg': cfg_m, 'kind': kind, 'y': y, 'origin': p.origin, 'base': base,
                                  'status': out.status, 'fidelity': out.fidelity, 'bound': bound}
        if self.on['wcs_feasibility'] and out.cid and kind not in ('refused', 'infra'):
            self._record_memory(engine, out, cfg_m, base, kind, cls)
            if kind in ('oom', 'thrash') and row is not None and (row.get('p_feasible') or 0.0) >= 0.3 \
                    and 'declared' in (row.get('why_feasible') or '') and self.spend_scale < SPEND_MAX:
                self.spend_scale = min(SPEND_MAX, self.spend_scale + SPEND_STEP)   # the declarations were optimistic
                self.counts['calibrations'] += 1
        if self.on['wcs_antagonism'] and out.cid:
            self._record_pairs(cfg_m, base, kind, out)
        if kind not in ('ok', 'slower', 'refused', 'infra') and out.cid:
            o = cls.get('oom') or {}
            self.failures.append({'candidate': out.cid, 'class': kind, 'mechanism': p.mechanism[:120],
                                  **({'detail': f'tried to allocate {o["tried_gib"]:.2f} GiB' + (
                                      f': short by at least {o["short_gib"]:.2f} GiB' if 'short_gib' in o else '')}
                                     if o else {'detail': cls['line'][:200]} if cls.get('line') else {})})
        if y is not None and (out.status == 'screened' or kind != 'thrash'):
            feats = self._feats({k: cfg_m.get(k) for k in self.knobs}, [x for x in ids])
            sd = engine.sigma_run() * (ABORT_SD if out.status == 'aborted' else 1.0)
            if bound:                                    # a bound never raises a belief already below it
                y, sd = min(y, Surrogate.value(feats, self.surr.mean())), max(sd, ABORT_BOUND_SD)
            measured = set(feats)                        # a suggestion about any value now measured is superseded
            self.surr.retire(lambda tag, fs, _m=measured: tag[0] == 'config' and bool(set(fs) & _m))   # (WCS_r2 4)
            self.surr.observe(feats, y, sd)
        if row is not None:
            row['status'] = {'screened': 'measured', 'aborted': 'slower', 'infeasible': kind, 'invalid': kind,
                             'rejected': 'refused', 'probed': 'probed', 'short': 'measured'}.get(out.status, out.status)
            row['candidate'] = out.cid
            if y is not None:
                row['bound_below_pct' if bound else 'measured_gain_pct'] = \
                    round(100 * math.expm1(y - (self.runs.get(base, {}).get('y') or 0.0)), 2)
            if out.reasons and not out.valid:
                row['why'] = str(out.reasons[0])[:200]
            if out.cid is not None and engine.incumbent() == out.cid:
                row['moved'] = True
        if hid.startswith('patch|'):
            self._observe_patch(engine, p, out, kind, y, base)
        self._forget()                                   # (the measured moves changed above: the caches go again)
        for pk in self.picks:
            if pk['id'] == hid:
                pk['outcome'] = row['status'] if row else out.status
                if y is not None:
                    pk['bound_below_pct' if bound else 'gain_vs_reference_pct'] = round(100 * math.expm1(y), 2)

    def _jitter(self, engine, p, out, cfg_m, base, kind, cls=None) -> str:
        """The cliff read without a peak (FABLE_HLS3_FIXES_R3 F23): a valid or aborted run whose update intervals scatter
        (MAD / median at least JITTER and JITTER_X times its base's) on a configuration that spends memory against its
        base is thrash - at the memory edge, not a slow move."""
        if kind not in ('ok', 'slower') or not self.on['wcs_feasibility'] or not out.cid or out.fidelity != 'full':
            return kind
        if not self._spends(self.runs[base]['cfg'] if base in self.runs else self._mcfg(engine, base), cfg_m)[0]:
            return kind
        t, tb = engine.t(out.cid), engine.t(base)
        if kind == 'ok' and not (t and tb and t > tb):
            return kind                                  # F23 reads a slowdown: a run as fast as its base is no cliff
        peak = (cls or {}).get('peak_gib')
        if isinstance(peak, (int, float)) and self.mem.cap and peak < NEAR_CAP * self.mem.cap:
            return kind                                  # a measured peak with headroom contradicts the memory edge
        j, jb = self._scatter(engine, out.cid), self._scatter(engine, base)
        if j is None or j < JITTER or (jb is not None and j < JITTER_X * jb):
            return kind
        self.counts['jitter_thrash'] += 1
        return 'thrash'

    @staticmethod
    def _scatter(engine, cid) -> float | None:
        c = engine.s.candidates.get(cid)
        rec = getattr(c, 'receipt', None) or {}
        ev = rec.get('evidence') if isinstance(rec.get('evidence'), dict) else {}
        ps = [r['t'] for r in ev.get('per_step') or [] if isinstance(r, dict) and isinstance(r.get('t'), (int, float))]
        gaps = [b - a for a, b in zip(ps, ps[1:])][2:]
        if len(gaps) < 5:
            return None
        med = statistics.median(gaps)
        if med <= 0:
            return None
        mad = statistics.median(abs(g - med) for g in gaps)
        return mad / med

    def _record_memory(self, engine, out, cfg_m, base, kind, cls) -> None:
        status = {'oom': 'oom', 'thrash': 'thrash', 'ok': 'ok', 'slower': 'ok', 'numerics': 'ok'}.get(kind)
        if status is None:
            return
        peak = cls.get('peak_gib')
        self.mem.record(out.cid, cfg_m, status, peak=peak, oom=cls.get('oom'))
        # the directions the runs show: an out-of-memory or thrashing run against a run that fitted, one move apart,
        # spends. Two failed runs one move apart say nothing about that move's direction (WCS_r1 D2: both failing
        # establishes that neither fits, not that the move is neutral): they stay failure bounds.
        runs = list(self.mem.runs.values())
        for ra in runs:
            for rb in runs:
                if ra is rb or rb['status'] not in ('oom', 'thrash') or len(sig(ra['cfg'], rb['cfg'])) != 1:
                    continue
                if ra['status'] == 'ok':
                    self.mem.relate(ra['cfg'], rb['cfg'], 'spends')
        b = self.mem.runs.get(base)
        if b is not None and b['status'] == 'ok' and isinstance(b.get('peak'), (int, float)) and isinstance(peak, (int, float)):
            d = peak - b['peak']                         # a measured peak against its base's (D3): the move's direction
            tol = max(1.0, 0.02 * (self.mem.cap or 0.0))
            self.mem.relate(b['cfg'], cfg_m, 'spends' if d > tol else 'frees' if d < -tol else 'neutral')

    def _record_pairs(self, cfg_m, base, kind, out) -> None:
        """The pair table: a run that failed at its start (a class a combination can cause) marks the pairs among the
        values it added to its base (few: strongly; many: weakly) and a value added alone; a valid run clears every
        pair and value it holds."""
        self._forget()
        new = sorted(set(self._levels(cfg_m)) - set(self._levels(self.runs[base]['cfg'] if base in self.runs else {})))
        if kind in TOGETHER:
            if len(new) == 1:
                self.alone[new[0]] = self.alone.get(new[0], 0) + 1
            elif 2 <= len(new) <= PAIR_SET_MAX:
                f = P_PAIR_FAILED if len(new) <= 3 else 0.5
                for a, b in itertools.combinations(new, 2):
                    rec = self.pairs.setdefault(frozenset((a, b)), {'p': 1.0, 'why': ''})
                    rec['p'] = min(rec['p'], max(P_PAIR_FAILED, rec['p'] * f))
                    rec['why'] = f'{out.cid} failed to start ({kind}) with both'
                # the values two failed starts share: their pairs are the suspects (a whole configuration adds many
                # values; what recurs across failures is what matters)
                for earlier in self.started:
                    common = sorted(set(new) & {(k, vk(v)) for m in earlier['members'] for k, v in m.items()})
                    if 2 <= len(common) <= 4:
                        for a, b in itertools.combinations(common, 2):
                            rec = self.pairs.setdefault(frozenset((a, b)), {'p': 1.0, 'why': ''})
                            rec['p'] = min(rec['p'], P_PAIR_FAILED)
                            rec['why'] = f'{earlier["run"]} and {out.cid} failed to start with both'
            if not any(k.startswith('patch:') for k, _ in new):          # (D4: knob values only travel to the store)
                self.started.append({'members': [{k: json.loads(v)} for k, v in new], 'kind': kind, 'run': out.cid})
        elif kind in ('ok', 'slower', 'thrash', 'numerics', 'oom'):
            held = set(self._levels(cfg_m))
            for pr in self.pairs:
                if pr <= held:
                    self.pairs[pr] = {'p': 1.0, 'why': f'{out.cid} ran with both'}
            for key in list(self.alone):
                if key in held:
                    del self.alone[key]

    def _levels(self, cfg_m: dict) -> list:
        return [(k, vk(v)) for k, v in cfg_m.items() if k.startswith('patch:') or (k in self.knobs and vk(v) != vk(self.ref.get(k)))]

    def _observe_patch(self, engine, p, out, kind, y, base) -> None:
        pid = p.hypothesis.split('|', 1)[1]
        rec = self.patches.get(pid)
        if rec is None:
            return
        self.queue = [q for q in self.queue if q.hypothesis != p.hypothesis]
        if kind == 'inert_at_start':
            f = self.proof_fail.get(out.cid) or {'reason': (out.reasons or [''])[0]}
            rec['status'] = 'inert at its probe'
            self.inert.append({'candidate': out.cid, 'mechanism': rec['mechanism'], 'site': rec.get('site'), 'at': 'probe',
                               'reason': f['reason'], **({'note': f['note']} if f.get('note') else {}),
                               'printed': f.get('lines') or [], **({'parsed': f['parsed']} if f.get('parsed') else {})})
            return
        if out.status != 'screened' or y is None:
            rec['status'] = f'{kind}' if kind not in ('ok',) else out.status
            return
        yb = (self.runs.get(base) or {}).get('y') or 0.0
        gain = y - yb
        rec.update(status='proven', candidate=out.cid, gain_pct=round(100 * math.expm1(gain), 2))
        sp = engine.sigma_pair()
        if gain > sp:                                    # adopted: every later configuration carries it
            if pid not in self.chain:
                self.chain.append(pid)
            c = engine.s.candidates.get(out.cid)
            self.overlay_best = dict(c.overlay or {}) if c is not None else None
            rec['adopted'] = True
        else:
            rec['adopted'] = False
            if rec['mu'] >= 2 * sp:
                self.inert.append({'candidate': out.cid, 'mechanism': rec['mechanism'], 'site': rec.get('site'),
                                   'predicted_pct': round(100 * math.expm1(rec['mu']), 2),
                                   'measured_pct': round(100 * math.expm1(gain), 2), 'at': 'full run'})

    # ------------------------------------------------------------ report
    def ledger_view(self) -> list[dict]:
        return list(self.ledger.values())

    def residual(self, engine) -> dict:
        mean = self.surr.mean()
        top = sorted(((f, m) for f, m in mean.items() if isinstance(f, tuple) and f[0] != 'pair'), key=lambda t: -t[1])[:6]
        return {'posterior_top': [{'value': f'{f[0]}={f[1]}', 'gain_pct': round(100 * math.expm1(m), 2)} for f, m in top],
                'observations': len(self.surr.obs), 'queued_patches': len(self.queue)}

    def _learned(self) -> dict:
        """D4: what this run learned of its starts, for runs.write_back (experience.learned.section): the sets of knob
        values that failed to start together, the pairs that stayed suspect (two or more failed starts, or one of a small
        set), the values that failed alone."""
        from ..experience.learned import section
        sets = [x for x in self.started if len(x['members']) >= 2]
        pairs = [{'members': [{k: json.loads(v)} for k, v in sorted(pr)], 'kind': 'start',
                  'run': rec['why'].split(' ', 1)[0], 'strikes': 2 if rec['p'] <= P_PAIR_FAILED else 1}
                 for pr, rec in self.pairs.items() if rec['p'] <= P_PAIR_FAILED and not rec['why'].startswith('an earlier arm')
                 and not rec['why'].startswith('declared') and not any(k.startswith('patch:') for k, _ in pr)]
        alone = [{'members': [{k: json.loads(v)}], 'kind': 'start'} for (k, v), n in sorted(self.alone.items())
                 if n and not k.startswith('patch:')]
        return section(failed_together=sets, pairs=pairs, alone=alone)

    def describe(self) -> dict:
        off = sorted(k for k in WCS_COMPONENTS if not self.on[k])
        sds = self.surr.sd() if self.surr.names else {}
        mean = self.surr.mean() if self.surr.names else {}
        out = {'name': self.name, 'proposer': self.proposer.describe() if hasattr(self.proposer, 'describe') else None,
               'wcs': {'off': off, 'prior': self.prior.describe() if self.prior else None,
                       'experience_absorbed': getattr(self.prior, 'absorbed', 0) if self.prior else 0,
                       'table': {'legal': len(self.table or []), 'whole': self.table_whole, 'cap': self.table_cap},
                       'counts': dict(self.counts), 'llm_calls': self._llm_used(), 'llm_calls_max': LLM_CALLS_MAX,
                       'picks': self.picks,
                       'runs': {cid: {k: r[k] for k in ('kind', 'origin', 'status', 'fidelity') if r.get(k) is not None}
                                | ({('bound_below_pct' if r.get('bound') else 'gain_vs_reference_pct'): round(100 * math.expm1(r['y']), 2)}
                                   if r.get('y') is not None else {})
                                for cid, r in self.runs.items() if cid != 'c0'},
                       'failures': self.failures, 'pairs': [{'pair': self._pair_words(pr), **rec} for pr, rec in self.pairs.items()
                                                            if rec['p'] < 1.0],
                       'alone_failed': {f'{k}={v}': n for (k, v), n in self.alone.items()},
                       'memory': {'capacity_gib': self.mem.cap, 'floor': self.mem.floor,
                                  'runs': {cid: {k: r[k] for k in ('status', 'peak') if r.get(k) is not None} | (
                                      {'short_gib': r['oom'].get('short_gib')} if r.get('oom') else {})
                                      for cid, r in self.mem.runs.items()},
                                  'measured_moves': [{'move': HLS3._words(s), 'direction': d} for s, d in self.mem.measured.items()]},
                       'posterior': sorted(({'value': f'{f[0]}={f[1]}' if f[0] != 'pair' else 'pair:' + '+'.join(
                           f'{k}={m}' for k, m in sorted(f[1])), 'gain_pct': round(100 * math.expm1(m), 2),
                                             'sd_pct': round(100 * sds.get(f, 0.0), 2)} for f, m in mean.items()),
                                           key=lambda r: -r['gain_pct'])[:16],
                       'patches': {pid: {k: v for k, v in rec.items() if k != 'files'} for pid, rec in self.patches.items()},
                       'chain': list(self.chain), 'inert': self.inert, 'gated': self.gated,
                       'from_store': self.from_store}}
        learned = self._learned()
        if learned:
            out['learned'] = learned                     # D4: runs.write_back writes it as constraints for a later arm
        return out
