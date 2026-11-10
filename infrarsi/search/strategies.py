"""Search strategies on the Engine (revamp design C; Fable 5.1 consult §2): what to test next, never how to measure or
when the budget is spent.

    greedy   the current agent's behaviour as a baseline: test the best-predicted mechanism on the incumbent, full
             fidelity only, and finish as soon as no prediction clears the noise ("further gains are unlikely")
    hls      headroom-ledger search: a ledger of hypotheses with predicted and measured gains, a calibration factor
             kappa (shrunk to 0.5 with 3 pseudo-observations) applied to every open LLM prediction (the only explicit
             calibration: the model sees measurements, not its own predictions of them), value/cost
             scheduling and the engine's numeric stop; within-noise positives are bundled and re-measured together;
             measured experience (infrarsi.experience) enters as seeded entries and one start composite, uncalibrated;
             when the incumbent's ledger, single-knob neighbourhood and bundles are all measured, the model is asked
             again (at most `reask` times per incumbent, not after an answer with nothing new) before the search ends;
             a knob that fails a numerics check on an incumbent carrying an earlier accepted change is tried once more
             with that change reverted (at most `retry` per incumbent: the failure may be theirs together)
    doe      a Plackett-Burman screening design over two-level factors at short fidelity, main effects, then one full
             run of the combination of the helpful factors
    hls2     hls with the search fix (FABLE_DESIGN_SEARCH_FIX, 2026-10-29) switched on: a never-tested knob entry follows
             the incumbent (C1), the model sees the ledger as the strategy can test it (C2), an untested move judged '+'
             gets an honest prior, the single-knob neighbourhood is priced on every incumbent and a judged knob goes
             before an idea below the noise (C3), kappa per origin (C4), a knob value that fails to start twice is
             unrunnable for the run (C5), same-anchor appends merge (C7), no reverts in the neighbourhood (C8); every
             switch off is hls
    hls3     hls2 with FABLE_TAKEAWAYS_20261030's general strategies (search/hls3.py: memory feasibility and kept probes,
             spender + enabler bundles, failures as information, stall probes and whole-configuration steps,
             re-basing, phase re-ranking, code sites, median decisions, card-size pruning); every component off is hls2
    chain    strategies in sequence (e.g. doe then hls)
    random / tpe   knob-space baselines through the same engine (fidelity plans included)
"""
from __future__ import annotations

import difflib
import functools
import hashlib
import json
import math
import random
from collections import Counter

from ..agent.search import START_FAILED
from .engine import Outcome, Proposal, Strategy
from .knob_space import choices
from .proposers import unpaired

PB_GENERATORS = {8: '+++-+--', 12: '++-+++---+-', 16: '++++-+-++--+---', 20: '++--++++-+-+----++-',
                 24: '+++++-+-++--++--+-+----'}


def pb_design(n_factors: int) -> list[list[int]]:
    """Plackett-Burman design for up to 23 two-level factors: rows of +1/-1, one column per factor."""
    n = next(k for k in sorted(PB_GENERATORS) if k - 1 >= n_factors)
    g = [1 if c == '+' else -1 for c in PB_GENERATORS[n]]
    rows = [[g[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n - 1)] + [[-1] * (n - 1)]
    return [r[:n_factors] for r in rows]


def _merge(changes) -> dict:
    out = {}
    for c in changes:
        out.update(c)
    return out


def merge3(base: str, a: str, b: str, *, append: bool = False) -> str | None:
    """Three-way line merge of two edits of `base`, conservative: the texts are cut at anchors - lines that both diffs
    leave unchanged and that occur once in each of the three texts, or once in each text's stretch between two anchors
    (found recursively, as in patience diff) - and each piece between two anchors may be changed by one side only (or
    identically by both); None otherwise. Cut at such lines the result cannot depend on which of several repeated lines
    a diff chose to align (between them it can); edits with no anchor in between conflict, edits of adjacent lines
    included (as in diff3).
    `append` (hls2, FABLE_DESIGN_SEARCH_FIX C7): where the base has no line between two anchors and both sides insert
    there (models append a function after a file's last call: Q9U c5/c6/c7, so their bundle never formed), both
    insertions are kept, `a`'s first (an insertion that extends the other is taken whole); a changed base line still
    conflicts."""
    bl, al, cl = (t.splitlines(keepends=True) for t in (base, a, b))
    pos: dict[int, dict] = {}                           # unchanged base line -> its line in a and in b
    for side, new in (('a', al), ('b', cl)):
        for i, j, n in difflib.SequenceMatcher(None, bl, new, autojunk=False).get_matching_blocks():
            for k in range(n):
                pos.setdefault(i + k, {})[side] = j + k

    def anchors(lo, hi):                                # strictly between two anchors (index triples)
        nb, na, nc = Counter(bl[lo[0] + 1:hi[0]]), Counter(al[lo[1] + 1:hi[1]]), Counter(cl[lo[2] + 1:hi[2]])
        inner = [(i, pos[i]['a'], pos[i]['b']) for i in range(lo[0] + 1, hi[0])
                 if len(pos.get(i, ())) == 2 and nb[bl[i]] == na[bl[i]] == nc[bl[i]] == 1]
        out = []
        for c in inner:
            out += anchors(lo, c) + [c]
            lo = c
        return out + anchors(lo, hi) if inner else []
    cuts = anchors((-1, -1, -1), (len(bl), len(al), len(cl)))
    out, prev = [], (-1, -1, -1)
    for cut in cuts + [(len(bl), len(al), len(cl))]:
        pb, pa, pc = bl[prev[0] + 1:cut[0]], al[prev[1] + 1:cut[1]], cl[prev[2] + 1:cut[2]]
        if pa == pb or pa == pc:
            out += pc
        elif pc == pb:
            out += pa
        elif append and not pb:
            out += pc if pc[:len(pa)] == pa else pa if pa[:len(pc)] == pc else pa + pc
        else:
            return None
        out += bl[cut[0]:cut[0] + 1]
        prev = cut
    return ''.join(out)


def merge_patches(base_files: dict, patches: list, *, append: bool = False) -> dict | None:
    """One patch {file: full text} from several patches of the same base candidate: different files are united, edits
    of one file are combined by a three-way merge against the base text; None when two edits conflict. `append`: see
    merge3."""
    out = {}
    for p in patches:
        for f, text in (p or {}).items():
            if f not in out or out[f] == text:
                out[f] = text
                continue
            m = merge3(base_files.get(f, ''), out[f], text, append=append)
            if m is None:
                return None
            out[f] = m
    return out


NUMERICS = ('change_class', 'loss_deviation_pct', 'loss_deviation_base_pct', 'mu_cap_pct', 'predicted_raw_pct')   # report-only
# checks a knob can fail because of what its base already changed (a crash, a workload or data change fail on every base)
NUMERICS_CHECKS = ('loss_within_tolerance', 'grad_norm_within_tolerance', 'no_errors')


def _numerics(cls, dev, dev_base=None) -> dict:
    """A ledger row's change class (the proposer's label) and the loss deviation its full run measured from the
    reference and from its base (unrounded: the report compares them with the repeat floor). Report-only: the rows the
    model sees leave them out (`_bare`)."""
    return {**({'change_class': cls} if cls else {}), **({'loss_deviation_pct': 100 * dev} if dev is not None else {}),
            **({'loss_deviation_base_pct': 100 * dev_base} if dev_base is not None else {})}


def _bare(h: dict) -> dict:
    return {k: v for k, v in h.items() if k not in NUMERICS}


def _gain(h: dict) -> float:
    """A ledger entry's gain for composition: measured when it was, else its prior (an open experience member)."""
    return h['measured'] if h.get('measured') is not None else h['mu']


class Greedy(Strategy):
    name = 'greedy'
    uses_engine_stop = False

    def __init__(self, proposer, *, finish_below: float | None = None):
        self.proposer, self.finish_below = proposer, finish_below
        self.history: list[dict] = []          # what it tried and what came of it, shown to the model as the ledger

    def describe(self) -> dict:
        return {'name': self.name, 'proposer': self.proposer.describe(), 'finish_below': self.finish_below}

    def propose(self, engine, k):
        inc = engine.incumbent()
        hyps = []
        for h in self.proposer.hypotheses(engine, inc, ledger=[_bare(h) for h in self.history[-40:]]):
            p = Proposal(change=h['change'], patch=h['patch'], base=inc, mu=h['mu'], sigma=h['sigma'],
                         p_feasible=h['p_feasible'], mechanism=h['mechanism'], origin='greedy', fidelity=('full',),
                         abortable=False, change_class=h.get('change_class'))
            if engine.key(p) not in engine.tried:
                hyps.append(p)
        if not hyps:
            return []
        best = max(hyps, key=lambda p: p.mu)
        thr = self.finish_below if self.finish_below is not None else 2 * engine.sigma_run()
        return [] if best.mu < thr else [best]

    def observe(self, engine, p, out: Outcome):
        """The old agent loop kept its tried moves in context; without them the model proposed moves that had already
        failed on an ancestor again (H2b arm B: 10 of 24 runs repeated FP8 / TE cross-entropy)."""
        if out.status in ('infra', 'duplicate'):
            return
        moved = out.cid is not None and engine.incumbent() == out.cid
        self.history.append({'mechanism': p.mechanism, 'change': dict(p.change or {}),
                             'patched_files': sorted(p.patch or {}) or None, 'predicted_gain_pct': round(100 * p.mu, 2),
                             'measured_gain_pct': None if out.gain is None else round(100 * out.gain, 2),
                             'status': 'adopted' if moved else 'not adopted' if out.status == 'screened' else out.status,
                             'failed_checks': out.failed_checks or None,
                             'why': out.reasons[:2] if out.status in ('rejected', 'invalid', 'infeasible') else None,
                             **_numerics(p.change_class, out.loss_deviation, out.loss_deviation_base)})


def _applied(config: dict, change: dict, *, native: bool) -> bool:
    """`change` already holds in `config` as the session applies it (agent/search.propose_config): without hooks an env
    change sets or (None) removes single variables, with hooks every value replaces the base's."""
    for k, v in change.items():
        if native and k == 'env' and isinstance(v, dict):
            env = config.get('env') or {}
            if any((n in env) if val is None else env.get(n) != val for n, val in v.items()):
                return False
        elif config.get(k) != v:
            return False
    return True


def _mkey(change: dict | None) -> str:
    """A knob change as a comparable key (the same move whatever its name or base)."""
    return json.dumps(change or {}, sort_keys=True, default=str)


def _pkey(patch: dict | None) -> str | None:
    return hashlib.sha256(json.dumps(sorted(patch.items()), default=str).encode()).hexdigest()[:16] if patch else None


def _knobs_set(change: dict, cfg: dict, *, native: bool) -> list:
    """(knob, value) pairs `change` sets to another value than `cfg` has; an env change (without hooks) per variable."""
    out = []
    for k, v in (change or {}).items():
        if native and k == 'env' and isinstance(v, dict):
            env = cfg.get('env') or {}
            out += [(f'env.{n}', val) for n, val in v.items() if env.get(n) != val]
        elif cfg.get(k) != v:
            out.append((k, v))
    return out


def _names(change: dict, native: bool) -> list:
    """The knobs a change sets (an env change without hooks: each variable as env.NAME)."""
    out = []
    for k, v in (change or {}).items():
        out += [f'env.{n}' for n in v] if native and k == 'env' and isinstance(v, dict) else [k]
    return out


def _value(cfg: dict, knob: str, *, native: bool):
    return (cfg.get('env') or {}).get(knob[4:]) if native and knob.startswith('env.') else cfg.get(knob)


MEASURED = ('confirmed', 'bundled', 'refuted', 'unresolved', 'invalid')   # a ledger entry that ran (or was refused)
UNTESTED = ('open', 'superseded')


class HLS(Strategy):
    name = 'hls'

    def __init__(self, proposer, *, kappa0: float = 0.5, n0: int = 3, probe_below: float = 0.8, explore: bool = True,
                 bundle: bool = True, split: bool = True, carry: bool = True, confirm: bool = True,
                 experience: dict | None = None, phase_cap: bool = False, reask: int = 2, retry: int = 2,
                 carry_open: bool = False, view_states: bool = False, prior_sigma: float | None = None,
                 explore_always: bool = False, backstop: bool = False, class_kappa: bool = False,
                 start_failures: int | None = None, cover_bundle: bool = False, append_merge: bool = False,
                 no_revert: bool = False):
        self.proposer, self.kappa0, self.n0, self.probe_below, self.explore = proposer, kappa0, n0, probe_below, explore
        self.bundle, self.split, self.carry, self.confirm = bundle, split, carry, confirm
        self.phase_cap = phase_cap                          # cap a prediction at what its phases can give (P0-2 step 2)
        self.reask = int(reask)                             # model calls per incumbent after its first, when the search runs dry
        self.reasks: dict[str, list] = {}                   # incumbent -> what each re-ask gave: 'new' | 'nothing' | 'failed'
        self.retry = int(retry)                             # knobs per incumbent tried again without its last change (_retry)
        self.retries: dict[str, list] = {}                  # incumbent -> the mechanisms of those entries
        # experience.store.plan: seeds (moves with a source gain), composite (the start point: knob names), known_ok
        # (moves measured within noise in scope: known to run), check (config -> warnings: constraints, memory)
        self.exp = experience or {}
        self.exp_fate: dict | None = None
        self.exp_bundle_screens = 0                         # full runs of bundles with an experience member
        self.ledger: dict[str, dict] = {}
        self.pairs: list[tuple[float, float]] = []          # (predicted, measured) log gains of resolved hypotheses
        self.refreshed_for = None
        self.bundles = 0                                    # bundle evaluations (duplicates not counted)
        self.bundle_numerics: list[dict] = []               # each full bundle run: its members' classes, its deviation
        self.bundle_ids: dict[frozenset, str] = {}          # member ledger keys -> bundle hypothesis id
        self.members: dict[str, list[str]] = {}             # bundle hypothesis id -> member ledger keys
        self.evaluated: set[frozenset] = set()
        self.failed_sets: list[frozenset] = []
        self.pending: dict[str, Proposal] = {}              # halves of failed bundles, until evaluated
        self.lost: dict[str, float] = {}                    # positive bundles that did not move the incumbent -> gain
        # the search fix (FABLE_DESIGN_SEARCH_FIX, 2026-10-29; strategy hls2): every switch off is the hls of before
        self.carry_open = carry_open                        # C1: a never-tested knob entry follows the incumbent
        self.view_states = view_states                      # C2/C9: the model's ledger as the strategy can test it
        self.prior_sigma = prior_sigma                      # C3a: sigma of an untested move judged '+' (None: off)
        self.explore_always = explore_always                # C3b: the neighbourhood on every incumbent, priced
        self.backstop = backstop                            # C3c: a judged knob before an idea below the noise
        self.class_kappa = class_kappa                      # C4: kappa per origin: knob / review / patch
        self.start_failures = int(start_failures or 0)      # C5: a value failing to start this often is unrunnable
        self.cover_bundle = cover_bundle                    # C6: one composite of cheap moves outnumbering the screens
        self.append_merge = append_merge                    # C7: insertions after the same anchor merge
        self.no_revert = no_revert                          # C8: the neighbourhood neither undoes nor tries a '-' move
        self.class_pairs: dict[str, list] = {}              # C4: origin class -> (predicted, measured), as `pairs`
        self.carried = 0                                    # C1: entries carried to a later incumbent
        self.backstops = 0                                  # C3c: proposals sets the backstop narrowed
        self.dead: dict[str, str] = {}                      # C5: 'knob=value' -> why it is unrunnable
        self.carry_patches = False                          # hls3 F18: an open patch follows a knob-only incumbent change
        self.carried_patches = 0                            # ... how many were carried
        self.merge = getattr(proposer, 'merge_patches', None) or (
            functools.partial(merge_patches, append=True) if append_merge else merge_patches)

    def describe(self) -> dict:
        out = {'name': self.name, 'proposer': self.proposer.describe(), 'kappa': round(self.kappa(), 4),
               'resolved': len(self.pairs), 'bundles': self.bundles, 'failed_bundles': len(self.failed_sets),
               'flags': {'bundle': self.bundle, 'split': self.split, 'carry': self.carry, 'confirm': self.confirm},
               'bundle_numerics': self.bundle_numerics, 'reasks': self.reasks, 'retries': self.retries}
        if self.exp:
            seeded = [h for h in self.ledger.values() if h.get('source') == 'experience']
            out['experience'] = {'seeded': len(seeded), **{st: sum(1 for h in seeded if h['status'] == st) for st in
                                                          ('confirmed', 'bundled', 'refuted', 'unresolved', 'invalid', 'open')
                                                          + (('superseded',) if self.carry_open else ())},
                                 'screens': sum(h.get('full_screens', 0) for h in seeded) + self.exp_bundle_screens,
                                 'composite': (self.exp.get('composite') or {}).get('members'), 'composite_fate': self.exp_fate}
        fix = self._switches()
        if fix:                                             # C9: what the fix did (hls: nothing added)
            out['fix'] = fix
            if self.carry_open:
                out.update(carried=self.carried, superseded=sum(1 for h in self.ledger.values() if h['status'] == 'superseded'))
            if self.backstop:
                out['backstop'] = self.backstops
            if self.start_failures:
                out['unrunnable'] = dict(self.dead)
            if self.class_kappa:
                out['kappa_by_class'] = self.class_kappas()
        return out

    def _switches(self) -> dict:
        on = {'carry_open': self.carry_open, 'view_states': self.view_states, 'prior_sigma': self.prior_sigma,
              'explore_always': self.explore_always, 'backstop': self.backstop, 'class_kappa': self.class_kappa,
              'start_failures': self.start_failures, 'cover_bundle': self.cover_bundle,
              'append_merge': self.append_merge, 'no_revert': self.no_revert}
        return on if any(on.values()) else {}

    def kappa(self, cls: str | None = None) -> float:
        """The calibration factor over every resolved prediction, or (C4) over one origin class's: each shrunk to kappa0
        with n0 pseudo-observations."""
        pairs = self.pairs if cls is None else self.class_pairs.get(cls) or []
        if not pairs:
            return self.kappa0
        pred = [max(p, 1e-3) for p, _ in pairs]
        mean_p = sum(pred) / len(pred)
        k = (sum(m for _, m in pairs) + self.kappa0 * self.n0 * mean_p) / (sum(pred) + self.n0 * mean_p)
        return min(1.5, max(0.05, k))

    def class_kappas(self) -> dict:
        return {c: {'kappa': round(self.kappa(c), 4), 'n': len(self.class_pairs.get(c) or [])}
                for c in ('knob', 'review', 'patch')}

    @staticmethod
    def _cls(h: dict) -> str:
        """C4: the origin a prediction is calibrated with: a review patch (the code review's), another patch (a
        hypothesis), or a knob change (the model's knob predictions ran at kappa 1.18 over the Q9U arms, its patches at
        0.19-0.43)."""
        if h.get('patch'):
            return 'review' if str(h.get('mechanism') or '').startswith('review:') else 'patch'
        return 'knob'

    def _pair(self, h: dict, pred: float, measured: float) -> None:
        self.pairs.append((pred, measured))
        self.class_pairs.setdefault(self._cls(h), []).append((pred, measured))

    def _kappa_for(self, h: dict, pooled: float) -> float:
        return self.kappa(self._cls(h)) if self.class_kappa else pooled

    def ledger_view(self) -> list[dict]:
        if self.view_states:
            return [self._row(h, trace=True) for h in self.ledger.values()
                    if not h.get('explore') or h['status'] != 'open' or h.get('judged') == '+']
        return [{'mechanism': h['mechanism'], 'change': h.get('change'), 'predicted_gain_pct': round(100 * h['mu'], 2),
                 'status': h['status'], 'measured_gain_pct': None if h.get('measured') is None else round(100 * h['measured'], 2),
                 'failed_checks': h.get('failed_checks'), 'candidate': h.get('candidate'), 'rebased': bool(h.get('rebased')),
                 'bundle_candidate': h.get('bundle_candidate'),
                 **_numerics(h.get('change_class'), h.get('loss_deviation'), h.get('loss_deviation_base')),
                 **({'mu_cap_pct': round(100 * h['mu_cap'], 2), 'predicted_raw_pct': round(100 * h['mu_raw'], 2)}
                    if h.get('mu_cap') is not None else {}),
                 **({'source': 'experience', 'transfer': h.get('transfer'), 'evidence_ids': h.get('evidence_ids')}
                    if h.get('source') == 'experience' else {}),
                 **({'warnings': h['warnings']} if h.get('warnings') else {})}
                for h in self.ledger.values() if not h.get('explore')
                or h['status'] != 'open']

    def _row(self, h: dict, *, trace: bool) -> dict:
        """C2/C9: a ledger row as hls shows it, with the incumbent it was proposed on (`base`) and why it was not measured
        (`why`); the trace's rows also say where a carried entry came from and a neighbour's transfer judgment."""
        return {'mechanism': h['mechanism'], 'change': h.get('change'), 'base': h['base'],
                'predicted_gain_pct': round(100 * h['mu'], 2), 'status': h['status'],
                'measured_gain_pct': None if h.get('measured') is None else round(100 * h['measured'], 2),
                'failed_checks': h.get('failed_checks'), 'candidate': h.get('candidate'), 'rebased': bool(h.get('rebased')),
                'bundle_candidate': h.get('bundle_candidate'),
                **_numerics(h.get('change_class'), h.get('loss_deviation'), h.get('loss_deviation_base')),
                **({'mu_cap_pct': round(100 * h['mu_cap'], 2), 'predicted_raw_pct': round(100 * h['mu_raw'], 2)}
                   if h.get('mu_cap') is not None else {}),
                **({'source': 'experience', 'transfer': h.get('transfer'), 'evidence_ids': h.get('evidence_ids')}
                   if h.get('source') == 'experience' else {}),
                **({'warnings': h['warnings']} if h.get('warnings') else {}),
                **({'why': h['why']} if h.get('why') and (h['status'] == 'superseded' or str(h['why']).startswith(
                    'unrunnable') or h['status'] != 'open' and h.get('source') != 'experience') else {}),
                **({'carried_from': h['carried_from']} if trace and h.get('carried_from') else {}),
                **({'judged': h['judged']} if trace and h.get('judged') else {})}

    def model_view(self) -> list[dict]:
        """The ledger as the model sees it: measurements without its own predictions of them. Shown its misses, a model
        discounts its next predictions and kappa shrinks them again (H2b arm A: bf16 main grads predicted 4.9 %, then
        0.8 % and 0.4 % after a miss, about 0.2 % after kappa; +3 % measured in arm B), so kappa is meant to be the only
        calibration (the model still learns about the job from the measurements); an experience entry's prior is data,
        not the model's prediction, and stays. The change class and loss deviation stay out (a report-only diagnostic)."""
        if self.view_states:
            return self._states_view(self.refreshed_for)
        return [_bare(h if h.get('source') == 'experience' else unpaired(h)) for h in self.ledger_view()]

    def _states_view(self, inc) -> list[dict]:
        """C2 (hls2): the model's ledger consistent with what the strategy can still test on `inc`. An open entry of the
        incumbent is "queued"; one of an earlier incumbent, never tested, is "superseded" (Q9U-code-env-opus: Liger was
        proposed on c0, orphaned when micro-batch 2 moved the incumbent, still shown "open", and the model wrote at
        20:10 that Liger was "already queued or open"); a value that failed to start twice is "unrunnable". One row per
        move (knobs and patch): the measured one, else the newest (experience re-seeding showed one seed five times in
        the 40 rows the model gets, and cut the first incumbent's rows); a superseded patch whose idea is queued again
        under its name (the review's patch, rebased on each incumbent) is left out."""
        rank = {'superseded': 0, 'queued': 1, 'deferred': 1}
        rows: dict[str, tuple] = {}
        for h in self.ledger.values():
            if h.get('explore') and h['status'] == 'open':
                continue                                    # a neighbour not yet run (the engine has it), as in hls
            r = self._row(h, trace=False)
            st = h['status']
            if st == 'open':
                st = 'queued' if h['base'] == inc else 'superseded'
                if st == 'superseded':
                    r['why'] = h.get('why') or 'not tested before the incumbent changed'
            elif st == 'invalid' and str(h.get('why') or '').startswith('unrunnable'):
                st = 'unrunnable'
            st = self._view_status(h, st)
            r['status'] = st
            r = _bare(r if h.get('source') == 'experience' else unpaired(r))
            key = json.dumps([_mkey(h.get('change')), _pkey(h.get('patch'))])
            old = rows.pop(key, None)
            if old is not None and rank.get(old[0]['status'], 2) > rank.get(st, 2):
                r, h = old                                  # a measurement outranks a later queued copy
            rows[key] = (r, h)
        out = list(rows.values())
        live = {(_mkey(h.get('change')), h['mechanism']) for r, h in out if r['status'] != 'superseded'}
        return [r for r, h in out if not (r['status'] == 'superseded' and h.get('patch')
                                          and (_mkey(h.get('change')), h['mechanism']) in live)]

    def _view_status(self, h: dict, st: str) -> str:
        """A row's state as the model sees it (hls3 adds 'deferred': predicted not to fit, or waiting for a separate run)."""
        return st

    def _warn(self, cfg: dict, change: dict) -> list:
        check = self.exp.get('check')                   # the transfer judgment may extend it
        return list(check({**cfg, **change}) or []) if check else []

    def _warn_p(self, engine, cfg: dict, change: dict, p: float, warn: list) -> float:
        """The feasibility of a move the experience plan warns about (a constraint, a memory estimate): 0.2 at most
        (hls3 overrides it, FABLE_HLS3_FIXES_R1 F7)."""
        return min(p, 0.2)

    def _seed(self, engine, row: dict, cfg: dict) -> dict | None:
        """A ledger entry from an experience row (experience decision 2.1): covered -> the source gain, partial -> half
        of it, with the transfer judgment's sign when it gave one ('0' / 'unknown': zero); sigma from the source size and
        the engine's noise; a warning (constraint, memory) or an 'infeasible' judgment makes it probe-first. Never scaled
        by kappa. With `prior_sigma` (C3a, hls2) a move without data that the judgment calls '+' has sigma at least
        prior_sigma (2 sigma_pair alone priced it as certain to do nothing: no-reshard waited for screen 8 behind +0.3 %
        patches), and one it calls '-' is not seeded (None)."""
        g, sp = float(row['gain']), engine.sigma_pair()
        a, sign = abs(g), row.get('sign')
        if self.prior_sigma is not None and a == 0 and sign == '-':
            return None
        if row['level'] == 'covered' and sign is None:
            mu, sigma, p = g, max(0.5 * a, 2 * sp), 0.9
        else:
            s = math.copysign(1.0, g) if sign is None else {'+': 1.0, '-': -1.0}.get(sign, 0.0)
            mu, sigma = 0.5 * s * a, max(a, 2 * sp)
            if self.prior_sigma is not None and a == 0 and sign == '+':
                sigma = max(sigma, float(self.prior_sigma))
            p = min(0.9, max(0.1, float(row.get('p_feasible', 0.7))))
        warn = self._warn(cfg, row['change']) if self.exp.get('check') else list(row.get('warnings') or [])
        if sign == 'infeasible':
            p = min(p, 0.2)
        elif warn:
            p = self._warn_p(engine, cfg, row['change'], p, warn)
        return {'mechanism': f'experience:{row["mechanism"]}', 'change': dict(row['change']), 'patch': None, 'mu': mu,
                'sigma': sigma, 'p_feasible': p, 'source': 'experience', 'transfer': 'llm' if sign else row['level'],
                'evidence_ids': list(row.get('evidence_ids') or []), 'source_gain': g, 'warnings': warn or None,
                **({'why': row['why']} if row.get('why') else {})}

    def _refresh(self, engine, inc):
        if self.refreshed_for is None and callable(self.exp.get('transfer')):
            self.exp['transfer'](engine.s)              # the LLM transfer judgment: once, charged to the session
        old = self.refreshed_for
        if self.carry and old is not None and old != inc:
            # unresolved positives of the old base move to the new one (not measured again: they join its next
            # bundle); a patch is rebased: its edit of the old base's text applied to the new incumbent's text
            hooks = engine.s.hooks
            c, o = engine.s.candidates[inc], engine.s.candidates[old]
            files = hooks.candidate_files(c.overlay) if hooks is not None else dict(c.overlay or {})
            old_files = hooks.candidate_files(o.overlay) if hooks is not None else dict(o.overlay or {})
            for h in list(self.ledger.values()):
                if h['base'] != old or h['status'] != 'unresolved' or (h.get('measured') or 0) <= 0:
                    continue
                same_cfg = all(c.config.get(k) == v for k, v in (h.get('change') or {}).items())
                patch = None
                if h.get('patch'):
                    cur = {f: files.get(f, '') for f in h['patch']}
                    patch = self.merge(old_files, [cur, h['patch']])
                    if patch is None:
                        continue                        # its edit conflicts with the new incumbent's
                    if patch == cur:
                        patch = None                    # the edit is already in the new incumbent
                if same_cfg and patch is None:
                    continue
                key = f'{inc}|' + (h['key'].split('|', 1)[1] if '|' in str(h.get('key')) else h['mechanism'])
                self.ledger.setdefault(key, dict(h, base=inc, key=key, status='unresolved', rebased=True, patch=patch,
                                                 p_feasible=min(h['p_feasible'], 0.7)))
            # a retry the incumbent moved past before it ran goes along (F1c-S1-gpt56terra: fused AdamW moved it first,
            # the retry of bf16 without compile was lost, and bf16 on the new incumbent reverted fused AdamW instead):
            # its change - the knob and the culprit reverted - set on the new incumbent
            for h in list(self.ledger.values()):
                if h['base'] != old or h['status'] != 'open' or not h.get('retry_of'):
                    continue
                if _applied(c.config, h['change'], native=hooks is None):
                    continue                            # the new incumbent already has it
                key = f'{inc}|' + h['key'].split('|', 1)[1]
                if key not in self.ledger:
                    self.ledger[key] = dict(h, base=inc, key=key, rebased=True)
                    self.retries.setdefault(inc, []).append(h['mechanism'])
        if self.carry_open and old is not None and old != inc:
            self._carry_open(engine, old, inc)
        cfg = engine.s.candidates[inc].config
        seeded = {}
        # on the first base the evidence is for this very move; on a later incumbent it is one step removed: still-open
        # moves are seeded again as partial (half the size, sigma the full size; toy T8 / T8b), or not ('reseed': 'none')
        ref = engine.s.candidates.get('c0')             # a first base the design (doe+hls) moved counts as later
        later = self.refreshed_for is not None or (ref is not None and ref.config != cfg)
        mode = self.exp.get('reseed', 'partial')
        rows = self.exp.get('seeds') or () if not later or mode != 'none' else ()
        for row in rows:
            mech = f'experience:{row["mechanism"]}'
            if all(cfg.get(n) == v for n, v in row['change'].items()) or any(
                    h['mechanism'] == mech and h['status'] not in UNTESTED for h in self.ledger.values()) or any(
                    cfg.get(k) != v for k, v in (row.get('valid_on') or {}).items()):
                continue                                # in the incumbent, measured (bundled) on an earlier base, or
                                                        # its evidence is for another starting point
            key = f'{inc}|{mech}'
            if key not in self.ledger:
                if self.carry_open and self._queued(inc, row['change']):
                    continue                            # a carried entry of the same move is queued already (C1)
                if self.start_failures and self._dead_in(engine, cfg, row['change']):
                    continue                            # it failed to start twice (C5)
                carried = later and mode == 'partial' and row['level'] == 'covered'
                src = dict(row, level='partial') if carried else row
                entry = self._seed(engine, src, cfg)
                if entry is None:                       # judged '-' without data (C3a)
                    continue
                self.ledger[key] = dict(entry, base=inc, status='open', key=key,
                                        **({'transfer': 'carried'} if carried else {}))
            seeded[row.get('knob')] = key
        if self.start_failures:
            self._retire_dead(engine, inc)              # (C5) before the model sees them as queued
        self._ask(engine, inc)
        comp = self.exp.get('composite')
        if self.cover_bundle and comp and comp.get('gain') is None:
            comp = None                                 # C6: a judged composite (no data) has mu 0 and never ran
        if comp and self.refreshed_for is None and 'bundle|exp' not in self.members:
            keys = [seeded[k] for k in comp['members'] if k in seeded]
            if len(keys) >= 2:                          # the start point, through the bundle path (decision 2.2)
                big = max(abs(self.ledger[k]['mu']) for k in keys)
                warn = self._warn(cfg, comp['change'])
                q = self._bundle_proposal(engine, inc, keys, p_feasible=self._warn_p(engine, cfg, comp['change'], 0.7, warn)
                                          if warn else 0.7,
                                          plan=('probe', 'full'), bid='bundle|exp', origin='experience-bundle',
                                          sigma=math.sqrt(len(keys)) * max(engine.sigma_pair(), 0.5 * big))
                if q is not None:
                    self.pending[q.hypothesis] = q
        if self.carry_open and old is not None and old != inc:
            self._carry_pending(engine, old, inc)
        if self.explore_always and self.explore:
            self._neighbourhood(engine, inc)            # C3b: priced neighbours on every incumbent, not only when dry
        if self.cover_bundle:
            self._cover(engine, inc)
        self.refreshed_for = inc

    # ------------------------------------------------------------ the search fix (hls2; FABLE_DESIGN_SEARCH_FIX)
    def _carry_open(self, engine, old, inc) -> None:
        """C1: the never-tested entries of the old incumbent when another one takes its place. A knob change moves to
        the new incumbent (open, its prediction kept, feasibility at most 0.9, `carried_from`); a change the new
        incumbent already holds, one that would undo a change it carries (the retry's job), a multi-knob change one knob
        of which changed since (its prediction was for that pair), a move queued there already and a patch (written
        against the old code) stay behind 'superseded' (the model sees them and may propose them again); experience
        seeds are seeded again on each incumbent. Before (M1): only measured positives moved, so the Liger knob,
        proposed on c0 and c1 of Q9U-code-env-opus, never ran (20 more model ideas were orphaned with it)."""
        native = engine.s.hooks is None
        c, o = engine.s.candidates[inc], engine.s.candidates[old]
        for h in list(self.ledger.values()):
            if h['base'] != old or h['status'] != 'open' or h.get('explore'):
                continue                                    # neighbours and retries belong to their incumbent
            ch = h.get('change') or {}
            if h.get('source') == 'experience':
                why = f'an experience move: seeded again on {inc} unless it is there or was measured'
            elif h.get('patch') and not (self.carry_patches and (c.overlay or {}) == (o.overlay or {})):
                why = f'a patch of {old}, not carried to {inc} (the code it was written against changed)'
            elif ch and _applied(c.config, ch, native=native):     # (a pure patch carried by hls3 F18 has no knobs)
                why = f'already in the incumbent {inc}'
            elif ch and self._undoes(engine, inc, ch):
                why = f'it would undo a change the incumbent {inc} carries'
            elif len(_names(ch, native)) > 1 and any(_value(c.config, k, native=native) != _value(o.config, k, native=native)
                                                     for k in _names(ch, native)):
                why = f'a knob of this combination changed on {inc}'
                rem = self._remainder(engine, h, old, inc)  # (hls3 F6: the part the incumbent did not adopt)
                if rem:
                    why += f'; the rest of it is carried as {rem}'
            elif ch and self._queued(inc, ch):
                why = f'the same move is queued on {inc}'
            else:
                self.carried_patches += bool(h.get('patch'))   # (hls3 F18: the code it was written against is unchanged)
                key = f'{inc}|{h["mechanism"]}'
                if key in self.ledger:
                    key = next(f'{key}#{k}' for k in range(2, 10 ** 6) if f'{key}#{k}' not in self.ledger)
                warn = self._warn(c.config, ch)
                p = min(h['p_feasible'], 0.9)
                self.ledger[key] = dict(h, base=inc, key=key, status='open', rebased=True, carried_from=old,
                                        p_feasible=self._warn_p(engine, c.config, ch, p, warn) if warn else p,
                                        warnings=warn or None)
                self.carried += 1
                why = f'carried to {inc}'
            h.update(status='superseded', why=why)

    def _remainder(self, engine, h: dict, old, inc) -> str | None:
        """The key of an entry carrying the part of a multi-knob entry the new incumbent did not adopt (hls3 F6), else
        None (hls / hls2: nothing is carried)."""
        return None

    def _carry_pending(self, engine, old, inc) -> None:
        """C1: a pending bundle of the old incumbent (a failed bundle's half, the experience start composite) is rebuilt
        on the new one from its members' entries there when at least two are still untested or unresolved, else
        dropped (before: it stayed pending on the old base, never proposed)."""
        for bid, q in list(self.pending.items()):
            if q.base != old:
                continue
            del self.pending[bid]
            keys = []
            for k in self.members.get(bid, []):
                e = self.ledger.get(f'{inc}|' + k.split('|', 1)[1])
                if e is not None and e['base'] == inc and e['status'] in ('open', 'unresolved'):
                    keys.append(e['key'])
            if len(keys) < 2:
                continue
            big = max(abs(self.ledger[k]['mu']) for k in keys)
            exp = bid == 'bundle|exp'
            nq = self._bundle_proposal(engine, inc, keys, p_feasible=q.p_feasible, plan=tuple(q.fidelity),
                                       bid=bid if exp else None, origin=q.origin,
                                       sigma=math.sqrt(len(keys)) * max(engine.sigma_pair(), 0.5 * big) if exp else None)
            if nq is not None:
                self.pending[nq.hypothesis] = nq

    def _queued(self, inc, change) -> bool:
        m = _mkey(change)
        return any(e['base'] == inc and e['status'] in ('open', 'unresolved') and not e.get('patch')
                   and _mkey(e.get('change')) == m for e in self.ledger.values())

    @staticmethod
    def _undoes(engine, inc, change) -> bool:
        """`change` sets a knob that the incumbent's lineage moved from the first incumbent's value to another value."""
        native = engine.s.hooks is None
        cfg = engine.s.candidates[inc].config
        first = (engine.s.candidates.get('c0') or engine.s.candidates[inc]).config
        return any(_value(cfg, k, native=native) != _value(first, k, native=native)
                   for k, _ in _knobs_set(change, cfg, native=native))

    @staticmethod
    def _lineage(engine, inc) -> dict:
        """knob -> the values it had from the first incumbent to `inc` (C8: going back to one of them undoes a change)."""
        cands, out, seen, cid = engine.s.candidates, {}, set(), inc
        chain = []
        while cid in cands and cid not in seen:
            seen.add(cid)
            chain.append(cands[cid])
            cid = getattr(cands[cid], 'parent', None)
        if 'c0' in cands and 'c0' not in seen:
            chain.append(cands['c0'])
        for cand in chain:
            for k, v in cand.config.items():
                vals = out.setdefault(k, [])
                if v not in vals:
                    vals.append(v)
        return out

    def _judgments(self) -> dict:
        """The transfer judgment's moves: change key -> (sign, p_feasible); empty without a judgment."""
        j = self.exp.get('judgment') or {}
        return {_mkey(a['change']): (a.get('sign'), a.get('p_feasible')) for a in j.get('assessment') or []
                if isinstance(a, dict) and isinstance(a.get('change'), dict)}

    def _dead_in(self, engine, cfg, change) -> str | None:
        """C5: 'knob=value: why' when `change` sets a value that failed to start twice in this search, else None."""
        f = getattr(engine.s, 'value_unrunnable', None)
        if f is None:
            return None
        for k, v in _knobs_set(change, cfg, native=engine.s.hooks is None):
            why = f(k, v)
            if why:
                return f'{k}={v}: {why}'
        return None

    def _retire_dead(self, engine, inc) -> None:
        """C5: the open entries of the incumbent that set an unrunnable value are not proposed: invalid, why unrunnable."""
        cfg = engine.s.candidates[inc].config
        for h in self.ledger.values():
            if h['status'] == 'open' and h['base'] == inc:
                why = self._dead_in(engine, cfg, h.get('change') or {})
                if why:
                    h.update(status='invalid', why=f'unrunnable: {why}')
                    self.dead.setdefault(why.split(':')[0], why)

    def _knob_move(self, p: Proposal, inc, judged: dict, thr: float) -> bool:
        """C3c: a proposal the backstop keeps: an open single-knob entry of the incumbent (no patch) that runs with
        probability >= 0.5, that the transfer judgment does not call '-', and that it calls '+' or whose own prediction
        (the model's, before kappa, or the experience's) is at least 2 sigma_pair - the moves the design's coverage
        criterion R1 requires to be measured. A move the model itself rates below the noise is not protected (ToyTrain:
        protecting every proposed knob put the model's predicted losses ahead of its ideas, frac -0.07 at 16 runs)."""
        if p.origin == 'hls-cover':                         # C6's composite of exactly such moves
            return True
        h = self.ledger.get(p.hypothesis or '')
        if h is None or h['status'] != 'open' or h['base'] != inc or p.patch or len(p.change or {}) != 1:
            return False
        sign = judged.get(_mkey(p.change), (None, None))[0]
        return sign != '-' and p.p_feasible >= 0.5 and (sign == '+' or h['mu'] >= thr)

    def _narrow(self, engine, inc, props: list) -> list:
        """C3c (the user's "it must at least try it", 2026-10-29): when the best proposal by value/cost is an idea below
        the pair noise (one run cannot tell it from nothing) while a single-knob move of the incumbent judged '+' or
        predicted at least 2 sigma_pair is still untested, the ideas below the noise wait until those moves ran (ideas
        at or above 2 sigma_pair stay). It defers, never ends the search: when the engine's stop rule would stop on the
        narrowed set, the full set goes to decide() as it is (review NM25: a deferred patch at value/cost 1.65, the kept
        knob at 0.48, and the search stopped after 4 of 10 runs). Counted in describe()['backstop']."""
        value, cost = getattr(engine, 'value', None), getattr(engine, 'expected_cost', None)
        if not props or value is None or cost is None:
            return props
        best = max(props, key=lambda p: value(p) / cost(p))     # the first of equals, as decide() takes it
        thr = 2 * engine.sigma_pair()
        if best.mu >= thr:
            return props
        judged = self._judgments()
        keep = [p for p in props if self._knob_move(p, inc, judged, thr)]
        if not keep or any(p is best for p in keep):
            return props
        narrowed = [p for p in props if p.mu >= thr or any(p is q for q in keep)]
        stops = getattr(engine, 'stops_at', None)
        if stops is not None and stops(max(value(p) / cost(p) for p in narrowed)):
            return props
        self.backstops += 1
        return narrowed

    def _cover(self, engine, inc) -> None:
        """C6 (off in hls2): when the untested cheap single-knob moves of the incumbent (judged '+', from experience, or
        rated at or above 2 sigma_pair) outnumber the full runs left, at most four of them (strongest prior first, one
        value per knob) are tested as one composite, probed first - random's many-knobs-at-once advantage, only when
        the budget cannot test them one by one; a failed one is split (four members) or leaves them to run alone."""
        left = getattr(getattr(engine.s, 'budget', None), 'screens', None)
        if left is None or not hasattr(engine, 'full_screens'):
            return
        left -= engine.full_screens()
        sp, kap, judged = engine.sigma_pair(), self.kappa(), self._judgments()
        moves = [h for h in self.ledger.values() if h['base'] == inc and h['status'] == 'open' and not h.get('patch')
                 and len(h.get('change') or {}) == 1 and (judged.get(_mkey(h['change']), (None, None))[0] == '+'
                                                          or h.get('source') == 'experience'
                                                          or self._kappa_for(h, kap) * h['mu'] >= 2 * sp)]
        if len(moves) <= left:
            return
        keys, knobs = [], set()
        for h in sorted(moves, key=lambda h: -(h['mu'] + h['sigma'])):
            k = next(iter(h['change']))
            if k not in knobs:
                keys.append(h['key'])
                knobs.add(k)
            if len(keys) == 4:
                break
        if len(keys) < 2:
            return
        ps = [self.ledger[k]['p_feasible'] for k in keys]
        q = self._bundle_proposal(engine, inc, keys, p_feasible=math.prod(ps) ** (1 / len(ps)), plan=('probe', 'full'),
                                  bid=f'bundle|cover|{inc}', origin='hls-cover',
                                  sigma=math.sqrt(len(keys)) * max(sp, max(self.ledger[k]['sigma'] for k in keys)))
        if q is not None:
            self.pending[q.hypothesis] = q

    def _ask(self, engine, inc) -> int:
        """One model call on `inc` with the ledger as the model sees it; its hypotheses join the ledger as open entries
        (an experience twin keeps its data-ordered entry and gets the prediction for calibration). Returns how many of
        them change something no entry of this base changes: the model had something new, not a repeat under another
        name (the engine runs no repeat; it is marked a duplicate at no cost). hls2: a hypothesis that makes the move of
        an open carried entry gives it the prediction made for this incumbent (C1), one that makes a neighbour's move
        replaces the neighbour's prior (C3b)."""
        cfg = engine.s.candidates[inc].config
        hyps = self.proposer.hypotheses(engine, inc, ledger=self._states_view(inc) if self.view_states else self.model_view())
        shares = self._class_shares(engine, inc) if self.phase_cap else None
        new = 0
        for h in hyps:
            key = f'{inc}|{h["mechanism"]}'
            mine = [e for e in self.ledger.values() if e['base'] == inc]
            if (self.carry_open or self.explore_always) and not h.get('patch'):
                old = next((e for e in mine if e['status'] == 'open' and not e.get('patch') and e.get('change') == h['change']
                            and (e.get('carried_from') or e.get('explore') and not e.get('retry_of'))), None)
                if old is not None and old.get('explore'):
                    del self.ledger[old['key']]
                    mine = [e for e in mine if e is not old]
                elif old is not None:
                    cap = self._cap(engine, h, shares) if shares else None
                    warn = self._warn(cfg, h['change'])
                    old.update({k: h[k] for k in ('mu', 'sigma', 'p_feasible', 'change_class', 'rationale', 'mem_dir',
                                                  'mem_gib', 'variance', 'site') if k in h}, rerated=True)
                    if warn:
                        old.update(p_feasible=self._warn_p(engine, cfg, h['change'], h['p_feasible'], warn), warnings=warn)
                    if cap is not None:
                        old.update(mu=cap, mu_raw=h['mu'], mu_cap=cap)
                    continue
            if self.carry_patches and h.get('patch'):  # hls3 F18: a carried patch the model proposes again on this
                old = next((e for e in mine if e['status'] == 'open' and e.get('carried_from')   # incumbent takes the
                            and e.get('patch') == h['patch'] and e.get('change') == h['change']), None)   # prediction
                if old is not None:                    # made for it here (as a carried knob move does, C1)
                    old.update({k: h[k] for k in ('mu', 'sigma', 'p_feasible', 'change_class', 'rationale', 'site',
                                                  'evidence_note', 'active_from_update', 'warmup_updates') if k in h},
                               rerated=True)
                    continue
            same = any(e.get('change') == h['change'] and e.get('patch') == h.get('patch') for e in mine)
            if key in self.ledger and not same:         # another change or patch under a name used on this base
                key = next(f'{key}#{k}' for k in range(2, 10 ** 6) if f'{key}#{k}' not in self.ledger)
            twin = None if h.get('patch') else next((e for e in mine if e.get('source') == 'experience'
                                                     and e['change'] == h['change']), None)
            cap = self._cap(engine, h, shares) if shares else None
            if twin is not None:                        # the data orders it; the LLM's prediction is still calibrated
                twin.setdefault('llm_mu', cap if cap is not None else h['mu'])
                continue
            if key not in self.ledger:
                new += not same
                warn = self._warn(cfg, h.get('change') or {})
                self.ledger[key] = dict(h, base=inc, status='open', key=key,
                                        **({'p_feasible': self._warn_p(engine, cfg, h.get('change') or {}, h['p_feasible'],
                                                                       warn), 'warnings': warn} if warn else {}),
                                        **({'mu': cap, 'mu_raw': h['mu'], 'mu_cap': cap} if cap is not None else {}))
        return new

    def _reask(self, engine, inc) -> bool:
        """Nothing is left to propose on the incumbent (its ledger measured, the single knobs tried, no bundle open)
        while budget remains: the model is asked again, with the ledger of everything measured since its last call, at
        most `reask` times per incumbent and not after an answer that added nothing new (it is out of ideas for this
        incumbent; a failed call is no answer and is retried within the cap). Before (E-C-IS2b): one failed call on the
        patch incumbent, two slower knob values, and the search ended `strategy_done` with 0.4 of its 2.85 GPU-hours
        spent. True when a call was made (propose looks at the ledger again)."""
        done = self.reasks.setdefault(inc, [])
        if len(done) >= self.reask or (done and done[-1] == 'nothing'):
            return False
        (getattr(engine, 'log', None) or (lambda m: None))(
            f'[search] hls: nothing left to test on {inc}: asking the model again ({len(done) + 1} of {self.reask})')
        failed = getattr(self.proposer, 'failed', 0)
        new = self._ask(engine, inc)
        done.append('new' if new else 'failed' if getattr(self.proposer, 'failed', 0) > failed else 'nothing')
        return True

    @staticmethod
    def _class_shares(engine, inc) -> dict | None:
        """The incumbent's update share per phase class (its receipt's phases, through the project hooks)."""
        f = getattr(engine.s.hooks, 'class_shares', None)
        return f(engine.s.candidates[inc].receipt) if f else None

    @staticmethod
    def _cap(engine, h: dict, shares: dict) -> float | None:
        """The most a knob change can gain when it only shortens its mechanisms' phase classes: -log(1 - their share
        of the update), only below the prediction. No cap for a patch, a knob with no mapped mechanism or one acting on
        every phase (layout, precision, recompute: all), a class the receipt did not measure (unknown is not zero), or
        classes that cover the whole update."""
        kp = getattr(engine.s.hooks, 'knob_phases', None) or {}
        sets = [kp.get(n) for n in (h.get('change') or {})]
        if h.get('patch') or not sets or any(s is None or s == 'all' for s in sets):
            return None
        classes = set().union(*sets)
        if any(c not in shares for c in classes):
            return None
        share = sum(shares[c] for c in classes)
        cap = -math.log(1 - share) if share < 1 else None
        return cap if cap is not None and h['mu'] > cap else None

    def propose(self, engine, k):
        inc = engine.incumbent()
        if self.start_failures and hasattr(engine.s, 'start_failure_limit') and not engine.s.start_failure_limit:
            engine.s.start_failure_limit = self.start_failures    # C5: the session counts start failures for this run
        if self.refreshed_for != inc:
            self._refresh(engine, inc)
        props = self._proposals(engine, inc)
        while not props and self._reask(engine, inc):
            props = self._proposals(engine, inc)
        return props

    def _proposals(self, engine, inc) -> list[Proposal]:
        """The incumbent's open ledger entries (kappa-scaled; experience as data), else its single-knob neighbourhood;
        plus the loose bundle and the pending halves. hls2: a neighbour is a prior, not scaled (C3b); each prediction
        is scaled by its origin's kappa (C4); an entry setting an unrunnable value is retired (C5); the backstop (C3c)."""
        if self.start_failures:
            self._retire_dead(engine, inc)
        kap = self.kappa()
        props = []
        for key, h in self.ledger.items():
            if h['status'] != 'open' or h['base'] != inc:
                continue
            plan = ('full',) if h.get('known_ok') else ('probe', 'full') if h['p_feasible'] < self.probe_below else ('full',)
            exp = h.get('source') == 'experience'       # a measurement with a known transfer error, not a prediction
            prior = self.explore_always and h.get('explore')
            scale = 1.0 if exp or prior else self._kappa_for(h, kap)
            props.append(Proposal(change=h['change'], patch=h['patch'], base=inc, mu=scale * h['mu'],
                                  sigma=max(h['sigma'] * scale, 0.005), p_feasible=h['p_feasible'], mechanism=h['mechanism'],
                                  hypothesis=key, origin='hls-experience' if exp else 'hls-explore' if prior else 'hls',
                                  fidelity=plan, change_class=h.get('change_class')))
        if not props and self.explore:
            props = self._neighbourhood(engine, inc)
        if self.bundle:
            b = self._loose_bundle(engine, inc)
            if b is not None:
                props.append(b)
            props += [p for p in self.pending.values() if p.base == inc]
        props = self._adjust(engine, inc, props)
        if self.backstop:
            props = self._narrow(engine, inc, props)
        return props

    def _adjust(self, engine, inc, props: list) -> list:
        """The proposals as a subclass screens them before the backstop (hls3: memory, antagonism, pruning, stalls)."""
        return props

    def receive(self, engine, items):
        """Positives an earlier strategy measured but could not resolve (a screening design's small main effects): within-
        noise entries on the current incumbent, bundled, confirmed or split like the ledger's own."""
        inc = engine.incumbent()
        cfg = engine.s.candidates[inc].config
        for it in items:
            if all(cfg.get(n) == v for n, v in it['change'].items()):
                continue                                    # already in the incumbent
            key = f'{inc}|{it["mechanism"]}'
            self.ledger.setdefault(key, {'mechanism': it['mechanism'], 'change': dict(it['change']), 'patch': None,
                                         'mu': it['measured'], 'sigma': 2 * engine.sigma_run(), 'p_feasible': 0.8,
                                         'base': inc, 'status': 'unresolved', 'key': key, 'measured': it['measured'],
                                         'rebased': True})

    def _loose_bundle(self, engine, inc):
        """The within-noise positives of this base, strongest first, as one composite; a set that failed before is
        not repeated (its weakest member is left out until no failed set is contained)."""
        loose = sorted((h for h in self.ledger.values() if h['status'] == 'unresolved' and h['base'] == inc
                        and (h.get('measured') or 0) > 0), key=lambda h: -h['measured'])
        keys = [h['key'] for h in loose]
        while len(keys) >= 2:
            bad = next((fs for fs in self.failed_sets if fs <= set(keys)), None)
            if bad is None:
                break
            keys.remove(min(bad, key=lambda k: self.ledger[k]['measured']))
        if len(keys) < 2 or frozenset(keys) in self.evaluated:
            return None
        return self._bundle_proposal(engine, inc, keys)

    def _bundle_proposal(self, engine, inc, keys, *, p_feasible=None, plan=None, bid=None, origin='hls-bundle', sigma=None):
        c = engine.s.candidates[inc]
        files = engine.s.hooks.candidate_files(c.overlay) if engine.s.hooks is not None else dict(c.overlay or {})
        kept, patch, change = [], None, {}
        for h in (self.ledger[k] for k in keys):           # strongest first; a member whose edit conflicts stays out
            ch = h.get('change') or {}
            if any(n in change and change[n] != v for n, v in ch.items()):
                continue                                    # another value of a knob a kept member sets
            if h.get('patch'):
                m = self.merge(files, [patch, h['patch']] if patch else [h['patch']])
                if m is None:
                    if self.append_merge:                   # C7: said, not silent
                        h['why'] = 'patch conflict: its edit conflicts with a stronger member of a bundle'
                    continue
                patch = m
            change.update(ch)
            kept.append(h)
        keys = [h['key'] for h in kept]
        if len(kept) < 2 or frozenset(keys) in self.evaluated:
            return None
        bid = self.bundle_ids.setdefault(frozenset(keys), bid or f'bundle|{len(self.bundle_ids)}')
        self.members[bid] = keys
        rebased = any(h.get('rebased') for h in kept)
        return Proposal(change=change, patch=patch, base=inc,
                        mu=sum(_gain(h) for h in kept), sigma=sigma or math.sqrt(len(kept)) * engine.sigma_pair(),
                        p_feasible=p_feasible if p_feasible is not None else (0.7 if rebased else 0.8),
                        mechanism='bundle:' + '+'.join(h['mechanism'] for h in kept)[:200], hypothesis=bid,
                        origin=origin, fidelity=plan or (('probe', 'full') if rebased else ('full',)),
                        confirm=self.confirm)

    def _neighbourhood(self, engine, inc) -> list[Proposal]:
        """Every single-knob change of the incumbent nobody predicted, with a zero-gain prior: the value/cost rule
        decides whether the remaining budget is worth spending on them (a minimum exploration the model cannot skip).
        hls2: built on every incumbent, not only once the model is out of ideas (C3b: Q9U-code-env-opus never got
        there), and priced by the transfer judgment: a move judged '+' and never tried in this search has sigma
        prior_sigma (at least 2 sigma_pair), any other 2 sigma_pair, with the judgment's feasibility; a move an entry
        of this incumbent makes already is left to it. No move back to a value the incumbent's lineage left, none the
        judgment calls '-' (C8: the dsv41 arms spent five screens on reverts, duplicates and a known timeout), none that
        failed to start twice (C5)."""
        cfg = engine.s.candidates[inc].config
        known = self.exp.get('known_ok') or ()          # measured within noise in scope: it runs (decision 2.1, null)
        judged = self._judgments() if self.explore_always or self.no_revert else {}
        lineage = self._lineage(engine, inc) if self.no_revert else {}
        mine = tried = ()
        if self.explore_always:
            mine = {_mkey(h.get('change')) for h in self.ledger.values()
                    if h['base'] == inc and not h.get('explore') and not h.get('patch')}
            tried = {_mkey(h.get('change')) for h in self.ledger.values() if h['status'] in MEASURED and not h.get('patch')}
        out = []
        for name, values in choices(engine.s.task).items():
            for v in values:
                if v == cfg.get(name):
                    continue
                key = f'{inc}|explore:{name}={v}'
                if key not in self.ledger:
                    move = _mkey({name: v})
                    sign, pj = judged.get(move, (None, None))
                    if self.no_revert and (sign == '-' or v in lineage.get(name, ())):
                        continue
                    if move in mine or self.start_failures and self._dead_in(engine, cfg, {name: v}):
                        continue
                    warn = self._warn(cfg, {name: v})
                    ok = not warn and {name: v} in known
                    sigma, p = 2 * engine.sigma_run(), self._warn_p(engine, cfg, {name: v}, 0.6, warn) if warn else \
                        0.9 if ok else 0.6
                    if self.explore_always:
                        sigma = 2 * engine.sigma_pair()
                        if sign == '+' and self.prior_sigma is not None and move not in tried:
                            sigma = max(sigma, float(self.prior_sigma))
                        if not warn and not ok and isinstance(pj, (int, float)):
                            p = min(0.9, max(0.1, float(pj)))
                    self.ledger[key] = {'mechanism': f'explore:{name}={v}', 'change': {name: v}, 'patch': None, 'mu': 0.0,
                                        'sigma': sigma, 'p_feasible': p,
                                        'base': inc, 'status': 'open', 'key': key, 'explore': True, 'known_ok': ok,
                                        **({'warnings': warn} if warn else {}),
                                        **({'judged': sign} if self.explore_always and sign else {})}
                h = self.ledger[key]
                if h['status'] == 'open':
                    out.append(Proposal(change=h['change'], base=inc, mu=0.0, sigma=h['sigma'], p_feasible=h['p_feasible'],
                                        mechanism=h['mechanism'], hypothesis=key, origin='hls-explore',
                                        fidelity=('full',) if h.get('known_ok') else ('probe', 'full')))
        return out

    def observe(self, engine, p, out: Outcome):
        if out.status == 'infra':                       # the hosts failed, not the hypothesis: it stays as it was
            return
        if p.hypothesis and p.hypothesis.startswith('bundle|'):
            self._observe_bundle(engine, p, out)
            return
        h = self.ledger.get(p.hypothesis or '')
        if h is None:
            return
        h['failed_checks'] = out.failed_checks or None
        h['candidate'] = out.cid
        if out.loss_deviation is not None:
            h['loss_deviation'] = out.loss_deviation
        if out.loss_deviation_base is not None:
            h['loss_deviation_base'] = out.loss_deviation_base
        if out.fidelity == 'full':
            h['full_screens'] = h.get('full_screens', 0) + 1
        if out.status in ('infeasible', 'invalid', 'rejected', 'duplicate'):
            h['status'] = 'invalid'
            first = str((out.reasons or [''])[0])
            if self.start_failures and out.status == 'rejected' and first.startswith(START_FAILED):
                what = first[len(START_FAILED):].strip(' :')   # C5: refused at no cost, 'knob=value: why'
                h['why'] = f'unrunnable: {what}'
                self.dead.setdefault(what.split(':')[0], what)
            if out.status in ('infeasible', 'invalid'):     # it ran (a proposal the session refused did not)
                self._retry(engine, h, out)
            return
        raw = h['mu']
        if out.status == 'aborted':
            h['status'], h['measured'] = 'refuted', -engine.sigma_run()
        else:
            g = out.gain if out.gain is not None else 0.0
            h['measured'] = g
            h['status'] = 'confirmed' if g > 2 * engine.sigma_pair() else ('refuted' if g <= 0 else 'unresolved')
        if h.get('source') == 'experience':         # data, not a prediction: only an LLM twin's prediction is scored
            if h.get('llm_mu') is not None:
                self._pair(h, h['llm_mu'], h['measured'])
        elif not h.get('explore'):                   # calibration is about the model's predictions only
            self._pair(h, raw, h['measured'])

    def _retry(self, engine, h: dict, out: Outcome):
        """A knob change that fails a numerics check on an incumbent carrying an earlier accepted change may fail only
        together with that change (F1-S1: bf16 broke the loss on the compile incumbent, 3.07 vs 1.00 over the probe's 2
        updates, and was never tried on another base; bf16 without compile ran x3.5 in the random arm). One more entry
        on the incumbent tests the knob with the incumbent's last change reverted (its parent's values): probe first,
        the zero prior of a neighbour, and the feasibility of a rebased entry (0.7: the knob ran and the parent ran, only
        their combination is new), which the value/cost rule puts ahead of the untried zero-prior neighbours (0.6). At
        most `retry` per incumbent; never for a retry's own failure, a patch, a crash or a workload change (those fail
        on every base), or when the last change is a patch (nothing to revert by knobs)."""
        failed = set(out.failed_checks or [])
        inc = h['base']
        if h.get('patch') or not h.get('change') or h.get('retry_of') or not failed or not failed <= set(NUMERICS_CHECKS):
            return
        if len(self.retries.get(inc, ())) >= self.retry:
            return
        c = engine.s.candidates.get(inc)
        p = engine.s.candidates.get(getattr(c, 'parent', None) or '') if c is not None else None
        if p is None or (c.overlay or {}) != (p.overlay or {}):
            return
        rev = {}
        for k in set(c.config) | set(p.config):             # the parent's values of what the last change set
            x, y, mine = c.config.get(k), p.config.get(k), h['change'].get(k)
            if x == y:
                continue
            if isinstance(x, dict) and isinstance(y, dict) and (mine is None or isinstance(mine, dict)):
                d = {n: y.get(n) for n in set(x) | set(y)   # per name (env): the knob's own names replace them anyway
                     if x.get(n) != y.get(n) and n not in (mine or {})}
                if d:
                    rev[k] = d
            elif k not in h['change']:                       # (the knob's own value replaces the incumbent's anyway)
                rev[k] = y
        if not rev:
            return
        change = dict(h['change'])
        for k, v in rev.items():
            change[k] = {**v, **change[k]} if isinstance(v, dict) and isinstance(change.get(k), dict) else v
        mech = f'{h["mechanism"]} without ' + ', '.join(f'{k}={c.config.get(k)}' for k in sorted(rev))
        key = f'{inc}|{mech}'
        if key in self.ledger:
            return
        self.ledger[key] = {'mechanism': mech, 'change': change, 'patch': None, 'mu': 0.0,
                            'sigma': 2 * engine.sigma_run(), 'p_feasible': 0.7, 'base': inc, 'status': 'open', 'key': key,
                            'explore': True, 'retry_of': h['key']}
        self.retries.setdefault(inc, []).append(mech)
        (getattr(engine, 'log', None) or (lambda m: None))(
            f'[search] hls: {h["mechanism"]} failed {", ".join(sorted(failed))} on {inc}: it will be tried without '
            + ', '.join(f'{k}={c.config.get(k)}' for k in sorted(rev)) + f' ({len(self.retries[inc])} of {self.retry} on {inc})')

    def _observe_bundle(self, engine, p, out: Outcome):
        members = [k for k in self.members.get(p.hypothesis, []) if k in self.ledger]
        self.pending.pop(p.hypothesis, None)
        if p.hypothesis == 'bundle|exp':
            self.exp_fate = {'status': out.status, 'gain': out.gain, 'candidate': out.cid,
                             'moved': out.cid is not None and engine.incumbent() == out.cid,
                             'failed_checks': out.failed_checks or None}
        if out.fidelity == 'full' and any(self.ledger[k].get('source') == 'experience' for k in members):
            self.exp_bundle_screens += 1
        if out.status == 'duplicate':                   # an existing candidate: not proposed again, members unchanged
            self.evaluated.add(frozenset(members))
            return
        self.bundles += 1
        self.evaluated.add(frozenset(members))
        if out.loss_deviation is not None:              # a combination can change the numerics its members did not
            self.bundle_numerics.append({'candidate': out.cid, 'members': [self.ledger[k]['mechanism'] for k in members],
                                         'classes': [self.ledger[k].get('change_class') for k in members],
                                         'status': out.status, 'loss_deviation_pct': 100 * out.loss_deviation,
                                         **({'loss_deviation_base_pct': 100 * out.loss_deviation_base}
                                            if out.loss_deviation_base is not None else {})})
        if out.cid is not None and engine.incumbent() == out.cid:          # moved (directly or after confirmation)
            for k in members:
                self.ledger[k].update(status='bundled', bundle_gain=out.gain, bundle_candidate=out.cid)
            return
        if out.status == 'screened' and (out.gain or 0) > 0:
            self.lost[p.hypothesis] = out.gain          # members stay unresolved; the bundle grows with the next positive
            return
        self.failed_sets.append(frozenset(members))     # failed, slower or not better: not repeated as a whole
        for k in members:
            if out.failed_checks:
                self.ledger[k]['failed_checks'] = out.failed_checks
        if self.split and len(members) >= 4:            # group testing: the halves, alternating by measured gain
            order = sorted(members, key=lambda k: -_gain(self.ledger[k]))   # (an open member: by its prior)
            hard = out.status in ('infeasible', 'invalid', 'rejected')
            for half in (order[0::2], order[1::2]):
                if len(half) >= 2 and frozenset(half) not in self.evaluated:
                    q = self._bundle_proposal(engine, p.base or engine.incumbent(), half, p_feasible=0.6,
                                              plan=('probe', 'full') if hard else ('full',))
                    if q is not None:
                        self.pending[q.hypothesis] = q

    def residual(self, engine) -> dict:
        kap = self.kappa()
        open_ = [h for h in self.ledger.values() if h['status'] == 'open' and h['base'] == engine.incumbent()]
        out = {'kappa': round(kap, 4), 'n_open': len(open_), 'sum_pred_gain': round(sum(max(0, kap * h['mu']) for h in open_), 4),
               'resolved': len(self.pairs), 'unresolved_positive': sum(1 for h in self.ledger.values()
                                                                       if h['status'] == 'unresolved' and (h.get('measured') or 0) > 0),
               'lost_bundle_gain_pct': round(100 * max(self.lost.values(), default=0.0), 2),
               'failed_bundles': len(self.failed_sets)}
        if self.class_kappa:
            out['kappa_by_class'] = self.class_kappas()
        return out


class HLS2(HLS):
    """hls with the search fix switched on (FABLE_DESIGN_SEARCH_FIX, 2026-10-29: Q9U-code-env-opus never tested the
    Liger knob in ten screens): the same class, C1-C5 and C7-C9 on, C6 (the budget composite) off."""
    name = 'hls2'
    SWITCHES = {'carry_open': True, 'view_states': True, 'prior_sigma': 0.03, 'explore_always': True, 'backstop': True,
                'class_kappa': True, 'start_failures': 2, 'cover_bundle': False, 'append_merge': True, 'no_revert': True}

    def __init__(self, proposer, **kw):
        super().__init__(proposer, **{**self.SWITCHES, **kw})


class DOE(Strategy):
    name = 'doe'
    uses_engine_stop = False

    def __init__(self, proposer, *, max_factors: int = 19, min_factors: int = 4, max_share: float = 0.35,
                 experience: dict | None = None, probe_first: bool = True):
        self.proposer, self.max_factors, self.min_factors, self.max_share = proposer, max_factors, min_factors, max_share
        self.exp, self.probe_first = experience or {}, probe_first
        self.why = None                                 # why no design ran (the report and the log say it)
        self.factors, self.design, self.base, self.y, self.state = None, None, None, {}, 'design'
        self.effects: dict | None = None
        self.resolution: float | None = None
        self.handoff: list[dict] = []                   # positive factors below the design's resolution
        self.queue: list[dict] = []                     # factors still to probe, one at a time, before the design
        self.probed: list[str] = []
        self.dropped: list[dict] = []                   # {'name', 'why'}: a known constraint, or its own probe failed

    def describe(self) -> dict:
        return {'name': self.name, 'factors': [f['name'] for f in self.factors or []], 'effects': self.effects,
                'resolution': self.resolution, 'below_resolution': [h['mechanism'][4:] for h in self.handoff],
                'probed': self.probed, 'dropped': self.dropped, 'why': self.why}

    def _not_run(self, engine, why: str) -> None:
        """No screening design: say why (D2-R201b: `--strategy doe` on a 16-GPU job ended `no_change` after 0 runs, its
        design did not fit the budget, and nothing said so)."""
        self.state, self.why = 'done', why
        (getattr(engine, 'log', None) or (lambda m: None))(f'[search] doe: no screening design: {why}')

    def _too_dear(self, need: float, left: float) -> str:
        return (f'it needs about {need:.2f} GPU-h, more than the {self.max_share:.0%} of the {left:.2f} GPU-h left that a '
                'design may use: raise search.budget.gpu_hours, or search with hls')

    def _known_to_run(self, change: dict) -> bool:
        """The store measured this move in scope (covered: a gain, null: within noise), so it runs here."""
        cov = self.exp.get('coverage') or {}
        return bool(change) and all((cov.get(k) or {}).get('level') in ('covered', 'null') and cov[k].get('value') == v
                                    for k, v in change.items())

    def _screen_factors(self, engine) -> bool:
        """Before the design (H1 block 1: one illegal flag crashed half of it; H2b arm D: 12 of 19 rows invalid and the
        main effects not estimable): drop the factors a known constraint or the measured memory look-up warns about
        (no GPU), and queue a 2-update probe of each factor the store has no run of, or that only the closed-form
        memory estimate (the plan's `soft` warnings) warns about. False when the block (probes + design) would starve
        the rest of the search."""
        cfg = engine.s.candidates[self.base].config
        check, soft, keep, probe = self.exp.get('check'), self.exp.get('soft') or (lambda w: False), [], set()
        for f in self.factors:
            warn = list(check({**cfg, **f['plus']}) or []) if check else []
            hard = [w for w in warn if not soft(w)]
            if hard:
                self.dropped.append({'name': f['name'], 'why': f'experience: {hard[0]}'[:240]})
            else:
                keep.append(f)
                if warn:                                   # an estimate only: its probe decides
                    probe.add(f['name'])
        self.factors = keep
        self.queue = [f for f in keep if f['name'] in probe or not self._known_to_run(f['plus'])]
        c = engine.costs()
        left = engine.s.budget.gpu_hours - engine.s.ledger.used['gpu_hours']
        block = len(self.queue) * c['probe'] + (len(pb_design(len(keep))) * c['short'] if keep else 0.0)
        if len(keep) < self.min_factors:
            self._not_run(engine, f'{len(keep)} factors are left after {len(self.dropped)} known failures (a design needs '
                          f'{self.min_factors})')
            return False
        if block > self.max_share * left:
            self._not_run(engine, f'the {len(self.queue)} factor probes and the {len(pb_design(len(keep)))}-run design of '
                          f'{len(keep)} factors at short fidelity: ' + self._too_dear(block, left))
            return False
        return True

    def _start_design(self, engine) -> bool:
        if len(self.factors) < self.min_factors:           # a design is not worth it for a few toggles
            self._not_run(engine, f'{len(self.factors)} factors are left after their probes (a design needs '
                          f'{self.min_factors})')
            return False
        self.design = pb_design(len(self.factors))
        left = engine.s.budget.gpu_hours - engine.s.ledger.used['gpu_hours']
        if len(self.design) * engine.costs()['short'] > self.max_share * left:   # it would starve the rest of the search
            self._not_run(engine, f'the {len(self.design)}-run design of {len(self.factors)} factors at short fidelity: '
                          + self._too_dear(len(self.design) * engine.costs()['short'], left))
            return False
        self.state = 'design'
        return True

    def propose(self, engine, k):
        if self.factors is None:
            self.base = engine.incumbent()
            self.factors = self.proposer.factors(engine, self.base)[:self.max_factors]
            if len(self.factors) < self.min_factors:       # a design is not worth it for a few toggles
                self._not_run(engine, f'the knobs give {len(self.factors)} two-level factors (a design needs '
                              f'{self.min_factors})')
                return []
            if self.probe_first and engine.multi_fidelity:
                if not self._screen_factors(engine):
                    return []
                self.state = 'probe'
            elif not self._start_design(engine):
                return []
        if self.state == 'probe':
            if self.queue:
                f = self.queue[0]
                return [Proposal(change=dict(f['plus']), base=self.base, mechanism=f'doe probe {f["name"]}',
                                 hypothesis=f'probe:{f["name"]}', origin='doe', fidelity=('probe',), abortable=False,
                                 p_feasible=0.7)]
            if not self._start_design(engine):
                return []
        if self.state == 'design':
            for i, row in enumerate(self.design):
                if i in self.y:
                    continue
                change = _merge(f['plus'] for f, s in zip(self.factors, row) if s > 0)
                if not change:                       # the all-minus row is the base itself
                    self.y[i] = engine.t_window(self.base, [engine.warmup + 1, engine.short_steps])   # the rows' updates
                    continue
                return [Proposal(change=change, base=self.base, mechanism=f'doe row {i}', hypothesis=f'row{i}', origin='doe',
                                 fidelity=('short',), abortable=False, p_feasible=0.7)]
            self.state = 'combine'
        if self.state == 'combine':
            self.state = 'done'
            self.effects = self._effects()
            # an N-run screening design spreads each two-factor interaction over the other columns (+-1/3 at N = 12), so
            # main effects below 4/N of the largest one, or within 2 sigma_pair, are not resolved: they are not combined
            # but handed on (to HLS in doe+hls) as within-noise positives, measured by their main effect
            gains = {f['name']: -self.effects[f['name']] for f in self.factors if math.isfinite(self.effects[f['name']])}
            self.resolution = max(4 / len(self.design) * max((abs(g) for g in gains.values()), default=0.0),
                                  2 * engine.sigma_pair())
            good = [f for f in self.factors if gains.get(f['name'], 0) >= self.resolution]
            self.handoff = [{'mechanism': f'doe:{f["name"]}', 'change': dict(f['plus']), 'measured': gains[f['name']]}
                            for f in self.factors if 0 < gains.get(f['name'], 0) < self.resolution]
            if good:
                return [Proposal(change=_merge(f['plus'] for f in good), base=self.base, mechanism='doe combination: '
                                 + ','.join(f['name'] for f in good), origin='doe', fidelity=('full',), abortable=False,
                                 confirm=True)]
        return []

    def observe(self, engine, p, out: Outcome):
        if p.hypothesis and p.hypothesis.startswith('probe:'):
            f = self.queue.pop(0) if self.queue and self.queue[0]['name'] == p.hypothesis[6:] else None
            if f is None or out.status in ('infra', 'duplicate'):
                return                                     # no evidence about the factor: it stays in the design
            self.probed.append(f['name'])
            if not out.valid:                              # crashed, out of memory, an invalid flag, the wrong job
                self.factors = [x for x in self.factors if x['name'] != f['name']]
                self.dropped.append({'name': f['name'], 'why': f'probe: {(out.reasons or [out.status])[0]}'[:240]})
            return
        if out.status == 'infra':
            return
        if p.hypothesis and p.hypothesis.startswith('row'):
            self.y[int(p.hypothesis[3:])] = out.step_time if out.valid else None

    def _effects(self) -> dict:
        """Main effect of each factor on log step time (+ level minus - level) over the rows that ran; a factor
        whose + level never ran gets +inf (assumed harmful)."""
        out = {}
        for j, f in enumerate(self.factors):
            plus = [math.log(self.y[i]) for i, r in enumerate(self.design) if r[j] > 0 and self.y.get(i)]
            minus = [math.log(self.y[i]) for i, r in enumerate(self.design) if r[j] < 0 and self.y.get(i)]
            out[f['name']] = (sum(plus) / len(plus) - sum(minus) / len(minus)) if plus and minus else math.inf
        return out


class Chain(Strategy):
    def __init__(self, *strategies):
        self.strategies, self.i = list(strategies), 0
        self.name = '+'.join(s.name for s in strategies)

    @property
    def uses_engine_stop(self):
        return self.strategies[self.i].uses_engine_stop if self.i < len(self.strategies) else True

    def ledger_view(self) -> list:
        return [h for s in self.strategies if hasattr(s, 'ledger_view') for h in s.ledger_view()]

    def describe(self) -> dict:
        return {'name': self.name, 'parts': [s.describe() for s in self.strategies]}

    def propose(self, engine, k):
        while self.i < len(self.strategies):
            props = self.strategies[self.i].propose(engine, k)
            if props:
                return props
            done, self.i = self.strategies[self.i], self.i + 1
            if self.i < len(self.strategies) and getattr(done, 'handoff', None) and                     hasattr(self.strategies[self.i], 'receive'):
                self.strategies[self.i].receive(engine, done.handoff)
        return []

    def observe(self, engine, p, out):
        if self.i < len(self.strategies):
            self.strategies[self.i].observe(engine, p, out)

    def residual(self, engine) -> dict:
        return self.strategies[-1].residual(engine)


class RandomKnobs(Strategy):
    name = 'random'
    uses_engine_stop = False

    def __init__(self, *, seed: int = 42, probes: bool = False):
        self.rng, self.probes = random.Random(seed), probes

    def propose(self, engine, k):
        space = choices(engine.s.task)
        for _ in range(2000):
            p = Proposal(change={n: self.rng.choice(v) for n, v in space.items()}, absolute=True, origin='random',
                         p_feasible=0.5, fidelity=('probe', 'full') if self.probes else ('full',), abortable=self.probes)
            if engine.key(p) not in engine.tried:
                return [p]
        return []


class TPEKnobs(Strategy):
    name = 'tpe'
    uses_engine_stop = False

    def __init__(self, *, seed: int = 42, probes: bool = False, n_startup_trials: int = 4):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction='minimize',
                                         sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials))
        self.probes, self.trial, self.warm = probes, None, False

    def propose(self, engine, k):
        from optuna.distributions import CategoricalDistribution
        from optuna.trial import create_trial
        space = choices(engine.s.task)
        if not self.warm:
            self.warm = True
            start = engine.s.candidates['c0']
            if start.verdict and start.verdict.get('valid') and all(start.config.get(n) in v for n, v in space.items()):
                self.study.add_trial(create_trial(params={n: start.config[n] for n in space},
                                                  distributions={n: CategoricalDistribution(v) for n, v in space.items()},
                                                  value=start.verdict['step_time']))
        for _ in range(500):
            self.trial = self.study.ask()
            p = Proposal(change={n: self.trial.suggest_categorical(n, v) for n, v in space.items()}, absolute=True,
                         origin='tpe', p_feasible=0.5, fidelity=('probe', 'full') if self.probes else ('full',),
                         abortable=self.probes)
            if engine.key(p) not in engine.tried:
                return [p]
            self.study.tell(self.trial, self._penalty(engine))
        return []

    def _penalty(self, engine) -> float:
        vals = [c.verdict['step_time'] for c in engine.s.candidates.values() if c.verdict and c.verdict.get('valid')
                and c.verdict.get('step_time')]
        return 2.0 * max(vals) if vals else 1e6

    def observe(self, engine, p, out: Outcome):
        if self.trial is not None:
            if out.status == 'infra':                  # no verdict: the trial fails without an objective value
                from optuna.trial import TrialState
                self.study.tell(self.trial, state=TrialState.FAIL)
            else:
                self.study.tell(self.trial, out.step_time if out.valid and out.step_time else self._penalty(engine))
            self.trial = None
