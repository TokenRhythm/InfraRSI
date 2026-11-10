"""esls: the Enabler/Spender Lattice Search (possible_search_algorithm/20 section 1, with section 2's memory ledger,
section 4's tail as a set and section 5's rung ladder).

The knob-side optimum of these jobs sits on the memory boundary: the largest speed moves spend memory and fit only with
the moves that free it, which alone may gain nothing or lose. esls searches the boundary, not the volume:

    prior      wcs's one prior call, asking also for memory terms (formulas over the knob values and the manifest's
               numbers) and each value's family; the memory ledger (search/ledger) built from it
    lattice    spenders: the values the ledger predicts to add memory; spend levels: the legal combinations of the
               spender knobs' values, kept on the Pareto front (more memory only for more predicted gain) and ordered
               by the ledger's memory as it stands at each set's choice - the chain; a level whose lower bound is above
               the card with every enabler is out. Enablers: the other values that free memory or of unknown direction
    enabler    one probe per enabler on the reference, the most promising first (its estimate plus its spread), while
               they take at most half of the low-fidelity allowance: its measured peak against the reference's is what
               it frees (an enabler's worth is the spend it unlocks, not its own time: one slower alone still enters)
    boundary   per enabler set (none, singles, pairs of the four that free most, all of them together; no pair the
               pair table rules out), the one whose best level has the largest expected improvement over the
               incumbent (P(fits) x the predicted gain above it; levels a valid screen already reached are done): a
               probabilistic bisection over the chain by probes at the level nearest P = 1/2 (out of memory, a peak at
               92 % of the card or a cliff: above; else below), the runs and the ledger deciding what they already
               know; one full screen at the boundary level (none at the reference's: that set unlocks nothing); the
               ledge - one level down - when that screen is slower than predicted near the card or fails
    code       wcs's code track, unchanged (patches only, the self-proof at the probe, a proven patch a pseudo-knob)
    lateral    one other composition at the boundary's memory, predicted faster: through the ladder
    tail       the small knobs rated positive, as one bundle on the incumbent (search/tail)
    hedge      when the lattice's screens all landed within 2 sigma_pair: one feasibility-weighted random draw
    fill       screens left: feasibility-weighted random draws through the ladder

Every probe and short run: not a screen, counted in GPU-hours, bounded by the low-fidelity allowance (search/ladder).
From the review CAND_r1: a spend level a workload invariant ties to another knob is completed from the legal table (the
manifest's coupled groups: micro-batch with its accumulation), not dropped; the hard prune needs every allowed enabler
assignment to leave the level above the card; a boundary probe the cliff rule rejects is an interval (thrash) in the
ledger, not a fit; the prior's extra fields are read only in the shapes asked for.
One proposal per step, its own order until the screens are spent (uses_engine_stop False). Nothing here names a knob,
a model or a library: the spenders, enablers, terms and families come from the prior, the ledger and the runs.
"""
from __future__ import annotations

import itertools
import math

from . import ESLS_COMPONENTS
from .cand_base import CandidateBase
from .ladder import LOWFI_SHARE
from .ledger import NEAR, for_wcs
from .rpp import weighted_draw
from .tail import TailScreen
from .wcs import _coerce, proposer_flags as wcs_flags, vk

LATTICE_SCREENS = 4             # full screens of the lattice phases at most (boundaries, ledges, lateral)
TAIL_SCREENS = 3                # the tail's bundle and its diagnosis at most
ENABLER_MAX = 6                 # enablers probed at most
ENABLER_SHARE = 0.5             # ... while their probes take at most this share of the low-fidelity allowance
CHAIN_MAX = 24                  # spend levels kept at most
SPEND_MIN = 0.01                # a value spends when the ledger adds at least this share of the card (and 1 GiB)
FREES_MIN = 0.02                # an enabler counts when it frees at least this share of the card (and 2 GiB)
LEDGE_PEAK = 0.85               # a boundary screen slower than predicted with a peak above this share: one level down
LATERAL_BAND = 0.05             # the lateral composition: within this share of the card of the boundary's memory
KNOWN = 0.99                    # the bisection skips a level the feasibility puts at this or 1 - this (else: a probe)
PRIOR_RULE = """
STRUCTURE: also give `mem_terms` - the terms of one GPU's peak memory that the knobs change, each {"name", "gib_expr",
"when"}: `gib_expr` an arithmetic expression giving GiB per GPU over the knob names (their values) and the numbers in
`memory_facts` (+ - * / ** min max abs, comparisons, `a if condition else b`), `when` an optional {knob: value or [values]}
under which the term applies; write the shape you know (how a term grows with each knob) and your best constants - the
search fits each term's scale from the runs. Give each value in `levels` a `family`: the cost it acts on, in one or two
words (activations, logits, optimizer, communication, input, kernels, allocator, ...)."""
PRIOR_TOP = {'mem_terms': {'type': ['array', 'null'], 'items': {'type': 'object', 'properties': {
    'name': {'type': ['string', 'null']}, 'gib_expr': {'type': ['string', 'null']}, 'when': {'type': ['object', 'null']}}}}}
PRIOR_LEVEL = {'family': {'type': ['string', 'null']}}


class ESLS(CandidateBase):
    name = 'esls'

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        """wcs's flags, the code track's by `esls_code`."""
        return wcs_flags(['wcs_code'] if 'esls_code' in set(off or ()) else [])

    def __init__(self, proposer, *, off=(), facts=None, accounting=None, experience=None, kappa0: float = 0.5,
                 seed: int = 42, lowfi_share: float = LOWFI_SHARE, lattice_screens: int = LATTICE_SCREENS,
                 tail_screens: int = TAIL_SCREENS, enabler_max: int = ENABLER_MAX, chain_max: int = CHAIN_MAX):
        off = set(off or ())
        self.esls_on = {k: k not in off for k in ESLS_COMPONENTS}
        off_wcs = [w for w, e in (('wcs_prior', 'esls_prior'), ('wcs_code', 'esls_code')) if not self.esls_on[e]]
        super().__init__(proposer, off_wcs=off_wcs, ledger=self.esls_on['esls_ledger'], short=self.esls_on['esls_ladder'],
                         lowfi_share=lowfi_share, facts=facts, accounting=accounting, experience=experience,
                         kappa0=kappa0, seed=seed)
        if self.esls_on['esls_prior'] and hasattr(proposer, 'prior_extra'):
            proposer.prior_extra = (PRIOR_RULE, PRIOR_TOP, PRIOR_LEVEL)
        self.seed = int(seed)
        self.lattice_screens, self.tail_screens = int(lattice_screens), int(tail_screens)
        self.enabler_max, self.chain_max = int(enabler_max), int(chain_max)
        self.tail = TailScreen()
        self.order_ledger = None                         # the ledger the lattice is ordered by (the feasibility's, or a
        #                                                  separate one when esls_ledger is off: esls-noledger)
        self.families: dict = {}
        self.spenders: list = []                         # (knob, vk)
        self.spender_knobs: list = []
        self.levels: list[dict] = []                     # every legal spend level: {'cfg', 'mem', 'gain'}
        self.lattice: list[dict] = []                    # the chain of spend levels (wcs's `chain` is the patch chain)
        self.legal_levels: list[dict] = []
        self._table_keys = None
        self.pruned: list[dict] = []
        self.enablers: dict = {}                         # (knob, vk) -> {'value', 'status', 'peak', 'frees', 'declared'}
        self.sets_done: list[dict] = []
        self.cur: dict | None = None                     # the enabler set being bisected
        self.boundaries: list[dict] = []
        self.lattice_used = 0
        self.tail_used = 0
        self.tail_base: tuple | None = None              # (the configuration the tail's members go onto, its gain)
        self.lateral_done = self.hedge_done = False
        self.fill_n = 0
        self.log: list[dict] = []
        self.full_keys: set = set()                      # configurations that ran in full (any outcome)
        self.ledge_of: dict = {}                         # a ledge's hypothesis -> (its set, its level, its level cfg)
        self._cliffs: dict = {}                          # a boundary probe's candidate -> the cliff rule's verdict

    # ------------------------------------------------------------ set-up: the prior's extras, the lattice
    def _init(self, engine) -> None:
        super()._init(engine)
        self.order_ledger = self.mledger if self.mledger is not None else for_wcs(self)
        answer = getattr(self.proposer, 'last_prior', None)
        rows = answer.get('levels') if isinstance(answer, dict) else None
        for row in rows if isinstance(rows, list) else []:          # (the model's answer as it gave it: the shapes
            try:                                                      #  checked, an unusable row adds no family -
                if isinstance(row, dict) and isinstance(row.get('knob'), str) and row['knob'] in self.knobs \
                        and isinstance(row.get('family'), str):       #  wcs degraded the same answer safely; C6)
                    v = _coerce(row.get('value'), self.knobs[row['knob']])
                    if v is not None:
                        self.families[(row['knob'], vk(v))] = row['family'].strip().lower()[:40]
            except Exception:                                         # noqa: BLE001
                continue
        self._build_lattice(engine)

    def _family(self, k, key) -> str | None:
        return self.families.get((k, key)) or next((f for (kk, _), f in self.families.items() if kk == k), None)

    def _gain(self, cfg: dict) -> float:
        """The surrogate's mean gain over the reference."""
        from .wcs import Surrogate
        return Surrogate.value(self._feats(cfg, ()), self.surr.mean())

    def _build_lattice(self, engine) -> None:
        led = self.order_ledger
        cap = self.mem.cap or 0.0
        base = led.predict(self.ref)[0]
        spend_min = max(1.0, SPEND_MIN * cap)
        self.spenders = []
        for k, vals in self.knobs.items():
            for v in vals:
                if vk(v) == vk(self.ref.get(k)):
                    continue
                e = (self.prior.levels.get((k, vk(v))) if self.prior else None) or {}
                d = led.predict({**self.ref, k: v})[0] - base
                if d >= spend_min or e.get('mem_dir') == 'spends':
                    self.spenders.append((k, vk(v)))
        self.spender_knobs = sorted({k for k, _ in self.spenders})
        enablers = []
        for k, vals in self.knobs.items():
            if k in self.spender_knobs:
                continue
            for v in vals:
                key = vk(v)
                if key == vk(self.ref.get(k)) or self.alone.get((k, key)):
                    continue
                e = (self.prior.levels.get((k, key)) if self.prior else None) or {}
                d = led.predict({**self.ref, k: v})[0] - base
                if e.get('mem_dir') in ('frees', None) or d <= -spend_min:
                    sd = led.predict({**self.ref, k: v})[1]
                    enablers.append(((k, key), v, e.get('mem_gib') if e.get('mem_dir') == 'frees' else None,
                                     max(0.0, -d) + sd))             # (what it may free: its estimate + its spread)
        enablers.sort(key=lambda t: -t[3])
        for (k, key), v, gib, _ in enablers[:self.enabler_max]:
            self.enablers[(k, key)] = {'knob': k, 'value': v, 'status': 'candidate', 'peak': None, 'frees': None,
                                       'declared': gib}
        self.legal_levels = []
        vals = [self.knobs[k] for k in self.spender_knobs]
        groups, seen = None, set()
        for combo in itertools.product(*vals) if self.spender_knobs else [()]:
            cfg = {**self.ref, **dict(zip(self.spender_knobs, combo))}
            if cfg != self.ref and not self._legal(engine, cfg):
                if groups is None:                       # (a level a workload invariant ties to another knob: completed
                    groups = self._spender_groups(engine)  #  from the legal table, not dropped - CAND_r1 C1)
                cfg = self._complete_level(cfg, groups.get(tuple(vk(v) for v in combo)) or [])
                if cfg is None:
                    continue
            key = self._ckey(cfg, ())
            if key not in seen:
                seen.add(key)
                self.legal_levels.append(cfg)
        self._build_chain()

    def _partners(self) -> set:
        """The knobs a workload invariant ties to a spender knob (memory_facts' coupled groups)."""
        out = set()
        for g in self.facts.get('coupled') or []:
            g = {x for x in g if isinstance(x, str)} if isinstance(g, (list, tuple, set)) else set()
            if g & set(self.spender_knobs):
                out |= g
        return (out & set(self.knobs)) - set(self.spender_knobs)

    def _spender_groups(self, engine) -> dict:
        """The legal table's rows by their spender knobs' values."""
        out: dict = {}
        for row in self._table(engine):
            out.setdefault(tuple(vk(row.get(k)) for k in self.spender_knobs), []).append(row)
        return out

    def _complete_level(self, cfg: dict, rows) -> dict | None:
        """An illegal spend level completed from the legal table (wcs's _complete rule, WCS_r1 D4): the legal row that
        holds its spender values and differs from it in the fewest knobs - only in knobs a workload invariant ties to a
        spender when the manifest names them; None when no legal row holds them."""
        partners = self._partners()
        best, bd = None, None
        for row in rows:
            diff = [k for k in self.knobs if vk(row.get(k)) != vk(cfg.get(k))]
            if partners and not set(diff) <= partners:
                continue
            if bd is None or len(diff) < bd:
                best, bd = {k: row.get(k) for k in self.knobs}, len(diff)
        return best

    def _legal(self, engine, cfg: dict) -> bool:
        """The session's own check (wcs's table when it holds the whole product)."""
        table = self._table(engine)
        if getattr(self, 'table_whole', False):
            if self._table_keys is None:
                self._table_keys = {self._ckey(c, ()) for c in table}
            return self._ckey(cfg, ()) in self._table_keys
        check = getattr(engine.s, 'check_config', None)
        if check is None:
            return True
        c0 = engine.s.candidates['c0']
        try:
            return check({**c0.config, **cfg}, c0)[0] is not None
        except Exception:                                # noqa: BLE001 - a check that fails rejects the configuration
            return False

    def _build_chain(self) -> None:
        """The chain: the legal spend levels from the reference up by the ledger's memory as it now stands, on the
        Pareto front of the surrogate's gain (more memory only for more gain); a level whose lower bound is above the
        card with every enabler: out."""
        led = self.order_ledger
        cap = self.mem.cap or 0.0
        ids = self._led_ids(self.chain)                  # (the levels run with the proven patches)
        levels = [{'cfg': cfg, 'mem': led.predict(cfg, ids=ids)[0], 'gain': self._gain(cfg)} for cfg in self.legal_levels]
        ref_mem = led.predict(self.ref, ids=ids)[0]
        levels = [r for r in levels if r['mem'] >= ref_mem - 1e-9 or r['cfg'] == self.ref]   # (spend levels: from the
        levels.sort(key=lambda r: (r['cfg'] != self.ref, r['mem'], -r['gain']))            #  reference up)
        self.levels = levels
        assigns = self._assignments() if cap else []
        chain, best = [], -math.inf
        self.pruned = []
        for r in levels:
            if r['gain'] <= best + 1e-9 and r['cfg'] != self.ref:
                continue
            if cap and r['cfg'] != self.ref:
                lbs = []                                 # (out only when no allowed enabler assignment brings its lower
                for s in assigns:                        #  bound under the card - CAND_r1 C7; the first that does ends it)
                    lb = led.lower_bound(self._with(s, r['cfg']), ids=ids)
                    if lb <= cap:
                        lbs = None
                        break
                    lbs.append(lb)
                if lbs:
                    self.pruned.append({'level': self._words(r['cfg'], ()), 'lower_bound_gib': round(min(lbs), 1)})
                    continue
            chain.append(r)
            best = max(best, r['gain'])
        if len(chain) > self.chain_max:                  # (evenly in memory order, the ends kept)
            step = (len(chain) - 1) / (self.chain_max - 1)
            chain = [chain[round(i * step)] for i in range(self.chain_max)]
        self.lattice = chain

    # ------------------------------------------------------------ the phases
    def propose(self, engine, k):
        if self.prior is None:
            self._init(engine)
        self.engine = engine
        left = self._left(engine)
        if left <= 0:
            return []
        inc = engine.incumbent()
        while self.pending:
            q = self.pending.pop(0)
            if engine._fresh(q) or tuple(q.fidelity) == ('probe',):
                return [q]
        for step in (self._step_enabler, self._step_boundary, self._step_code, self._step_lateral, self._step_tail,
                     self._step_hedge, self._step_fill):
            q = step(engine, inc, left)
            if q is not None:
                return [q]
        return []

    def _probe(self, engine, inc, cfg, kind, why) -> object:
        q = self._knob_q(engine, inc, cfg, kind=kind, plan=('probe',), why=why)
        q.gate = lambda *a, **kw: None                   # (a probe-only plan: the engine passes the detailed row, the
        return q                                         #  gate itself is never called; the cliff is read in observe)

    def _step_enabler(self, engine, inc, left):
        if not self.esls_on['esls_enabler_probe']:
            return None
        share = self.ladder.spent(engine) + self.ladder.next_cost(engine, 'probe') > \
            ENABLER_SHARE * self.ladder.allowance(engine) + 1e-12      # (of the knob side's pool)
        for key, e in self.enablers.items():
            if e['status'] != 'candidate':
                continue
            if not self.spender_knobs:                   # (no spender: an enabler's worth is the spend it unlocks, and
                e['status'], e['why'] = 'unprobed', 'no spend level to unlock'   # there is none to unlock)
                continue
            if share or not self.ladder.affordable(engine, 'probe'):
                e['status'] = 'declared'                 # (their share of the allowance is spent: the declaration or
                continue                                 #  the ledger's estimate stands)
            e['status'] = 'probing'
            q = self._probe(engine, inc, {**self.ref, e['knob']: e['value']}, 'enabler',
                            f'what {e["knob"]}={e["value"]} frees')
            e['probe'] = q.hypothesis
            return q
        return None

    def _frees_min(self) -> float:
        return max(2.0, FREES_MIN * (self.mem.cap or 0.0))

    def _effective(self) -> list:
        """The enablers that free memory: measured, else declared, else the ledger's estimate."""
        out = []
        led = self.order_ledger
        base = led.predict(self.ref)[0]
        for key, e in self.enablers.items():
            if e['status'] in ('failed',):
                continue
            frees = e['frees']
            if frees is None and e['status'] != 'probing':
                frees = e['declared'] if e['declared'] is not None else base - led.predict({**self.ref, e['knob']: e['value']})[0]
            if frees is not None and frees >= self._frees_min():
                out.append((key, frees))
        out.sort(key=lambda t: -t[1])
        return [k for k, _ in out]

    def _allowed(self, s) -> bool:
        """No two values of one knob, no pair the pair table rules out."""
        for a, b in itertools.combinations(s, 2):
            if a[0] == b[0]:
                return False
            rec = self.pairs.get(frozenset((a, b)))
            if rec and rec['p'] < 0.5:
                return False
        return True

    def _sets(self) -> list[tuple]:
        """The enabler sets: none, each one, the pairs among the four that free most, and all of them together (greedy
        by what each frees, skipping one the pair table rules out with those already in)."""
        eff = self._effective()
        sets = [()] + [(k,) for k in eff]
        sets += [s for s in itertools.combinations(eff[:4], 2) if self._allowed(s)]
        most: list = []
        for e in eff:
            if self._allowed(tuple(most) + (e,)):
                most.append(e)
        if len(most) >= 3:
            sets.append(tuple(most))
        done = {tuple(sorted(d['set'])) for d in self.sets_done}
        out, seen = [], set()
        for s in sets:
            key = tuple(sorted(s))
            if key not in done and key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def _assignments(self) -> list[tuple]:
        """Every allowed enabler assignment (the hard prune's 'with every enabler'): per enabler knob its reference value
        or one of its enabler values, no pair the pair table rules out; the empty one first."""
        byk: dict = {}
        for (k, key) in self.enablers:
            byk.setdefault(k, []).append((k, key))
        out = []
        for combo in itertools.product(*[[None] + v for v in byk.values()]):
            s = tuple(x for x in combo if x is not None)
            if self._allowed(s):
                out.append(s)
        return out

    def _with(self, eset, level_cfg) -> dict:
        cfg = dict(level_cfg)
        for (k, key) in eset:
            if vk(level_cfg.get(k)) == vk(self.ref.get(k)):   # (a knob the level itself moved - a coupled partner -
                cfg[k] = self.enablers[(k, key)]['value']     #  keeps the level's value)
        return cfg

    def _known_level(self, eset, i) -> str | None:
        """'fits' / 'fails' when a run or the feasibility already decides level i under the set, else None."""
        cfg = self._with(eset, self.lattice[i]['cfg'])
        st = self.mledger.ran_status(cfg, ids=self._led_ids(self.chain)) if self.mledger is not None else None
        if st is not None:
            return 'fits' if st == 'ok' else 'fails'
        p, _ = self._pf(cfg, tuple(self.chain))
        if p >= KNOWN:
            return 'fits'
        if p <= 1 - KNOWN:
            return 'fails'
        return None

    def _step_boundary(self, engine, inc, left):
        if not self.lattice or self.lattice_used >= self.lattice_screens:
            return None
        if any(e['status'] == 'probing' for e in self.enablers.values()):
            return None
        if self.cur is None:
            self._build_chain()                          # (the ledger and the surrogate as they now stand)
            y_inc = (self.runs.get(inc) or {}).get('y') or 0.0
            reached = max([self._gain(b.get('level_cfg') or {**self.ref, **{k: b['cfg'][k] for k in self.spender_knobs}})
                           for b in self.boundaries if b.get('gain') is not None], default=-math.inf)
            options = []
            for s in self._sets():
                value, b = 0.0, None
                for i in range(1, len(self.lattice)):
                    if self.lattice[i]['gain'] <= reached + 1e-9:     # (a spend level a valid screen reached or passed)
                        continue
                    cfg = self._with(s, self.lattice[i]['cfg'])
                    if self._screened(cfg):
                        continue
                    v = self._pf(cfg, tuple(self.chain))[0] * max(0.0, self._gain(cfg) - y_inc)
                    if v > value:
                        value, b = v, i
                if b is None:
                    self.sets_done.append({'set': s, 'level': None, 'why': 'no spend level under it beyond those reached '
                                                                           'is predicted above the incumbent'})
                    continue
                options.append((value, -len(s), s, b))
            if not options:
                return None
            options.sort(key=lambda t: (-t[0], -t[1]))
            _, _, s, b = options[0]
            lo, hi = 0, len(self.lattice)
            for i in range(len(self.lattice)):
                st = self._known_level(s, i)
                if st == 'fits':
                    lo = max(lo, i)
                elif st == 'fails' and i > lo:
                    hi = min(hi, i)
            self.cur = {'set': s, 'lo': lo, 'hi': hi, 'probes': 0, 'ledge': False}
        cur = self.cur
        while cur['hi'] - cur['lo'] > 1:
            for i in range(cur['lo'] + 1, cur['hi']):
                if self._known_level(cur['set'], i) == 'fits':
                    cur['lo'] = i
            for i in range(cur['lo'] + 1, cur['hi']):
                if self._known_level(cur['set'], i) == 'fails':
                    cur['hi'] = i
                    break
            if cur['hi'] - cur['lo'] <= 1:
                break
            ps = {i: self._pf(self._with(cur['set'], self.lattice[i]['cfg']), tuple(self.chain))[0]
                  for i in range(cur['lo'] + 1, cur['hi'])}
            mid = min(ps, key=lambda i: (abs(ps[i] - 0.5), -i))
            if not self.ladder.affordable(engine, 'probe'):
                cur['lo' if ps[mid] >= 0.5 else 'hi'] = mid
                continue
            cur['mid'] = mid
            cur['probes'] += 1
            return self._probe(engine, inc, self._with(cur['set'], self.lattice[mid]['cfg']), 'bound',
                               f'the boundary of {self._set_words(cur["set"])}: level {mid} of {len(self.lattice)}')
        return self._screen_boundary(engine, inc)

    def _screened(self, cfg: dict) -> bool:
        """Whether the configuration ran in full (a probe of it does not count)."""
        return self._ckey(cfg, tuple(self.chain)) in self.full_keys

    def _set_words(self, s) -> str:
        return ' + '.join(f'{k}={self.enablers[(k, key)]["value"]}' for k, key in s) or 'no enabler'

    def _screen_boundary(self, engine, inc):
        cur = self.cur
        level = cur['lo']
        cfg = self._with(cur['set'], self.lattice[level]['cfg'])
        if level == 0 or cfg == self.ref:
            self.sets_done.append({'set': cur['set'], 'level': level, 'why': 'unlocks no spend level'})
            self.cur = None
            return None
        if self._screened(cfg):
            self.sets_done.append({'set': cur['set'], 'level': level, 'why': 'its boundary ran'})
            self.cur = None
            return None
        p, why = self._pf(cfg, tuple(self.chain))
        plan = ('full',) if cur.get('probed_at') == level else \
            self.ladder.plan(engine, p, probe=True, short=False)     # (a boundary point: probed first, if affordable)
        q = self._knob_q(engine, inc, cfg, kind='boundary', plan=plan, p=p, why=why)
        cur['screen'] = q.hypothesis
        cur['level'] = level
        cur['level_cfg'] = self.lattice[level]['cfg']    # (its spend level as completed: the reached spend)
        cur['pred'] = self._gain(cfg)                    # (the prediction before its run: the ledge reads it)
        self.lattice_used += 1
        return q

    def _step_code(self, engine, inc, left):
        if not self.on['wcs_code']:
            return None
        code = [q for q in self._code(engine, inc, left) if engine._fresh(q) and self._lowfi_ok(engine, q)]
        return code[0] if code else None

    def _step_lateral(self, engine, inc, left):
        if not self.esls_on['esls_lateral'] or self.lateral_done or not self.boundaries or \
                self.lattice_used >= self.lattice_screens:
            return None
        self.lateral_done = True
        best = max((b for b in self.boundaries if b.get('gain') is not None and b.get('set') is not None),
                   key=lambda b: b['gain'], default=None)
        if best is None:
            return None
        cap = self.mem.cap or 0.0
        bcfg = best['cfg']
        bmem = self.order_ledger.predict(bcfg)[0]
        bgain = self._gain(bcfg)
        eset = best['set']
        options = []
        for r in self.levels:
            cfg = self._with(eset, r['cfg'])
            if {k: cfg[k] for k in self.spender_knobs} == {k: bcfg[k] for k in self.spender_knobs}:
                continue
            if self._screened(cfg):
                continue
            m = self.order_ledger.predict(cfg)[0]
            g = self._gain(cfg)
            if abs(m - bmem) <= LATERAL_BAND * cap and g > bgain + 2 * engine.sigma_pair():
                options.append((g, cfg))
        if not options:
            return None
        g, cfg = max(options, key=lambda t: t[0])
        p, why = self._pf(cfg, tuple(self.chain))
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None)
        if plan[-1] == 'full':
            self.lattice_used += 1
        return self._knob_q(engine, inc, cfg, kind='lateral', plan=plan, p=p, why=why)

    def _step_tail(self, engine, inc, left):
        if not self.esls_on['esls_tail'] or self.tail.done or self.tail_used >= self.tail_screens or left < 1:
            return None
        if self.tail.testing is not None and self.tail_base is not None and self.tail_base[2] == 'waiting':
            return None
        base_cfg = {k: v for k, v in self._knob_cfg(engine, inc).items()}
        if self.tail_base is None:
            skip = set(self.spender_knobs) | {k for k in self.knobs if vk(base_cfg.get(k)) != vk(self.ref.get(k))}
            mean = self.surr.mean()
            values = []
            for (k, key), mu in mean.items() if mean else []:
                if not isinstance(k, str) or k in skip or k not in self.knobs:
                    continue
                v = next((x for x in self.knobs[k] if vk(x) == key), None)
                if v is not None:
                    values.append((k, v, mu, self._family(k, key)))
            members = self.tail.start(values)
            if not members:
                return None
            self.tail_base = (base_cfg, self.runs.get(inc, {}).get('y') or 0.0, 'ready')
        cfg = {**self.tail_base[0], **{k: v for k, v, *_ in self.tail.testing}}
        if self._screened(cfg):
            self.tail.done = True
            return None
        p, why = self._pf(cfg, tuple(self.chain))
        plan = self.ladder.plan(engine, p, short=False)
        self.tail_used += 1
        self.tail_base = (self.tail_base[0], self.tail_base[1], 'waiting')
        return self._knob_q(engine, inc, cfg, kind='tail', plan=plan, p=p, why=why)

    def _knob_cfg(self, engine, cid) -> dict:
        c = engine.s.candidates[cid]
        cfg = self._mcfg(engine, cid)
        return {k: cfg.get(k, c.config.get(k)) for k in self.knobs}

    def _step_hedge(self, engine, inc, left):
        if not self.esls_on['esls_hedge'] or self.hedge_done:
            return None
        gains = [b['gain'] for b in self.boundaries if b.get('gain') is not None]
        if len(gains) < 2 or max(gains) - min(gains) > 2 * engine.sigma_pair():
            return None
        self.hedge_done = True
        got = weighted_draw(self, engine, inc, a=1.0, T=math.inf, salt=f'hedge|{len(self.log)}', p_min=self._p_min(engine))
        if got is None:
            return None
        cfg, ids, p, why, info = got
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None)
        return self._knob_q(engine, inc, cfg, kind='hedge', plan=plan, p=p, why=why)

    def _p_min(self, engine) -> float | None:
        """No probe left in the allowance: draw only what is predicted to fit."""
        return None if self.ladder.affordable(engine, 'probe') else self.ladder.probe_below

    def _step_fill(self, engine, inc, left):
        got = weighted_draw(self, engine, inc, a=1.0, T=math.inf, salt=f'fill|{self.fill_n}', p_min=self._p_min(engine))
        self.fill_n += 1
        if got is None:
            return None
        cfg, ids, p, why, info = got
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None)
        return self._knob_q(engine, inc, cfg, kind='fill', plan=plan, p=p, why=why)

    # ------------------------------------------------------------ outcomes
    def _cliff(self, engine, out) -> bool:
        """A passed probe at the memory edge: its peak at NEAR of the card or more, or its updates past the warm-up at
        the speed gate's ratio against the reference's (wcs's gate, on the detailed row)."""
        c = engine.s.candidates.get(out.cid)
        row = ((c.results or {}).get('probe') or {}) if c is not None else {}
        peak = row.get('memory_peak_measured_gib') or row.get('memory_peak_gib')
        if isinstance(peak, (int, float)) and self.mem.cap and peak >= NEAR * self.mem.cap:
            return True
        return bool(row.get('intervals')) and self._speed_gate(engine, out.cid, 'c0', row, True, False, 0) is not None

    def _evidence(self, engine, p, out, kind=None):
        """A boundary probe the cliff rule rejects goes to the ledger as an interval (thrash), not a fit (CAND_r1 C4)."""
        if kind is None and (p.hypothesis or '').startswith('bound|') and out.status == 'probed' and out.cid:
            self._cliffs[out.cid] = self._cliff(engine, out)
            if self._cliffs[out.cid]:
                kind = 'thrash'
        return super()._evidence(engine, p, out, kind=kind)

    def observe(self, engine, p, out):
        cls = super().observe(engine, p, out)
        hid = p.hypothesis or ''
        if out.fidelity == 'full' and out.cid and out.status not in ('duplicate', 'rejected', 'infra') and p.absolute:
            self.full_keys.add(self._ckey(self._cfg_of(p), tuple(self._patch_ideas(p))))
        kind = hid.split('|', 1)[0]
        k = (cls or {}).get('kind')
        if kind == 'enabler':
            e = next((x for x in self.enablers.values() if x.get('probe') == hid), None)
            if e is not None:
                if out.status == 'probed':
                    peak = self._peak_of(engine, out)
                    ref_peak = (self.mem.runs.get('c0') or {}).get('peak')
                    e.update(status='measured', peak=peak,
                             frees=(ref_peak - peak) if isinstance(peak, (int, float)) and isinstance(ref_peak, (int, float))
                             else None)
                    if e['frees'] is None:
                        e['status'] = 'ran'                  # (no peak measured: its declaration stands)
                else:
                    e.update(status='failed', why=k or out.status)
        elif kind == 'bound' and self.cur is not None:
            cur = self.cur
            mid = cur.get('mid')
            cliff = self._cliffs.pop(out.cid) if out.cid in self._cliffs else \
                out.status == 'probed' and self._cliff(engine, out)
            fits = out.status == 'probed' and not cliff
            if mid is not None:
                if fits:
                    cur['lo'] = mid
                    cur['probed_at'] = mid
                else:
                    cur['hi'] = mid
            self.log.append({'phase': 'bound', 'set': self._set_words(cur['set']), 'level': mid, 'fits': fits,
                             'outcome': k or out.status})
        elif kind == 'boundary' and self.cur is not None and self.cur.get('screen') == hid:
            cur = self.cur
            level = cur['level']
            gain = self.runs.get(out.cid, {}).get('y') if out.cid else None
            peak = self._peak_of(engine, out)
            rec = {'set': cur['set'], 'level': level, 'cfg': self._cfg_of(p), 'outcome': k or out.status,
                   'gain': gain if out.status == 'screened' else None, 'level_cfg': cur.get('level_cfg')}
            self.boundaries.append(rec)
            self.sets_done.append({'set': cur['set'], 'level': level, 'outcome': rec['outcome']})
            pred = cur.get('pred', self._gain(rec['cfg']))
            slow = gain is not None and gain < pred - 2 * max(engine.sigma_pair(), 1e-3)
            near = isinstance(peak, (int, float)) and self.mem.cap and peak >= LEDGE_PEAK * self.mem.cap
            failed = out.status in ('invalid', 'infeasible', 'aborted') or k in ('oom', 'thrash')
            if self.esls_on['esls_ledge'] and level > 0 and (slow and near or failed) and \
                    self.lattice_used < self.lattice_screens:
                cfg = self._with(cur['set'], self.lattice[level - 1]['cfg'])
                if cfg != self.ref and not self._screened(cfg):
                    q = self._knob_q(engine, engine.incumbent(), cfg, kind='ledge', plan=('full',))
                    self.lattice_used += 1
                    self.pending.append(q)
                    self.ledge_of[q.hypothesis] = (cur['set'], level - 1, self.lattice[level - 1]['cfg'])
            self.cur = None
        elif kind == 'ledge' and hid in self.ledge_of:
            s, lv, lcfg = self.ledge_of.pop(hid)
            self.boundaries.append({'set': s, 'level': lv, 'cfg': self._cfg_of(p), 'outcome': k or out.status,
                                    'ledge': True, 'gain': self.runs.get(out.cid, {}).get('y') if out.status == 'screened'
                                    else None, 'level_cfg': lcfg})
        elif kind == 'tail' and self.tail_base is not None and out.status == 'duplicate':
            self.tail.done = True
        elif kind == 'tail' and self.tail_base is not None:
            y = self.runs.get(out.cid, {}).get('y') if out.cid and out.status == 'screened' else None
            nxt = self.tail.observe(engine, None if y is None else y - self.tail_base[1])
            self.tail_base = (self.tail_base[0], self.tail_base[1], 'ready')
            if nxt is None:
                self.tail.done = True
        if tuple(p.fidelity)[-1] == 'short' and out.fidelity == 'short' and p.absolute:
            if self.ladder.promote(engine, out, label=p.mechanism):
                self.pending.append(self._knob_q(engine, engine.incumbent(), self._cfg_of(p), kind=f'{kind}-full',
                                                 plan=('full',), p=1.0, why='its short run may beat the incumbent'))

    # ------------------------------------------------------------ report
    def describe(self) -> dict:
        view, learned, prop = self.candidate_view()
        led = self.order_ledger
        out = {'name': self.name, 'proposer': prop,
               'esls': {'off': sorted(k for k, v in self.esls_on.items() if not v),
                        'spenders': [f'{k}={key}' for k, key in self.spenders],
                        'lattice': [{'level': self._words(r['cfg'], ()), 'mem_gib': round(led.predict(r['cfg'])[0], 1)
                                     if led else None, 'gain_pct': round(100 * math.expm1(self._gain(r['cfg'])), 2)}
                                    for r in self.lattice] if self.prior is not None else [],
                        'pruned': self.pruned,
                        'enablers': {f'{k}={key}': {kk: e.get(kk) for kk in ('status', 'peak', 'frees', 'declared', 'why')
                                                    if e.get(kk) is not None} for (k, key), e in self.enablers.items()},
                        'families': {f'{k}={key}': f for (k, key), f in self.families.items()},
                        'boundaries': [{'set': self._set_words(b['set']) if b['set'] else None, 'level': b['level'],
                                        'configuration': self._words(b['cfg'], ()), 'outcome': b['outcome'],
                                        **({'ledge': True} if b.get('ledge') else {}),
                                        **({'gain_pct': round(100 * math.expm1(b['gain']), 2)} if b.get('gain') is not None
                                           else {})} for b in self.boundaries],
                        'sets': [{'set': self._set_words(d['set']), **{kk: d[kk] for kk in ('level', 'why', 'outcome')
                                                                      if d.get(kk) is not None}} for d in self.sets_done],
                        'probes_bisection': self.log[-40:], 'tail': self.tail.describe(),
                        'screens_lattice': self.lattice_used, 'screens_tail': self.tail_used, **view}}
        if learned:
            out['learned'] = learned
        return out
