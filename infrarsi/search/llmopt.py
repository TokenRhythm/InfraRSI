"""What opro and ltune (possible_search_algorithm/10 section 04: OPRO, lambda-Tune) share on top of the candidates' base
(search/cand_base: wcs's prior call, table, surrogate, pair table, run dominance and code track; the memory ledger; the
rung ladder).

    the call     proposers.config_batch: `k` whole configurations from the task view (knobs, workload invariants,
                 constraints, hardware), the knob table and - for opro's rounds - the history; for ltune's race, diverse
    admission    a proposed configuration's values matched to the table (the reference's for a knob it leaves out; a
                 value a knob of the table does not have: not run), the knobs it leaves out completed from the legal
                 table when its values need it (a coupled knob), every value it names kept - the reference-valued ones
                 too (C4_r1 4) - else not legal, not run; one that ran, or that the runs rule out (the
                 memory ledger, wcs's measured dominance, the pair table; one that holds every change of a run that
                 failed for a reason other than memory), dropped and shown to the model as such; one proposed again while
                 open moves to the newest call
    history      every measured configuration, best first (its knob values off the reference, its patches, the measured
                 speed over the reference or how it failed - out of memory and by how much, thrash, a slow start,
                 aborted as slower -, its measured peak), then the probes, the failures, the proposals ruled out before
                 running
    budget       one LLM budget for the knob and the code calls: at most CALLS_MAX backend attempts (the prior, the
                 configuration calls, the review, the patch asks, the proposer's re-asks and the backend's own JSON
                 re-asks: `BudgetDriver` counts every attempt and lets the backend re-ask only within the attempts left)
                 and USD_MAX dollars (FABLE_PICK_TWO_20261031 W4; every call capped at min(the session's remainder,
                 USD_MAX - spent), recomputed before every call; C4_r1 2, 3)
    code         wcs's code track (patches only, the self-proof at the probe, a proven patch a pseudo-knob), asked when
                 the strategy says so, at most wcs's three asks, within the budget
    tail         the small values the time model rates above the incumbent's own, as one bundle on the incumbent
                 (search/tail: accepted from max(2 %, 2 sigma_pair), a regression bisected at most twice)

Nothing here names a knob, a model or a library: every value comes from the table, the model's answer and the runs.
"""
from __future__ import annotations

import dataclasses
import json
import math

from ..agent.backends.base import LLMResult
from ..agent.optimize import MIN_CALL_USD
from ..agent.state import Stateful
from .cand_base import CandidateBase
from .engine import Proposal, no_gate
from .rpp import weighted_draw
from .tail import TailScreen
from .wcs import SD_CONFIG, Surrogate, _coerce, _num, vk

CALLS_MAX, USD_MAX = 10, 8.0    # the model calls and dollars of a search in all, the knob and the code side together
K_BATCH = 5                     # configurations asked per call
HISTORY_MAX = 40                # history rows shown at most
P_MIN = 0.05                    # a proposal below this feasibility is not run while another reaches it
STALL = 2                       # the knob side stalls when the incumbent stayed the same over this many full screens
BAD_KINDS = ('numerics', 'crash', 'compiler', 'collective')   # failures not of memory: what holds all their changes fails
_NO_MATCH = object()            # no value of the knob matches what the model wrote (a legal null matches as None)


def _match(v, values):
    """wcs's `_coerce` with a legal null told apart from no match (C4_r2): the knob's value that the model's matches -
    None for a null the knob has (written null, "null" or "none") -, else _NO_MATCH."""
    w = _coerce(v, values)
    if w is not None or not any(x is None for x in values):
        return _NO_MATCH if w is None else w
    if v is None:
        return None
    if isinstance(v, str):
        try:
            if json.loads(v) is None:
                return None
        except ValueError:
            pass
        if v.strip().lower() == 'none':
            return None
    return _NO_MATCH


class BudgetDriver(Stateful):
    """The proposer's backend within one strategy's model budget (C4_r1 2, 3): every backend attempt - the backend's own
    re-asks of a reply that is not the JSON asked for included (`LLMResult.attempts`) - counts against `calls_max`; the
    backend may re-ask only within the attempts left (its `max_retries` for the call: one attempt left, no second
    request); every call's dollars are capped at min(the session's remainder, `usd_max` - spent), recomputed before
    every call (the backends keep their own re-asks within the call's cap); a call that does not fit is not sent.
    Everything else is the backend's."""
    CHECKPOINT_LIVE = ('_inner', '_owner')               # (the counters - attempts, usd, refused - are its state)

    def __init__(self, inner, owner):
        self.__dict__.update(_inner=inner, _owner=owner, attempts=0, usd=0.0, refused=0)

    def __getattr__(self, name):
        return getattr(self.__dict__['_inner'], name)

    def usd_left(self) -> float:
        st = self._owner
        spent = max(self.usd, st._usd(st.engine) if st.engine is not None else 0.0)
        return st.usd_max - spent

    def complete(self, system, prompt, **kw):
        st, inner = self._owner, self._inner
        left, usd = st.calls_max - self.attempts, self.usd_left()
        if left < 1 or usd < MIN_CALL_USD:
            self.refused += 1
            return LLMResult(action=None, text=None, llm_tokens=0, llm_usd=0.0, wall_seconds=0.0,
                             accounting_complete=True, error='the strategy\'s model budget is spent; not sent',
                             model=str(getattr(inner, 'model', '') or ''), observed_models=[], attempts=0)
        cap = kw.get('max_usd')
        kw['max_usd'] = usd if cap is None else min(float(cap), usd)
        old = getattr(inner, 'max_retries', None)
        if isinstance(old, int) and not isinstance(old, bool):
            inner.max_retries = max(0, min(old, left - 1))
        try:
            res = inner.complete(system, prompt, **kw)
        finally:
            if isinstance(old, int) and not isinstance(old, bool):
                inner.max_retries = old
        n = getattr(res, 'attempts', 1)
        self.attempts += max(1, int(n)) if isinstance(n, int) and not isinstance(n, bool) else 1
        u = getattr(res, 'llm_usd', None)
        if isinstance(u, (int, float)) and not isinstance(u, bool) and math.isfinite(u) and u > 0:
            self.usd += float(u)
        st._allowance()                                  # (the proposer's own count follows the attempts: its next
        return res                                       #  call of a sequence - review, ask, re-ask - stops in time)


class LLMOptBase(CandidateBase):
    """The shared part of opro and ltune (module docstring); a subclass sets `history_on` / `filter_on` and its own
    propose."""
    history_on = True
    filter_on = True
    CHECKPOINT_LIVE = ('_bdriver',)                      # search-state checkpoints: the budget driver is a part (its counters)

    def checkpoint_parts(self) -> dict:
        return {**super().checkpoint_parts(), 'budget_driver': getattr(self, '_bdriver', None)}

    def __init__(self, proposer, *, k: int = K_BATCH, calls_max: int = CALLS_MAX, usd_max: float = USD_MAX, **kw):
        super().__init__(proposer, **kw)
        self.seed = int(kw.get('seed', 42))
        self.k, self.calls_max, self.usd_max = int(k), int(calls_max), float(usd_max)
        self.archive: list[dict] = []                    # every admitted proposal, in the order admitted
        self.rounds: list[dict] = []                     # every configuration call (and the prior's round)
        self.rejected: list[dict] = []                   # proposals not run, with why (shown to the model)
        self.by_hid: dict[str, dict] = {}                # proposal id -> its archive item
        self.short_y: dict[str, float] = {}              # candidate -> its short run's gain over the reference
        self.inc_hist: list[str] = []                    # the incumbent before the first and after each full screen
        self.tail = TailScreen()
        self.tail_base: tuple | None = None              # (the configuration its members go onto, its gain, state)
        self.code_screens = 0
        self.last_got: int | None = None
        self.short_of: dict[str, Proposal] = {}          # candidate -> the proposal whose short run it is
        self.promoted: dict[str, Proposal] = {}          # a promotion's id -> that short run's proposal (C4_r1 1)
        self._bdriver = None
        if getattr(proposer, 'driver', None) is not None:    # (the model's calls within this strategy's budget)
            self._bdriver = BudgetDriver(proposer.driver, self)
            proposer.driver = self._bdriver

    def _init(self, engine) -> None:
        self.engine = engine                             # (the budget driver reads the session's dollars from it)
        super()._init(engine)
        self.inc_hist = [engine.incumbent()]

    # ------------------------------------------------------------ the budget
    def _llm_used(self) -> int:
        """The model calls made: the backend attempts the budget driver counted (every re-ask included), else wcs's."""
        d = getattr(self, '_bdriver', None)
        return d.attempts if d is not None else super()._llm_used()

    def _calls_left(self) -> int:
        return self.calls_max - self._llm_used()

    @staticmethod
    def _usd(engine) -> float:
        used = getattr(getattr(engine.s, 'ledger', None), 'used', None)
        x = used.get('llm_usd') if isinstance(used, dict) else None
        return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0

    def _budget_ok(self, engine, reserve: int = 0) -> bool:
        d = getattr(self, '_bdriver', None)
        usd_left = d.usd_left() if d is not None else self.usd_max - self._usd(engine)
        return self._calls_left() > reserve and usd_left >= MIN_CALL_USD

    def _allowance(self) -> None:
        """What the proposer may still spend of the calls (its re-asks included)."""
        if hasattr(self.proposer, 'calls_left'):
            self.proposer.calls_left = max(0, self.calls_max - self._llm_used())

    # ------------------------------------------------------------ the code track (wcs's, on demand, in the budget)
    def _code(self, engine, inc, left, ask: bool = True) -> list:
        """wcs's code track: the queued patch proposals, rebased on the incumbent when it changed; a new ask (when `ask`)
        when none is pending, a screen is left, a full screen ran since the last ask, wcs's asks (three) are not spent and
        the budget has a call and dollars left."""
        out = []
        for q in list(self.queue):
            pid = q.hypothesis.split('|', 1)[1]
            want = frozenset(self.chain) | {pid}
            if q.base != inc or self.ideas.get(q.hypothesis) != want:
                patch = self._rebase(engine, inc, q.patch, q.base)
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
        n = engine.full_screens()
        if out or not ask or left < 1 or n < 1 or self.last_ask_at == n or self.asks >= self.code_asks:
            return out
        if not self._can_code():
            return out                                   # (no model: no code track)
        if not self._budget_ok(engine):
            return out
        self.asks += 1
        self.last_ask_at = n
        self._allowance()
        try:
            hyps = self.proposer.hypotheses(engine, inc, ledger=self.model_view())
        except Exception as exc:                         # noqa: BLE001
            (getattr(engine, 'log', None) or (lambda m: None))(f'[search] {self.name}: the patch call failed: {str(exc)[:200]}')
            hyps = []
        self.llm_calls = self._llm_used()
        m = 0
        for h in hyps or []:
            if not isinstance(h, dict) or not h.get('patch') or (h.get('change') or {}):
                continue                                 # patches only
            if m >= self.patches_per_ask:
                break
            m += 1
            q = self._patch_proposal(engine, inc, h)
            if q is not None:
                self.queue.append(q)
                out.append(q)
        return out

    def _can_code(self) -> bool:
        return hasattr(self.proposer, 'hypotheses') and getattr(self.proposer, 'driver', True) is not None

    def _code_step(self, engine, inc, left, ask: bool = True):
        code = [q for q in self._code(engine, inc, left, ask=ask) if engine._fresh(q) and self._lowfi_ok(engine, q)]
        if code:
            self.code_screens += 1
            return code[0]
        return None

    # ------------------------------------------------------------ the configuration call and its admission
    def _ask(self, engine, inc, purpose: str = 'round') -> list[dict]:
        """One configuration call (within the budget): its configurations admitted (`_admit`), the round recorded;
        `last_got`: the configurations the model gave (None: no call was made)."""
        self.last_got = None
        if not hasattr(self.proposer, 'config_batch') or getattr(self.proposer, 'driver', True) is None:
            return []
        if not self._budget_ok(engine):
            return []
        self._allowance()
        history_rows = self._history(engine) if self.history_on else []
        try:
            ans = self.proposer.config_batch(engine, 'c0', k=self.k, history=history_rows, purpose=purpose)
        except Exception as exc:                         # noqa: BLE001 - a failed call is an empty round, not a dead search
            (getattr(engine, 'log', None) or (lambda m: None))(f'[search] {self.name}: the configuration call failed: '
                                                              f'{str(exc)[:200]}')
            ans = None
        self.llm_calls = self._llm_used()
        rows = ans.get('configs') if isinstance(ans, dict) else None
        rnd = len(self.rounds)
        got, out = 0, []
        for i, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict) or not isinstance(row.get('change'), dict):
                continue
            got += 1
            item = self._admit(engine, row['change'], source=purpose, rnd=rnd, rank=i,
                               gain=_num(row.get('gain_pct'), -90.0, 400.0), why=str(row.get('why') or '')[:160])
            if item is not None:
                out.append(item)
        self.rounds.append({'round': rnd, 'purpose': purpose, 'screens_done': engine.full_screens(), 'asked': self.k,
                            'got': got, 'admitted': len(out), 'history_rows': len(history_rows)})
        self.last_got = got
        return out

    def _admit(self, engine, change: dict, *, source: str, rnd: int, rank: int, gain=None, why: str = '',
               observe: bool = True) -> dict | None:
        """A proposed configuration into the archive (module docstring), or None (recorded in `rejected` when it is a
        real rejection, not a repeat). `gain`: its stated gain in percent, a wide observation of the time model unless
        `observe` is False (the prior's configurations: wcs's _init observed them already)."""
        ch, bad = {}, {}
        for k, v in (change or {}).items():
            if k in self.knobs:
                w = _match(v, self.knobs[k])
                if w is _NO_MATCH:
                    bad[k] = v
                else:
                    ch[k] = w
        words = {k: v for k, v in ch.items() if vk(v) != vk(self.ref.get(k))}
        if bad:                                          # (running the rest would run another configuration)
            self.rejected.append({'change': {**words, **bad}, 'source': source,
                                  'why': 'not a value of the knob table: ' + ', '.join(f'{k}={v}' for k, v in bad.items())})
            return None
        if not words:
            return None                                  # (nothing off the reference: the reference ran)
        legal = self._complete(engine, {**self.ref, **ch}, set(ch))
        if legal is None or any(vk(legal.get(k)) != vk(v) for k, v in ch.items()):
            self.rejected.append({'change': words, 'why': 'no legal configuration holds these values', 'source': source})
            return None
        cfg = {k: legal.get(k) for k in self.knobs}
        ids = tuple(self.chain)
        key = self._ckey(cfg, ids)
        kk = self._ckey(cfg, ())
        if key in self.tried or kk in self._ran_full() or any(a['key'] == key and a['status'] != 'open'
                                                              for a in self.archive):
            return None                                  # (a repeat: what ran in full - with any patches - or was proposed)
        again = next((a for a in self.archive if a['status'] == 'open' and self._ckey(a['cfg'], ()) == kk), None)
        if again is not None:                            # proposed again while still open: it moves to this call
            again.update(round=rnd, rank=rank, source=source, key=key, why=why or again['why'],
                         gain=again['gain'] if gain is None else math.log1p(max(-0.9, gain / 100.0)))
            return again
        p, why_p = self._pf(cfg, ids) if self.filter_on else (1.0, '')
        if p <= 0.0:
            self.rejected.append({'change': words, 'why': f'ruled out by the runs: {why_p}'[:200], 'source': source})
            return None
        g = None if gain is None else math.log1p(max(-0.9, gain / 100.0))
        item = {'cfg': cfg, 'key': key, 'round': rnd, 'rank': rank, 'gain': g, 'why': why, 'p': p, 'why_p': why_p[:160],
                'status': 'open', 'source': source}
        self.archive.append(item)
        if g is not None and observe:                    # the stated gain as a wide observation of the time model (as
            self.surr.observe(self._feats(cfg, ()), g, SD_CONFIG, tag=('config', self._ckey(cfg, ())))   # wcs's prior
        return item                                      # configurations; a measurement retires it)

    def _prior_items(self, engine, rnd: int) -> list[dict]:
        """The prior's whole configurations as a round (no call: wcs's prior call made them)."""
        out = []
        for i, c in enumerate((self.prior.configs if self.prior else None) or []):
            mu = c.get('mu')
            item = self._admit(engine, c.get('change') or {}, source='prior', rnd=rnd, rank=i,
                               gain=None if mu is None else 100.0 * math.expm1(mu), why=str(c.get('name') or ''),
                               observe=False)
            if item is not None:
                out.append(item)
        return out

    def _pf(self, cfg: dict, ids=()) -> tuple[float, str]:
        """The candidates' feasibility (the ledger, wcs's measured dominance, the pair table), and 0 for a configuration
        that holds every change of a run that failed for a reason other than memory (numerics, a crash, the compiler, a
        collective): one of those changes, or several together, made it fail."""
        p, why = super()._pf(cfg, ids)
        if p <= 0.0:
            return p, why
        bad = self._bad_sets()
        if bad:
            have = {(k, vk(cfg.get(k))) for k in self.knobs if vk(cfg.get(k)) != vk(self.ref.get(k))}
            have |= {(f'patch:{x}', 'true') for x in ids}
            for s, cid, kind in bad:
                if s <= have:
                    return 0.0, f'it holds every change of {cid}, which failed ({kind})'
        return p, why

    def _ran_full(self) -> set:
        """The knob configurations (patches left out) that ran in full; built once per observation."""
        stamp = len(self.runs)
        if getattr(self, '_full_stamp', None) != stamp:
            self._full = {self._ckey({k: (r.get('cfg') or {}).get(k) for k in self.knobs}, ())
                          for r in self.runs.values() if r.get('fidelity') == 'full'}
            self._full_stamp = stamp
        return self._full

    def _bad_sets(self) -> list[tuple]:
        """(the changes off the reference, the run, its class) of every run that failed for a reason other than memory;
        built once per observation."""
        stamp = (len(self.runs), len(self.failures))
        if getattr(self, '_bad_stamp', None) != stamp:
            out = []
            for cid, r in self.runs.items():
                if r.get('kind') in BAD_KINDS:
                    c = r.get('cfg') or {}
                    s = {(k, vk(c.get(k))) for k in self.knobs if vk(c.get(k)) != vk(self.ref.get(k))}
                    s |= {(k, 'true') for k, v in c.items() if isinstance(k, str) and k.startswith('patch:') and v}
                    if s:
                        out.append((frozenset(s), cid, r['kind']))
            self._bad, self._bad_stamp = out, stamp
        return self._bad

    def _open(self) -> list[dict]:
        """The archive's items not run yet (the key with the patches carried now), their feasibility refreshed."""
        ids = tuple(self.chain)
        out = []
        for a in self.archive:
            if a['status'] != 'open' or self._ckey(a['cfg'], ids) in self.tried:
                continue
            if self._refresh(a) > 0.0:
                out.append(a)
        return out

    def _refresh(self, item: dict) -> float:
        """The item's feasibility now (the runs change it); ruled out: closed and shown to the model."""
        if not self.filter_on:
            return item['p']
        p, why = self._pf(item['cfg'], tuple(self.chain))
        item['p'], item['why_p'] = p, why[:160]
        if p <= 0.0:
            item['status'] = 'ruled out'
            self.rejected.append({'change': self._words_of(item['cfg']), 'why': f'ruled out by the runs: {why}'[:200],
                                  'source': item['source']})
        return p

    def _words_of(self, cfg: dict) -> dict:
        return {k: cfg.get(k) for k in self.knobs if vk(cfg.get(k)) != vk(self.ref.get(k))}

    def _stated(self, item: dict) -> float:
        """The item's gain over the reference: as the model stated it, else the time model's."""
        if item.get('gain') is not None:
            return item['gain']
        return Surrogate.value(self._feats(item['cfg'], tuple(self.chain)), self.surr.mean())

    def _promote_q(self, engine, cid: str, *, kind: str, why: str):
        """The full run of a short run's own candidate `cid`: no configuration or overlay resubmitted (with proven patches
        carried, a resubmitted configuration comes back as a duplicate of that very candidate and is refused; C4_r1 1),
        the short run's patch ideas, words and gain kept; `observe` reads it as that configuration's full run."""
        sp = self.short_of[cid]
        self._n += 1
        hid = f'{kind}|{self._n}'
        ids = frozenset(self._patch_ideas(sp))
        self.ideas[hid] = ids
        self.promoted[hid] = sp
        cfg = self._cfg_of(sp)
        q = Proposal(change={}, patch=None, base=cid, mu=sp.mu, sigma=sp.sigma, p_feasible=0.99, mechanism=sp.mechanism,
                     hypothesis=hid, origin=self.name, fidelity=('full',), absolute=False)
        self.ledger[hid] = {'id': hid, 'kind': kind, 'mechanism': sp.mechanism, 'patches': sorted(ids), 'base': 'c0',
                            'change': self._words_of(cfg), 'predicted_gain_pct': round(100 * math.expm1(sp.mu), 2),
                            'p_feasible': 0.99, 'why_feasible': why[:200], 'plan': ['full'], 'status': 'open',
                            'promoted_from': cid}
        self.picks.append({'id': hid, 'kind': kind, 'screens_done': engine.full_screens(), 'mechanism': sp.mechanism,
                           'p_feasible': 0.99, 'predicted_gain_pct': round(100 * math.expm1(sp.mu), 2)})
        return q

    def _run_item(self, engine, inc, item: dict, *, kind: str, plan: tuple | None = None, probe_ok: bool = True):
        """A proposal for an archive item: the ladder's plan (probe first below its threshold, within the allowance; no
        short rung) unless given."""
        p = item['p']
        if plan is None:
            plan = self.ladder.plan(engine, p, warmup=self._warmup(item['cfg']) or None, probe=None if probe_ok else False,
                                    short=False)
        why = (item['why'] or item['source']) + (f'; {item["why_p"]}' if item.get('why_p') else '')
        q = self._knob_q(engine, inc, item['cfg'], kind=kind, plan=plan, p=p, why=why)
        if tuple(plan) == ('probe',):
            q.gate = no_gate                             # (a probe-only plan: no gate)
        item['status'], item['hid'] = 'proposed', q.hypothesis
        self.by_hid[q.hypothesis] = item
        return q

    # ------------------------------------------------------------ the history the model reads
    def _history(self, engine) -> list[dict]:
        ok, probes, fails = [], [], []
        for cid, r in self.runs.items():
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
            kind, st = r.get('kind'), r.get('status')
            y = self.short_y.get(cid) if st == 'short' else r.get('y')
            if st in ('screened', 'short') and y is not None and kind != 'thrash':
                row['outcome'] = f'x{math.exp(y):.3f} the reference speed' + (' (a short run)' if st == 'short' else '')
                ok.append((y, row))
                continue
            if kind == 'oom':
                o = m.get('oom') if isinstance(m.get('oom'), dict) else {}
                short = o.get('short_gib')
                row['outcome'] = 'out of memory' + (f', short by {short:.1f} GiB' if isinstance(short, (int, float)) else '')
            elif kind == 'thrash':
                row['outcome'] = 'thrash at the memory edge (slow near the card)'
            elif st == 'aborted':
                row['outcome'] = 'aborted: slower than the best so far' + (
                    f' (about x{math.exp(y):.3f} the reference speed)' if y is not None and not r.get('bound') else '')
            elif st == 'probed':
                row['outcome'] = 'its probe ran (it fits; the speed was not measured)'
                probes.append(row)
                continue
            elif kind in ('start_slow', 'start_cap', 'hang'):
                row['outcome'] = 'a slow start: stopped at its start cap'
            elif kind == 'slow_probe':
                row['outcome'] = 'its first updates ran much slower than the reference: not run in full'
            elif kind in ('inert_at_start', 'patch_inert'):
                row['outcome'] = 'its patch did not take effect'
            else:
                row['outcome'] = f'failed: {kind or st}'
            fails.append(row)
        rows = [row for _, row in sorted(ok, key=lambda t: -t[0])] + probes + fails
        rows += [{'change': x['change'], 'outcome': f'not run: {x["why"]}'} for x in self.rejected[-10:]]
        return rows[:HISTORY_MAX]

    # ------------------------------------------------------------ the tail and the fallback draw
    def _tail_step(self, engine, inc, left):
        """The tail as a set on the incumbent: per knob the value the time model rates highest above the incumbent's own,
        none that spends memory by a known move (declared or measured); full when it fits by the ladder's threshold,
        else probe first."""
        if self.tail.done or left < 1 or (self.tail_base is not None and self.tail_base[2] == 'waiting'):
            return None
        if self.tail_base is None:
            base_cfg = self._knob_cfg(engine, inc)
            mean = self.surr.mean()
            known = self._known()
            values = []
            for k in self.knobs:
                cur = mean.get((k, vk(base_cfg.get(k))), 0.0)
                for v in self.knobs[k]:
                    g = mean.get((k, vk(v)), 0.0) - cur
                    if vk(v) == vk(base_cfg.get(k)) or g <= 0.0:
                        continue
                    if self.mem.split(base_cfg, {**base_cfg, k: v}, known)['spends']:
                        continue
                    values.append((k, v, g, None))
            if not self.tail.start(values):
                return None
            self.tail_base = (base_cfg, (self.runs.get(inc) or {}).get('y') or 0.0, 'ready')
        cfg = {**self.tail_base[0], **{k: v for k, v, *_ in self.tail.testing}}
        legal = self._complete(engine, cfg, {k for k, *_ in self.tail.testing})
        ids = tuple(self.chain)
        if legal is None or self._ckey(legal, ids) in self.tried:
            self.tail.done = True
            return None
        p, why = self._pf(legal, ids)
        if p <= 0.0:
            self.tail.done = True
            return None
        plan = self.ladder.plan(engine, p, short=False)
        self.tail_base = (self.tail_base[0], self.tail_base[1], 'waiting')
        return self._knob_q(engine, inc, legal, kind='tail', plan=plan, p=p, why=why)

    def _knob_cfg(self, engine, cid) -> dict:
        c = engine.s.candidates[cid]
        cfg = self._mcfg(engine, cid)
        return {k: cfg.get(k, c.config.get(k)) for k in self.knobs}

    def _fallback(self, engine, inc, short: bool = False):
        """Nothing proposed is left: a draw weighted by feasibility and the time model (rpp's), through the ladder."""
        got = weighted_draw(self, engine, inc, a=1.0, T=0.02, salt=f'{self.name}|{len(self.rounds)}|{engine.full_screens()}',
                            p_min=None if self.ladder.affordable(engine, 'probe') else self.ladder.probe_below)
        if got is None:
            return None
        cfg, ids, p, why, info = got
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None, short=short)
        return self._knob_q(engine, inc, cfg, kind='draw', plan=plan, p=p, why=why)

    def _stalled(self) -> bool:
        return len(self.inc_hist) > STALL and len(set(self.inc_hist[-(STALL + 1):])) == 1

    # ------------------------------------------------------------ outcomes
    def observe(self, engine, p, out):
        hid = p.hypothesis or ''
        sp = self.promoted.get(hid)
        if sp is not None:                               # (the bookkeeping reads it as the configuration's own full run)
            p = dataclasses.replace(p, change=dict(sp.change or {}), patch=sp.patch, absolute=sp.absolute, base=sp.base)
        cls = super().observe(engine, p, out)
        if out.status == 'short' and out.cid and out.gain is not None:
            self.short_y[out.cid] = out.gain
            self.short_of[out.cid] = p
        item = self.by_hid.get(hid)
        if item is not None:
            if out.status == 'duplicate':
                item['status'] = 'repeat'
            else:
                item['status'] = out.status
                item['outcome'] = (cls or {}).get('kind') or out.status
                if item['outcome'] in BAD_KINDS:         # its stated gain no longer speaks for its values
                    tag = ('config', self._ckey(item['cfg'], ()))
                    self.surr.retire(lambda t, fs, _t=tag: t == _t)
                if out.status == 'screened' and out.cid:
                    item['y'] = (self.runs.get(out.cid) or {}).get('y')
                elif out.status == 'short' and out.cid:
                    item['y_short'] = out.gain
        if out.fidelity == 'full' and out.cid and out.status in ('screened', 'aborted', 'invalid'):
            self.inc_hist.append(engine.incumbent())
        if hid.startswith('tail|') and self.tail_base is not None:
            if out.status == 'duplicate':
                self.tail.done = True
            else:
                y = (self.runs.get(out.cid) or {}).get('y') if out.cid and out.status == 'screened' else None
                nxt = self.tail.observe(engine, None if y is None else y - self.tail_base[1])
                self.tail_base = (self.tail_base[0], self.tail_base[1], 'ready')
                if nxt is None:
                    self.tail.done = True
        return cls

    # ------------------------------------------------------------ report
    def llmopt_view(self) -> dict:
        def row(a):
            d = {k: a[k] for k in ('round', 'rank', 'source', 'status', 'outcome', 'stage') if a.get(k) is not None}
            d.update({'change': self._words_of(a['cfg']), 'p_feasible': round(a['p'], 3)})
            if a.get('gain') is not None:
                d['stated_gain_pct'] = round(100 * math.expm1(a['gain']), 2)
            if a.get('y_short') is not None:
                d['short_gain_pct'] = round(100 * math.expm1(a['y_short']), 2)
            if a.get('y') is not None:
                d['measured_gain_pct'] = round(100 * math.expm1(a['y']), 2)
            return d
        return {'rounds': self.rounds, 'archive': [row(a) for a in self.archive[-40:]], 'rejected': self.rejected[-20:],
                'tail': self.tail.describe(), 'llm_calls': self._llm_used(), 'llm_calls_max': self.calls_max,
                'llm_usd_max': self.usd_max, 'code_screens': self.code_screens, 'code_asks': self.asks}
