"""ltune: lambda-Tune-style search (possible_search_algorithm/10 section 04: Giannakouris and Trummer, "lambda-Tune:
Harnessing Large Language Models for Automated Database System Tuning") on the candidates' base (search/llmopt).

One model call generates about five diverse whole configurations (proposers.config_batch, purpose 'race'): the task
(its knobs, the workload invariants and constraints - the rules of the legal table -, the hardware) and the knob table.
They are admitted as opro's (legality; one the runs rule out is not run) and raced with increasing run lengths:

    probe     a configuration below 0.7 feasibility gets a probe-only run first (its feasibility, its measured peak); one
              that fails it is out (within the low-fidelity allowance; none left: it stays, by its feasibility); the
              feasibility is read again after every run of the race (an out-of-memory short run lowers the others');
              one below 0.05 is not raced while another reaches it
    short     each survivor gets a short run (its gain over the reference on the same updates; within the allowance)
    capping   a survivor whose short gain is below the best one's by max(2 sigma_short, 2 %) is cut - the deterministic
              part of ParamILS's capping: it cannot win the race by the short runs' own spread
    extend    the survivors by their short gain, best first, run in full while their short gain + sigma_short reaches
              the incumbent's gain (the ladder's promotion rule) - the short run's own candidate, nothing resubmitted
              (C4_r1 1); every full run is capped by the engine's early abort against the incumbent (the engine's rule,
              unchanged). Without short runs (none affordable, or ltune_short off) the survivors run in full in the
              order of feasibility x the stated gain. At most two of the race's configurations run in full in all
              (full_max; C4_r1 5).

After the race the remaining screens alternate between the code track (wcs's: patches only, the self-proof at the probe,
a proven patch a pseudo-knob; asked within the budget) and the local refinement of the winner: the tail (the small values
the time model rates positive, as one bundle on the incumbent), then the incumbent's neighbours (one or two knobs off it)
by feasibility x the time model's gain plus a quarter of its spread, through the rung ladder (probe -> short -> full, a
short run promoted only when it may beat the incumbent); nothing left: rpp's feasibility-weighted draw. Probes and short
runs stay within the low-fidelity allowance (GPU-hours, not screens). Budget: at most 10 model calls and 8 USD - the
race's one call, the code track's asks and their re-asks (with ltune_one_call off, the prior call too).

    ltune_one_call  the knob side's one call (the race), no prior call; off: wcs's prior call too (its means and memory
                    declarations for the feasibility and the local refinement, its configurations added to the race)
    ltune_probe     the race's probe-only runs below 0.7 feasibility; off: no probe in the race
    ltune_short     the short rung (the race's first run, the local refinement's ladder); off: full runs only
    ltune_cap       the race's capping of the short runs; off: only the extension rule cuts
    ltune_ledger    the memory ledger as the feasibility where the runs do not decide; off: wcs's
    ltune_code      wcs's code track after the race, alternating with the knob side
    ltune_tail      the tail bundle on the winner before its neighbours
    ltune_local     the winner's neighbours by feasibility x the time model; off: the feasibility-weighted draw

Nothing here names a knob, a model or a library.
"""
from __future__ import annotations

import math

from . import LTUNE_COMPONENTS
from .engine import Outcome
from .ladder import LOWFI_SHARE
from .llmopt import CALLS_MAX, K_BATCH, P_MIN, USD_MAX, LLMOptBase
from .wcs import Surrogate, vk
from .wcs import proposer_flags as wcs_flags

FULL_MAX = 2                    # raced configurations extended to full runs, at most
CAP_MIN = 0.02                  # the capping margin: at least 2 % (and 2 sigma_short)
LOCAL_DIST = 2                  # the neighbours: one or two knobs off the incumbent
LOCAL_EXPLORE = 0.25            # a neighbour's score: the time model's gain + this x its spread, x its feasibility
PRIOR_EXTRA = 2                 # with the prior call, its configurations join the race: at most this many beyond K


class LTune(LLMOptBase):
    name = 'ltune'

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        """wcs's flags, the code track's by `ltune_code`."""
        return wcs_flags(['wcs_code'] if 'ltune_code' in set(off or ()) else [])

    def __init__(self, proposer, *, off=(), k: int = K_BATCH, calls_max: int = CALLS_MAX, usd_max: float = USD_MAX,
                 facts=None, accounting=None, experience=None, kappa0: float = 0.5, seed: int = 42,
                 lowfi_share: float = LOWFI_SHARE, name: str | None = None, full_max: int = FULL_MAX):
        off = set(off or ())
        self.lt_on = {c: c not in off for c in LTUNE_COMPONENTS}
        off_wcs = (['wcs_prior'] if self.lt_on['ltune_one_call'] else []) + ([] if self.lt_on['ltune_code'] else ['wcs_code'])
        super().__init__(proposer, k=k, calls_max=calls_max, usd_max=usd_max, off_wcs=off_wcs,
                         ledger=self.lt_on['ltune_ledger'], short=self.lt_on['ltune_short'], lowfi_share=lowfi_share,
                         facts=facts, accounting=accounting, experience=experience, kappa0=kappa0, seed=seed)
        if name:
            self.name = name
        self.full_max = int(full_max)
        self.race: list[dict] | None = None
        self.race_done = False
        self.fulls = 0                                   # race configurations run in full
        self.fulls_raced = 0                             # ... of them after a short run
        self.turn = 'code'                               # after the race: the side whose turn it is
        self.race_log: list[dict] = []
        self.steps: list[dict] = []

    # ------------------------------------------------------------ the race
    def _build_race(self, engine, inc) -> None:
        items = self._ask(engine, inc, purpose='race')
        extra = []
        if self.on['wcs_prior'] and self.prior is not None and self.prior.configs:
            rnd = len(self.rounds)
            extra = self._prior_items(engine, rnd=rnd)
            self.rounds.append({'round': rnd, 'purpose': 'prior', 'screens_done': engine.full_screens(), 'asked': 0,
                                'got': len(self.prior.configs), 'admitted': len(extra), 'history_rows': 0})
        race = (items + extra)[:self.k + (PRIOR_EXTRA if extra else 0)]
        for a in self.archive:
            a['stage'] = 'new' if any(a is r for r in race) else 'not raced'
        self.race = race

    def _promote(self, engine, it) -> bool:
        """The ladder's promotion rule on the item's short run (gain + z sigma_short >= the incumbent's gain)."""
        o = Outcome(it.get('short_cid'), 'short', True, None, it['gs'], 'short', 0.0, [], [])
        return self.ladder.promote(engine, o, label=self._words(it['cfg'], ()))

    def _race_step(self, engine, inc, left):
        if self.race is None:
            self._build_race(engine, inc)
        race = self.race
        for it in race:                                  # feasibility re-read after every run of the race
            if it['stage'] in ('new', 'fit') and self._refresh(it) <= 0.0:
                it['stage'] = 'out'
        live = [it for it in race if it['stage'] in ('new', 'fit')]
        if any(it['p'] >= P_MIN for it in live):         # below P_MIN only when nothing else is left
            for it in live:
                if it['p'] < P_MIN:
                    it['stage'] = 'unlikely'
        for it in race:                                  # the probes
            if it['stage'] not in ('new', 'fit'):
                continue
            p = it['p']
            if self.lt_on['ltune_probe'] and not it.get('probed') and p < self.ladder.probe_below \
                    and self.ladder.affordable(engine, 'probe'):
                it['stage'] = 'probing'
                return self._run_item(engine, inc, it, kind='race', plan=('probe',))
            it['stage'] = 'fit'
        for it in race:                                  # the short runs
            if it['stage'] != 'fit':
                continue
            if self._refresh(it) <= 0.0:
                it['stage'] = 'out'
                continue
            long_warm = self._warmup(it['cfg']) > int(getattr(engine, 'warmup', 3) or 3)
            if self.lt_on['ltune_short'] and not long_warm and self.ladder.affordable(engine, 'short'):
                it['stage'] = 'shorting'
                self.ladder.counts['short'] += 1
                return self._run_item(engine, inc, it, kind='race', plan=('short',))
            it['stage'] = 'unraced'
        raced = [it for it in race if it['stage'] == 'raced']
        if raced and self.lt_on['ltune_cap']:            # the capping: below the best short gain by the margin
            best = max(it['gs'] for it in raced)
            margin = max(2 * self.ladder.sigma_short(engine), math.log1p(CAP_MIN))
            for it in raced:
                if it['gs'] < best - margin:
                    it['stage'] = 'capped'
        for it in sorted((it for it in race if it['stage'] == 'raced'), key=lambda it: -it['gs']):
            if self.fulls >= self.full_max:
                it['stage'] = 'not extended'
                continue
            if self._refresh(it) <= 0.0:
                it['stage'] = 'out'
                continue
            if not self._promote(engine, it):
                it['stage'] = 'not promoted'
                continue
            it['stage'] = 'full'
            self.fulls += 1
            self.fulls_raced += 1
            if it.get('short_cid') in self.short_of:     # its short run's own candidate in full (C4_r1 1)
                q = self._promote_q(engine, it['short_cid'], kind='race', why='the race: its short run may win')
                it['status'], it['hid'] = 'proposed', q.hypothesis
                self.by_hid[q.hypothesis] = it
                return q
            return self._run_item(engine, inc, it, kind='race', plan=('full',))
        rest = [it for it in race if it['stage'] == 'unraced']
        for it in sorted(rest, key=lambda it: (-(it['p'] * self._stated(it)), it['rank'])):
            if self._refresh(it) <= 0.0:
                it['stage'] = 'out'
                continue
            if self.fulls >= self.full_max:              # every race full run counts against full_max (C4_r1 5)
                it['stage'] = 'not extended'
                continue
            it['stage'] = 'full'
            self.fulls += 1
            plan = ('full',) if it.get('probed') or not self.lt_on['ltune_probe'] else None
            return self._run_item(engine, inc, it, kind='race', plan=plan)
        return None

    # ------------------------------------------------------------ after the race
    def _code_side(self, engine, inc, left):
        if not self.on['wcs_code']:
            return None
        q = self._code_step(engine, inc, left)
        if q is not None:
            self.turn = 'knob'
            self.steps.append({'screens_done': engine.full_screens(), 'pick': 'code'})
        return q

    def _knob_side(self, engine, inc, left):
        q, pick = (self._tail_step(engine, inc, left), 'tail') if self.lt_on['ltune_tail'] else (None, '')
        if q is None and self.lt_on['ltune_local']:
            q, pick = self._local(engine, inc), 'local'
        if q is not None:
            self.turn = 'code'
            self.steps.append({'screens_done': engine.full_screens(), 'pick': pick})
        return q

    def _local(self, engine, inc):
        """The incumbent's best neighbour (one or two knobs off it, legal, not run): feasibility x (the time model's gain
        over the incumbent + LOCAL_EXPLORE x its spread); below P_MIN only when nothing else is left; once no probe is
        left, only what is predicted to fit when any is; through the ladder."""
        base = self._knob_cfg(engine, inc)
        ids = tuple(self.chain)
        mean = self.surr.mean()
        y_inc = Surrogate.value(self._incumbent_feats(engine, inc), mean)
        rows = []
        for cfg in self._table(engine):
            d = sum(1 for k in self.knobs if vk(cfg.get(k)) != vk(base.get(k)))
            if not 1 <= d <= LOCAL_DIST or self._ckey(cfg, ids) in self.tried:
                continue
            p, why = self._pf(cfg, ids)
            if p <= 0.0:
                continue
            x = self._feats(cfg, ids)
            mu = Surrogate.value(x, mean) - y_inc + LOCAL_EXPLORE * math.sqrt(max(1e-6, self.surr.var(x)))
            rows.append((p * mu if mu > 0 else mu, -d, p, cfg, why))
        if any(r[2] >= P_MIN for r in rows):
            rows = [r for r in rows if r[2] >= P_MIN]
        if not self.ladder.affordable(engine, 'probe') and any(r[2] >= self.ladder.probe_below for r in rows):
            rows = [r for r in rows if r[2] >= self.ladder.probe_below]
        if not rows:
            return None
        _, _, p, cfg, why = max(rows, key=lambda r: (r[0], r[1]))
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None)
        return self._knob_q(engine, inc, cfg, kind='local', plan=plan, p=p, why=why or 'a neighbour of the incumbent')

    # ------------------------------------------------------------ proposals
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
            if engine._fresh(q):
                return [q]
        if not self.race_done:
            q = self._race_step(engine, inc, left)
            if q is not None:
                return [q]
            self.race_done = True
            self.steps.append({'screens_done': engine.full_screens(), 'pick': 'race done', 'incumbent': inc})
        sides = (self._code_side, self._knob_side) if self.turn == 'code' else (self._knob_side, self._code_side)
        for side in sides:
            q = side(engine, inc, left)
            if q is not None:
                return [q]
        q = self._fallback(engine, inc, short=self.lt_on['ltune_short'])
        if q is not None:
            self.steps.append({'screens_done': engine.full_screens(), 'pick': 'draw'})
        return [q] if q is not None else []

    # ------------------------------------------------------------ outcomes
    def observe(self, engine, p, out):
        cls = super().observe(engine, p, out)
        hid = p.hypothesis or ''
        it = self.by_hid.get(hid)
        if it is not None and it.get('stage') in ('probing', 'shorting', 'full'):
            st = it['stage']
            if out.status == 'duplicate':
                it['stage'] = 'out'
            elif st == 'probing':
                it['probed'] = out.status == 'probed'
                it['stage'] = 'fit' if it['probed'] else 'out'
            elif st == 'shorting':
                if out.status == 'short' and out.gain is not None:
                    it['stage'], it['gs'], it['short_cid'] = 'raced', out.gain, out.cid
                else:
                    it['stage'] = 'out'
            else:
                it['stage'] = 'done'
            self.race_log.append({'id': hid, 'rung': st, 'status': out.status, 'candidate': out.cid,
                                  **({'gain_pct': round(100 * math.expm1(out.gain), 2)} if out.gain is not None else {}),
                                  'outcome': (cls or {}).get('kind') or out.status})
        elif out.fidelity == 'short' and p.absolute and tuple(p.fidelity)[-1:] == ('short',) and it is None:
            if self.ladder.promote(engine, out, label=p.mechanism) and out.cid in self.short_of:   # (the local ladder:
                self.pending.append(self._promote_q(engine, out.cid, kind='promoted',            # its own candidate,
                                                    why='its short run may beat the incumbent'))  # C4_r1 1)
        return cls

    # ------------------------------------------------------------ report
    def describe(self) -> dict:
        view, learned, prop = self.candidate_view()
        out = {'name': self.name, 'proposer': prop,
               'ltune': {'name': self.name, 'off': sorted(k for k, v in self.lt_on.items() if not v), **view,
                         **self.llmopt_view(), 'race_full_runs': self.fulls, 'race_log': self.race_log[-40:],
                         'steps': self.steps[-60:]}}
        if learned:
            out['learned'] = learned                     # D4: written back for a later arm
        return out
