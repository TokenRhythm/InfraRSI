"""opro: OPRO-style LLM-as-optimizer over whole configurations (possible_search_algorithm/10 section 04: Yang et al.,
"Large Language Models as Optimizers") on the candidates' base (search/llmopt).

Each round the model gets the task (its knobs and their values, the workload invariants and the constraints - the rules
of the legal table -, the hardware), the knob table and the history of every measured configuration, best first: its
outcome (the measured speed over the reference; out of memory and by how much, thrash, a slow start, aborted as slower,
a crash) and its measured peak, then the proposals ruled out before running with why. It returns K whole configurations
with short reasons (proposers.config_batch). They are filtered for legality and for measured / arithmetic feasibility
(the memory ledger, wcs's measured dominance, the pair table: one the runs rule out is not run and is shown to the
model as such) and ordered by feasibility x the stated gain - the latest round first, one below 0.05 feasibility only
when nothing else is left (a new round is asked first when the budget pays for one), once the low-fidelity allowance has
no probe left only what is predicted to fit when any is. The best runs - probed first when its feasibility is below 0.7,
within the allowance - and the loop goes on. A round runs the best m of its batch, m = the screens left over the rounds
the budget still pays for, so that the rounds span the search; a batch with nothing admissible is asked again once (its
rejections are in the history). The first round is the prior call's own whole configurations (wcs's prior call, which
also gives the time model its means).

The code track is wcs's (patches only, the self-proof at the probe, a proven patch a pseudo-knob), interleaved when the
knob side stalls: the incumbent unchanged over the last two full screens, two knob screens since the last ask. One
budget for both sides: at most 10 model calls and 8 USD (the prior, the configuration calls, the patch asks, their
re-asks), of which the rounds leave 2 calls to the code track (at most its asks left of wcs's three) while 3 or more
screens are left. When no proposal is left and no round can be paid: the tail (the small values the time model rates
positive as one bundle on the incumbent), then rpp's feasibility-weighted draw.

    opro_prior    the prior call: its configurations are round 0, its means the time model; off: a flat prior
    opro_history  the history in every call; off: the task and the knob table only (independent batches)
    opro_filter   the feasibility filter and order; off: legality only, the model's order
    opro_ledger   the memory ledger as the feasibility where the runs do not decide; off: wcs's
    opro_probe    probe first below 0.7 feasibility; off: full runs only (a code proof keeps its probe)
    opro_code     wcs's code track when the knob side stalls
    opro_tail     the tail bundle before the draw when nothing is proposed; off: the draw only

Nothing here names a knob, a model or a library.
"""
from __future__ import annotations

import math

from . import OPRO_COMPONENTS
from .ladder import LOWFI_SHARE
from .llmopt import CALLS_MAX, K_BATCH, P_MIN, STALL, USD_MAX, LLMOptBase
from .wcs import proposer_flags as wcs_flags

CODE_CALLS = 2                  # calls the rounds leave to the code track (its asks left, while 3 or more screens are left)
ASKS_PER_STEP = 2               # configuration calls in one step at most (a batch with nothing admissible: once more)


class OPRO(LLMOptBase):
    name = 'opro'

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        """wcs's flags, the code track's by `opro_code`."""
        return wcs_flags(['wcs_code'] if 'opro_code' in set(off or ()) else [])

    def __init__(self, proposer, *, off=(), k: int = K_BATCH, calls_max: int = CALLS_MAX, usd_max: float = USD_MAX,
                 facts=None, accounting=None, experience=None, kappa0: float = 0.5, seed: int = 42,
                 lowfi_share: float = LOWFI_SHARE, name: str | None = None):
        off = set(off or ())
        self.opro_on = {c: c not in off for c in OPRO_COMPONENTS}
        off_wcs = [w for w, c in (('wcs_prior', 'opro_prior'), ('wcs_code', 'opro_code')) if not self.opro_on[c]]
        super().__init__(proposer, k=k, calls_max=calls_max, usd_max=usd_max, off_wcs=off_wcs,
                         ledger=self.opro_on['opro_ledger'], short=False, lowfi_share=lowfi_share, facts=facts,
                         accounting=accounting, experience=experience, kappa0=kappa0, seed=seed)
        if name:
            self.name = name
        self.history_on = self.opro_on['opro_history']
        self.filter_on = self.opro_on['opro_filter']
        if not self.opro_on['opro_probe']:
            self.ladder.probe_below = 0.0                # (no knob configuration is probed first)
        self.round_left = 0                              # configurations the current round may still run
        self.knob_since_ask = 0                          # knob full screens since the code track's last ask
        self.steps: list[dict] = []                      # each step's pick and why

    def _init(self, engine) -> None:
        super()._init(engine)
        if self.on['wcs_prior'] and self.prior is not None and self.prior.configs:
            items = self._prior_items(engine, rnd=0)
            self.rounds.append({'round': 0, 'purpose': 'prior', 'screens_done': 0, 'asked': 0,
                                'got': len(self.prior.configs), 'admitted': len(items), 'history_rows': 0})
            self.round_left = self._per_round(engine, self._left(engine)) if items else 0

    # ------------------------------------------------------------ the rounds
    def _rounds_payable(self, engine, left: int) -> int:
        """The configuration calls the budget still pays for (the code track's calls held back while it may use them)."""
        if self._usd(engine) >= self.usd_max:
            return 0
        reserve = max(0, min(CODE_CALLS, self.code_asks - self.asks)) if self.on['wcs_code'] and self._can_code() \
            and left >= 3 else 0
        return max(0, self._calls_left() - reserve)

    def _per_round(self, engine, left: int) -> int:
        """The configurations a round runs: the screens left over this round and the rounds still payable after it."""
        return max(1, math.ceil(left / (self._rounds_payable(engine, left) + 1)))

    def _score(self, a: dict) -> float:
        g = self._stated(a)
        return a['p'] * g if g > 0 else g

    def _ranked(self, engine) -> list[dict]:
        """The open proposals in the order they run (module docstring)."""
        items = self._open()
        if not items:
            return []
        if self.filter_on:
            if any(a['p'] >= P_MIN for a in items):
                items = [a for a in items if a['p'] >= P_MIN]
            if not self.ladder.affordable(engine, 'probe') and any(a['p'] >= self.ladder.probe_below for a in items):
                items = [a for a in items if a['p'] >= self.ladder.probe_below]
        top = max(a['round'] for a in items)
        if self.filter_on:
            return sorted(items, key=lambda a: (a['round'] != top, -self._score(a), a['round'], a['rank']))
        return sorted(items, key=lambda a: (a['round'] != top, -a['round'], a['rank']))

    def _round_step(self, engine, inc, left):
        items, asks, empty = self._ranked(engine), 0, False
        while (self.round_left <= 0 or not items or empty or self.filter_on and items[0]['p'] < P_MIN) \
                and asks < ASKS_PER_STEP and self._rounds_payable(engine, left) > 0:
            asks += 1
            new = self._ask(engine, inc)
            self.round_left = self._per_round(engine, left)
            empty = not new
            items = self._ranked(engine)
            if empty and not self.last_got:
                break                                    # (no call was made, or the model gave nothing: no re-ask)
        if not items:
            return None
        a = items[0]
        self.round_left -= 1
        self.steps.append({'screens_done': engine.full_screens(), 'pick': 'round', 'round': a['round'], 'rank': a['rank'],
                           'p_feasible': round(a['p'], 3)})
        return self._run_item(engine, inc, a, kind='opro')

    def _code_due(self) -> bool:
        return self._stalled() and self.knob_since_ask >= STALL

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
        if self.on['wcs_code'] and (self.queue or self._code_due()):
            n0 = self.asks
            q = self._code_step(engine, inc, left, ask=self._code_due())
            if self.asks != n0:
                self.knob_since_ask = 0
                self.steps.append({'screens_done': engine.full_screens(), 'pick': 'code ask', 'stalled': self._stalled()})
            if q is not None:
                return [q]
        q = self._round_step(engine, inc, left)
        if q is not None:
            return [q]
        if self.opro_on['opro_tail']:
            q = self._tail_step(engine, inc, left)
            if q is not None:
                self.steps.append({'screens_done': engine.full_screens(), 'pick': 'tail'})
                return [q]
        q = self._fallback(engine, inc)
        if q is not None:
            self.steps.append({'screens_done': engine.full_screens(), 'pick': 'draw'})
        return [q] if q is not None else []

    # ------------------------------------------------------------ outcomes
    def observe(self, engine, p, out):
        cls = super().observe(engine, p, out)
        if out.fidelity == 'full' and out.cid and not (p.hypothesis or '').startswith('patch|'):
            self.knob_since_ask += 1
        return cls

    # ------------------------------------------------------------ report
    def describe(self) -> dict:
        view, learned, prop = self.candidate_view()
        out = {'name': self.name, 'proposer': prop,
               'opro': {'name': self.name, 'off': sorted(k for k, v in self.opro_on.items() if not v), **view,
                        **self.llmopt_view(), 'steps': self.steps[-60:]}}
        if learned:
            out['learned'] = learned                     # D4: written back for a later arm
        return out
