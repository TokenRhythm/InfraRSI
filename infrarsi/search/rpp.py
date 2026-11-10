"""rpp and rpp-ctl: Random++ (possible_search_algorithm/20 section 6) - random's smallest generalisation, and its
control setting.

Each step, when no rung of an earlier draw is pending, one configuration is drawn from the legal table less what ran,
with probability proportional to

    w(c) = P_feas(c)^a x exp(mu'(c) / T_k),   T_k = T0 / (1 + k)

(P_feas: the memory ledger - a configuration that ran is its own result - x the pair table, search/cand_base; mu': wcs's
surrogate gain over the incumbent, times P_feas when it is a gain - a run that does not fit gains nothing; k: the valid
full runs so far; T0: the prior's median spread, at least 0.02); with a >= 1 a configuration below 0.05 is not drawn
while any reaches it, and once the low-fidelity allowance has no probe left only one predicted to fit (P_feas >= 0.7) is
drawn when any is. The weights are those of every configuration left (CAND_r1 C2: no pre-sampled pool), so the draw is
the stated distribution and the arm ends only when the table is spent.
The step's random numbers are seeded by (seed, step): rpp arms on one task share them. The
draw then climbs the rung ladder (search/ladder): probe first below 0.7, a short run, the full run only when the short
run may beat the incumbent.

    rpp       a = 1, T annealed, the model's prior, the experience store, the code track (wcs's, unchanged)
    rpp-ctl   a = 1, T = inf - filtered and weighted by feasibility only, the same ladder - with no model call, no
              experience store and no code track: the control the literature asks for (random + the same filter + the
              same rungs), so that a candidate's edge over random can be split into "fewer failed runs" and "search"

a = 0 and T = inf is uniform without replacement over the legal table: random's distribution (not the draw sequence of
`knob_space.random_search`, which decodes mixed-radix indices). Nothing here names a knob, a model or a library.
"""
from __future__ import annotations

import math
import random

from . import RPP_COMPONENTS
from .cand_base import CandidateBase, median_sd
from .ladder import LOWFI_SHARE
from .wcs import Surrogate, proposer_flags as wcs_flags

P_MIN = 0.05                    # with a >= 1, a configuration below this feasibility is not drawn while any reaches it
T0_MIN = 0.02


class RPP(CandidateBase):
    name = 'rpp'

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        """wcs's flags, the code track's by `rpp_code`."""
        return wcs_flags(['wcs_code'] if 'rpp_code' in set(off or ()) else [])

    def __init__(self, proposer, *, off=(), a: float = 1.0, temperature: float | None = None, name: str | None = None,
                 facts=None, accounting=None, experience=None, kappa0: float = 0.5, seed: int = 42,
                 lowfi_share: float = LOWFI_SHARE):
        off = set(off or ())
        self.rpp_on = {k: k not in off for k in RPP_COMPONENTS}
        off_wcs = [w for w, r in (('wcs_prior', 'rpp_prior'), ('wcs_code', 'rpp_code')) if not self.rpp_on[r]]
        super().__init__(proposer, off_wcs=off_wcs, ledger=self.rpp_on['rpp_ledger'], short=self.rpp_on['rpp_ladder'],
                         lowfi_share=lowfi_share, facts=facts, accounting=accounting, experience=experience,
                         kappa0=kappa0, seed=seed)
        if name:
            self.name = name
        self.a = float(a)
        self.temperature = temperature                   # None: annealed from T0; math.inf: no weighting by the time model
        self.T0 = None
        self.seed = int(seed)
        self.k_valid = 0
        self.draws: list[dict] = []

    def _init(self, engine) -> None:
        super()._init(engine)
        self.T0 = max(T0_MIN, median_sd(self.prior.levels if self.prior else {}))

    def _temperature(self) -> float:
        if not self.rpp_on['rpp_anneal'] or self.temperature == math.inf:
            return math.inf
        if self.temperature is not None:
            return float(self.temperature)
        return self.T0 / (1 + self.k_valid)

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
        if self.on['wcs_code']:
            code = [q for q in self._code(engine, inc, left) if engine._fresh(q) and self._lowfi_ok(engine, q)]
            if code:
                return [code[0]]
        q = self._draw(engine, inc)
        return [q] if q is not None else []

    def _draw(self, engine, inc):
        got = weighted_draw(self, engine, inc, a=self.a, T=self._temperature(), salt=len(self.draws),
                            p_min=None if self.ladder.affordable(engine, 'probe') else self.ladder.probe_below)
        if got is None:
            return None
        cfg, ids, p, why, info = got
        plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None)
        q = self._knob_q(engine, inc, cfg, kind='draw', plan=plan, p=p, why=why)
        self.draws.append({'id': q.hypothesis, 'screens_done': engine.full_screens(), 'p_feasible': round(p, 3),
                           'plan': list(plan), **info})
        return q

    # ------------------------------------------------------------ outcomes
    def observe(self, engine, p, out):
        super().observe(engine, p, out)
        if out.status == 'screened':
            self.k_valid += 1
        hid = p.hypothesis or ''
        if out.fidelity == 'short' and p.absolute and tuple(p.fidelity)[-1] == 'short':
            if self.ladder.promote(engine, out, label=p.mechanism):
                q = self._knob_q(engine, engine.incumbent(), self._cfg_of(p), kind='promoted', plan=('full',), p=1.0,
                                 why='its short run may beat the incumbent')
                self.pending.append(q)
        for d in self.draws:
            if d['id'] == hid:
                d['outcome'] = out.status

    # ------------------------------------------------------------ report
    def describe(self) -> dict:
        view, learned, prop = self.candidate_view()
        T = self._temperature() if self.T0 is not None else None
        out = {'name': self.name, 'proposer': prop,
               'rpp': {'name': self.name, 'a': self.a, 'temperature': 'inf' if T is None or not math.isfinite(T) else round(T, 4),
                       'T0': self.T0, 'off': sorted(k for k, v in self.rpp_on.items() if not v),
                       'draws': self.draws[-60:], **view}}
        if learned:
            out['learned'] = learned                     # D4: written back for a later arm
        return out


def weighted_draw(st, engine, inc, *, a: float, T: float, salt, p_min: float | None = None) -> tuple | None:
    """One configuration of a wcs-based candidate's table less what ran, drawn with probability proportional to
    P_feas^a x exp(mu' / T) over every configuration left (T = inf: no time weighting; a = 0: no feasibility), mu' the
    surrogate's gain over the incumbent, times P_feas when it is a gain (a run that does not fit gains nothing); with
    a >= 1 the configurations below P_MIN drop out while any reaches it (none does: the draw is among all, by weight - the
    arm does not end on it), and below `p_min` too when one is given and some configuration reaches it (no probe left:
    only what is predicted to fit); a weight of 0 (ruled out by the runs) is never drawn. The random numbers seeded by
    (the strategy's seed, `salt`). (cfg, ids, p, why, info) or None when nothing is left to draw."""
    cands = st._candidates(engine, inc)
    if not cands:
        return None
    rng = random.Random(f'{st.seed}|{salt}')
    mean = st.surr.mean() if math.isfinite(T) else None
    y_inc = Surrogate.value(st._incumbent_feats(engine, inc), mean) if mean is not None else 0.0
    rows = []
    for cfg, ids in cands:
        if a > 0:
            p, why = st._pf(cfg, ids)
            w = p ** a
        else:
            p, why, w = 1.0, '', 1.0
        if mean is not None and w > 0:
            mu = Surrogate.value(st._feats(cfg, ids), mean) - y_inc
            mu = p * mu if mu > 0 else mu
            w *= math.exp(max(-50.0, min(50.0, mu / T)))
        if w > 0:
            rows.append((w, cfg, ids, p, why))
    if a >= 1:
        rows = [r for r in rows if r[3] >= P_MIN] or rows
    if a >= 1 and p_min is not None and any(r[3] >= p_min for r in rows):
        rows = [r for r in rows if r[3] >= p_min]
    total = sum(r[0] for r in rows)
    if not rows:
        return None
    x = rng.random() * total
    for w, cfg, ids, p, why in rows:
        x -= w
        if x <= 0:
            break
    return cfg, ids, p, why, {'weight_share': round(w / total, 4), 'pool': len(cands), 'eligible': len(rows),
                              'temperature': None if not math.isfinite(T) else round(T, 4)}
