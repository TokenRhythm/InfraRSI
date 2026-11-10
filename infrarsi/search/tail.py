"""The tail as a set: the small knobs screened together (possible_search_algorithm/20 section 4) - a shared component.

On the incumbent, every small knob value the time model rates positive goes into one configuration (one full screen).
Its gain over the configuration it was added to decides: at least max(2 %, 2 sigma_pair) - accepted; a regression of
that size - one bisection by family (the prior's labels; without them, halves by the time model's mean), at most
`diag_max` more screens; within the threshold - nothing more (the values stay as they were). No recovery guarantee is
claimed: signed effects can cancel inside a bundle (the literature's point); it is a bounded bundle with a bounded
diagnosis. The tail only decides what to test next; the host strategy builds the proposals and reports the gains.

    tail = TailScreen()
    members = tail.start(values)                 # values: [(knob, value, mu, family)]; the members to test, or None
    members = tail.observe(engine, gain)         # the gain over the configuration they were added to; the next members
"""
from __future__ import annotations

import math

THRESHOLD_MIN = 0.02            # a bundle's gain is read only from this much (and 2 sigma_pair) up
DIAG_MAX = 2                    # diagnostic screens after a regression, at most


class TailScreen:
    def __init__(self, *, threshold_min: float = THRESHOLD_MIN, diag_max: int = DIAG_MAX):
        self.threshold_min, self.diag_max = float(threshold_min), int(diag_max)
        self.members: list[tuple] = []
        self.testing: list[tuple] | None = None
        self.queue: list[list[tuple]] = []
        self.accepted: list[tuple] = []
        self.history: list[dict] = []
        self.diag = 0
        self.done = False

    def threshold(self, engine) -> float:
        return max(math.log1p(self.threshold_min), 2 * engine.sigma_pair())

    def start(self, values) -> list[tuple] | None:
        """The bundle: one value per knob (the best by the time model's mean), those with a positive mean."""
        best: dict = {}
        for k, v, mu, fam in values or []:
            if mu is not None and mu > 0 and (k not in best or mu > best[k][2]):
                best[k] = (k, v, mu, fam)
        self.members = sorted(best.values(), key=lambda m: -m[2])
        if not self.members:
            self.done = True
            return None
        self.testing = list(self.members)
        return self.testing

    @staticmethod
    def _split(members: list[tuple]) -> tuple[list, list]:
        """Two groups: by family (families kept together, the higher-rated families first) when there are two or more
        families; else the halves by the time model's mean."""
        fams: dict = {}
        for m in members:
            fams.setdefault(m[3] or '', []).append(m)
        if len(fams) >= 2:
            order = sorted(fams.values(), key=lambda g: -sum(m[2] for m in g))
            a, b = [], []
            for g in order:
                (a if sum(m[2] for m in a) <= sum(m[2] for m in b) else b).extend(g)
            return a, b
        half = (len(members) + 1) // 2
        return members[:half], members[half:]

    def observe(self, engine, gain: float | None) -> list[tuple] | None:
        """The gain of `testing` over the configuration it was added to (None: it failed); the next members to test, or
        None when the tail is done."""
        thr = self.threshold(engine)
        g = gain if gain is not None else -math.inf
        verdict = 'accepted' if g >= thr else 'regressed' if g <= -thr else 'within'
        self.history.append({'members': [f'{k}={v}' for k, v, *_ in self.testing or []], 'diagnostic': self.diag,
                             **({'gain_pct': round(100 * math.expm1(gain), 2)} if gain is not None else {'gain_pct': None}),
                             'threshold_pct': round(100 * math.expm1(thr), 2), 'verdict': verdict})
        if verdict == 'accepted':
            self.accepted = list(self.testing or [])
            self.done = True
            return None
        if verdict == 'regressed' and self.diag < self.diag_max and len(self.testing or []) >= 2 and not self.queue:
            a, b = self._split(self.testing)
            self.queue = [g_ for g_ in (a, b) if g_]
        if self.queue and self.diag < self.diag_max:
            self.diag += 1
            self.testing = self.queue.pop(0)
            return self.testing
        self.done = True
        return None

    def describe(self) -> dict:
        return {'members': [f'{k}={v}' for k, v, *_ in self.members], 'accepted': [f'{k}={v}' for k, v, *_ in self.accepted],
                'diagnostic_screens': self.diag, 'history': self.history, 'done': self.done}
