"""Search budget (design 1.6): screens, LLM tokens (all roles), wall clock and GPU-hours; any exhausted -> stop.

Loading, compilation, failed and verification screens count. GPU-hours for the locked scoring pair
({FR, candidate} on plan B) are part of the search budget, so admission keeps a reserve of
`scoring_reserve_screens` mean screen costs.
"""
from __future__ import annotations

import dataclasses
import math
import time

from .state import Stateful


# Finite sentinel for optional project caps; exact in JSON/IEEE-754 and large enough
# to be unbounded in practice while preserving the engine's numeric budget protocol.
UNLIMITED = 2 ** 53 - 1


def target_value(name: str, value):
    """Validate a stopping target and normalize MFU percentages to fractions."""
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if name == 'target_mfu':
        if 1 < value <= 100:
            value /= 100
        if not 0 < value <= 1:
            raise ValueError('target_mfu must be in (0, 1] or a percentage in (1, 100]')
    elif value <= 1:
        raise ValueError('target_speedup must be greater than 1.0')
    return value


@dataclasses.dataclass(frozen=True)
class Budget:
    screens: int
    llm_tokens: int
    wall_seconds: float
    gpu_hours: float
    llm_usd: float | None = None
    scoring_reserve_screens: float = 2.0
    target_speedup: float | None = None
    target_mfu: float | None = None

    def __post_init__(self):
        for k in ('screens', 'llm_tokens', 'wall_seconds', 'gpu_hours'):
            v = getattr(self, k)
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError(f'budget {k}={v!r}')
        for k in ('target_speedup', 'target_mfu'):
            object.__setattr__(self, k, target_value(k, getattr(self, k)))

    @classmethod
    def from_task(cls, task, *, wall_multiplier: float = 1.0, run: dict | None = None, **overrides) -> 'Budget':
        """Task-spec search_budget; a RUN_MATRIX row may lower (never raise) screens/tokens."""
        raw = task.raw if hasattr(task, 'raw') else task
        sb = raw['search_budget']
        screens, tokens = int(sb['screens']), int(sb['llm_tokens'])
        if run:
            if run.get('screens') is not None:
                screens = min(screens, int(run['screens']))
            if run.get('llm_tokens_cap') is not None:
                tokens = min(tokens, int(run['llm_tokens_cap']))
        kw = dict(screens=screens, llm_tokens=tokens, wall_seconds=float(sb['wall_hours']) * 3600 * wall_multiplier,
                  gpu_hours=float(sb['gpu_hours_cap']), target_speedup=sb.get('target_speedup'),
                  target_mfu=sb.get('target_mfu'))
        kw.update(overrides)
        return cls(**kw)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class VirtualClock:
    """Deterministic clock for tests: runners and scripted backends advance it."""
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


class SearchLedger(Stateful):
    # the budget and the clock are the resuming run's; the wall clock goes on from the wall used so far (below)
    CHECKPOINT_LIVE = ('budget', 'clock', 't0', 'allow_token_free')

    def __init__(self, budget: Budget, clock=None):
        self.budget, self.clock = budget, clock or time.monotonic
        self.allow_token_free = False  # Engine enables this only for strategies with a zero-token proposal path
        self.t0 = self.clock()
        self.used = {'screens': 0, 'llm_tokens': 0, 'llm_usd': 0.0, 'gpu_hours': 0.0}
        self.accounting_complete = True
        self.screen_costs: list[tuple[float, float]] = []      # (gpu_hours, wall_seconds) per screen
        self.prior: tuple[float, float] | None = None          # screen cost estimate before the first screen

    def wall(self) -> float:
        return self.clock() - self.t0

    def state_dict(self) -> dict:
        out = super().state_dict()
        out['wall_used'] = self.wall()
        return out

    def load_state_dict(self, state: dict) -> None:
        state = dict(state)
        wall = state.pop('wall_used', None)
        super().load_state_dict(state)
        if wall is not None:
            self.t0 = self.clock() - float(wall)             # the wall clock of the resumed run starts at the wall used

    def snapshot(self) -> dict:
        return {**self.used, 'wall_seconds': round(self.wall(), 3), 'accounting_complete': self.accounting_complete}

    def fractions(self) -> dict:
        b = self.budget
        return {'screens': self.used['screens'] / b.screens if b.screens else 1.0,
                'llm_tokens': self.used['llm_tokens'] / b.llm_tokens if b.llm_tokens else 0.0,      # 0 = no LLM arm
                'gpu_hours': self.used['gpu_hours'] / b.gpu_hours if b.gpu_hours else 1.0,
                'wall_seconds': self.wall() / b.wall_seconds if b.wall_seconds else 1.0}

    def charge_llm(self, tokens: int | None, usd: float | None, complete: bool) -> None:
        """Values must be finite and non-negative; anything else (or None) leaves the accounting incomplete and
        charges nothing for that field (backends already report an upper bound when usage is unknown)."""
        ok = complete is True
        if type(tokens) is int and tokens >= 0:
            self.used['llm_tokens'] += tokens
        else:
            ok = False
        if type(usd) in (int, float) and math.isfinite(usd) and usd >= 0:
            self.used['llm_usd'] += float(usd)
        else:
            ok = False                                   # also usd=None: cost unknown
        if not ok:
            self.accounting_complete = False

    def charge_screen(self, gpu_hours: float, wall_seconds: float, known: bool = True, count: bool = True) -> None:
        """count=False: a lower-fidelity trial (GPU-hours and wall clock, not the screen count)."""
        if count:
            self.used['screens'] += 1
        else:
            self.used['trials'] = self.used.get('trials', 0) + 1
        if not (type(gpu_hours) in (int, float) and math.isfinite(gpu_hours) and gpu_hours >= 0):
            gpu_hours, known = 0.0, False
        if not (type(wall_seconds) in (int, float) and math.isfinite(wall_seconds) and wall_seconds >= 0):
            wall_seconds, known = 0.0, False
        self.used['gpu_hours'] += gpu_hours
        if count:
            self.screen_costs.append((gpu_hours, wall_seconds))
        if not known:
            self.accounting_complete = False

    def mean_screen(self) -> tuple[float, float] | None:
        if not self.screen_costs:
            return None
        n = len(self.screen_costs)
        return sum(c[0] for c in self.screen_costs) / n, sum(c[1] for c in self.screen_costs) / n

    def exhausted(self, *, allow_token_free: bool | None = None) -> list[str]:
        b, out = self.budget, []
        allow_token_free = self.allow_token_free if allow_token_free is None else allow_token_free
        if self.used['screens'] >= b.screens:
            out.append('screens')
        if not allow_token_free and self.used['llm_tokens'] >= b.llm_tokens and (b.llm_tokens > 0 or self.used['llm_tokens'] > 0):
            out.append('llm_tokens')                     # a zero-token budget forbids LLM calls, not screens
        if b.llm_usd is not None and self.used['llm_usd'] >= b.llm_usd:
            out.append('llm_usd')
        if self.used['gpu_hours'] >= b.gpu_hours:
            out.append('gpu_hours')
        if self.wall() >= b.wall_seconds:
            out.append('wall_seconds')
        return out

    def admit_screen(self, estimate: tuple[float, float] | None = None) -> str | None:
        """None if one more screen fits while keeping `scoring_reserve_screens` screen costs for the locked scoring
        pair (GPU-hours and wall clock), else the limiting dimension. The estimate is the mean measured screen cost;
        before the first measured screen it is `prior` (the start/reference receipt's cost or the task's plan)."""
        ex = self.exhausted()
        if ex:
            return ex[0]
        est = estimate or self.mean_screen() or self.prior
        if est is None:
            return None
        gpu_h, wall_s = est
        reserve = self.budget.scoring_reserve_screens
        if self.used['gpu_hours'] + gpu_h * (1 + reserve) > self.budget.gpu_hours:
            return 'gpu_hours'
        if self.wall() + wall_s * (1 + reserve) > self.budget.wall_seconds:
            return 'wall_seconds'
        return None
