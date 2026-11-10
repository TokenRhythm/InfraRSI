"""Adaptive prompt sizes and human-readable token/time budgets."""
from __future__ import annotations

import math
import re
from decimal import Decimal


def parse_tokens(val: str | int) -> int:
    """Non-negative integral tokens, optionally suffixed with k or M (decimal units)."""
    if type(val) is int and val >= 0:
        return val
    if not isinstance(val, str):
        raise ValueError('token budget must be a non-negative integer or a value such as 500k or 1.5M')
    m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*', val)
    if not m:
        raise ValueError(f'invalid token budget {val!r}; use an integer, 500k, or 1.5M')
    n = Decimal(m[1]) * {'': 1, 'k': 1000, 'm': 1000000}[m[2].lower()]
    if n != n.to_integral_value():
        raise ValueError('token budget must resolve to a whole number of tokens')
    return int(n)


def parse_duration_hours(val: str | float) -> float:
    """Non-negative hours, or a duration suffixed with h, m, or s."""
    if type(val) in (int, float):
        hours = float(val)
    elif isinstance(val, str):
        m = re.fullmatch(r'\s*((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([hHmMsS]?)\s*', val)
        if not m:
            raise ValueError(f'invalid duration {val!r}; use hours, 2h, 90m, or 3600s')
        hours = float(m[1]) * {'': 1.0, 'h': 1.0, 'm': 1 / 60, 's': 1 / 3600}[m[2].lower()]
    else:
        raise ValueError('duration must be hours or a value such as 2h, 90m, or 3600s')
    if not math.isfinite(hours) or hours < 0:
        raise ValueError('duration must be finite and non-negative')
    return hours


class TokenGovernor:
    """A fresh view of the ledger for each request; no state survives a budget increase on resume."""

    def __init__(self, ledger, max_hypotheses: int = 12):
        self.used_tokens = getattr(ledger, 'used', {}).get('llm_tokens', 0)
        self.budget_tokens = getattr(getattr(ledger, 'budget', None), 'llm_tokens', 0)
        self.remaining = max(0, self.budget_tokens - self.used_tokens)
        self.exhausted = self.remaining == 0 and (self.budget_tokens > 0 or self.used_tokens > 0)
        fraction = self.used_tokens / self.budget_tokens if self.budget_tokens > 0 else 0.0
        self.zone = 'green'
        if self.budget_tokens > 0:
            if fraction >= 0.88 or self.remaining < 12000:
                self.zone = 'wind_down'
            elif fraction >= 0.65 or self.remaining < 32000:
                self.zone = 'economy'
        self.code_tokens, self.history_scale, self.allow_revise = {
            'green': (12000, 1.0, True), 'economy': (4000, 0.5, False), 'wind_down': (1500, 0.25, False),
        }[self.zone]
        self.max_hypotheses = min(max_hypotheses, {'green': max_hypotheses, 'economy': 4, 'wind_down': 2}[self.zone])

    def history_limit(self, limit: int) -> int:
        return max(1, int(limit * self.history_scale))
