"""Frozen: the loss of the toy (the verifier checks this file is never changed)."""
import math

PRECISION_ERROR = {'fp32': 0.0, 'bf16': 0.002, 'fp8': 0.035}


def loss_of(step: int, batch: list, precision: str) -> tuple[float, float]:
    base = 2.5 * math.exp(-0.02 * step) + 0.05 * (sum(batch) / len(batch) - 0.5)
    gn = 1.0 + 0.5 * math.exp(-0.05 * step) + 0.02 * (batch[0] - 0.5)
    e = PRECISION_ERROR[precision]
    return base * (1 + e), gn * (1 + e)
