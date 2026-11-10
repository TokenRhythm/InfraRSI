"""Frozen: the loss definition of the stand-in job (the verifier checks it is never changed)."""
import math


def loss_of(step: int) -> float:
    return 2.5 * math.exp(-0.02 * step)
