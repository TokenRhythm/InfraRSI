"""Mutable 'kernel' module of the toy: the search may patch it."""


def attention_cost(seq_len: int) -> float:
    """Modelled time per sample of the attention kernel (naive: quadratic chunks)."""
    chunks = max(1, seq_len // 128)
    return 0.02 * chunks * chunks / 4
