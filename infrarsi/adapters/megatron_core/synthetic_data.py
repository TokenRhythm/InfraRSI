"""Deterministic synthetic packed THD data with the adapters.data interface (INTERFACES section 4b).

Serves E0-SMOKE-S1-SYN and every run whose task data block asks for synthetic input. Each pack is generated
from ``numpy.random.default_rng([seed, pack])`` so any rank can materialise any pack without I/O.
``pt_stream`` packs are one fully supervised window; ``sft_packed`` packs hold several segments (lengths a
multiple of 8, position ids reset, a quarter of each segment unsupervised prompt, last position unsupervised).
Labels are already aligned next-token targets, as in the real packed data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from infrarsi.contracts import digest

SCHEMA = 'infrarsi-synthetic-packed-v1'
PLAN_SEEDS = {'A': 42, 'B': 1042}
PAD_MULTIPLE = 8


class SyntheticPackedTHDDataset:
    def __init__(self, manifest_path: str | None = None, plan: str = 'A', *, seq_len: int | None = None,
                 packs_total: int | None = None, vocab_size: int | None = None, kind: str = 'pt_stream',
                 seed: int = 1234):
        if manifest_path:
            m = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            if m.get('schema') != SCHEMA:
                raise ValueError(f'{manifest_path}: not a synthetic manifest')
            seq_len, packs_total, vocab_size, kind, seed = (m['seq_len'], m['packs_total'], m['vocab_size'],
                                                            m['kind'], m['seed'])
        if plan not in PLAN_SEEDS:
            raise ValueError(f'input plan {plan!r}')
        if kind not in ('pt_stream', 'sft_packed') or seq_len % PAD_MULTIPLE:
            raise ValueError(f'synthetic kind {kind!r} / seq_len {seq_len}')
        self.seq_len, self.packs_total, self.vocab_size = int(seq_len), int(packs_total), int(vocab_size)
        self.kind, self.seed, self.plan = kind, int(seed), plan
        self.manifest = {'schema': SCHEMA, 'seq_len': self.seq_len, 'packs_total': self.packs_total,
                         'vocab_size': self.vocab_size, 'kind': kind, 'seed': self.seed, 'synthetic': True}
        # Same plan generator as adapters/data: numpy RandomState(seed).permutation(packs_total).
        self._order = np.random.RandomState(PLAN_SEEDS[plan]).permutation(self.packs_total)

    def __len__(self) -> int:
        return self.packs_total

    @property
    def order_digest(self) -> str:
        """Identity of what this plan feeds (same role as adapters/data PackedTHDDataset.order_digest)."""
        return digest({'manifest': self.manifest, 'plan': self.plan, 'order': self._order.tolist()})

    def order(self) -> np.ndarray:
        return self._order.copy()

    def _segments(self, rng) -> list[int]:
        if self.kind == 'pt_stream':
            return [self.seq_len]
        lengths, left = [], self.seq_len
        while left > 0:
            n = PAD_MULTIPLE * int(rng.integers(32, max(33, self.seq_len // (2 * PAD_MULTIPLE)) + 1))
            n = min(n, left)
            if left - n < 64:          # never leave a sliver segment
                n = left
            lengths.append(n)
            left -= n
        return lengths

    def pack(self, i: int) -> dict:
        if not 0 <= i < self.packs_total:
            raise IndexError(i)
        rng = np.random.default_rng([self.seed, int(i)])
        stream = rng.integers(0, self.vocab_size, size=self.seq_len + 1, dtype=np.int64).astype(np.int32)
        tokens = stream[:-1].copy()
        labels = np.full(self.seq_len, -100, dtype=np.int32)
        positions = np.zeros(self.seq_len, dtype=np.int32)
        cu = [0]
        for n in self._segments(rng):
            lo, hi = cu[-1], cu[-1] + n
            positions[lo:hi] = np.arange(n, dtype=np.int32)
            if self.kind == 'pt_stream':
                labels[lo:hi] = stream[lo + 1:hi + 1]
            else:
                prompt = n // 4
                labels[lo + prompt:hi - 1] = tokens[lo + prompt + 1:hi]
            cu.append(hi)
        return {'pack_id': int(i), 'tokens': tokens, 'labels': labels, 'position_ids': positions,
                'cu_seqlens': np.asarray(cu, dtype=np.int32), 'max_seqlen': int(np.diff(cu).max()),
                'supervised_tokens': int((labels != -100).sum()), 'physical_tokens': self.seq_len}

    def update_assignment(self, update_index, packs_per_update, dp_rank, dp_size, micro_batch_size) -> list[list[int]]:
        """Pack ids per micro-batch of this DP rank in update ``update_index`` (0-based), as adapters/data."""
        if (int(update_index) + 1) * int(packs_per_update) > self.packs_total:
            raise IndexError(f'update {update_index} needs packs beyond packs_total {self.packs_total}')
        slots = update_assignment(update_index, packs_per_update, dp_rank, dp_size, micro_batch_size)
        return [[int(self._order[s]) for s in mb] for mb in slots]

    def update_token_counts(self, update_index, packs_per_update) -> dict:
        ids = [p for mb in self.update_assignment(update_index, packs_per_update, 0, 1, 1) for p in mb]
        return {'packs': ids, 'physical_tokens': len(ids) * self.seq_len,
                'supervised_tokens': sum(self.pack(p)['supervised_tokens'] for p in ids)}


def update_assignment(update_index: int, packs_per_update: int, dp_rank: int, dp_size: int,
                      micro_batch_size: int) -> list[list[int]]:
    """Plan-order positions per micro-batch of ``dp_rank`` in update ``update_index`` (0-based).

    Same layout as adapters.data.update_slots (Megatron pretraining sampler): update u owns positions
    [u*P, (u+1)*P) and micro-batch g of rank r takes u*P + (g*dp_size + r)*MBS + [0, MBS).
    """
    if packs_per_update % (dp_size * micro_batch_size) or not 0 <= dp_rank < dp_size:
        raise ValueError('packs_per_update must be divisible by dp_size * micro_batch_size')
    base, dp, mbs = update_index * packs_per_update, dp_size, micro_batch_size
    return [list(range(base + (g * dp + dp_rank) * mbs, base + (g * dp + dp_rank + 1) * mbs))
            for g in range(packs_per_update // (dp * mbs))]
