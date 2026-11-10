"""Read side of the packed data (INTERFACES §4b): PackedTHDDataset under input plan A or B, the pack assignment of
each update, and task -> manifest resolution through INDEX.json. numpy only (no torch); pack arrays are copies."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from infrarsi.contracts import RemotePaths, digest, file_digest
from .packing import IGNORE_INDEX, SCHEMA_PACKED


def _int(name, v) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise TypeError(f'{name} must be an int, got {type(v).__name__}')
    return int(v)


def update_assignment(update_index, packs_per_update, dp_rank, dp_size, micro_batch_size, order=None):
    """Pack ids each micro-batch of `dp_rank` consumes in update `update_index` (0-based) -> list[list[int]].

    Update u owns plan positions [u*ppu, (u+1)*ppu); micro-batch g of rank r takes the consecutive slice
    u*ppu + (g*dp_size + r)*mbs + [0, mbs) (Megatron pretraining-sampler layout, also used by
    megatron_core.synthetic_data), GA = ppu/(dp_size*mbs). Positions map to pack ids through `order` (the plan
    permutation); without `order` the plan positions themselves are returned.
    """
    u, ppu, r, dp, mbs = (_int(n, v) for n, v in (('update_index', update_index), ('packs_per_update',
                          packs_per_update), ('dp_rank', dp_rank), ('dp_size', dp_size), ('micro_batch_size',
                          micro_batch_size)))
    if u < 0 or dp < 1 or mbs < 1 or ppu < 1 or not 0 <= r < dp:
        raise ValueError(f'bad assignment request u={u} ppu={ppu} dp_rank={r} dp_size={dp} mbs={mbs}')
    if ppu % (dp * mbs):
        raise ValueError(f'packs_per_update {ppu} is not divisible by dp_size*micro_batch_size = {dp * mbs}')
    base = u * ppu
    slots = [list(range(base + (g * dp + r) * mbs, base + (g * dp + r + 1) * mbs)) for g in range(ppu // (dp * mbs))]
    if order is None:
        return slots
    if (u + 1) * ppu > len(order):
        raise IndexError(f'update {u} needs plan positions beyond the {len(order)} prepared packs')
    return [[int(order[s]) for s in mb] for mb in slots]


def update_slots(update_index, packs_per_update, dp_rank, dp_size, micro_batch_size):
    """Plan-order positions only (update_assignment without an order)."""
    return update_assignment(update_index, packs_per_update, dp_rank, dp_size, micro_batch_size)


def manifest_for_task(task: dict, index_path=None) -> str:
    """Manifest path of a task (spec dict) from INDEX.json, checked against the task's data contract."""
    index_path = Path(index_path or f'{RemotePaths.datasets_packed}/INDEX.json')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    tid = task['id']
    e = index['tasks'].get(tid)
    if e is None:
        why = index.get('pending', {}).get(tid, {}).get('reason', 'task not in INDEX.json')
        raise LookupError(f'{tid}: no prepared data in {index_path} ({why}); '
                          f'rerun python -m infrarsi.adapters.data.build_index')
    w = task['workload']
    want = {'model': task['model']['repo_id'], 'data_key': task['data']['key'], 'seq_len': int(w['seq_len']),
            'packs_per_update': int(w['packs_per_update']), 'packs_total': int(w['packs_total'])}
    got = {k: e[k] for k in want}
    if got != want:
        raise ValueError(f'{tid}: INDEX.json entry {got} does not match the task {want}')
    if file_digest(e['manifest']) != e['manifest_sha256']:
        raise ValueError(f'{e["manifest"]}: sha256 differs from INDEX.json; rerun build_index')
    return e['manifest']


class PackedTHDDataset:
    """One prepared dataset seen through input plan 'A' (seed 42) or 'B' (seed 1042): same packs, other order.

    pack(i) is addressed by pack id; order() is the plan's permutation of pack ids; update_assignment maps
    (update, DP rank, micro-batch) to pack ids, so every parallel layout consumes the same packs per update.
    Labels are already shifted (labels[j] is the target of tokens[j]): never shift again; loss mask = labels != -100.
    """

    def __init__(self, manifest_path, plan='A', verify=False):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = m = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if m.get('schema') != SCHEMA_PACKED:
            raise ValueError(f'{manifest_path}: schema {m.get("schema")!r} != {SCHEMA_PACKED}')
        if plan not in m['plans']:
            raise ValueError(f'plan {plan!r} not in {sorted(m["plans"])}')
        self.plan, self.kind = plan, m['kind']
        self.seq_len, self.packs_total = int(m['seq_len']), int(m['packs_total'])
        self._order = np.asarray(m['plans'][plan]['order'], dtype=np.int64)
        if not np.array_equal(np.sort(self._order), np.arange(self.packs_total)):
            raise ValueError(f'plan {plan} order is not a permutation of {self.packs_total} packs')
        for name, f in m['files'].items():
            if (self.root / name).stat().st_size != f['bytes']:
                raise ValueError(f'{self.root / name}: {(self.root / name).stat().st_size} bytes != {f["bytes"]}')
        if verify:
            self.verify()
        self._maps, self._pid = None, None

    def __len__(self):
        return self.packs_total

    def __getstate__(self):          # memmaps are reopened lazily in each process
        state = self.__dict__.copy()
        state['_maps'] = state['_pid'] = None
        return state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._maps = None

    def _open(self):
        if self._maps is None or self._pid != os.getpid():
            P, S, files = self.packs_total, self.seq_len, self.manifest['files']

            def mm(name, dtype, shape):
                return np.memmap(self.root / name, dtype=dtype, mode='r', shape=shape)

            self._maps = {'tokens': mm('tokens.bin', '<i4', (P, S)), 'labels': mm('labels.bin', '<i4', (P, S)),
                          'position_ids': mm('position_ids.bin', '<i4', (P, S)),
                          'cu': mm('cu_seqlens.bin', '<i4', (files['cu_seqlens.bin']['bytes'] // 4,)),
                          'cu_off': np.fromfile(self.root / 'cu_offsets.bin', dtype='<i8'),
                          'segments': mm('segments.bin', '<i8', (files['segments.bin']['bytes'] // 24, 3))}
            self._pid = os.getpid()
        return self._maps

    def verify(self):
        for name, f in self.manifest['files'].items():
            if file_digest(self.root / name) != f['sha256']:
                raise ValueError(f'{self.root / name}: sha256 differs from manifest')

    def order(self) -> list[int]:
        return self._order.tolist()

    @property
    def order_digest(self) -> str:
        """Identity of what this plan feeds: the packs' file digests and the plan order."""
        return digest({'files': {n: f['sha256'] for n, f in sorted(self.manifest['files'].items())},
                       'plan': self.plan, 'order': self.order()})

    def pack(self, i) -> dict:
        """tokens/labels/position_ids int32 [seq_len]; cu_seqlens int32 (0 .. seq_len); padding_mask True at padding."""
        i = _int('i', i)
        if not 0 <= i < self.packs_total:
            raise IndexError(i)
        m = self._open()
        lo, hi = int(m['cu_off'][i]), int(m['cu_off'][i + 1])
        cu = np.array(m['cu'][lo:hi], dtype=np.int32)
        segs = np.array(m['segments'][lo - i:hi - i - 1])
        labels = np.array(m['labels'][i])
        padding = np.ones(self.seq_len, dtype=bool)
        for start, valid in zip(cu[:-1].tolist(), segs[:, 2].tolist()):
            padding[start:start + valid] = False
        return {'pack_id': i, 'tokens': np.array(m['tokens'][i]), 'labels': labels,
                'position_ids': np.array(m['position_ids'][i]), 'cu_seqlens': cu, 'max_seqlen': int(np.diff(cu).max()),
                'padding_mask': padding, 'source_ids': segs[:, 1].tolist(),
                'supervised_tokens': int((labels != IGNORE_INDEX).sum()), 'physical_tokens': self.seq_len,
                'real_tokens': int(segs[:, 2].sum())}

    __getitem__ = pack

    def update_assignment(self, update_index, packs_per_update, dp_rank, dp_size, micro_batch_size) -> list[list[int]]:
        """Pack ids per micro-batch of this DP rank in update `update_index` (0-based) under this plan."""
        return update_assignment(update_index, packs_per_update, dp_rank, dp_size, micro_batch_size,
                                 order=self._order)

    def update_token_counts(self, update_index, packs_per_update) -> dict:
        """Physical and supervised tokens of one whole update (the invariants a receipt must report)."""
        ids = [p for mb in self.update_assignment(update_index, packs_per_update, 0, 1, 1) for p in mb]
        labels = self._open()['labels']
        return {'packs': ids, 'physical_tokens': len(ids) * self.seq_len,
                'supervised_tokens': int(sum(int((labels[p] != IGNORE_INDEX).sum()) for p in ids))}
