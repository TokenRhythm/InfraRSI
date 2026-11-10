"""Data source for the trainer: T-data's PackedTHDDataset (adapters/data) or its synthetic twin.

``dataset.update_assignment`` maps (update, DP rank, micro-batch) to pack ids (Megatron pretraining-sampler
layout over the plan order). Every DP rank of every layout sees the same global pack set per update, which is
what keeps the loss comparable across layouts.
Micro-batches with MBS > 1 concatenate packs into one THD row; context parallelism keeps, per segment, the
two zig-zag chunks (r, 2*CP-1-r) that Transformer Engine's THD context-parallel attention expects.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from infrarsi.contracts import RemotePaths, digest, file_digest
from infrarsi.adapters.megatron_core import synthetic_data

PACK_KEYS = ('tokens', 'labels', 'position_ids', 'cu_seqlens', 'supervised_tokens', 'physical_tokens')


class DataUnavailable(RuntimeError):
    pass


def is_synthetic(task: dict) -> bool:
    data = task.get('data') or {}
    return bool(data.get('synthetic')) or str(data.get('key', '')).startswith('D-SYN')


def resolve_manifest(task: dict, root: str = RemotePaths.datasets_packed, data_manifest: str | None = None) -> str:
    """Packed-data manifest of a task.

    ``data_manifest`` (RunRequest.data_manifest) is used when given, but with ``<root>/INDEX.json`` present its
    sha256 must equal the task's INDEX entry. Otherwise, with the index present, adapters/data's
    ``manifest_for_task`` is authoritative (model, data key, seq_len, packs per update, packs total and manifest
    sha256 checked against the task); any mismatch fails. Without an index (test roots), an explicit
    ``task.data.manifest`` or the single matching directory is used.
    """
    data, w = task.get('data') or {}, task['workload']
    index = Path(root) / 'INDEX.json'
    if data_manifest:
        if index.exists():
            entry = json.loads(index.read_text(encoding='utf-8')).get('tasks', {}).get(task.get('id'))
            if not entry or Path(data_manifest).resolve() != Path(entry.get('manifest', '')).resolve() \
                    or file_digest(data_manifest) != entry.get('manifest_sha256'):
                raise DataUnavailable(f'{task.get("id")}: data_manifest {data_manifest} is not the INDEX.json '
                                      f'manifest of the task (path and manifest_sha256 must match)')
        return str(data_manifest)
    if index.exists():
        from infrarsi.adapters.data import manifest_for_task
        try:
            return str(manifest_for_task(task, index))
        except (LookupError, ValueError, KeyError, OSError) as e:
            raise DataUnavailable(f'{task.get("id")}: packed data rejected by {index}: {e}') from e
    if data.get('manifest'):
        return str(data['manifest'])
    pattern = f'{root}/{data["key"]}__*__s{int(w["seq_len"])}__p{int(w["packs_total"])}/manifest.json'
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise DataUnavailable(f'{task.get("id")}: expected one prepared manifest for {pattern}, found {hits}')
    return hits[0]


def zigzag_index(cu_seqlens: np.ndarray, cp_size: int, cp_rank: int) -> np.ndarray:
    """Token indices this CP rank keeps: chunks r and 2*CP-1-r of every segment (TE THD convention)."""
    parts = []
    for lo, hi in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        n = int(hi - lo)
        if n % (2 * cp_size):
            raise ValueError(f'segment length {n} not divisible by 2*CP={2 * cp_size}')
        c = n // (2 * cp_size)
        parts.append(np.arange(lo + cp_rank * c, lo + (cp_rank + 1) * c))
        parts.append(np.arange(lo + (2 * cp_size - 1 - cp_rank) * c, lo + (2 * cp_size - cp_rank) * c))
    return np.concatenate(parts)


def concat_packs(packs: list[dict]) -> dict:
    """One THD row from MBS packs: tokens/labels/positions concatenated, cu_seqlens offset per pack."""
    cu, offset = [0], 0
    for p in packs:
        cu.extend(int(x) + offset for x in p['cu_seqlens'][1:])
        offset += len(p['tokens'])
    return {'tokens': np.concatenate([p['tokens'] for p in packs]).astype(np.int64),
            'labels': np.concatenate([p['labels'] for p in packs]).astype(np.int64),
            'position_ids': np.concatenate([p['position_ids'] for p in packs]).astype(np.int64),
            'cu_seqlens': np.asarray(cu, dtype=np.int32)}


def local_arrays(packs: list[dict], cp_size: int = 1, cp_rank: int = 0) -> dict:
    """Host arrays for one micro-batch on this rank; ``cu_seqlens`` stays global (unpartitioned)."""
    row = concat_packs(packs)
    labels = row['labels']
    out = {'tokens': row['tokens'], 'position_ids': row['position_ids'],
           'labels': np.where(labels == -100, 0, labels), 'loss_mask': labels != -100}
    if cp_size > 1:
        idx = zigzag_index(row['cu_seqlens'], cp_size, cp_rank)
        out = {k: v[idx] for k, v in out.items()}
    out = {k: np.ascontiguousarray(v[None]) for k, v in out.items()}
    out['cu_seqlens'] = row['cu_seqlens']
    return out


class DataSource:
    def __init__(self, task: dict, plan: str, vocab_size: int, data_manifest: str | None = None):
        w = task['workload']
        self.seq_len, self.ppu, self.packs_total = int(w['seq_len']), int(w['packs_per_update']), int(w['packs_total'])
        if is_synthetic(task):
            kind = (task.get('data') or {}).get('kind') or 'pt_stream'
            self.ds = synthetic_data.SyntheticPackedTHDDataset(None, plan, seq_len=self.seq_len,
                                                               packs_total=self.packs_total,
                                                               vocab_size=vocab_size, kind=kind)
            self.source, self.manifest_digest = 'synthetic', digest(self.ds.manifest)
        else:
            try:
                from infrarsi.adapters import data as tdata
            except ImportError as e:
                raise DataUnavailable(f'adapters.data not importable: {e}') from e
            manifest = resolve_manifest(task, data_manifest=data_manifest)
            self.ds = tdata.PackedTHDDataset(manifest, plan)
            self.source, self.manifest_digest = manifest, file_digest(manifest)
            if int(self.ds.seq_len) != self.seq_len:
                raise DataUnavailable(f'{manifest}: seq_len {self.ds.seq_len} != task {self.seq_len}')
        self.plan = plan
        self.order = np.asarray(self.ds.order(), dtype=np.int64)
        if len(self.ds) != self.packs_total or len(self.order) != self.packs_total:
            raise DataUnavailable(f'pack count {len(self.ds)}/{len(self.order)} != packs_total {self.packs_total}')
        self._checked = False

    @property
    def max_updates(self) -> int:
        return self.packs_total // self.ppu

    def microbatches(self, update_index: int, dp_rank: int, dp_size: int, mbs: int) -> list[list[int]]:
        """Pack ids per micro-batch of this DP rank for update ``update_index`` (0-based)."""
        if update_index >= self.max_updates:
            raise ValueError(f'update {update_index + 1} exceeds the {self.max_updates} prepared updates')
        return [[int(p) for p in mb] for mb in self.ds.update_assignment(update_index, self.ppu, dp_rank, dp_size, mbs)]

    def update_packs(self, update_index: int, dp_size: int, mbs: int) -> list[int]:
        return [p for d in range(dp_size) for mb in self.microbatches(update_index, d, dp_size, mbs) for p in mb]

    def verify_payload(self) -> None:
        """sha256 of every packed binary against the manifest (adapters/data verify); synthetic data has none."""
        if hasattr(self.ds, 'verify'):
            try:
                self.ds.verify()
            except ValueError as e:
                raise DataUnavailable(f'packed payload differs from its manifest: {e}') from e

    def update_supervised_tokens(self, update_index: int) -> int:
        """Supervised tokens of one whole update from the data alone (the receipt's expected value)."""
        return int(self.ds.update_token_counts(update_index, self.ppu)['supervised_tokens'])

    def pack(self, pid: int) -> dict:
        p = self.ds.pack(int(pid))
        if not self._checked:
            missing = [k for k in PACK_KEYS if k not in p]
            cu = np.asarray(p.get('cu_seqlens', []))
            if missing or len(p['tokens']) != self.seq_len or cu[0] != 0 or cu[-1] != self.seq_len:
                raise DataUnavailable(f'pack {pid} violates the packed-THD contract (missing={missing})')
            self._checked = True
        return p

    @property
    def order_digest(self) -> str:
        """Plan identity in the INDEX.json ``data_order_digest[<plan>]`` format (files + plan + order)."""
        return str(self.ds.order_digest)

    def describe(self) -> dict:
        return {'source': self.source, 'plan': self.plan, 'manifest_digest': self.manifest_digest,
                'seq_len': self.seq_len, 'packs_per_update': self.ppu, 'packs_total': self.packs_total}
