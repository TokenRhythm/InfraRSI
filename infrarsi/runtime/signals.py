"""Per-rank signal files: one JSON line per update per rank (INTERFACES section 3.4).

``signals/rank<r>.jsonl`` holds the per-rank timing and memory row; rank 0 also carries the global scalars.
The driver re-derives every window statistic from these files, so they are the primary timing record.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# microbatches: forward-backward micro-batches this rank executed in the update (counted in frozen code: the
# forward step and a gradient hook on its output), = packs_per_update / (dp * mbs).
RANK_KEYS = ('step', 'whole_update_s', 'fwd_bwd_s', 'optim_s', 'data_s', 'peak_alloc_gib', 'peak_reserved_gib',
             'alloc_retries', 'alloc_ooms', 'utc_start_s', 'utc_end_s', 'microbatches')
GLOBAL_KEYS = ('main_ce', 'aux_loss', 'grad_norm', 'lr', 'physical_tokens', 'supervised_tokens', 'optimizer_steps',
               'clip_calls')
# inter_step_s: start(k) - start(k-1) on this rank (None for a process's first update); cuda_whole_s: CUDA-event
# span of the same update. utc_start_s/utc_end_s are POSIX seconds at the aligned start and the end of the update.
EXTRA_KEYS = ('inter_step_s', 'cuda_whole_s')


_INF = float('inf')


def _value(v) -> str:
    """JSON text of one scalar using builtin-type methods only (they cannot be monkeypatched from Python)."""
    if v is None:
        return 'null'
    if v is True:
        return 'true'
    if v is False:
        return 'false'
    if type(v) is int:
        return int.__repr__(v)
    if type(v) is float:
        if v != v:
            return 'NaN'
        if v == _INF or v == -_INF:
            return 'Infinity' if v > 0 else '-Infinity'
        return float.__repr__(v)
    if type(v) is str and v.isascii() and v.isprintable() and '"' not in v and '\\' not in v:
        return '"' + v + '"'
    raise TypeError(f'signal value {v!r} is not a plain scalar')


def dump_row(row: dict) -> str:
    """Deterministic one-line JSON of a flat signal row; does not go through the (patchable) json module."""
    keys = [k for k in row]
    keys.sort()
    return '{' + ', '.join('"' + k + '": ' + _value(row[k]) for k in keys) + '}'


class SignalWriter:
    """Append-only writer; one file per rank and per phase directory (``signals`` or ``signals_resume``)."""

    def __init__(self, out_dir: str | Path, rank: int, name: str = 'signals'):
        self.path = Path(out_dir) / name / f'rank{rank}.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, 'w', encoding='utf-8')

    def write(self, row: dict) -> None:
        missing = [k for k in RANK_KEYS if k not in row]
        if missing:
            raise KeyError(f'signal row missing {missing}')
        self._f.write(dump_row(row) + '\n')
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()


def signal_files(signals_dir: str | Path) -> dict[int, Path]:
    files = {}
    for p in Path(signals_dir).glob('rank*.jsonl'):
        try:
            files[int(p.stem[4:])] = p
        except ValueError:
            continue
    return dict(sorted(files.items()))


def read_signals(signals_dir: str | Path) -> dict[int, list[dict]]:
    """Rows per rank; a torn last line (crash mid-write) is ignored."""
    out = {}
    for rank, path in signal_files(signals_dir).items():
        rows = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
        out[rank] = rows
    return out


def signals_digest(signals_dir: str | Path) -> str:
    """sha256 over the concatenated bytes of signals/rank<r>.jsonl in increasing rank order."""
    h = hashlib.sha256()
    for _, path in signal_files(signals_dir).items():
        h.update(path.read_bytes())
    return h.hexdigest()
