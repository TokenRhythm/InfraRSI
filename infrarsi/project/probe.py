"""A dependency-free probe a training loop may call to give InfraRSI exact per-update signals (optional).

    from infrarsi.project.probe import Probe
    probe = Probe()                       # writes to $INFRARSI_PROBE (set by the runner); a no-op when unset
    probe.meta(trainable_params=n, precision={'param': 'bf16', 'grad': 'fp32'})
    for step, batch in enumerate(loader, 1):
        ...
        probe.step(step, loss=loss.item(), grad_norm=gn, lr=sched.get_last_lr()[0], tokens=n_tokens,
                   data=Probe.digest(batch['input_ids']))         # call once per update, on rank 0, after the update

Each update appends one JSON line {step, t, ...}; `t` is the wall clock right after the update, so the step time
InfraRSI derives covers the whole update. `digest` hashes a tensor / array / list cheaply; give the digest of the
global batch (or of rank 0's shard when the data-parallel size never changes).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

ENV = 'INFRARSI_PROBE'


class Probe:
    def __init__(self, path: str | None = None, rank: int | None = None):
        self.path = path or os.environ.get(ENV)
        if rank is None:
            rank = int(os.environ.get('RANK', '0') or 0)
        self.enabled = bool(self.path) and rank == 0
        self._f = open(self.path, 'a', encoding='utf-8', buffering=1) if self.enabled else None

    def _write(self, obj: dict) -> None:
        if self._f is not None:
            self._f.write(json.dumps(obj, separators=(',', ':'), default=float) + '\n')

    def step(self, step: int, **signals) -> None:
        """loss, grad_norm, lr, tokens, samples, seq_len, data, mem_gib, step_time (seconds), phases ({name: ms}, where
        the update's time went, e.g. {'data': 3.1, 'forward_backward': 410.0, 'optimizer': 22.5}) - all optional."""
        self._write({'step': int(step), 't': time.time(), **{k: v for k, v in signals.items() if v is not None}})

    def meta(self, **facts) -> None:
        """trainable_params, model_digest, precision {...}, memory_peak_gib - anything constant for the run."""
        self._write({'meta': facts})

    @staticmethod
    def digest(obj) -> str:
        """sha256 prefix of a tensor, numpy array or nested list (moved to CPU; no torch import needed)."""
        if hasattr(obj, 'detach'):
            obj = obj.detach().cpu()
        if hasattr(obj, 'numpy'):
            obj = obj.numpy()
        data = obj.tobytes() if hasattr(obj, 'tobytes') else json.dumps(obj, default=str).encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
