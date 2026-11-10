"""Phase guards. Development: no held-out task may be read or run. Held-out: only the frozen A_R (by digest) runs,
memory is read-only and write_memory is a logged no-op; results are never written back (G4).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from ..contracts import load_all_tasks
from .package import compute_digest


def split_ids(split: str) -> tuple[str, ...]:
    return tuple(sorted(k for k, t in load_all_tasks().items() if t.raw.get('split') == split))


class ReadWhitelist:
    """Which task ids a phase may load, search or read receipts of."""

    def __init__(self, phase: str, allowed: tuple[str, ...]):
        self.phase, self.allowed = phase, tuple(allowed)

    @classmethod
    def dev(cls) -> 'ReadWhitelist':
        return cls('dev', split_ids('dev'))

    @classmethod
    def heldout(cls, task_id: str) -> 'ReadWhitelist':
        if task_id not in split_ids('heldout'):
            raise ValueError(f'{task_id} is not a held-out task')
        return cls('heldout', (task_id,))

    @classmethod
    def frozen_phase(cls, task_id: str) -> 'ReadWhitelist':
        """Any non-development task (held-out, flagship, cross-GPU): only that task, frozen package, read-only."""
        tasks = load_all_tasks()
        split = tasks[task_id].raw.get('split') if task_id in tasks else None
        if split in (None, 'dev'):
            raise ValueError(f'{task_id} is a development task or unknown; no frozen phase applies')
        return cls(split, (task_id,))

    def check_task(self, task_id: str) -> None:
        if task_id not in self.allowed:
            raise PermissionError(f'{self.phase} phase may not read task {task_id} (allowed: {self.allowed})')

    def check_receipt(self, receipt: dict) -> None:
        self.check_task(receipt.get('task_id', ''))


class HeldoutSession:
    def __init__(self, task_id: str, frozen_digest: str, log_path: str | Path):
        self.whitelist = ReadWhitelist.frozen_phase(task_id)
        self.task_id, self.frozen_digest = task_id, frozen_digest
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def check(self, pkg, task_id: str) -> None:
        self.whitelist.check_task(task_id)
        if pkg.digest != self.frozen_digest:
            raise PermissionError(f'held-out runs only the frozen package {self.frozen_digest[:12]}, got {pkg.digest[:12]}')
        actual = compute_digest(pkg.root)
        if actual != self.frozen_digest:
            raise PermissionError(f'package content changed after freeze ({actual[:12]})')
        self._log({'event': 'heldout_start', 'task_id': task_id, 'package_digest': pkg.digest})

    def log_noop(self, entry: dict) -> None:
        self._log({'event': 'write_memory_noop', 'task_id': self.task_id, 'entry': entry})

    def _log(self, row: dict) -> None:
        row = {'utc': _dt.datetime.now(_dt.timezone.utc).isoformat(), **row}
        with self.log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
