"""Package lineage of the R arm: A0 -> A_1..A_8 (one reflect per dev task), snapshots A1 (after task 4) and
A2 = A_R (after task 8), freeze by digest at G3; any later change is detectable (G4).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .package import OptimizerPackage, compute_digest, copy_package

STATUSES = ('initial', 'changed', 'unchanged', 'invalid_fallback', 'snapshot', 'frozen')


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class Lineage:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / 'lineage.jsonl'

    def rows(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(x) for x in self.log.read_text(encoding='utf-8').splitlines() if x.strip()]

    def _append(self, row: dict) -> dict:
        row = {'utc': _utc(), 'index': len(self.rows()), **row}
        if row.get('status') not in STATUSES:
            raise ValueError(f'status {row.get("status")!r}')
        with self.log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        return row

    def register(self, pkg: OptimizerPackage, *, status: str, task_id: str | None = None, parent_digest: str | None = None,
                 reflect: dict | None = None) -> dict:
        if self.frozen() and status not in ('snapshot',):
            raise PermissionError(f'lineage frozen at {self.frozen()["digest"][:12]}; no new versions after G3')
        return self._append({'status': status, 'digest': pkg.digest, 'label': pkg.label, 'path': str(pkg.root),
                             'task_id': task_id, 'parent_digest': parent_digest, 'reflect': reflect or {}})

    def current(self) -> dict | None:
        rows = [r for r in self.rows() if r['status'] in ('initial', 'changed', 'unchanged', 'invalid_fallback')]
        return rows[-1] if rows else None

    def snapshot(self, label: str, pkg: OptimizerPackage) -> OptimizerPackage:
        dst = self.root / 'snapshots' / label
        copy_package(pkg.root, dst)
        snap = OptimizerPackage.load(dst)
        if snap.digest != pkg.digest:
            raise RuntimeError('snapshot digest mismatch')
        self._append({'status': 'snapshot', 'digest': snap.digest, 'label': label, 'path': str(dst)})
        return snap

    def freeze(self, label: str) -> dict:
        snaps = [r for r in self.rows() if r['status'] == 'snapshot' and r['label'] == label]
        if not snaps:
            raise KeyError(f'no snapshot {label}')
        snap = snaps[-1]
        actual = compute_digest(snap['path'])
        if actual != snap['digest']:
            raise RuntimeError(f'snapshot {label} changed before freeze')
        record = {'label': label, 'digest': snap['digest'], 'path': snap['path'], 'frozen_utc': _utc()}
        (self.root / f'FROZEN_{label}.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
        self._append({'status': 'frozen', 'digest': snap['digest'], 'label': label, 'path': snap['path']})
        return record

    def frozen(self, label: str | None = None) -> dict | None:
        files = sorted(self.root.glob('FROZEN_*.json')) if label is None else [self.root / f'FROZEN_{label}.json']
        files = [f for f in files if f.exists()]
        return json.loads(files[-1].read_text(encoding='utf-8')) if files else None

    def verify_frozen(self, label: str) -> bool:
        rec = self.frozen(label)
        return bool(rec) and compute_digest(rec['path']) == rec['digest']
