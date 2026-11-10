"""E6 R-arm development sequence (design E6, FINAL_PLAN 3.6): fixed order S1 -> S2 -> D1 -> D2 -> D3 -> D4 -> D5 -> D6,
one reflect after each task, snapshot A1 after the 4th task (D2), A2 = A_R after the 8th (D6), then freeze (G3).

The GPU searches themselves are dispatched elsewhere (they need leased nodes); `advance` is called once a task's
search trace exists and returns the package for the next task.
"""
from __future__ import annotations

import json
from pathlib import Path

from .lineage import Lineage
from .package import OptimizerPackage
from .reflect import ReflectResult, reflect
from .selfcheck import selfcheck
from .audit import audit_package

DEV_ORDER = ('S1', 'S2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6')
SNAPSHOTS = {'D2': 'A1', 'D6': 'A_R'}
STOP_LOSS_ORDER = ('S1', 'S2', 'D1', 'D2')            # design 6 stop-loss 1: dense-only sequence
STOP_LOSS_SNAPSHOTS = {'S2': 'A1', 'D2': 'A_R'}


def history_receipts(trace_dirs) -> list:
    """(receipt, parent, None) triples from finished development search out_dirs, for selfcheck's tool runs.
    Fail closed: a trace or receipt of any non-development task raises PermissionError (G3 isolation)."""
    from .heldout import ReadWhitelist
    dev = ReadWhitelist.dev()
    out = []
    for d in trace_dirs:
        d = Path(d)
        if not (d / 'trace.json').exists():
            continue
        trace = json.loads((d / 'trace.json').read_text(encoding='utf-8'))
        dev.check_task(trace.get('task_id', ''))
        parents = {c['id']: c.get('parent') for c in trace['candidates']}
        recs = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in (d / 'receipts').glob('*.json')}
        for rec in recs.values():
            if isinstance(rec, dict) and rec.get('task_id') is not None:
                dev.check_receipt(rec)
        for cid, rec in recs.items():
            out.append((rec, recs.get(parents.get(cid) or ''), None))
    return out


def advance(lineage: Lineage, task_id: str, trace: dict, package: OptimizerPackage, driver, out_dir: str | Path, *,
            ledger=None, trace_dirs=(), order: tuple = DEV_ORDER, snapshots: dict = SNAPSHOTS) -> tuple[OptimizerPackage, ReflectResult]:
    """Reflect on `task_id`'s trace, register A_{k+1}, snapshot/freeze at the design points. Enforces the order."""
    done = [r['task_id'] for r in lineage.rows() if r['status'] in ('changed', 'unchanged', 'invalid_fallback')]
    expected = order[len(done)] if len(done) < len(order) else None
    if task_id != expected:
        raise ValueError(f'development order: expected {expected}, got {task_id} (done {done})')
    if lineage.frozen():
        raise PermissionError('lineage already frozen')
    k = len(done) + 1
    res = reflect(task_id, trace, package, driver, Path(out_dir) / f'reflect_{k:02d}_{task_id}', label=f'A_{k}',
                  history_receipts=history_receipts(trace_dirs), lineage=lineage, ledger=ledger)
    nxt = OptimizerPackage.load(res.package_dir)
    if task_id in snapshots:
        label = snapshots[task_id]
        snap = lineage.snapshot(label, nxt)
        if label == 'A_R':
            check, audit = selfcheck(snap.root, history_receipts=history_receipts(trace_dirs)), audit_package(snap.root)
            if not (check.ok and audit.passed):
                raise RuntimeError(f'G3 blocked: selfcheck {check.errors[:3]} audit {audit.findings[:3]}')
            lineage.freeze(label)
        nxt = snap
    return nxt, res
