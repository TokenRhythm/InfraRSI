"""Probe chains: drive a planner state machine (fr_v0 OOM escalation, ED top-k) with 5-update probe attempts."""
from __future__ import annotations

import statistics
from typing import Callable

from infrarsi.contracts import RunRequest, TaskSpec

from .runner import run_screen


def probe_outcome(receipt: dict) -> str:
    """ok | oom | error (anything else: launch failure, NCCL, exception, timeout - not an OOM escalation trigger)."""
    if receipt.get('valid'):
        return 'ok'
    kind = (receipt.get('harness') or {}).get('failure_kind')
    return 'oom' if kind == 'oom' else 'error'


def probe_step_time(receipt: dict) -> float | None:
    """Median max-rank whole-update time of updates 2..n (the first update carries compile/warm-up)."""
    t = [p['max_rank_whole_s'] for p in receipt.get('per_step') or [] if p.get('update', 0) >= 2
         and isinstance(p.get('max_rank_whole_s'), (int, float))]
    return statistics.median(t) if t else None


def gpus_per_node(task: TaskSpec) -> int:
    return task.gpus if task.nodes == 1 else task.gpus // task.nodes


def run_probe_chain(chain, task: TaskSpec, nodes: list[str], run_id: str, *, runner: Callable = run_screen,
                    wall_seconds_cap: int = 3600, max_probes: int = 8, **runner_kw) -> list[dict]:
    """chain: .current() -> config | None ; .advance(outcome, receipt). Returns the probe receipts in order."""
    receipts = []
    while len(receipts) < max_probes:
        cfg = chain.current()
        if cfg is None:
            break
        req = RunRequest(run_id=run_id, purpose='probe', task=task.raw, config=cfg, nodes=list(nodes),
                         gpus_per_node=gpus_per_node(task), input_plan='A', wall_seconds_cap=wall_seconds_cap)
        rec = runner(req, **runner_kw)
        receipts.append(rec)
        chain.advance(probe_outcome(rec), rec)
    return receipts
