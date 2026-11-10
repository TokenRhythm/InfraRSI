"""ScreenRunner over the driver's request-level runner (builder D: dispatch Context.runner / harness.run_screen).

    runner(RunRequest, *, reference=<dict|None>, overlay_files=<{name: bytes}|None>) -> validated receipt

`HarnessScreenRunner.run(task, config, overlay, purpose, run_id, role='candidate')` builds the RunRequest and returns
the harness-validated, signed receipt unchanged:
  - numerics_role (signed into the receipt): 'candidate' runs are compared with a reference receipt of the same plan;
    'reference' is only for the task's base runs (FR, or ED when FR is infeasible), launched without one. A candidate
    is never launched without a reference, except a candidate-first locked-pair score (`recheck=True`), which the
    caller re-checks with `harness.recheck_against` against the paired plan-B reference once the pair is complete.
  - purpose 'screen' / 'probe' -> input plan A, reference = the task's plan-A reference screen (FR, or ED fallback);
    'score' -> plan B, reference = the paired plan-B reference run when it ran first (`score_reference`);
    'lifecycle' -> plan B with the task's lifecycle block, reference = the paired plan-B reference run.
  - gpus_per_node = task.gpus on one node, else task.gpus // task.nodes; overlays {name: text} go up as UTF-8 bytes.
  - Training failures come back as valid=false receipts (the harness never raises for them). A harness exception
    (OccupiedError cannot happen under the dispatcher's run lease, but is handled) becomes an invalid record charged
    world x elapsed GPU-seconds with cost.accounting_complete = false, so the search ledger marks the run incomplete.
Budget fields of validated receipts: receipt['harness']['gpu_hours'] and ['wall_s'] (agent/judge.py).
"""
from __future__ import annotations

import time
from typing import Callable

from ..contracts import SCHEMA_RECEIPT, RunRequest, TaskSpec, utc_now

PLANS = {'screen': 'A', 'probe': 'A', 'score': 'B', 'lifecycle': 'B'}


def gpus_per_node(task) -> int:
    return task.gpus if task.nodes == 1 else task.gpus // task.nodes


def wire_config(config: dict) -> dict:
    """The config as sent to WINH: planner-only '_' keys never leave the driver."""
    return {k: v for k, v in (config or {}).items() if not (type(k) is str and k.startswith('_'))}


class HarnessScreenRunner:
    stop_grace_s = 300.0            # harness.run_screen: a STOP at the wall cap is followed by up to this grace

    def __init__(self, runner: Callable, nodes, gpus_per_node: int | None = None, *, reference_receipt_path: str | None = None,
                 reference_receipt: dict | None = None, wall_seconds_cap: int = 3600, clock=time.monotonic, log=None):
        self.runner, self.nodes = runner, list(nodes)
        self.gpn = gpus_per_node
        self.reference_path, self.reference = reference_receipt_path, reference_receipt
        self.score_reference: tuple[str | None, dict] | None = None      # (path, receipt) of the paired FR plan-B run
        self.wall_seconds_cap = int(wall_seconds_cap)
        self.clock, self.log = clock, log or (lambda m: None)
        self.attempts: list[dict] = []

    def _gpn(self, task) -> int:
        return int(self.gpn) if self.gpn else gpus_per_node(task)

    # -------- worst-case admission (an attempt is indivisible: it may use its whole wall cap plus the stop grace)
    def worst_case_gpu_hours(self, world: int, wall_cap: float | None = None) -> float:
        return int(world) * (float(wall_cap or self.wall_seconds_cap) + self.stop_grace_s) / 3600.0

    def bounded_cap(self, gpu_hours_left: float, world: int, wall_left: float, expected_s: float) -> int | None:
        """Largest wall cap whose worst case fits the GPU-hours and wall left; None when it would be shorter than the
        expected attempt length (a screen cut by its cap is wasted)."""
        cap = min(float(self.wall_seconds_cap), gpu_hours_left * 3600.0 / max(1, int(world)) - self.stop_grace_s,
                  wall_left - self.stop_grace_s)
        return int(cap) if cap >= max(60.0, float(expected_s)) else None

    def pair_wall_cap(self, expected_s: float) -> int:
        """Wall cap of each locked-pair member (reserved before every screen, used for the pair)."""
        return int(min(float(self.wall_seconds_cap), max(1.5 * float(expected_s), float(expected_s) + 300.0)))

    def _reference(self, purpose: str, role: str) -> tuple[str | None, dict | None]:
        """(path, receipt) of the numerics reference a run of this purpose and role is compared with."""
        if role == 'reference':
            return None, None
        if purpose in ('screen', 'probe'):
            return self.reference_path, self.reference
        return self.score_reference if self.score_reference else (None, None)       # score / lifecycle: plan B

    def request(self, task, config: dict, purpose: str, run_id: str, *, role: str = 'candidate') -> RunRequest:
        if purpose not in PLANS:
            raise ValueError(f'purpose {purpose!r}')
        raw = task.raw
        ref_path = self._reference(purpose, role)[0]
        kw = {}
        if purpose == 'lifecycle':          # AMENDMENT-002: deterministic kernels on every lifecycle (driver-fixed)
            wl = raw['workload']
            kw['lifecycle'] = {'save_at': int(wl['screen_steps']), 'resume_steps': int(wl['resume_continuation_steps']),
                               'export': True, 'forward_compare': True, 'deterministic': True}
        return RunRequest(run_id=run_id, purpose=purpose, task=raw, config=wire_config(config), nodes=list(self.nodes),
                          gpus_per_node=self._gpn(task), input_plan=PLANS[purpose], reference_receipt=ref_path,
                          wall_seconds_cap=self.wall_seconds_cap, numerics_role=role, **kw)

    def run(self, task, config: dict, overlay: dict | None, purpose: str, run_id: str, *, role: str = 'candidate',
            recheck: bool = False, wall_cap: int | None = None) -> dict:
        task = task if hasattr(task, 'raw') else TaskSpec(raw=task)
        reference = self._reference(purpose, role)[1]
        if role == 'candidate' and reference is None and not (recheck and purpose == 'score'):
            raise ValueError(f'{purpose} candidate run without a numerics reference (only base runs may be launched as '
                             f"numerics_role='reference')")
        req = self.request(task, config, purpose, run_id, role=role)
        if wall_cap is not None:                       # the session's admission shrank this attempt's cap
            req.wall_seconds_cap = int(min(int(wall_cap), req.wall_seconds_cap))
        files = {str(n): t.encode('utf-8') for n, t in overlay.items()} if overlay else None
        t0 = self.clock()
        try:
            rec = self.runner(req, reference=reference, overlay_files=files)
            if not isinstance(rec, dict):
                raise TypeError(f'runner returned {type(rec).__name__}')
        except Exception as exc:                                       # noqa: BLE001 - recorded, charged, never raised
            rec = self.failure_record(req, exc, self.clock() - t0)
        from .judge import gpu_seconds, wall_seconds
        h = rec.get('harness') if isinstance(rec.get('harness'), dict) else {}
        g = gpu_seconds(rec)
        self.attempts.append({'attempt_id': rec.get('attempt_id'), 'purpose': purpose, 'input_plan': req.input_plan,
                              'numerics_role': role, 'valid': rec.get('valid'), 'gpu_hours': None if g is None else g / 3600.0,
                              'wall_s': wall_seconds(rec), 'failure_kind': h.get('failure_kind') or (rec.get('runner_error') or {}).get('kind'),
                              'utc': utc_now()})
        self.log(f'{run_id}/{rec.get("attempt_id")}: {purpose} plan {req.input_plan} valid={rec.get("valid")}')
        return rec

    def failure_record(self, req: RunRequest, exc: BaseException, elapsed: float) -> dict:
        """An invalid receipt-shaped record for a harness exception (nothing validated, nothing signed)."""
        try:
            from ..harness.leases import OccupiedError
            occupied = isinstance(exc, OccupiedError)
        except ImportError:                                            # pragma: no cover - harness always shipped
            occupied = False
        kind = 'occupied' if occupied else 'harness_error'
        elapsed = max(0.0, float(elapsed))
        return {'schema': SCHEMA_RECEIPT, 'run_id': req.run_id, 'attempt_id': req.attempt_id, 'purpose': req.purpose,
                'task_id': req.task['id'], 'input_plan': req.input_plan, 'world_size': req.world_size, 'nodes': list(req.nodes),
                'steps': req.steps, 'steps_completed': 0, 'valid': False,
                'invalid_reasons': [f'{kind}: {type(exc).__name__}: {str(exc)[:400]}'], 'step_time': None,
                'cost': {'gpu_seconds': round(req.world_size * elapsed, 3), 'wall_seconds': round(elapsed, 3),
                         'accounting_complete': False, 'basis': 'world_size x elapsed (harness exception)'},
                'runner_error': {'kind': kind, 'type': type(exc).__name__, 'message': str(exc)[:2000]}}
