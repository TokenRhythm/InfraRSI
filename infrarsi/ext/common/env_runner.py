"""EnvRunner base (EXT_INTERFACES §2): an agent.runner.ScreenRunner for an external environment.

    run(task, config, overlay, purpose, run_id, **kw) -> EnvReceipt (schema ext-receipt-v1, signed on this PC)

The base holds the node for the attempt (GpuLease: occupancy check, node lease or per-GPU claims, owner
infrarsi-ext), calls the environment's `execute` (materialize the candidate in its own directory from the
hash-verified base tree, run the environment's scripts and judge, collect, apply guards, delete bulky outputs), then
assembles, signs and stores the receipt. valid = judge_valid and not guard_rejections; judge_valid is the environment
judge's verdict as returned (a judge success without a finite metric adds the guard `output_integrity`); the metric
(and step_time) is null when invalid. Failures never raise: an occupied node or a runner error becomes an invalid
receipt whose cost is world x held seconds with accounting_complete false. kw: candidate_id (the session's id;
directory name), wall_cap (bounded admission: the attempt must end within it).

Admission (Codex H-05): an environment declares the conservative wall of one attempt, worst_case() (its protocol
timeouts + judge + collection; X2/X3 give theirs). SearchSession then admits an attempt only when that worst case fits
the wall and GPU-hours left (bounded_cap; GPU = world x worst case) and passes the cap on; execute() receives it as
wall_cap and sets its timeouts to min(protocol timeout, what the cap leaves). The locked re-score is not part of an ext
arm's budget, so no pair is reserved. A runner without a worst case keeps the ledger's mean-cost admission. The
basic agent admits its experiment tools with the same numbers (basic_agent.EnvTool.worst_case).
Containment (Codex B-02): after `execute`, in a finally and before any lease or claim is released, every process of the
trust-domain uids in `domain_uids()` is killed (remote_exec.uid_kill); if one survives, the lease is kept and the
receipt is invalid (guard `containment`). Lease releases retry; unreleased paths are reported in the receipt (M-02).
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

from ...contracts import digest, new_attempt_id, utc_now
from ...harness import signing
from ...harness.leases import OccupiedError
from .guards import GuardResult
from .remote_exec import ContainmentError, GpuLease, NodeOps, uid_kill
from .session_hooks import sha_text

SCHEMA_RECEIPT = 'ext-receipt-v1'


@dataclasses.dataclass
class EnvRun:
    """What an environment's execute() returns to the base runner."""
    candidate_id: str
    judge: dict | None                    # the environment judge's own output, verbatim
    judge_valid: bool
    metric_value: float | None            # the minimized metric according to the judge
    metric_raw: dict                      # environment-native values (e.g. TFLOPS, ln-seconds score, printed val_bpb)
    judge_reasons: list                   # why the judge says invalid (empty when valid)
    guards: GuardResult
    phases: dict                          # seconds per phase
    candidate_digest: str | None = None   # digest of the effective candidate (rendered files); base default otherwise
    extra: dict = dataclasses.field(default_factory=dict)    # environment fields copied into the receipt


def candidate_digest(config: dict | None, overlay: dict | None) -> str:
    return digest({'config': config or {}, 'files': {p: sha_text(t) for p, t in sorted((overlay or {}).items())}})


def _finite_pos(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v) and v > 0


def jsonable(obj):
    """JSON-safe copy (non-finite floats become the strings 'nan'/'inf'/'-inf'; the signature needs strict JSON)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


class EnvRunner:
    env = ''                     # environment name (subclasses)
    deviations: tuple = ()       # registered deviation ids that apply to every run of this runner
    worst_case_s: float | None = None    # conservative wall of one attempt, or override worst_case() (subclasses)
    sibling_uids: tuple = ()             # co-located arms' trust domains: their processes elsewhere never block
    stop_grace_s = 0.0                   # SearchSession admission API (agent/harness_runner.py semantics)

    def __init__(self, ops: NodeOps, *, gpus, lease_mode: str = 'own', held_run_id: str | None = None,
                 out_dir: str | Path | None = None, clock=time.monotonic, log=None):
        self.ops, self.node, self.gpus = ops, ops.node, [int(g) for g in gpus]
        self.lease_mode, self.held_run_id = lease_mode, held_run_id
        self.out_dir = Path(out_dir) if out_dir else None
        self.clock, self.log = clock, log or (lambda msg: None)
        self.attempts: list[dict] = []

    # ------------------------------------------------------------ to implement per environment
    def execute(self, task, config: dict, overlay: dict, purpose: str, run_id: str, attempt_id: str,
                candidate_id: str, **kw) -> EnvRun:
        raise NotImplementedError

    def describe(self) -> dict:
        return {'kind': type(self).__name__, 'env': self.env, 'node': self.node, 'gpus': self.gpus,
                'lease_mode': self.lease_mode, 'deviations': list(self.deviations), 'worst_case_s': self.worst_case()}

    def domain_uids(self) -> list[int]:
        """Trust-domain uids whose processes must all be dead before the GPUs are released (subclasses)."""
        return []

    # ------------------------------------------------------------ bounded admission (SearchSession._admission)
    def worst_case(self) -> float | None:
        """Conservative wall seconds of one attempt (None: not declared, mean-cost admission)."""
        return float(self.worst_case_s) if self.worst_case_s else None

    def worst_case_gpu_hours(self, world: int, wall_cap: float | None = None) -> float:
        cap = self.worst_case() if wall_cap is None else wall_cap
        return int(world) * float(cap or 0.0) / 3600.0

    def pair_wall_cap(self, expected_s: float) -> int:
        return 0                                   # the locked re-score is not part of an ext arm's budget

    @property
    def bounded_cap(self):
        """SearchSession's bounded admission hook, present only when the environment declares its worst case."""
        return self._bounded_cap if self.worst_case() is not None else None

    def _bounded_cap(self, gpu_hours_left: float, world: int, wall_left: float, expected_s: float) -> int | None:
        """The worst case as the attempt's wall cap when it fits the wall and GPU-hours left, else None (not started)."""
        need = self.worst_case()
        room = min(float(wall_left), float(gpu_hours_left) * 3600.0 / max(1, int(world)))
        return int(math.ceil(need)) if room >= need else None

    def admits(self, ledger, budget, world: int | None = None) -> str | None:
        """Basic-agent admission of one attempt of this runner (None = admitted, else the limiting dimension)."""
        ex = ledger.exhausted()
        if ex:
            return ex[0]
        need = self.worst_case()
        if need is None:
            return ledger.admit_screen()
        world = int(world or len(self.gpus))
        if budget.wall_seconds - ledger.wall() < need:
            return 'wall_seconds'
        if budget.gpu_hours - ledger.used['gpu_hours'] < world * need / 3600.0:
            return 'gpu_hours'
        return None

    def contain(self) -> dict:
        """Kill every process of the trust-domain uids; raises ContainmentError when one survives."""
        return {str(u): uid_kill(self.ops, u) for u in self.domain_uids()}

    # ------------------------------------------------------------ ScreenRunner
    def run(self, task, config: dict, overlay: dict | None, purpose: str, run_id: str, **kw) -> dict:
        attempt_id = new_attempt_id()
        candidate_id = str(kw.pop('candidate_id', None) or attempt_id)
        t0, utc_start = self.clock(), utc_now()
        lease = GpuLease(self.ops, env=self.env, gpus=self.gpus, mode=self.lease_mode, held_run_id=self.held_run_id,
                         sibling_uids=self.sibling_uids)
        result, failure, held, containment = None, None, 0.0, {}
        try:
            t_held = self.clock()                    # charged from the first claim RPC to the last release RPC
            try:
                lease.acquire(run_id, attempt_id)
            except BaseException:
                # R3-03: a partial acquire held claims until its rollback ended - charged conservatively (the cost of
                # a failed attempt is incomplete anyway)
                if lease.taken or (lease.rollback or {}).get('ambiguous'):
                    held = max(0.0, self.clock() - t_held)
                raise
            try:
                result = self.execute(task, dict(config or {}), dict(overlay or {}), purpose, run_id, attempt_id, candidate_id, **kw)
            finally:
                try:
                    containment = {'ok': True, 'uids': self.contain()}
                except Exception as exc:             # noqa: BLE001 - survivors: keep the GPUs leased, record it
                    containment = {'ok': False, 'error': f'{type(exc).__name__}: {str(exc)[:500]}'}
                try:
                    if containment['ok']:
                        lease.release()
                finally:
                    held = max(0.0, self.clock() - t_held)       # X3 #6: the release RPCs and their retries are held time
        except OccupiedError as exc:
            failure = ('occupied_sibling' if (lease.report or {}).get('sibling_only') else 'occupied', exc)
        except Exception as exc:                     # noqa: BLE001 - recorded and charged, never raised
            failure = ('runner_error', exc)
        receipt = self.receipt(task, config, overlay, purpose, run_id, attempt_id, candidate_id, result, failure,
                               utc_start=utc_start, wall=max(0.0, self.clock() - t0), held=held,
                               occupancy=lease.report)
        receipt['containment'] = containment
        receipt['lease_release'] = {'kept': list(lease.acquired), 'failures': list(lease.release_failures)}
        if lease.rollback is not None:
            receipt['lease_rollback'] = lease.rollback
        if lease.acquired:                           # X3 #6: a kept claim keeps costing after this receipt
            why = ('containment failed, the claims are kept' if containment and not containment.get('ok')
                   else 'the release failed')
            receipt['cost']['accounting_complete'] = False
            receipt['cost']['incomplete_reasons'] = [f'{why}; still held: {sorted(lease.acquired)}']
        if containment and not containment.get('ok'):
            receipt['valid'] = False
            receipt['metric']['value'] = receipt['step_time']['mean_window_max_rank_s'] = None
            receipt['invalid_reasons'] = list(receipt['invalid_reasons']) + [f'containment: {containment["error"]}']
            receipt['guard_rejections'] = list(receipt['guard_rejections']) + [{'id': 'containment', 'detail': containment['error']}]
        signing.sign(receipt)
        self._store(receipt)
        self.attempts.append({'attempt_id': attempt_id, 'candidate_id': candidate_id, 'purpose': purpose,
                              'valid': receipt['valid'], 'metric': receipt['metric']['value'], 'utc': receipt['utc_end']})
        self.log(f'{run_id}/{candidate_id}/{attempt_id}: {purpose} valid={receipt["valid"]} '
                 f'{task.metric_name}={receipt["metric"]["value"]}')
        return receipt

    def receipt(self, task, config, overlay, purpose, run_id, attempt_id, candidate_id, result: EnvRun | None,
                failure, *, utc_start: str, wall: float, held: float, occupancy: dict | None) -> dict:
        world = int(task.world_size)
        base = {'schema': SCHEMA_RECEIPT, 'env': self.env or task.env, 'task_id': task.id, 'arm': task.arm, 'run_id': run_id,
                'attempt_id': attempt_id, 'candidate_id': candidate_id, 'purpose': purpose,
                'candidate_digest': (result.candidate_digest if result and result.candidate_digest else candidate_digest(config, overlay)),
                'frozen_digest': task.frozen_digest, 'task_digest': task.digest, 'node': self.node, 'gpus': list(self.gpus),
                'world_size': world, 'deviations': sorted(set(self.deviations) | set((result.extra if result else {}).get('deviations', ())))}
        if result is None:
            kind, exc = failure
            reasons = [f'{kind}: {type(exc).__name__}: {str(exc)[:600]}']
            base.update(valid=False, invalid_reasons=reasons, step_time={'mean_window_max_rank_s': None},
                        metric={'name': task.metric_name, 'value': None, 'direction': 'min', 'raw': {}},
                        judge=None, judge_valid=False, guard_rejections=[], guard_reports=[], phases={},
                        runner_error={'kind': kind, 'type': type(exc).__name__, 'message': str(exc)[:2000]})
            complete = False
        else:
            rejections, reports = list(result.guards.rejections), list(result.guards.reports)
            judge_valid = bool(result.judge_valid)            # the environment judge's own verdict, never rewritten
            if judge_valid and not _finite_pos(result.metric_value):
                # A judge success without a usable metric is an integrity failure of the outputs, recorded as a guard.
                rejections.append({'id': 'output_integrity',
                                   'detail': f'judge reported success without a finite positive metric ({result.metric_value!r})'})
            valid = judge_valid and not rejections
            reasons = [] if judge_valid else ([f'judge: {r}' for r in result.judge_reasons] or ['judge: invalid'])
            reasons += [f'guard:{g["id"]}: {g["detail"]}' for g in rejections]
            value = float(result.metric_value) if valid else None
            base.update(valid=valid, invalid_reasons=reasons, step_time={'mean_window_max_rank_s': value},
                        metric={'name': task.metric_name, 'value': value, 'direction': 'min',
                                'raw': {**(result.metric_raw or {}), 'judge_value': result.metric_value}},
                        judge=result.judge, judge_valid=judge_valid, guard_rejections=rejections, guard_reports=reports,
                        phases={**(result.phases or {})})
            for k, v in (result.extra or {}).items():
                if k not in base and k != 'deviations':
                    base[k] = v
            complete = True
        base['phases']['total'] = round(wall, 3)
        base.update(utc_start=utc_start, utc_end=utc_now(), wall_s=round(wall, 3),
                    cost={'gpu_seconds': round(world * held, 3), 'wall_seconds': round(wall, 3), 'accounting_complete': complete,
                          'basis': 'world_size x seconds the GPUs were held (materialize, run, judge, collect)'},
                    occupancy={'problems': (occupancy or {}).get('problems'), 'lease': (occupancy or {}).get('lease'),
                               'apps': ((occupancy or {}).get('probe') or {}).get('apps'),
                               'hostname': ((occupancy or {}).get('probe') or {}).get('hostname')})
        return json.loads(json.dumps(jsonable(base), allow_nan=False))

    def _store(self, receipt: dict) -> None:
        if self.out_dir:
            d = self.out_dir / 'receipts'
            d.mkdir(parents=True, exist_ok=True)
            (d / f'{receipt["attempt_id"]}.json').write_text(json.dumps(receipt, indent=1, ensure_ascii=False), encoding='utf-8')
