"""Runner glue of the ENV-C arms (design §4.3): C-IS, C-RND, C-NAT-free, C-NAT-infra, the locked re-score.

The orchestrator runs the arms; these functions only wire the same runner, judge, budget and trace format:
  run_c_is      InfraRSI A0 via agent.optimize with AutoresearchHooks (patch-only, verifier on the §4.2 checklist)
  run_c_rnd     the 128-point knob space in a seed-42 permutation through SearchSession (no LLM)
  run_c_nat     the basic agent on program.md (free or infra track; native.run_native)
  locked_pair   the arm's final train.py and the baseline back to back on the same GPU (order alternates by arm)
Every arm holds its GPU for its whole run (arm_lease): a node lease (or the orchestrator's co-location lease) plus a
run-scoped GPU claim; the runner's attempts then use lease mode 'held'. Every arm is one trust domain (task.ARM_UIDS,
Codex W9 B-02/B-03): its uid must be unused when the arm starts (no passwd entry, no process), its agent's shell and all
its trainings run under it, and every process of it is killed (and verified gone) before the arm's claims are released.
Arm entry points accept only arm tasks with the D-AR-7 loader (task.require_arm_task, B-04).
"""
from __future__ import annotations

import contextlib

from ...agent.optimize import optimize
from ...agent.search import SearchSession
from ..common.env_judge import EnvJudge
from ..common.remote_exec import ContainmentError, GpuLease, LeaseError, assert_uid_unused, uid_kill
from . import knobs, native, source
from .hooks import AutoresearchHooks
from .runner import AutoresearchRunner
from .task import CALIB_UIDS, ENVC_UIDS, require_arm_task


def contain_all(ops, uids) -> list[dict]:
    """uid_kill for every uid (all are tried); raises ContainmentError naming the uids that kept a process."""
    out, bad = [], []
    for u in uids:
        try:
            out.append({'kind': 'contain', **uid_kill(ops, u)})
        except Exception as exc:                    # noqa: BLE001 - recorded; the others are still killed
            bad.append(u)
            out.append({'kind': 'contain_failed', 'uid': u, 'error': f'{type(exc).__name__}: {exc}'[:500]})
    if bad:
        raise ContainmentError(f'{ops.node}: processes of uids {bad} survived: {out}')
    return out


@contextlib.contextmanager
def arm_lease(ops, run_id: str, gpus, *, group_run_id: str | None = None, uids=(), events: list | None = None):
    """Hold the arm's GPUs for the whole run. group_run_id: the orchestrator's node lease for co-located arms. uids: the
    arm's trust domain(s) - unused at the start, all their processes killed before the claims are released (the
    claims are kept when one survives). `events` (also lease.events) receives the containment and release records;
    the last one is the final lease state. Fail closed (Codex W9 M-02): an incomplete acquire rollback, surviving
    processes or claims that could not be released raise (LeaseError / ContainmentError) after being recorded."""
    lease = GpuLease(ops, env='autoresearch', gpus=gpus, mode='held' if group_run_id else 'own', held_run_id=group_run_id,
                     sibling_uids=ENVC_UIDS)
    lease.events = events if events is not None else []
    for u in uids:
        assert_uid_unused(ops, u)
    try:
        lease.acquire(run_id, f'run:{run_id}', scope='run')
    except BaseException as exc:
        lease.events.append({'kind': 'acquire_failed', 'error': f'{type(exc).__name__}: {str(exc)[:500]}', **lease.state()})
        raise
    try:
        yield lease
    finally:
        try:
            lease.events += contain_all(ops, uids)
        except ContainmentError as exc:             # survivors: the GPUs stay claimed, the run reports it
            lease.events.append({'kind': 'claims_kept', 'error': str(exc)[:1000], **lease.state(), 'complete': False})
            raise
        released = lease.release()
        lease.events.append({'kind': 'release', 'released': released, **lease.state()})
        if lease.acquired:
            raise LeaseError(f'arm {run_id}: claims not released: {lease.acquired}', kept=lease.acquired,
                             failures=lease.release_failures)


def make_runner(ops, task, *, gpu: int, run_id: str, out_dir=None, colocated: bool = False, calib_uid: int | None = None,
                **kw) -> AutoresearchRunner:
    """The arm's runner; its attempts run under the arm's lease (mode 'held', held_run_id = the arm's node lease run) and
    the arm's trust-domain uid (calib_uid: one of task.CALIB_UIDS for the calibration's concurrent runs)."""
    require_arm_task(task)
    if calib_uid is not None and calib_uid not in CALIB_UIDS:
        raise ValueError(f'calibration uid {calib_uid} not in {CALIB_UIDS}')
    return AutoresearchRunner(ops, gpus=(gpu,), track=task.raw.get('track', 'infra'), expected=task.raw.get('expected_runtime'),
                              reference_losses=task.raw.get('reference_losses'), colocated=colocated,
                              train_uid=calib_uid if calib_uid is not None else task.raw['uid'],
                              lease_mode='held', held_run_id=kw.pop('held_run_id', run_id), out_dir=out_dir, **kw)


def _check(task, runner) -> None:
    require_arm_task(task)
    if getattr(runner, 'train_uid', None) != task.raw['uid']:
        raise ValueError(f'{task.arm}: runner uid {getattr(runner, "train_uid", None)} != the arm uid {task.raw["uid"]}')


def run_c_is(task, package, driver, runner, *, run_id: str, out_dir=None, frozen_package_digest: str | None = None, **kw):
    _check(task, runner)
    hooks = AutoresearchHooks(task, frozen_package_digest=frozen_package_digest)
    return optimize(task, package, driver, task.search_budget(), {}, runner, judge=EnvJudge('autoresearch'), run_id=run_id,
                    arm='C-IS', out_dir=out_dir, mode='strong', hooks=hooks, **kw)


def run_c_rnd(task, runner, *, run_id: str, out_dir=None, seed: int = 42, clock=None):
    _check(task, runner)
    hooks = AutoresearchHooks(task)
    session = SearchSession(task=task, budget=task.search_budget(), runner=runner, judge=EnvJudge('autoresearch'), run_id=run_id,
                            arm='C-RND', out_dir=out_dir, start_config={}, mode='strong', layers=('L0',), clock=clock,
                            hooks=hooks, driver_info={'kind': 'C-RND', 'llm': None, 'seed': seed})
    session.screen('c0')
    stop, drawn = None, 0
    for point in knobs.sample_order(seed):
        if session.stop_flag or session.ledger.exhausted():
            break
        drawn += 1
        fb = session.propose_config(config=point, base='c0', origin='random')
        if 'candidate_id' in fb:
            session.screen(fb['candidate_id'])
    else:
        stop = 'space_exhausted'
    stop = stop or session.stop_flag or ('budget:' + session.ledger.exhausted()[0] if session.ledger.exhausted() else 'unknown')
    screened = sum(1 for s in session.screens if s.get('candidate_id') != 'c0')
    size = len(knobs.sample_order(seed))
    return session.finalize(stop_reason=stop, extra={'seed': seed, 'drawn': drawn, 'space': knobs.SPACE, 'space_size': size,
                                                     'space_coverage': round(screened / size, 4) if size else None,
                                                     'knob_space_sha256': (task.raw.get('knob_space') or {}).get('sha256')})


def run_c_nat(task, driver, runner, ops, *, run_id: str, out_dir=None, tag: str | None = None, **kw) -> dict:
    if not (task.arm or '').startswith('C-NAT'):
        raise ValueError(f'{task.arm}: not a native arm')
    _check(task, runner)
    return native.run_native(task, driver, runner, ops, run_id=run_id, out_dir=out_dir, tag=tag, **kw)


def baseline_skip(candidate_text: str) -> dict | None:
    """D20 (the AMENDMENT_EXT_003 Q2 principle): when the arm's final selection is the unchanged baseline (C-RND drew no
    better point, the agent's best is the base train.py, C-IS selected c0, nothing valid), no locked pair is run - no
    A/A pair; the locked value is the reference: y = 0, invalid_accept_locked 0, both halves null."""
    if candidate_text != source.base_train():
        return None
    return {'skipped': 'selection_is_baseline', 'y': 0.0, 'invalid_accept_locked': 0, 'baseline': None, 'candidate': None,
            'order': None, 'selection_sha256': source.sha256_text(candidate_text)}


def locked_pair(runner, task, candidate_text: str, *, run_id: str, arm_index: int) -> dict:
    """Locked re-score (design §1.4): {baseline, candidate} back to back on the same GPU; odd arm index = candidate first.
    This is a functional acceptance of the arm's final choice, not a seed repeat. Both members run sequentially under
    the arm's runner (its trust-domain uid; D-AR-7 loader). A selection that is the baseline runs nothing (D20)."""
    _check(task, runner)
    skip = baseline_skip(candidate_text)
    if skip:
        return skip
    order = ('candidate', 'baseline') if arm_index % 2 else ('baseline', 'candidate')
    out = {}
    for which in order:
        overlay = {'train.py': candidate_text} if which == 'candidate' and candidate_text != source.base_train() else {}
        out[which] = runner.run(task, {}, overlay, 'score', run_id, candidate_id=f'lock-{which[:4]}')
    out['order'] = list(order)
    return out
