"""Runner glue of the ENV-A arms (design §2, AMENDMENT_EXT_001): A-NAT, A-ISW (weak), A-RND, and the locked re-score.

The orchestrator runs the arms; these functions wire the same runner, judge, budget and trace format. Every arm holds
its node for the whole run (arm_lease: a run-scoped node lease owned by infrarsi-ext plus a run-scoped GPU claim), so
the runner's per-score attempts use lease mode 'held'; before the arm lease is released every process of the arm's own
uid pair (candidate uid, judge uid; Codex W9 B-02 / R2 N-01) is killed and verified gone, and the lease's containment
and release outcome is returned for the arm result and its events (Codex W9 R2 M-02): anything retained -- a survivor,
an incomplete release, an incomplete acquire rollback -- raises ArmLeaseError carrying the retained paths. Each arm
first runs the runner's CPU-only `prepare` (safetensors-only reference, judge-owned base-model copy, uid availability
record). The two LLM arms share one metered weak driver (DeepSeek-V4-Flash via vLLM); tokens are reported, not
charged (D1).
"""
from __future__ import annotations

import contextlib
import copy
import os

from ...agent.optimize import optimize
from ...agent.search import SearchSession
from ..common.env_judge import EnvJudge
from ..common.remote_exec import GpuLease, assert_uid_unused, uid_kill
from . import knobs, native
from . import tools as tools_mod
from .driver import LlmMeter, MeteringDriver, RequestSeed, NATIVE_CONTEXT, phase_e_driver
from .phase_e import run_a_is2
from .hooks import EnvAHooks
from .runner import RebenchLlmfRunner
from .task import WEAK_ENDPOINT_CONTEXT, build_env_a_task

WINDOW_8H_S = 28_800


# ---------------------------------------------------------------- driver (A-NAT / A-ISW)
def weak_driver(*, base_url: str | None = None, served_model: str | None = None,
                endpoint_context: int = WEAK_ENDPOINT_CONTEXT, timeout_seconds: float = 600,
                temperature: float | None = None, request_seed: int | None = None, seed_per_call: bool = True) -> tuple:
    """`agent.arms.weak_driver(structured=True)` with the amendment overrides (timeout 600 s, backend context 98_304),
    wrapped in a MeteringDriver. Returns (metered_driver, route, meter). base_url / served_model override via env the
    way `agent.arms.selfhost_settings` reads them. AMENDMENT_EXT_006 (A-NAT restart): `temperature` replaces the
    weak driver's greedy 0.0 (1.0 = METR's default sampling); `request_seed` puts a seed into every request body
    (RequestSeed: base + call index unless seed_per_call=False). Nothing else changes (JSON mode, thinking off,
    max_tokens, top_p)."""
    from ...agent.arms import weak_driver as _weak_driver
    if base_url:
        os.environ['INFRARSI_WEAK_URL'] = base_url
    if served_model:
        os.environ['INFRARSI_WEAK_MODEL'] = served_model
    route, backend = _weak_driver(structured=True, selfhost=bool(base_url or served_model))   # a named endpoint
    if hasattr(backend, 'timeout_seconds'):
        backend.timeout_seconds = float(timeout_seconds)
    if endpoint_context and hasattr(backend, 'context_tokens'):
        backend.context_tokens = int(endpoint_context)
    if temperature is not None:
        backend.temperature = float(temperature)
    inner = RequestSeed(backend, request_seed, per_call=seed_per_call) if request_seed is not None else backend
    meter = LlmMeter(serving_gpus=int((getattr(backend, 'selfhost', None) or {}).get('gpus') or 0))
    return MeteringDriver(inner, meter), route, meter


# ---------------------------------------------------------------- lease / runner
class ArmLeaseError(RuntimeError):
    """The arm lease was not cleaned up completely; `state` (the arm_lease record) lists the retained paths."""

    def __init__(self, msg: str, state: dict):
        super().__init__(msg)
        self.state = state


@contextlib.contextmanager
def arm_lease(ops, run_id: str, gpus, *, uids=(), group_run_id: str | None = None, events: list | None = None):
    """Hold the arm's GPUs for the whole run; group_run_id = the orchestrator's node lease for co-located arms. `uids`
    (the runner's domain_uids(): the arm's own pair) must be unused at the start; on exit every process of them is
    killed and verified gone before the lease is released.

    Yields `state`, which the caller puts into the arm result: acquired paths, then on exit the per-uid containment
    records, the released and retained (`kept`) paths, `release_failures` and `complete`. `events` (optional list)
    receives the same records. Fail closed: ArmLeaseError carrying `state` when a process survived (claims kept), when
    the final release left a path behind, or when a failed acquire's rollback did."""
    lease = GpuLease(ops, env='rebench_llmf', gpus=gpus, mode='held' if group_run_id else 'own', held_run_id=group_run_id)
    events = events if events is not None else []
    state = {'run_id': run_id, 'node': ops.node, 'gpus': [int(g) for g in gpus], 'mode': lease.mode, 'uids': list(uids),
             'acquired': [], 'containment': [], 'released': [], 'kept': [], 'release_failures': [], 'complete': False}
    for u in uids:
        assert_uid_unused(ops, u)
    try:
        lease.acquire(run_id, f'run:{run_id}', scope='run')
    except BaseException as exc:
        state.update(kept=list(lease.acquired), release_failures=list(lease.release_failures),
                     error=f'acquire: {type(exc).__name__}: {str(exc)[:400]}')
        events.append({'kind': 'arm_lease_acquire_failed', **state})
        if lease.acquired or lease.release_failures:
            raise ArmLeaseError(f'{ops.node}: arm {run_id}: acquire failed and its rollback retained {lease.acquired}',
                                state) from exc
        raise
    state['acquired'] = list(lease.acquired)
    events.append({'kind': 'arm_lease_acquired', 'acquired': list(lease.acquired)})
    try:
        yield state
    finally:
        survivors = []
        for u in uids:
            try:
                state['containment'].append({'kind': 'contain', **uid_kill(ops, u)})
            except Exception as exc:                       # noqa: BLE001 - survivors (or no proof) keep the GPUs leased
                survivors.append(u)
                state['containment'].append({'kind': 'contain_failed', 'uid': u, 'error': f'{type(exc).__name__}: {exc}'[:500]})
        if survivors:
            state.update(kept=list(lease.acquired), error=f'processes of uids {survivors} survived; claims kept')
            events.append({'kind': 'arm_lease_kept', **state})
            raise ArmLeaseError(f'{ops.node}: arm {run_id}: processes of uids {survivors} survived; retained {lease.acquired}',
                                state)
        state['released'] = lease.release()
        state.update(kept=list(lease.acquired), release_failures=list(lease.release_failures), complete=not lease.acquired)
        events.append({'kind': 'arm_lease_release', **state})
        if lease.acquired:
            raise ArmLeaseError(f'{ops.node}: arm {run_id}: final release incomplete; retained {lease.acquired}', state)


def make_runner(ops, task, *, gpus=(0, 1, 2, 3), run_id: str, reference_src: str | None = None, out_dir=None,
                enforce_g7: bool = False, group_run_id: str | None = None, g4_share: bool = False,
                timeout_s: int = 1200, base_solution: dict | None = None) -> RebenchLlmfRunner:
    """The arm's runner; attempts run under the arm's held lease (held_run_id = the arm's node-lease run)."""
    return RebenchLlmfRunner(ops, task=task, gpus=gpus, enforce_g7=enforce_g7, reference_src=reference_src,
                             g4_share=g4_share, timeout_s=timeout_s, base_solution=base_solution, lease_mode='held',
                             held_run_id=group_run_id or run_id, out_dir=out_dir)


# ---------------------------------------------------------------- arms
def run_a_nat(task, driver, runner, ops, *, run_id: str, out_dir=None, raw_instructions=None,
              context_tokens: int = native.CONTEXT_TOKENS, keep_last: int = native.KEEP_LAST, clock=None,
              dry_run: bool = False, restart_of: str | None = None) -> dict:
    if (task.arm or 'A-NAT') != 'A-NAT':
        raise ValueError(f'{task.arm}: not A-NAT')
    return native.run_native(task, driver, runner, ops, run_id=run_id, out_dir=out_dir,
                             raw_instructions=raw_instructions, context_tokens=context_tokens, keep_last=keep_last,
                             clock=clock, dry_run=dry_run, restart_of=restart_of)


def run_a_nat2(task, driver, runner, ops, *, run_id: str, out_dir=None, raw_instructions=None,
               clock=None, dry_run: bool = False) -> dict:
    """Phase E native arm: EXT_006 basic-agent loop, Qwen3, and an initial charged S0 score."""
    if task.arm != 'A-NAT2':
        raise ValueError(f'{task.arm}: not A-NAT2')
    from .task import arm_deviations
    runner.deviations = arm_deviations(task.arm)
    return native.run_native(task, driver, runner, ops, run_id=run_id, out_dir=out_dir,
                             raw_instructions=raw_instructions, context_tokens=NATIVE_CONTEXT,
                             clock=clock, dry_run=dry_run)


def _weak_package(package, *, max_turns: int, dst):
    """A working copy of the A0 package with modes.weak.max_turns overridden (the frozen file is not edited); its
    policy flows into optimize so the trace's policy field records max_turns (AMENDMENT Q1/Q2). The copy's manifest is
    refreshed so its content digest stays consistent (hooks.split_guard verifies it)."""
    import yaml
    from ...rsi.package import OptimizerPackage, copy_package, write_manifest
    src = package if isinstance(package, OptimizerPackage) else OptimizerPackage.load(package)
    root = copy_package(src.root, dst)
    pol_path = root / 'policy.yaml'
    doc = yaml.safe_load(pol_path.read_text(encoding='utf-8'))
    doc['modes']['weak']['max_turns'] = int(max_turns)
    pol_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding='utf-8')
    write_manifest(root, label='A0-isw-max_turns', parent_digest=src.digest,
                   producer={'override': 'modes.weak.max_turns', 'value': int(max_turns), 'by': 'ext.rebench_llmf.arms'})
    return OptimizerPackage.load(root)


def run_a_isw(task, package, driver, runner, *, run_id: str, out_dir=None, context_limit: int = 32_000,
              max_turns: int = 200, package_work_dir=None, frozen_package_digest: str | None = None,
              dry_run: bool = False, clock=None, **kw) -> dict:
    """A-ISW: InfraRSI A0 in weak mode (DeepSeek-V4-Flash) via agent.optimize with the ENV-A hooks. mode='weak',
    context_limit 32K, verifier/skills/memory on, run-level max_turns=200 recorded in the trace policy."""
    if task.arm != 'A-ISW':
        raise ValueError(f'{task.arm}: not A-ISW')
    hooks = EnvAHooks(task, frozen_package_digest=frozen_package_digest)
    pkg = package
    if max_turns is not None and package_work_dir is not None:
        pkg = _weak_package(package, max_turns=max_turns, dst=package_work_dir)
    if dry_run:
        return {'dry_run': True, 'arm': 'A-ISW', 'mode': 'weak', 'context_limit': context_limit,
                'max_turns': max_turns, 'layers': list(hooks.layers), 'budget': task.budget,
                'package_digest': getattr(pkg, 'digest', None), 'uid': runner.uid}
    prep = runner.prepare(run_id)
    trace = optimize(task, pkg, driver, task.search_budget(), {}, runner, judge=EnvJudge('rebench_llmf'),
                     run_id=run_id, arm='A-ISW', out_dir=out_dir, mode='weak', hooks=hooks, context_limit=context_limit,
                     use_verifier=True, use_skills=True, use_memory=True, clock=clock, **kw)
    locked = locked_rescore_selection(runner, task, trace, run_id=run_id)
    return {'arm': 'A-ISW', 'trace': trace.to_dict() if hasattr(trace, 'to_dict') else trace, 'locked': locked,
            'prepare': prep}


def run_a_rnd(task, runner, *, run_id: str, out_dir=None, seed: int = 42, n: int = 60, clock=None,
              dry_run: bool = False) -> dict:
    """A-RND: random sampling of the §2.4 knob space (seed 42) through SearchSession, no LLM; locked re-score of best."""
    if task.arm != 'A-RND':
        raise ValueError(f'{task.arm}: not A-RND')
    hooks = EnvAHooks(task, layers=('L0',))
    points = knobs.sample_order(seed, n)
    if dry_run:
        return {'dry_run': True, 'arm': 'A-RND', 'seed': seed, 'n_points': len(points), 'budget': task.budget,
                'uid': runner.uid}
    prep = runner.prepare(run_id)
    session = SearchSession(task=task, budget=task.search_budget(), runner=runner, judge=EnvJudge('rebench_llmf'),
                            run_id=run_id, arm='A-RND', out_dir=out_dir, start_config={}, mode='strong', layers=('L0',),
                            clock=clock, hooks=hooks, driver_info={'kind': 'A-RND', 'llm': None, 'seed': seed})
    session.screen('c0')
    stop, drawn = None, 0
    for point in points:
        if session.stop_flag or session.ledger.exhausted():
            break
        drawn += 1
        fb = session.propose_config(config=point, base='c0', origin='random')
        if 'candidate_id' in fb:
            session.screen(fb['candidate_id'])
    stop = session.stop_flag or ('budget:' + session.ledger.exhausted()[0] if session.ledger.exhausted() else 'space_exhausted')
    trace = session.finalize(stop_reason=stop, extra={'seed': seed, 'drawn': drawn, 'space_size': knobs.SPACE_SIZE})
    locked = locked_rescore_selection(runner, task, trace, run_id=run_id)
    return {'arm': 'A-RND', 'trace': trace.to_dict() if hasattr(trace, 'to_dict') else trace, 'locked': locked,
            'prepare': prep}


def selection_is_start(trace, base: dict | None = None) -> bool:
    """Decision D20: the search's selection is the unchanged starting solution S0 -- the start candidate itself, or a
    selection whose rendered finetune.sh / .yaml (and no extra overlay file) are the starting files."""
    t = trace.to_dict() if hasattr(trace, 'to_dict') else (trace or {})
    sel = t.get('selected')
    if not sel:
        return False
    if sel == (t.get('start') or 'c0'):
        return True
    from .runner import render_candidate
    from .task import read_base_solution
    base = base or read_base_solution()
    knob_cfg = {k: v for k, v in (t.get('selected_config') or {}).items() if k in knobs.KNOBS}
    norm, errs, _ = knobs.validate(knob_cfg) if knob_cfg else ({}, [], [])
    if errs:
        return False
    sh, yml, extra = render_candidate(base, norm, t.get('selected_overlay') or None)
    return sh == base['finetune.sh'] and yml == base['finetune.yaml'] and not extra


def d20_skip(selected, node: str | None) -> dict:
    """Decision D20 record: no locked re-score of the unchanged start; its locked value is the node's S0 reference
    (filled in from the node references by the block / summary), InvalidAccept 0."""
    return {'locked': None, 'skipped': 'selection_is_start', 'selected': selected, 'node': node,
            'locked_value': 'node S0 reference', 'invalid_accept': 0, 'rule': 'D20'}


def locked_rescore_selection(runner, task, trace, *, run_id: str) -> dict:
    """Locked re-score (design §1.4, with G4, clean slate on the node) of a search's selection: run its selected
    knobs/overlay at purpose 'score'. The decision value of A-ISW / A-RND. D20: not run when the selection is the
    unchanged start (`skipped: selection_is_start`)."""
    t = trace.to_dict() if hasattr(trace, 'to_dict') else trace
    if selection_is_start(t, getattr(runner, 'base', None)):
        return d20_skip(t.get('selected'), getattr(runner, 'node', None))
    config = t.get('selected_config') or {}
    overlay = t.get('selected_overlay') or None
    knob_cfg = {k: v for k, v in config.items() if k in knobs.KNOBS}
    receipt = runner.run(task, knob_cfg, overlay, 'score', run_id, run_g4=True, candidate_id='locked')
    from .guards import judge_metrics
    m = judge_metrics(receipt.get('judge'))
    return {'locked': {'valid': bool(receipt.get('valid')), 'execution_time': m.get('execution_time'),
                       'score': m.get('score'), 'norm_diff': m.get('norm_diff'),
                       'guard_rejections': [r['id'] for r in receipt.get('guard_rejections', [])]},
            'selected': t.get('selected'), 'attempt_id': receipt.get('attempt_id')}


# ---------------------------------------------------------------- report helpers (AMENDMENT Q1/Q3)
def t_best_search_at(trace, *, seconds: int = WINDOW_8H_S) -> float | None:
    """Best (lowest) valid execution_time among scorings completed within the first `seconds` of wall clock."""
    t = trace.to_dict() if hasattr(trace, 'to_dict') else (trace or {})
    best = None
    for s in t.get('screens') or []:
        cum = (s.get('cum') or {}).get('wall_seconds')
        st = s.get('step_time')
        if s.get('valid') and type(st) in (int, float) and type(cum) in (int, float) and cum <= seconds:
            best = st if best is None else min(best, st)
    return best


def llm_report(meter: LlmMeter | None) -> dict:
    """The reported-only ENV-A LLM aggregates (D1): never charged. Empty for A-RND (no LLM)."""
    return meter.summary() if meter is not None else {'llm_calls': 0, 'llm_tokens_total': 0, 'llm_tokens_uncached': 0,
                                                      'llm_wall_hours': 0.0, 'llm_gpu_hours_h20': 0.0, 'llm_usd': 0}
