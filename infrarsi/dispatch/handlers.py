"""Run handlers by RUN_MATRIX arm. D provides FR, ED, FR_lifecycle, harness_selftest, FR_smoke_synthetic and
lifecycle_promoted, plus explicit 'deferred (P2, not in v0.1)' handlers for FR+ED_at_64_GPUs, S_match_1024_quality and
rebench_llm_foundry; other builders register theirs (infrarsi.agent / rsi / redteam / baselines expose
DISPATCH_HANDLERS = {arm: fn}).

A handler is fn(run: dict, nodes: list[str], ctx: Context) -> result dict; it blocks until the run is complete.
terminal_state(arm, result) maps the result to the dispatcher's terminal states:
  done        - the run completed (result status 'done' / 'feasible' / 'pass' / 'ok', or no status: search arms);
  infeasible  - FR or ED found no feasible configuration (design 1.4/1.6: a recorded outcome, t_ref falls back);
  invalid     - the run produced an invalid receipt (status 'invalid' / 'fail');
  failed      - anything else (status 'error', unknown status, infeasible for another arm, or an exception).
Only 'done' - and 'infeasible' of an FR/ED run - satisfies a dependency (queue.dep_ok).
"""
from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from infrarsi.contracts import EXECUTION_DIR, REPO_ROOT, RemotePaths, RunRequest, TaskSpec, digest, utc_now

EXTERNAL_MODULES = ('infrarsi.agent', 'infrarsi.rsi', 'infrarsi.redteam', 'infrarsi.baselines')
TERMINAL_STATES = ('done', 'infeasible', 'failed', 'invalid')
INFEASIBLE_OK_ARMS = ('FR', 'ED')                 # design 1.6: t_ref = FR -> ED -> first feasible
_STATUS = {'done': 'done', 'feasible': 'done', 'pass': 'done', 'ok': 'done', 'infeasible': 'infeasible',
           'not_applicable': 'infeasible', 'invalid': 'invalid', 'fail': 'invalid', 'failed': 'failed', 'error': 'failed'}
DEFERRED_NOTE = 'deferred (P2, not in v0.1)'
DEFERRED_ARMS = {'FR+ED_at_64_GPUs': 'planner-only check at 64 GPUs (P2-T3)',
                 'S_match_1024_quality': '1,024-step S_match quality run (E7, not executed by default)',
                 'rebench_llm_foundry': 'external RE-Bench task (P2-H7)'}
SELFTEST_OUT = EXECUTION_DIR / 'E0' / 'selftest.json'
SYN_PREFIX = 'D-SYN-'


@dataclasses.dataclass
class Context:
    store: object
    remote: object = None
    runner: Callable | None = None
    hw: dict | None = None
    log: Callable = print
    searches_root: object = None

    def search_dir(self, run_id: str, create: bool = True):
        """Out dir for an agent search of this run (SearchSession events.jsonl / trace.json); the monitor sums
        the LLM spend found under execution/searches/."""
        d = Path(self.searches_root or EXECUTION_DIR / 'searches') / run_id
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d


class Deferred(RuntimeError):
    """A RUN_MATRIX arm that v0.1 deliberately does not implement (P2)."""


def terminal_state(arm: str, result) -> tuple[str, str | None]:
    """(terminal state, reason) of a handler result; see the module docstring."""
    if not isinstance(result, dict):
        return 'failed', f'handler returned {type(result).__name__}, not a result dict'
    st = result.get('status')
    if st is None:
        return 'done', None
    t = _STATUS.get(str(st))
    if t is None:
        return 'failed', f'unknown handler status {st!r}'
    if t == 'infeasible' and arm not in INFEASIBLE_OK_ARMS:
        return 'failed', f'{arm} reported {st!r}; infeasibility is a recorded outcome only for FR / ED runs'
    return t, (None if t == 'done' else f'handler status {st!r}')


def attempt_cap(run: dict, parts: int = 3) -> int:
    return int(max(1800, float(run.get('wall_hours_cap') or 1) * 3600 / parts))


def _gpn(task: TaskSpec) -> int:
    return task.gpus if task.nodes == 1 else task.gpus // task.nodes


def _facts(task, ctx):
    from infrarsi.discover import workload_facts
    return workload_facts(task, remote=ctx.remote)


def _runner(ctx):
    if ctx.runner is not None:
        return ctx.runner
    from infrarsi.harness import run_screen
    return lambda req, **kw: run_screen(req, remote=ctx.remote, **kw)


def _remote(ctx):
    if ctx.remote is None:
        from infrarsi.harness.remote import default_remote
        ctx.remote = default_remote()
    return ctx.remote


def _read(ctx, nodes, path):
    return _remote(ctx).read_json((list(nodes) or ['WINH0'])[0], path) if path else None


def verified_receipt(ctx, nodes, path: str | None, task: TaskSpec, plan: str, *, role: str = 'reference',
                     purpose: str | None = None) -> tuple[dict | None, list[str]]:
    """Read a stored receipt and authenticate it with the harness's reference loader (validate.reference_problems)."""
    from infrarsi.harness import validate as V
    if not path:
        return None, ['no receipt recorded']
    try:
        rec = _read(ctx, nodes, path)
    except Exception as e:                                             # noqa: BLE001 - reported as a problem
        return None, [f'{path} unreadable: {e!r}'[:300]]
    problems = V.reference_problems(rec, task=task.raw, plan=plan, steps=task.screen_steps, window=task.window, role=role,
                                    purpose=purpose)
    return (rec if not problems else None), problems


def _screen_and_score(run, nodes, ctx, task, config, ref_screen=None, ref_score=None, role='candidate'):
    """Plan-A screen and plan-B score. role 'reference' only for the task's numerics reference (FR, or ED when FR
    is infeasible); a candidate compares against ref_screen / ref_score of the same plan."""
    rs = _runner(ctx)
    cap = attempt_cap(run)
    screen = rs(RunRequest(run_id=run['run_id'], purpose='screen', task=task.raw, config=config, nodes=list(nodes),
                           gpus_per_node=_gpn(task), input_plan='A', reference_receipt=ref_screen, wall_seconds_cap=cap,
                           numerics_role=role))
    score = rs(RunRequest(run_id=run['run_id'], purpose='score', task=task.raw, config=config, nodes=list(nodes),
                          gpus_per_node=_gpn(task), input_plan='B', reference_receipt=ref_score, wall_seconds_cap=cap,
                          numerics_role=role))
    return screen, score


def reference_paths(ref: dict) -> tuple[str | None, str | None, str | None]:
    """(plan-A screen, plan-B score, kind) of the task's numerics reference: FR when FR is feasible and valid, ED
    only after it ran as the reference because FR was infeasible; (None, None, None) while there is none. The plan-A
    screen alone qualifies FR as the screen reference (the plan-B score may be absent: locked pairs run their own)."""
    if ref.get('fr_status') == 'feasible' and ref.get('fr_screen'):
        return ref['fr_screen'], ref.get('fr_score'), 'FR'
    if ref.get('numerics_reference') == 'ED' and ref.get('ed_status') == 'feasible' and ref.get('ed_screen'):
        return ref['ed_screen'], ref.get('ed_score'), 'ED'
    return None, None, None


def _summary(rec):
    return None if rec is None else {'attempt_id': rec['attempt_id'], 'valid': rec['valid'],
                                     'step_time': (rec.get('step_time') or {}).get('mean_window_max_rank_s'),
                                     'rsd': (rec.get('step_time') or {}).get('rsd_window'),
                                     'failure_kind': (rec.get('harness') or {}).get('failure_kind'),
                                     'receipt': ((rec.get('harness') or {}).get('stored')
                                                 or f'{RemotePaths.receipts}/{rec["attempt_id"]}.json')}


def run_fr(run, nodes, ctx: Context) -> dict:
    from infrarsi.harness.probes import run_probe_chain
    from infrarsi.planner import FRChain
    task = TaskSpec.load(run['task'])
    chain = FRChain(task, _facts(task, ctx), ctx.hw)
    probes = run_probe_chain(chain, task, nodes, run['run_id'], runner=_runner(ctx), wall_seconds_cap=attempt_cap(run, 4))
    res = {'fr': chain.result(), 'probes': [_summary(p) for p in probes]}
    if chain.status != 'feasible':
        ctx.store.set_reference(task.id, fr_status=chain.status, fr_run=run['run_id'])
        return dict(res, status=chain.status)
    screen, score = _screen_and_score(run, nodes, ctx, task, chain.config, role='reference')
    res.update(screen=_summary(screen), score=_summary(score), status='feasible' if screen['valid'] and score['valid'] else 'invalid')
    ctx.store.set_reference(task.id, fr_status=res['status'], fr_run=run['run_id'], fr_config=chain.config,
                            fr_screen=_summary(screen)['receipt'], fr_score=_summary(score)['receipt'],
                            fr_score_rsd=_summary(score)['rsd'],
                            **({'numerics_reference': 'FR'} if res['status'] == 'feasible' else {}))
    return res


def run_ed(run, nodes, ctx: Context) -> dict:
    from infrarsi.discover import default_profile
    from infrarsi.harness.probes import run_probe_chain
    from infrarsi.planner import EDChain
    task = TaskSpec.load(run['task'])
    ref = ctx.store.references().get(task.id, {})
    # Numerics: compare with the FR reference; only when FR is infeasible does ED become the task's reference.
    if ref.get('fr_status') == 'feasible' and ref.get('fr_screen') and ref.get('fr_score'):
        role, ref_screen, ref_score = 'candidate', ref['fr_screen'], ref['fr_score']
        problems = []
        for path, plan in ((ref_screen, 'A'), (ref_score, 'B')):     # authenticated before any probe runs
            problems += [f'FR plan {plan}: {x}' for x in verified_receipt(ctx, nodes, path, task, plan)[1]]
        if problems:
            raise RuntimeError(f'{task.id}: FR reference receipts unusable ({"; ".join(problems[:4])}); not launching ED')
    elif ref.get('fr_status') in ('infeasible', 'not_applicable'):
        role, ref_screen, ref_score = 'reference', None, None
    else:
        raise RuntimeError(f'{task.id}: no FR numerics reference (fr_status {ref.get("fr_status")!r}); run or fix FR first')
    chain = EDChain(task, _facts(task, ctx), ctx.hw or default_profile(task.nodes))
    probes = run_probe_chain(chain, task, nodes, run['run_id'], runner=_runner(ctx), wall_seconds_cap=attempt_cap(run, 4))
    res = {'ed': chain.result(), 'probes': [_summary(p) for p in probes]}
    if chain.status != 'feasible':
        ctx.store.set_reference(task.id, ed_status=chain.status)
        return dict(res, status=chain.status)
    screen, score = _screen_and_score(run, nodes, ctx, task, chain.config, ref_screen, ref_score, role=role)
    res.update(screen=_summary(screen), score=_summary(score), status='feasible' if screen['valid'] and score['valid'] else 'invalid',
               numerics_role=role)
    extra = {'numerics_reference': 'ED'} if role == 'reference' and res['status'] == 'feasible' else {}
    ctx.store.set_reference(task.id, ed_status=res['status'], ed_config=chain.config, ed_screen=_summary(screen)['receipt'],
                            ed_score=_summary(score)['receipt'], ed_numerics_role=role, **extra)
    return res


def _lifecycle_block(task: TaskSpec) -> dict:
    """AMENDMENT-002: every lifecycle asks for deterministic kernels (fixed by the driver, never an agent option); the
    trainer falls back to the default kernels only for Gated DeltaNet and says so (deterministic_unavailable)."""
    wl = task.raw['workload']
    return {'save_at': int(wl['screen_steps']), 'resume_steps': int(wl['resume_continuation_steps']),
            'export': True, 'forward_compare': True, 'deterministic': True}


def run_lifecycle(run, nodes, ctx: Context) -> dict:
    """FR lifecycle (E0.3): only on a verified, feasible FR - its signed, valid plan-B reference receipt - and exactly
    the configuration that receipt ran (config digest). Continuity is checked inside the attempt (signals_resume/)."""
    task = TaskSpec.load(run['task'])
    ref = ctx.store.references().get(task.id, {})
    if ref.get('fr_status') != 'feasible' or not ref.get('fr_config') or not ref.get('fr_score'):
        raise RuntimeError(f'{task.id}: FR is not verified feasible (fr_status {ref.get("fr_status")!r}, config '
                           f'{"set" if ref.get("fr_config") else "missing"}, plan-B receipt {ref.get("fr_score")!r}); '
                           'FR_lifecycle needs the FR calibration first')
    fr_b, problems = verified_receipt(ctx, nodes, ref['fr_score'], task, 'B')
    if fr_b is not None and digest(ref['fr_config']) != fr_b.get('config_digest'):
        problems.append('the recorded fr_config does not hash to the FR plan-B receipt config_digest')
    if problems:
        raise RuntimeError(f'{task.id}: FR plan-B receipt {ref["fr_score"]} unusable for the lifecycle: '
                           + '; '.join(problems[:6]))
    req = RunRequest(run_id=run['run_id'], purpose='lifecycle', task=task.raw, config=ref['fr_config'], nodes=list(nodes),
                     gpus_per_node=_gpn(task), input_plan='B', wall_seconds_cap=attempt_cap(run, 1),
                     lifecycle=_lifecycle_block(task), reference_receipt=None, numerics_role='reference')
    rec = _runner(ctx)(req)
    return {'lifecycle': _summary(rec), 'status': 'pass' if rec['valid'] else 'fail', 'detail': rec.get('lifecycle'),
            'fr_score_attempt': fr_b.get('attempt_id'), 'config_digest': fr_b.get('config_digest')}


# ---------------------------------------------------------------- E0 / E7 arms

def _selftest_python() -> str:
    """The driver venv when present (E:/tr/tyc_sftv2aliyun/infrarsi/.venv), else this interpreter."""
    for rel in ('.venv/Scripts/python.exe', '.venv/bin/python'):
        p = REPO_ROOT / rel
        if p.exists():
            return str(p)
    return sys.executable


def _junit_counts(path: Path) -> dict | None:
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == 'testsuite' else list(root.iter('testsuite'))
    tot = {k: sum(int(s.get(k, 0) or 0) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    failed = []
    for case in root.iter('testcase'):
        if case.find('failure') is not None or case.find('error') is not None:
            failed.append(f'{case.get("classname")}::{case.get("name")}')
    return {'total': tot['tests'], 'failed': tot['failures'], 'errors': tot['errors'], 'skipped': tot['skipped'],
            'passed': tot['tests'] - tot['failures'] - tot['errors'] - tot['skipped'], 'failures': failed[:200]}


def run_selftest(run, nodes, ctx: Context, *, out_path: Path | None = None, python: str | None = None,
                 run_fn: Callable = subprocess.run) -> dict:
    """E0.1 (harness_selftest): `pytest -q -p no:cacheprovider tests/cpu` on the driver PC with the driver venv; the
    counts and failing test ids go to execution/E0/selftest.json. Any failure or error (or no test at all) fails the
    run."""
    out_path = Path(out_path or SELFTEST_OUT)
    py = python or _selftest_python()
    cap = attempt_cap(run, 1)
    with tempfile.TemporaryDirectory(prefix='isst') as td:
        junit = Path(td) / 'junit.xml'
        cmd = [py, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/cpu', f'--junitxml={junit}']
        t0 = time.monotonic()
        try:
            proc = run_fn(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=cap)
            rc, stdout, stderr = proc.returncode, proc.stdout or '', proc.stderr or ''
        except subprocess.TimeoutExpired as e:
            rc, stdout, stderr = None, str(e.stdout or ''), f'timeout after {cap} s'
        counts = _junit_counts(junit)
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    rec = {'schema': 'infrarsi-selftest-v1', 'run_id': run.get('run_id'), 'utc': utc_now(), 'cmd': cmd[:-1],
           'python': py, 'cwd': str(REPO_ROOT), 'returncode': rc, 'wall_s': round(time.monotonic() - t0, 1),
           'counts': counts, 'summary': lines[-1] if lines else None, 'stdout_tail': '\n'.join(lines[-40:]),
           'stderr_tail': stderr[-4000:]}
    try:
        from infrarsi.harness.codesync import code_digest
        rec['code_digest'] = code_digest()
    except Exception as e:                                             # noqa: BLE001 - informative only
        rec['code_digest'] = f'unavailable: {e!r}'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding='utf-8')
    bad = (rc != 0 or counts is None or counts['failed'] or counts['errors'] or not counts['total'])
    if bad:
        raise RuntimeError(f'CPU selftest failed (exit {rc}, counts {None if counts is None else {k: counts[k] for k in ("passed", "failed", "errors", "skipped")}}); '
                           f'see {out_path}')
    return {'status': 'done', 'selftest': str(out_path), 'counts': {k: counts[k] for k in ('passed', 'failed', 'errors', 'skipped', 'total')},
            'summary': rec['summary']}


def synthetic_task(task: TaskSpec) -> TaskSpec:
    """A copy of `task` on synthetic data (data.synthetic and a D-SYN-* key): INDEX checks are skipped by design
    and the task digest differs from the real task's, so no receipt of it can ever be the task's reference."""
    raw = copy.deepcopy(task.raw)
    data = dict(raw.get('data') or {})
    key = str(data.get('key') or 'DATA')
    if not key.startswith(SYN_PREFIX.rstrip('-')):
        key = SYN_PREFIX + (key[2:] if key.startswith('D-') else key)
    data.update(synthetic=True, key=key)
    raw['data'] = data
    out = TaskSpec(raw=raw)
    if out.digest == task.digest:                                      # pragma: no cover - guarded by construction
        raise RuntimeError('synthetic task copy has the real task digest')
    return out


def run_smoke_synthetic(run, nodes, ctx: Context) -> dict:
    """E0-SMOKE-S1-SYN (FR_smoke_synthetic): the FR probe chain plus one plan-A screen of the FR configuration on a
    synthetic copy of the task. Never recorded in references.json (it is no task reference)."""
    from infrarsi.harness.probes import run_probe_chain
    from infrarsi.planner import FRChain
    base = TaskSpec.load(run['task'])
    task = synthetic_task(base)
    chain = FRChain(task, _facts(base, ctx), ctx.hw)
    probes = run_probe_chain(chain, task, nodes, run['run_id'], runner=_runner(ctx), wall_seconds_cap=attempt_cap(run, 4))
    res = {'task': base.id, 'synthetic_data_key': task.raw['data']['key'], 'task_digest': task.digest,
           'real_task_digest': base.digest, 'fr': chain.result(), 'probes': [_summary(p) for p in probes],
           'note': 'synthetic smoke: never a task reference'}
    if chain.status != 'feasible':
        return dict(res, status='error' if chain.status in ('probing', 'error') else chain.status)
    screen = _runner(ctx)(RunRequest(run_id=run['run_id'], purpose='screen', task=task.raw, config=chain.config,
                                     nodes=list(nodes), gpus_per_node=_gpn(task), input_plan='A',
                                     wall_seconds_cap=attempt_cap(run), numerics_role='reference'))
    return dict(res, screen=_summary(screen), status='done' if screen['valid'] else 'invalid')


def _search_of(run: dict) -> str:
    from .queue import is_marker
    deps = [d for d in run.get('depends_on') or [] if not is_marker(d)]
    if len(deps) != 1:
        raise RuntimeError(f'{run["run_id"]}: expected exactly one search run dependency, got {deps}')
    return deps[0]


def run_lifecycle_promoted(run, nodes, ctx: Context) -> dict:
    """E7 (lifecycle_promoted): the task lifecycle on exactly the configuration and overlay of the E7 search's
    promoted candidate - read from that search's result.json / locked receipts and the candidate's own plan-B request
    (config digest and overlay file sha256 re-verified) - with the search's paired reference plan-B receipt (FR, or ED
    when FR is infeasible) as the numerics reference. Promotion is decided from the signed receipts, not from the
    search's 'promoted' flag: the locked candidate must be verified (t_cand_locked) and differ from the reference in its
    effective configuration or effective overlay set (a start candidate c0 that differs from FR is a promotion).
    Nothing to do when no verified candidate differs from the reference."""
    from infrarsi.harness import validate as V
    task = TaskSpec.load(run['task'])
    search_id = _search_of(run)
    sdir = Path(ctx.search_dir(search_id, create=False))
    result = json.loads((sdir / 'result.json').read_text(encoding='utf-8'))
    pair = result.get('locked_pair') or {}
    base = {'search_run': search_id, 'search_dir': str(sdir), 'selected': pair.get('selected'),
            'search_promoted_flag': pair.get('promoted')}
    if pair.get('t_cand_locked') is None:
        return dict(base, status='done', promoted=False,
                    note='the search has no verified locked candidate; there is no promoted configuration to run')
    kind = pair.get('reference_kind')
    cand_b = json.loads((sdir / 'locked' / 'candidate_plan_b.json').read_text(encoding='utf-8'))
    ref_b = json.loads((sdir / 'locked' / f'{str(kind).lower()}_plan_b.json').read_text(encoding='utf-8'))
    problems = [f'reference: {x}' for x in V.reference_problems(ref_b, task=task.raw, plan='B', steps=task.screen_steps,
                                                                 window=task.window)]
    problems += [f'candidate: {x}' for x in V.reference_problems(cand_b, task=task.raw, plan='B', steps=task.screen_steps,
                                                                  window=task.window, role='candidate')]
    if not problems and digest(cand_b.get('effective_config')) == digest(ref_b.get('effective_config'))             and cand_b.get('overlay_digest') == ref_b.get('overlay_digest'):
        return dict(base, status='done', promoted=False,
                    note='the selected configuration and overlay are the reference itself; no lifecycle to run')
    num = (cand_b.get('harness') or {}).get('numerics') or {}
    if num.get('compared') is not True or num.get('reference_attempt_id') != ref_b.get('attempt_id'):
        problems.append('candidate plan-B receipt was not verified against the paired reference')
    if cand_b.get('attempt_id') != (pair.get('candidate') or {}).get('attempt_id'):
        problems.append('locked candidate receipt is not the attempt result.json names')
    req = None
    if not problems:
        req = _read(ctx, nodes, f'{RemotePaths.runs}/{cand_b["run_id"]}/{cand_b["attempt_id"]}/request.json')
        if not isinstance(req, dict):
            problems.append('the locked candidate request.json is unreadable')
        elif digest(req.get('config')) != cand_b.get('config_digest') or digest(req.get('task')) != cand_b.get('task_digest'):
            problems.append('the locked candidate request.json does not hash to its receipt (config / task digest)')
    files = None
    if not problems and (req.get('overlay') or {}).get('files'):
        import hashlib
        ov = req['overlay']
        files = {}
        for name, sha in sorted(ov['files'].items()):
            data = _remote(ctx).get_bytes((list(nodes) or ['WINH0'])[0], f'{ov["dir"]}/{name}')
            if data is None or hashlib.sha256(data).hexdigest() != sha:
                problems.append(f'overlay file {name} missing or changed since the locked run')
            files[name] = data
    if problems:
        raise RuntimeError(f'{run["run_id"]}: promoted candidate of {search_id} cannot be bound: ' + '; '.join(problems[:6]))
    ref_path = (ref_b.get('harness') or {}).get('stored') or f'{RemotePaths.receipts}/{ref_b["attempt_id"]}.json'
    lreq = RunRequest(run_id=run['run_id'], purpose='lifecycle', task=task.raw, config=req['config'], nodes=list(nodes),
                      gpus_per_node=_gpn(task), input_plan='B', wall_seconds_cap=attempt_cap(run, 1),
                      lifecycle=_lifecycle_block(task), reference_receipt=ref_path, numerics_role='candidate')
    rec = _runner(ctx)(lreq, reference=ref_b, overlay_files=files)
    bind = [k for k in ('task_digest', 'config_digest', 'overlay_digest') if rec.get(k) != cand_b.get(k)]
    out = dict(base, promoted=True, reference_kind=kind, lifecycle=_summary(rec), detail=rec.get('lifecycle'),
               candidate_attempt=cand_b['attempt_id'], reference_attempt=ref_b['attempt_id'], unbound=bind,
               code_digest_same=rec.get('code_digest') == cand_b.get('code_digest'),
               search_promotion_lifecycle=result.get('promotion_lifecycle'))
    out['status'] = 'pass' if rec.get('valid') and not bind else 'fail'
    return out


def deferred_handler(arm: str) -> Callable:
    def handler(run, nodes, ctx):
        raise Deferred(f'{run.get("run_id")} ({arm}: {DEFERRED_ARMS.get(arm, "")}): {DEFERRED_NOTE}')
    handler.deferred = DEFERRED_NOTE
    handler.__name__ = f'deferred_{arm}'
    return handler


def is_deferred(handler) -> bool:
    return bool(getattr(handler, 'deferred', None))


def search_handler(search_fn: Callable, start: str = 'ED') -> Callable:
    """Adapter for agent-search arms (A0, A_R, baselines): search_fn(run=, task=, nodes=, start_config=, start_receipt=,
    reference_receipt=, runner=, out_dir=, ctx=) -> dict. The search writes events.jsonl / trace.json under
    execution/searches/<run_id>/ (the monitor sums LLM spend there); screens go through run_screen on the run's nodes
    (inside the dispatcher's run-scoped lease); the numerics reference is the task's reference screen receipt (plan A):
    FR's, or ED's when FR was infeasible. Without one reference_receipt is None and the arm must refuse to start."""
    key = {'ED': 'ed', 'FR': 'fr'}[start]

    def handler(run, nodes, ctx: Context) -> dict:
        task = TaskSpec.load(run['task'])
        ref = ctx.store.references().get(task.id, {})
        if not ref.get(f'{key}_config'):
            raise RuntimeError(f'{task.id}: no {start} config recorded yet (run E0-CAL/E1 {start} first)')
        out_dir = ctx.search_dir(run['run_id'])
        ref_screen, _, _ = reference_paths(ref)
        res = search_fn(run=run, task=task, nodes=list(nodes), start_config=ref[f'{key}_config'],
                        start_receipt=_read(ctx, nodes, ref.get(f'{key}_screen')),
                        reference_receipt=_read(ctx, nodes, ref_screen), runner=_runner(ctx), out_dir=out_dir, ctx=ctx)
        return {'search_dir': str(out_dir), 'start': start, 'result': res}
    return handler


BUILTIN = {'FR': run_fr, 'ED': run_ed, 'FR_lifecycle': run_lifecycle, 'harness_selftest': run_selftest,
           'FR_smoke_synthetic': run_smoke_synthetic, 'lifecycle_promoted': run_lifecycle_promoted,
           **{arm: deferred_handler(arm) for arm in DEFERRED_ARMS}}


def handlers() -> dict:
    """Built-ins plus any DISPATCH_HANDLERS other builders expose (imported lazily; absence is fine)."""
    out = dict(BUILTIN)
    for mod in EXTERNAL_MODULES:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        out.update(getattr(m, 'DISPATCH_HANDLERS', {}) or {})
    return out
