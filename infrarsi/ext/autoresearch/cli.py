"""Command line for the ENV-C arms (run from E:\\tr\\tyc_sftv2aliyun\\infrarsi; driver side, no torch).

  c-base       --run-id R [--gpu 0] [--loader-check]          the unmodified baseline once (C-BASE / locked reference)
  c-rnd        --run-id R [--gpu 0] [budget] [--seed 42]      128-point infra knob space, seed-42 permutation
  c-nat-free   --run-id R [--gpu 2] [budget] [--keep-revert auto|git|copy]   program.md, free track
  c-nat-infra  --run-id R [--gpu 4] [budget] [--keep-revert ...]             program.md + infra appendix
  c-is         --run-id R [--gpu 6] [budget] [--package A0]   InfraRSI A0 (optimize + AutoresearchHooks)
  locked       --run-id R --arm A --from <trace.json | train.py> --arm-index N [--gpu G]   {baseline, final} back to back
  calib        --run-id R [--gpus 0,2,4,6] [--gate 0.02]      co-location calibration (design §4.3): C-BASE with the
               loader check (AMENDMENT_EXT_002 b) on the first GPU, concurrently C-RND's first 3 points on the others;
               gate: C-BASE steps within 2 % of the anchor's
  group-lease   --run-id G [--gpus 0,2,4,6]   take the co-location node lease (owner infrarsi-ext, scope run) after the
                occupancy check and the check that no ENV-C uid is in use; the arms then run with --group-run-id G and
                claim only their GPUs
  group-release --run-id G                     kill every ENV-C uid's processes (verified), then release the lease
                                               (only a lease of that run id; kept when a process survives)
Common: --node WINH0, --group-run-id G (the orchestrator's co-location node lease: the command then only claims its
GPUs; without it the command holds the node lease itself, owner infrarsi-ext), --prepare-variant h2d_sync (D-AR-7,
the only accepted value), --dry-run (print the plan, touch nothing). Budget flags (design §4.3): --screens 36
--wall-seconds 14400 --llm-tokens 3000000 --llm-usd 40 --gpu-hours 4. LLM arms use Opus 5.5 (agent.arms.strong_driver).
Trust domains (Codex W9 B-02/B-03): every command runs its untrusted processes under its arm's uid (task.ARM_UIDS:
C-BASE 64000, C-NAT-free 64001, C-NAT-infra 64002, C-IS 64003, C-RND 64004; locked: the --arm's uid; calib:
task.CALIB_UIDS, one per GPU); the uid must be unused when the command starts and is killed before its claims go.
Records: <EXECUTION_DIR>/ext_bench/autoresearch/<run_id>/ (result.json, trace.json, events.jsonl, runner/receipts/,
agent/ for the native arms). result.json is written on every outcome with `lease_events`, `lease_final` and
`lease_complete`; a command whose claims could not be released, or whose processes survived, exits with an error
(fail closed), and `group-release` exits with status 3 when the node lease is not released.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from ...contracts import EXECUTION_DIR
from . import arms, knobs, source
from .task import ARM_GPUS, ARM_PREPARE, ARM_UIDS, BUDGET, CALIB_GATE, CALIB_UIDS, DIAG_UID, NODE, env_c_task

ALL_UIDS = tuple(sorted(set(ARM_UIDS.values()) | set(CALIB_UIDS) | {DIAG_UID}))

EXEC_ROOT = EXECUTION_DIR / 'ext_bench' / 'autoresearch'
LLM_ARMS = ('C-NAT-free', 'C-NAT-infra', 'C-IS')
CMD_ARM = {'c-base': 'C-BASE', 'c-rnd': 'C-RND', 'c-nat-free': 'C-NAT-free', 'c-nat-infra': 'C-NAT-infra', 'c-is': 'C-IS'}


def _log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def _brief(r: dict) -> dict:
    s = r.get('summary') or {}
    return {'attempt_id': r.get('attempt_id'), 'candidate_id': r.get('candidate_id'), 'purpose': r.get('purpose'),
            'gpus': r.get('gpus'), 'valid': r.get('valid'), 'val_bpb': (r.get('metric') or {}).get('value'),
            'val_bpb_printed': (r.get('metric') or {}).get('raw', {}).get('val_bpb_printed'), 'num_steps': s.get('num_steps'),
            'checks': s.get('checks'), 'invalid_reasons': r.get('invalid_reasons'), 'deviations': r.get('deviations'),
            'cost': r.get('cost'), 'reports': [g for g in r.get('guard_reports') or []
                                               if g.get('id') in ('judge_loader_equivalence', 'printed_vs_judge')]}


def budget_of(args) -> dict:
    return {'screens': args.screens, 'llm_tokens': args.llm_tokens, 'wall_seconds': args.wall_seconds,
            'gpu_hours': args.gpu_hours, 'llm_usd': args.llm_usd}


def task_for(arm: str, args):
    return env_c_task(arm, budget=budget_of(args), node=args.node, prepare_variant=args.prepare_variant)


def gpus_of(args, arm: str | None) -> list[int]:
    if getattr(args, 'gpus', None):
        return [int(g) for g in str(args.gpus).split(',')]
    return [int(args.gpu) if args.gpu is not None else ARM_GPUS.get(arm, 0)]


def uids_of(args, arm: str | None) -> list[int]:
    """The trust-domain uids of a command (calib: one per GPU)."""
    if args.cmd == 'calib':
        return list(CALIB_UIDS[:len(gpus_of(args, None))])
    return [ARM_UIDS[arm]]


def plan(args) -> dict:
    arm = CMD_ARM.get(args.cmd) or getattr(args, 'arm', None) or 'C-BASE'
    task = task_for(arm if arm in CMD_ARM.values() else 'C-BASE', args)
    p = {'cmd': args.cmd, 'arm': arm, 'run_id': args.run_id, 'node': args.node, 'gpus': gpus_of(args, arm),
         'lease': f'GPU claims under the group node lease {args.group_run_id}' if args.group_run_id
         else 'node lease (owner infrarsi-ext) + GPU claims for the run', 'out': str(EXEC_ROOT / args.run_id),
         'prepare_variant': task.raw['prepare_variant'], 'deviations': task.raw['deviations'], 'budget': task.budget,
         'anchor': {k: task.raw['anchor'][k] for k in ('val_bpb', 'num_steps', 'attempt_id', 'node')},
         'driver': 'claude-opus-5-5 (agent.arms.strong_driver)' if arm in LLM_ARMS else None, 'uids': uids_of(args, arm)}
    if args.cmd in ('c-nat-free', 'c-nat-infra'):
        p.update(uid=ARM_UIDS[arm], keep_revert=args.keep_revert)
    if args.cmd == 'calib':
        p.update(points=knobs.sample_order(args.seed)[:max(0, len(p['gpus']) - 1)], gate=args.gate,
                 gate_reference_steps=task.raw['anchor']['num_steps'])
    if args.cmd == 'locked':
        p.update(order=['candidate', 'baseline'] if args.arm_index % 2 else ['baseline', 'candidate'], source=args.from_path)
        try:
            p['selection_is_baseline'] = arms.baseline_skip(candidate_text(args.from_path)) is not None
        except (OSError, ValueError) as exc:
            p['selection_is_baseline'] = f'unknown ({type(exc).__name__})'
    return p


def candidate_text(path: str) -> str:
    """The final train.py of an arm: a train.py file, a SearchTrace (C-IS overlay / C-RND knobs) or a basic-agent trace
    (its selected experiment's saved text)."""
    p = Path(path)
    if p.suffix == '.py':
        return p.read_text(encoding='utf-8')
    tr = json.loads(p.read_text(encoding='utf-8'))
    if tr.get('schema') in ('infrarsi-search-trace-v1', 'infra' + 'smith-search-trace-v1'):
        text = (tr.get('selected_overlay') or {}).get('train.py') or source.base_train()
        cfg = {k: v for k, v in (tr.get('selected_config') or {}).items() if k in knobs.SPACE}
        return knobs.apply(text, cfg) if cfg else text
    if tr.get('schema') == 'ext-basic-agent-trace-v1':
        if not tr.get('selected'):
            return source.base_train()                     # nothing valid: the arm falls back to the baseline
        return (p.parent / 'experiments' / f'{tr["selected"]}.train.py').read_text(encoding='utf-8')
    raise ValueError(f'{path}: not a train.py, SearchTrace or basic-agent trace')


def _runner(ops, task, args, gpu: int, out: Path, **kw):
    return arms.make_runner(ops, task, gpu=gpu, run_id=args.run_id, out_dir=out / 'runner', colocated=bool(args.group_run_id),
                            held_run_id=args.group_run_id or args.run_id, log=_log, **kw)


def calibrate(ops_factory, args, out: Path) -> dict:
    """C-BASE (loader check) on gpus[0] and the first C-RND points on the other GPUs, all at once."""
    gpus = gpus_of(args, None)
    base_task, rnd_task = task_for('C-BASE', args), task_for('C-RND', args)
    points = knobs.sample_order(args.seed)[:len(gpus) - 1]
    # One trust domain per concurrent run: ending one run kills its uid, which must not stop the others.
    jobs = [(gpus[0], base_task, {}, 'c0', {'loader_check': True, 'calib_uid': CALIB_UIDS[0]})] + [
        (g, rnd_task, pt, f'p{i + 1}', {'calib_uid': CALIB_UIDS[i + 1]}) for i, (g, pt) in enumerate(zip(gpus[1:], points))]
    results: dict = {}

    def one(gpu, task, config, cid, kw):
        runner = _runner(ops_factory(args.node), task, args, gpu, out, **kw)
        results[cid] = runner.run(task, config, None, 'screen', args.run_id, candidate_id=cid)
    threads = [threading.Thread(target=one, args=job, daemon=True) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    base = results.get('c0') or {}
    steps, ref = (base.get('summary') or {}).get('num_steps'), base_task.raw['anchor']['num_steps']
    dev = abs(steps - ref) / ref if isinstance(steps, int) else None
    loader = next((g for g in base.get('guard_reports') or [] if g.get('id') == 'judge_loader_equivalence'), None)
    return {'gate': {'reference_steps': ref, 'reference': base_task.raw['anchor']['attempt_id'], 'c_base_steps': steps,
                     'rel_dev': dev, 'limit': args.gate, 'passed': dev is not None and dev <= args.gate and base.get('valid') is True},
            'loader_equivalence': loader, 'points': points, 'runs': {k: _brief(v) for k, v in sorted(results.items())}}


def group_lease(ops, run_id: str, gpus, *, release: bool = False) -> dict:
    """The co-location node lease of a group of ENV-C arms (no GPU claims: each arm claims its own). Acquire: no ENV-C
    uid may be in use; release: every ENV-C uid is killed first (ContainmentError, lease kept, if one survives)."""
    from ...harness.leases import OccupiedError
    from ..common.remote_exec import assert_uid_unused, ext_lease, node_lease_path, occupancy_problems
    path = node_lease_path(ops.node)
    if release:
        contained = arms.contain_all(ops, ALL_UIDS)
        st = ops.call('release_status', path=path, attempt_id=f'run:{run_id}')
        return {'released': st.get('state') in ('removed', 'absent'), 'state': st.get('state'), 'holder': st.get('holder'),
                'lease': path, 'contained': [{k: c.get(k) for k in ('uid', 'killed', 'zombies')} for c in contained]}
    for u in ALL_UIDS:
        assert_uid_unused(ops, u)
    current = ops.read_json(path)
    why = occupancy_problems(ops.probe(), gpus, lease=current, lease_ok=lambda l: l.get('attempt_id') == f'run:{run_id}')
    if why:
        raise OccupiedError({ops.node: '; '.join(why)})
    lease = ext_lease(run_id, f'run:{run_id}', ops.node, 'autoresearch', gpus, scope='run')
    try:
        res = ops.acquire(path, lease)
    except Exception as exc:                       # the answer may be lost after the lease was written: roll back
        from ..common.remote_exec import LeaseError
        st = ops.call('release_status', path=path, attempt_id=f'run:{run_id}')        # owner-checked
        if st.get('state') not in ('removed', 'absent', 'foreign'):
            raise LeaseError(f'{ops.node}: group lease acquire failed and its rollback is unknown', kept=[path]) from exc
        raise
    if not res['ok']:
        raise OccupiedError({ops.node: f'lease taken: {res.get("existing")}'})
    return {'acquired': path, 'lease': lease}


def main(argv=None, *, ops_factory=None, driver_factory=None, package_factory=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(prog='infrarsi.ext.autoresearch.cli', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('group-lease', 'group-release'):
        p = sub.add_parser(name)
        p.add_argument('--run-id', required=True)
        p.add_argument('--node', default=NODE)
        p.add_argument('--gpus', default='0,2,4,6')
    for name in ('c-base', 'c-rnd', 'c-nat-free', 'c-nat-infra', 'c-is', 'locked', 'calib'):
        p = sub.add_parser(name)
        p.add_argument('--run-id', required=True)
        p.add_argument('--node', default=NODE)
        p.add_argument('--group-run-id', default=None)
        p.add_argument('--prepare-variant', default=ARM_PREPARE, choices=(ARM_PREPARE,),
                       help='D-AR-7 only (the upstream loader is a named diagnostic: check.py --diagnostic-upstream-loader)')
        p.add_argument('--dry-run', action='store_true')
        p.add_argument('--screens', type=int, default=BUDGET['screens'])
        p.add_argument('--wall-seconds', type=float, default=BUDGET['wall_seconds'])
        p.add_argument('--llm-tokens', type=int, default=BUDGET['llm_tokens'])
        p.add_argument('--llm-usd', type=float, default=BUDGET['llm_usd'])
        p.add_argument('--gpu-hours', type=float, default=BUDGET['gpu_hours'])
        p.add_argument('--seed', type=int, default=42)
        if name == 'calib':
            p.add_argument('--gpus', default='0,2,4,6')
            p.add_argument('--gate', type=float, default=CALIB_GATE)
        else:
            p.add_argument('--gpu', type=int, default=None)
        if name == 'c-base':
            p.add_argument('--loader-check', action='store_true', help='second trusted judge with the D-AR-7 loader')
        if name in ('c-nat-free', 'c-nat-infra'):
            p.add_argument('--keep-revert', default='auto', choices=('auto', 'git', 'copy'))
        if name == 'c-is':
            p.add_argument('--package', default='A0')
        if name == 'locked':
            p.add_argument('--arm', required=True, choices=sorted(set(CMD_ARM.values()) | {'C-IS2'}))   # C-IS2: 10-24
            p.add_argument('--from', dest='from_path', required=True)
            p.add_argument('--arm-index', type=int, required=True)
    args = ap.parse_args(argv)
    if args.cmd in ('group-lease', 'group-release'):
        if ops_factory is None:
            from ..common.remote_exec import WinhNodeOps as ops_factory
        res = group_lease(ops_factory(args.node), args.run_id, [int(g) for g in args.gpus.split(',')],
                          release=args.cmd == 'group-release')
        print(json.dumps(res, indent=1, default=str))
        return 3 if args.cmd == 'group-release' and not res.get('released') else 0
    if args.dry_run:
        print(json.dumps(plan(args), indent=1, default=str))
        return 0
    if ops_factory is None:
        from ..common.remote_exec import WinhNodeOps
        ops_factory = WinhNodeOps
    if driver_factory is None:
        from ...agent.arms import strong_driver as driver_factory
    out = EXEC_ROOT / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plan.json').write_text(json.dumps(plan(args), indent=1, default=str), encoding='utf-8')
    if args.cmd == 'locked':                       # D20: a baseline selection runs nothing (no lease, no node, exit 0)
        skip = arms.baseline_skip(candidate_text(args.from_path))
        if skip:
            result = {**skip, 'arm': args.arm, 'arm_index': args.arm_index, 'source': args.from_path, 'lease_events': [],
                      'lease_final': None, 'lease_complete': True}
            (out / 'result.json').write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
            print(json.dumps(result, indent=1, default=str))
            return 0
    ops = ops_factory(args.node)
    arm = CMD_ARM.get(args.cmd) or getattr(args, 'arm', None)
    gpus = gpus_of(args, arm)
    events: list = []
    result, error = {}, None
    try:
        with arms.arm_lease(ops, args.run_id, gpus, group_run_id=args.group_run_id, uids=uids_of(args, arm), events=events):
            result = _command(args, ops, ops_factory, driver_factory, package_factory, arm, gpus, out)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {str(exc)[:2000]}'
        raise
    finally:
        final = next((e for e in reversed(events) if e.get('kind') in ('release', 'claims_kept', 'acquire_failed')), None)
        result = {**(result or {}), 'lease_events': events, 'lease_final': final,
                  'lease_complete': bool(final) and final.get('kind') == 'release' and final.get('complete') is True,
                  **({'error': error} if error else {})}
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    print(json.dumps(result, indent=1, default=str))
    return 0


def _command(args, ops, ops_factory, driver_factory, package_factory, arm, gpus, out) -> dict:
    """One ENV-C command inside the arm lease; returns its result dict."""
    if args.cmd == 'calib':
        result = calibrate(ops_factory, args, out)
    elif args.cmd == 'c-base':
        task = task_for('C-BASE', args)
        r = _runner(ops, task, args, gpus[0], out, loader_check=args.loader_check).run(task, {}, None, 'screen', args.run_id,
                                                                                        candidate_id='c0')
        result = _brief(r)
    elif args.cmd == 'c-rnd':
        task = task_for('C-RND', args)
        tr = arms.run_c_rnd(task, _runner(ops, task, args, gpus[0], out), run_id=args.run_id, out_dir=out, seed=args.seed).to_dict()
        result = {k: tr.get(k) for k in ('stop_reason', 'selected', 'selected_config', 'selected_step_time', 'used')}
    elif args.cmd in ('c-nat-free', 'c-nat-infra'):
        task = task_for(arm, args)
        tr = arms.run_c_nat(task, driver_factory(), _runner(ops, task, args, gpus[0], out), ops, run_id=args.run_id,
                            out_dir=out / 'agent', keep_revert=args.keep_revert)
        result = {k: tr.get(k) for k in ('stop_reason', 'selected', 'selected_value', 'used', 'turns')}
    elif args.cmd == 'c-is':
        task = task_for('C-IS', args)
        if package_factory is None:
            from ...agent.arms import resolve_package as package_factory
        pkg = package_factory(args.package)
        tr = arms.run_c_is(task, pkg, driver_factory(), _runner(ops, task, args, gpus[0], out), run_id=args.run_id,
                           out_dir=out, frozen_package_digest=pkg.digest).to_dict()
        result = {k: tr.get(k) for k in ('stop_reason', 'selected', 'selected_step_time', 'used')}
    else:                                           # locked
        task = task_for(args.arm, args)
        pair = arms.locked_pair(_runner(ops, task, args, gpus[0], out), task, candidate_text(args.from_path),
                                run_id=args.run_id, arm_index=args.arm_index)
        result = {'order': pair['order'], **{k: _brief(pair[k]) for k in ('baseline', 'candidate')}}
    return result


if __name__ == '__main__':
    sys.exit(main())
