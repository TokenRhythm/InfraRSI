"""ENV-A arm launcher (Follow-up 1 #4, AMENDMENT_EXT_001):

    python -m infrarsi.ext.rebench_llmf.cli {a-nat,a-isw,a-rnd,a-nat2,a-is2,context-check} --node WINHk --gpus 0,1,2,3 --run-id ID [flags]

Builds the node ops, task, runner and (for LLM arms) the metered weak driver, holds the arm's node lease, runs the arm,
records under execution/ext_bench/rebench_llmf/<run_id>/ and prints a JSON summary (arm result + the reported-only LLM
aggregates + t_best_search@8h). `--dry-run` builds everything and stops before the first LLM call / scoring. No LLM arm
starts until the orchestrator launches it after the Codex review; tokens are reported, not charged (D1).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from ...contracts import EXECUTION_DIR
from . import arms, native
from .task import BUDGET_LLM, WEAK_ENDPOINT_CONTEXT, NODE_TASK, build_env_a_task, read_base_solution

RECORDS = EXECUTION_DIR / 'ext_bench' / 'rebench_llmf'          # execution/ext_bench/<env>/<run_id>/ (EXT_INTERFACES §5)
DEFAULT_REF = '/data/infrarsi/ext/runs/rebench_llmf/r1_20261021c/home_agent/trained_model_reference'


def _ops(a):
    """WinhNodeOps for a real run; a LocalNodeOps on a temp dir for `--local` (dry-run without a node)."""
    if getattr(a, 'on_node', False):
        import os
        from ..common import node_ext
        from ..common.remote_exec import NodeOps
        if getattr(a, 'local', False):
            raise ValueError('--on-node and --local are mutually exclusive')
        if os.name != 'posix' or os.geteuid() != 0:
            raise ValueError('--on-node requires root on the selected Linux WINH node (the existing uid/lease protocol)')

        class OnNodeOps(NodeOps):
            def call(self, func, _timeout=300, **kw):
                return getattr(node_ext, func)(**kw)

        return OnNodeOps(a.node)
    if getattr(a, 'local', False):
        import tempfile
        from ..common.remote_exec import LocalNodeOps
        return LocalNodeOps(tempfile.mkdtemp(prefix='isx2cli'), node=a.node)
    from ..common.remote_exec import WinhNodeOps
    return WinhNodeOps(a.node)


def _gpus(s: str) -> tuple:
    return tuple(int(x) for x in str(s).split(',') if x != '')


def _budget_overrides(a) -> dict | None:
    over = {}
    if a.scores is not None:
        over['screens'] = int(a.scores)
    if a.wall_s is not None:
        over['wall_seconds'] = float(a.wall_s)
    if getattr(a, 'tokens_cap', None) is not None:
        over['llm_tokens'] = int(a.tokens_cap)
    if getattr(a, 'gpu_hours', None) is not None:
        over['gpu_hours'] = float(a.gpu_hours)
    return over or None


def _task(arm: str, a, override: dict | None):
    budget = None
    if override:
        from .task import BUDGET_LLM, BUDGET_RND
        budget = {**(BUDGET_RND if arm == 'A-RND' else BUDGET_LLM), **override}
    return build_env_a_task(arm=arm, action_space='knobs+patch' if arm in ('A-ISW', 'A-IS2') else 'knobs', budget=budget,
                            src_dir=NODE_TASK if getattr(a, 'on_node', False) else None)


def _out_dir(a) -> Path:
    return Path(a.out_dir) if a.out_dir else (RECORDS / a.run_id)


def _emit(summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / 'result.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
    print(json.dumps(summary, indent=1, default=str))


EXIT_LEASE_INCOMPLETE = 3


def _record_lease(out: Path, state: dict | None, events: list) -> None:
    """The arm lease's containment / release record: <out>/arm_lease.json and one line per event in events.jsonl."""
    out.mkdir(parents=True, exist_ok=True)
    (out / 'arm_lease.json').write_text(json.dumps({'state': state, 'events': events}, indent=1, default=str), encoding='utf-8')
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
        for e in events:
            f.write(json.dumps({'utc': utc, 'kind': 'arm_lease', 'payload': e}, default=str) + '\n')


def _leased(a, ops, runner, out: Path, arm: str, fn) -> dict:
    """fn() under the arm lease (Codex W9 R2 M-02): the lease's containment / release state goes into the result and
    the events; an incomplete cleanup returns exit_code 3 with the retained paths (the claims stay on the node)."""
    events, holder = [], {}
    try:
        with arms.arm_lease(ops, a.run_id, _gpus(a.gpus), uids=runner.domain_uids(), group_run_id=a.group_run_id,
                            events=events) as state:
            holder['state'] = state
            res = fn()
    except arms.ArmLeaseError as exc:
        _record_lease(out, exc.state, events)
        ctx = exc.__context__
        return {'arm': arm, 'error': str(exc), 'arm_error': repr(ctx)[:1000] if ctx else None, 'arm_lease': exc.state,
                'retained': exc.state.get('kept'), 'exit_code': EXIT_LEASE_INCOMPLETE, 'records': str(out)}
    except BaseException:
        _record_lease(out, holder.get('state'), events)
        raise
    _record_lease(out, holder['state'], events)
    return {**res, 'arm_lease': holder['state']}


HARD_STOP_MARGIN_S = 55 * 60       # AMENDMENT_EXT_006: wall = (hard stop - 55 min) - t_restart when WINH1 is needed


def _wall_rule(a, now=None) -> dict | None:
    """AMENDMENT_EXT_006: the full 57,600 s unless --hard-stop leaves less: wall = (hard stop - 55 min) - now."""
    if not getattr(a, 'hard_stop', None):
        return None
    now = now or datetime.datetime.now(datetime.timezone.utc)
    stop = datetime.datetime.fromisoformat(a.hard_stop)
    room = (stop - now).total_seconds() - HARD_STOP_MARGIN_S
    full = float(a.wall_s) if a.wall_s is not None else float(BUDGET_LLM['wall_seconds'])
    wall = max(0.0, min(full, room))
    return {'hard_stop': a.hard_stop, 'margin_s': HARD_STOP_MARGIN_S, 'start_utc': now.isoformat(), 'room_s': round(room, 1),
            'wall_seconds': round(wall, 1), 'shortened': wall < full}


def abandon_record(records: Path, *, restart_run_id: str, reason: str) -> dict:
    """AMENDMENT_EXT_006: mark an aborted A-NAT run `abandoned` (its records stay) with cause and numbers from its
    events: turns, scorings, LLM calls / tokens / wall, the most repeated (tool, args, output) signature."""
    import collections
    import hashlib
    rows = []
    if (records / 'events.jsonl').exists():
        for line in (records / 'events.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
    turns = [r['payload'] for r in rows if r.get('kind') == 'turn']
    llm = [r['payload'] for r in rows if r.get('kind') == 'llm_call']

    def sig(p):
        act = {k: v for k, v in (p.get('action') or {}).items() if k not in ('tool', 'note')}
        res = {k: v for k, v in (p.get('result') or {}).items() if k != 'seconds'} if isinstance(p.get('result'), dict) else p.get('result')
        return hashlib.sha256(json.dumps([p.get('tool'), act, res], sort_keys=True, default=str).encode('utf-8')).hexdigest()
    sigs = [sig(p) for p in turns]
    top, n_top = (collections.Counter(sigs).most_common(1)[0] if sigs else (None, 0))
    run = best = 0
    for i, x in enumerate(sigs):
        run = run + 1 if i and x == sigs[i - 1] else 1
        best = max(best, run)
    first = next((turns[i].get('turn') for i, x in enumerate(sigs) if x == top), None)
    rec = {'status': 'abandoned', 'run_id': records.name, 'restarted_as': restart_run_id, 'reason': reason,
           'turns': len(turns), 'scorings': sum(1 for r in rows if r.get('kind') == 'screen'),
           'llm_calls': len(llm), 'llm_tokens': sum(p.get('llm_tokens') or 0 for p in llm),
           'llm_wall_seconds': round(sum(p.get('wall_seconds') or 0 for p in llm), 1),
           'most_repeated_signature': {'count': n_top, 'first_turn': first, 'max_consecutive': best},
           'first_event_utc': rows[0]['utc'] if rows else None, 'last_event_utc': rows[-1]['utc'] if rows else None,
           'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'ruling': 'AMENDMENT_EXT_006'}
    (records / 'ABANDONED.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    return rec


def cmd_a_nat(a) -> dict:
    out = _out_dir(a)
    wall_rule = _wall_rule(a)
    if wall_rule and wall_rule['shortened']:
        a.wall_s = wall_rule['wall_seconds']
    task = _task('A-NAT', a, _budget_overrides(a))
    ops = _ops(a)
    runner = arms.make_runner(ops, task, gpus=_gpus(a.gpus), run_id=a.run_id, reference_src=a.reference_src,
                              out_dir=out, enforce_g7=False, group_run_id=a.group_run_id)
    sampled = a.temperature is not None and float(a.temperature) != 0.0
    if a.restart_of or sampled:                                   # AMENDMENT_EXT_006: every receipt of the restart
        runner.deviations = tuple(runner.deviations) + ('D-RB-16',)
    driver, route, meter = arms.weak_driver(base_url=a.base_url, served_model=a.served_model,
                                            endpoint_context=a.endpoint_context, timeout_seconds=600,
                                            temperature=a.temperature, request_seed=a.request_seed,
                                            seed_per_call=not a.seed_constant)
    meta = {'restart_of': a.restart_of, 'wall_rule': wall_rule, 'temperature': a.temperature, 'request_seed': a.request_seed,
            'request_seed_mode': None if a.request_seed is None else ('constant' if a.seed_constant else 'base + call index')}
    if a.dry_run:
        raw = '/home/agent/solution LOCAL DRY-RUN PLACEHOLDER (instructions rendered on the node in a real run)' if a.local else None
        res = arms.run_a_nat(task, driver, runner, ops, run_id=a.run_id, out_dir=out, restart_of=a.restart_of,
                             context_tokens=a.context_tokens, raw_instructions=raw, dry_run=True)
        return {**res, **meta, 'route': route, 'records': str(out)}
    if a.restart_of and (RECORDS / a.restart_of).is_dir() and not (RECORDS / a.restart_of / 'ABANDONED.json').exists():
        meta['abandoned'] = abandon_record(RECORDS / a.restart_of, restart_run_id=a.run_id,
                                           reason='deterministic loop at T = 0 (AMENDMENT_EXT_006); driver stopped by O')
    res = _leased(a, ops, runner, out, 'A-NAT', lambda: arms.run_a_nat(task, driver, runner, ops, run_id=a.run_id,
                                                                        out_dir=out, context_tokens=a.context_tokens,
                                                                        restart_of=a.restart_of))
    if res.get('exit_code'):                                     # lease cleanup incomplete: fail closed
        return {**res, **meta, 'route': route, 'llm': meter.summary()}
    return {**res, **meta, 'route': route, 'llm': meter.summary(), 't_best_search_8h': arms.t_best_search_at(res),
            'records': str(out)}


def cmd_a_isw(a) -> dict:
    out = _out_dir(a)
    task = _task('A-ISW', a, _budget_overrides(a))
    ops = _ops(a)
    runner = arms.make_runner(ops, task, gpus=_gpus(a.gpus), run_id=a.run_id, reference_src=a.reference_src,
                              out_dir=out, enforce_g7=True, group_run_id=a.group_run_id)
    driver, route, meter = arms.weak_driver(base_url=a.base_url, served_model=a.served_model,
                                            endpoint_context=a.endpoint_context, timeout_seconds=600)
    from ...rsi import build_a0
    pkg = build_a0(out / 'package' / 'A0')
    kw = dict(run_id=a.run_id, out_dir=out, context_limit=a.context_limit, max_turns=a.max_turns,
              package_work_dir=out / 'package' / 'A0_isw')
    if a.dry_run:
        res = arms.run_a_isw(task, pkg, driver, runner, dry_run=True, **kw)
        return {**res, 'route': route, 'records': str(out)}
    res = _leased(a, ops, runner, out, 'A-ISW', lambda: arms.run_a_isw(task, pkg, driver, runner, **kw))
    if res.get('exit_code'):                                     # lease cleanup incomplete: fail closed
        return {**res, 'route': route, 'llm': meter.summary()}
    trace = res.get('trace')
    return {**res, 'route': route, 'llm': meter.summary(), 't_best_search_8h': arms.t_best_search_at(trace),
            'records': str(out)}


def _phase_e(a, arm: str) -> dict:
    out = _out_dir(a)
    # Fail before node setup if VLLM_URL is missing; this constructor makes no network request.
    driver, route, meter = arms.phase_e_driver(native=arm == 'A-NAT2', out_dir=out)
    task = _task(arm, a, _budget_overrides(a))
    ops = _ops(a)
    if a.on_node:
        # _ops validated root on Linux. Protect all driver copies before any records are written;
        # the arm's home/solution stays outside this directory with its existing access modes.
        from .node.score_runner import private_dir
        private_dir(out)
    runner = arms.make_runner(ops, task, gpus=_gpus(a.gpus), run_id=a.run_id, reference_src=a.reference_src,
                              out_dir=out, enforce_g7=arm == 'A-IS2', group_run_id=a.group_run_id,
                              base_solution=read_base_solution(NODE_TASK) if a.on_node else None)
    # Also carry arm deviations on failed receipts, whose node execution returned no extras.
    from .task import arm_deviations
    runner.deviations = arm_deviations(arm)

    def run():
        if arm == 'A-IS2':
            return arms.run_a_is2(task, driver, runner, run_id=a.run_id, out_dir=out, dry_run=a.dry_run)
        raw = 'LOCAL DRY-RUN PLACEHOLDER' if a.local and a.dry_run else None
        return arms.run_a_nat2(task, driver, runner, ops, run_id=a.run_id, out_dir=out,
                               raw_instructions=raw, dry_run=a.dry_run)

    res = run() if a.dry_run else _leased(a, ops, runner, out, arm, run)
    trace = res.get('trace') if arm == 'A-IS2' else res
    return {**res, 'route': route, 'llm': meter.summary(), 'records': str(out),
            't_best_search_8h': arms.t_best_search_at(trace)}


def cmd_a_is2(a) -> dict:
    return _phase_e(a, 'A-IS2')


def cmd_a_nat2(a) -> dict:
    return _phase_e(a, 'A-NAT2')


def cmd_a_rnd(a) -> dict:
    out = _out_dir(a)
    task = _task('A-RND', a, _budget_overrides(a))
    ops = _ops(a)
    runner = arms.make_runner(ops, task, gpus=_gpus(a.gpus), run_id=a.run_id, reference_src=a.reference_src,
                              out_dir=out, enforce_g7=False, group_run_id=a.group_run_id)
    if a.dry_run:
        res = arms.run_a_rnd(task, runner, run_id=a.run_id, out_dir=out, seed=a.seed, n=a.n, dry_run=True)
        return {**res, 'records': str(out)}
    res = _leased(a, ops, runner, out, 'A-RND', lambda: arms.run_a_rnd(task, runner, run_id=a.run_id, out_dir=out,
                                                                        seed=a.seed, n=a.n))
    if res.get('exit_code'):                                     # lease cleanup incomplete: fail closed
        return {**res, 'llm': arms.llm_report(None)}
    return {**res, 'llm': arms.llm_report(None), 't_best_search_8h': arms.t_best_search_at(res.get('trace')),
            'records': str(out)}


def cmd_context_check(a) -> dict:
    if a.dry_run:
        system, prompt = native.context_check_prompt(a.context_tokens)
        from ...agent.context import estimate_tokens
        return {'dry_run': True, 'arm': 'context-check', 'context_tokens': a.context_tokens,
                'estimate_units_prompt': estimate_tokens(system + prompt), 'prompt_bytes': len(prompt.encode('utf-8'))}
    driver, route, _ = arms.weak_driver(base_url=a.base_url, served_model=a.served_model,
                                        endpoint_context=a.endpoint_context, timeout_seconds=600)
    return {'arm': 'context-check', 'route': route, **native.context_check(driver, context_tokens=a.context_tokens)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='rebench_llmf.cli')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp, *, llm=True):
        sp.add_argument('--node', default='WINH1')
        sp.add_argument('--gpus', default='0,1,2,3')
        sp.add_argument('--run-id', required=True)
        sp.add_argument('--out-dir', default=None)
        sp.add_argument('--reference-src', default=DEFAULT_REF)
        sp.add_argument('--group-run-id', default=None, help='the orchestrator node lease for co-located arms (held mode)')
        sp.add_argument('--scores', type=int, default=None, help='override budget screens (default 60)')
        sp.add_argument('--wall-s', type=float, default=None, help='override budget wall_seconds (default 57600)')
        sp.add_argument('--gpu-hours', type=float, default=None)
        sp.add_argument('--dry-run', action='store_true', help='build everything, stop before the first LLM call / scoring')
        sp.add_argument('--local', action='store_true', help='dry-run without a node (no WinhNodeOps)')
        if llm:
            sp.add_argument('--base-url', default=None, help='weak endpoint, e.g. http://127.0.0.1:18000/v1')
            sp.add_argument('--served-model', default=None, help='e.g. deepseek-ai/DeepSeek-V4-Flash')
            sp.add_argument('--endpoint-context', type=int, default=WEAK_ENDPOINT_CONTEXT)
            sp.add_argument('--tokens-cap', type=int, default=None, help='override llm_tokens (sentinel default; not charged)')

    nat = sub.add_parser('a-nat'); common(nat)
    nat.add_argument('--context-tokens', type=int, default=native.CONTEXT_TOKENS)
    nat.add_argument('--temperature', type=float, default=None,
                     help='sampling temperature (default: the weak driver greedy 0.0); 1.0 for the EXT_006 restart')
    nat.add_argument('--request-seed', type=int, default=None,
                     help='seed in every request body (base + call index; vLLM honours it)')
    nat.add_argument('--seed-constant', action='store_true', help='use the same request seed for every call')
    nat.add_argument('--restart-of', default=None, help='the abandoned run this restarts (records ABANDONED.json, D-RB-16)')
    nat.add_argument('--hard-stop', default=None,
                     help='ISO time the node is needed back: wall = min(57,600 s, hard stop - 55 min - now)')
    nat.set_defaults(fn=cmd_a_nat)

    isw = sub.add_parser('a-isw'); common(isw)
    isw.add_argument('--context-limit', type=int, default=32_000)
    isw.add_argument('--max-turns', type=int, default=200)
    isw.set_defaults(fn=cmd_a_isw)

    rnd = sub.add_parser('a-rnd'); common(rnd, llm=False)
    rnd.add_argument('--seed', type=int, default=42)
    rnd.add_argument('--n', type=int, default=60)
    rnd.set_defaults(fn=cmd_a_rnd)

    for name, fn in (('a-is2', cmd_a_is2), ('a-nat2', cmd_a_nat2)):
        sp = sub.add_parser(name); common(sp, llm=False)
        sp.add_argument('--on-node', action='store_true',
                        help='run on this Linux node via the existing node operations (root required); --node labels it')
        sp.set_defaults(fn=fn)

    cc = sub.add_parser('context-check')
    cc.add_argument('--base-url', default=None)
    cc.add_argument('--served-model', default=None)
    cc.add_argument('--endpoint-context', type=int, default=WEAK_ENDPOINT_CONTEXT)
    cc.add_argument('--context-tokens', type=int, default=native.CONTEXT_TOKENS)
    cc.add_argument('--dry-run', action='store_true')
    cc.set_defaults(fn=cmd_context_check)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    summary = a.fn(a)
    if getattr(a, 'out_dir', None):
        _emit(summary, Path(a.out_dir))
    elif getattr(a, 'dry_run', False) or not getattr(a, 'run_id', None):
        print(json.dumps(summary, indent=1, default=str))       # dry-runs / context-check never write into the records
    else:
        _emit(summary, RECORDS / a.run_id)
    rc = int(summary.get('exit_code') or 0) if isinstance(summary, dict) else 0
    if rc:                                                       # fail closed: the operator must see what was kept
        print(f'ARM LEASE CLEANUP INCOMPLETE on {a.node}: retained {summary.get("retained")}; {summary.get("error")}',
              file=sys.stderr)
    return rc


if __name__ == '__main__':
    sys.exit(main())
