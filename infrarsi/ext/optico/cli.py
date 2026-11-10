"""Command line for ENV-B (run from E:\\tr\\tyc_sftv2aliyun\\infrarsi with the project venv).

No GPU:  bed-plan W | rnd-seq W [--n 20] | view W
One trial on a leased node:  trial W --node N --run-id R --params JSON [--purpose P] [--arm A]
Arms (design §3.3-§3.6; budgets as flags, defaults = design §3.5 per arm and workload):
  b-def  W --node N --run-id R                              script defaults (B-DEF; the G-B2 reference)
  b-ed   W --node N --run-id R [--gb2-default R0]           planner projection, one trial
  b-nat  W --node N --run-id R [--cycles 21 --max-trials 26 --wall-cap-s S --tokens-cap 1500000 --usd-cap 30]
  b-rnd  W --node N --run-id R [--screens 21 --wall-cap-s S --gpu-hours H]
  b-is   W --node N --run-id R [--weak] [--screens 21 --wall-cap-s S --tokens-cap 1500000 --usd-cap 30 --gpu-hours H]
  locked W --node N --run-id R --arm A (--from trace.json|result.json|receipt.json | --params JSON)
Noise margin (AMENDMENT_EXT_002 Q1, no GPU):  margin W [--receipts DIR_OR_FILE ...] [--out DIR]
  -> margin_<W>.json from the protocol receipts (default: every run under <EXECUTION_DIR>/ext_bench/optico)
Every arm takes --gb2-default <B-DEF receipt> [--gb2-bed <B-ED receipt>] (G-B2 enabling rule, design §3.4).
Run records: <EXECUTION_DIR>/ext_bench/optico/<run_id>/ unless --out is given.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ...contracts import EXECUTION_DIR
from . import b_ed, b_rnd, params as pm, task as taskmod, workloads as wl

EXEC_ROOT = EXECUTION_DIR / 'ext_bench' / 'optico'


def _out(args) -> Path:
    return Path(args.out) if args.out else EXEC_ROOT / args.run_id


def _log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def _common(p, *, node=True):
    p.add_argument('workload')
    if node:
        p.add_argument('--node', required=True)
        p.add_argument('--run-id', required=True)
        p.add_argument('--out')
        p.add_argument('--gb2-default', default=None, help='B-DEF receipt (G-B2 reference loss)')
        p.add_argument('--gb2-bed', default=None, help='B-ED receipt (with --gb2-default: the enabling rule decides the veto)')
    return p


def _brief(r: dict) -> dict:
    keys = ('attempt_id', 'candidate_id', 'purpose', 'valid', 'judge_valid', 'invalid_reasons', 'guard_rejections', 'summary',
            'cost', 'deviations', 'sanitizer')
    return {**{k: r.get(k) for k in keys}, 'metric': r.get('metric')}


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(prog='infrarsi.ext.optico.cli', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    _common(sub.add_parser('bed-plan'), node=False)
    _common(sub.add_parser('rnd-seq'), node=False).add_argument('--n', type=int, default=20)
    _common(sub.add_parser('view'), node=False)
    p = _common(sub.add_parser('margin'), node=False)
    p.add_argument('--receipts', nargs='*', default=None, help='run directories / receipt files (default: all runs)')
    p.add_argument('--out', default=None, help='where margin_<W>.json goes (default: <EXECUTION_DIR>/ext_bench/optico)')
    p = _common(sub.add_parser('trial'))
    p.add_argument('--params', default='{}', help='OptiCo knobs (JSON); missing keys = script defaults')
    p.add_argument('--candidate-id', default='c1')
    p.add_argument('--purpose', default='screen')
    p.add_argument('--arm', default='B-IS', choices=sorted(taskmod.ARMS))
    p.add_argument('--keep-workspace', action='store_true')
    p.add_argument('--gb2-ref', type=float, default=None, help='explicit iteration-1 reference loss (instead of receipts)')
    p.add_argument('--gb2-hard', action='store_true')
    _common(sub.add_parser('b-def'))
    _common(sub.add_parser('b-ed'))
    for name in ('b-nat', 'native'):
        p = _common(sub.add_parser(name))
        p.add_argument('--cycles', type=int, default=21)
        p.add_argument('--max-trials', type=int, default=26)
        p.add_argument('--wall-cap-s', type=float, default=None, help='default: the workload arm wall (3 h, W3 1.5 h)')
        p.add_argument('--tokens-cap', type=int, default=1_500_000)
        p.add_argument('--usd-cap', type=float, default=30.0)
        p.add_argument('--poll-s', type=float, default=15.0)
        p.add_argument('--purpose', default='native')
        p.add_argument('--prompt-transform', default='strip_think_format', choices=['strip_think_format', 'none'],
                       help='D-OC-15 (declared, D7) is the default; none = verbatim prompts (diagnosis only)')
    p = _common(sub.add_parser('b-rnd'))
    p.add_argument('--screens', type=int, default=None, help='default 21 = the default start + 20 new trials')
    p.add_argument('--wall-cap-s', type=float, default=None)
    p.add_argument('--gpu-hours', type=float, default=None)
    p = _common(sub.add_parser('b-is'))
    p.add_argument('--weak', action='store_true', help='B-ISW: the self-hosted weak driver (D-OC-10)')
    p.add_argument('--screens', type=int, default=None, help='default 21 = the default start + 20 new trials')
    p.add_argument('--wall-cap-s', type=float, default=None)
    p.add_argument('--tokens-cap', type=int, default=None)
    p.add_argument('--usd-cap', type=float, default=None)
    p.add_argument('--gpu-hours', type=float, default=None)
    p = _common(sub.add_parser('b-is2'))
    p.add_argument('--noexp', action='store_true', help='B-IS2-noexp: the experience packs off')
    p.add_argument('--screens', type=int, default=None, help='default 21 = the default start + 20 new trials')
    p.add_argument('--wall-cap-s', type=float, default=None)
    p.add_argument('--tokens-cap', type=int, default=None)
    p.add_argument('--usd-cap', type=float, default=None)
    p.add_argument('--gpu-hours', type=float, default=None)
    p = _common(sub.add_parser('locked'))
    p.add_argument('--arm', required=True, choices=sorted(taskmod.ARMS))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--from', dest='from_path', help="the arm's trace.json / B-NAT result.json / a receipt")
    g.add_argument('--params', help='OptiCo knobs (JSON)')
    args = ap.parse_args(argv)
    w = wl.get(args.workload)

    if args.cmd == 'bed-plan':
        print(json.dumps(b_ed.pick(w), indent=1))
        return
    if args.cmd == 'rnd-seq':
        for i, p in enumerate(b_rnd.OptiCoRandomSampler(w).sequence(args.n)):
            print(i + 1, pm.format_params(p), json.dumps(p))
        return
    if args.cmd == 'view':
        print(json.dumps(taskmod.OptiCoHooks(taskmod.make_task(w, 'B-IS')).task_view(), indent=1, default=str))
        return
    if args.cmd == 'margin':
        from . import margin
        path = margin.write(w, margin.load_receipts(args.receipts or [EXEC_ROOT]), Path(args.out) if args.out else EXEC_ROOT)
        rec = json.loads(path.read_text(encoding='utf-8'))
        print(json.dumps({k: rec.get(k) for k in ('workload', 'estimator', 'n', 'mean', 's', 'rsd', 'rsd_used', 'margin',
                                                  'pooled_df')} | {'S_w': [(x['kind'], x['T']) for x in rec['S_w']],
                                                                   'excluded': len(rec['excluded']), 'file': str(path)},
                         indent=1, default=str))
        return

    from ..common.remote_exec import WinhNodeOps
    from . import arms
    ops, out = WinhNodeOps(args.node), _out(args)
    out.mkdir(parents=True, exist_ok=True)
    if getattr(args, 'gb2_ref', None) is not None:
        gb2 = arms.gb2_policy(ref=args.gb2_ref, hard=args.gb2_hard)
    else:
        gb2 = arms.gb2_policy(args.gb2_default, args.gb2_bed) if args.gb2_default else None
    if gb2 is not None:
        _log(f'G-B2 policy: reference {gb2.reference_loss1} hard={gb2.hard} ({gb2.rule})')
    if args.cmd == 'trial':
        from .runner import OptiCoRunner
        runner = OptiCoRunner(ops, w, out_dir=out, keep_workspace=args.keep_workspace, gb2=gb2, log=_log)
        r = runner.run(taskmod.make_task(w, args.arm), json.loads(args.params), None, args.purpose, args.run_id,
                       candidate_id=args.candidate_id)
        (out / f'receipt_{args.candidate_id}.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
        print(json.dumps(_brief(r), indent=1, default=str))
    elif args.cmd == 'b-def':
        r = arms.run_trial(w, ops, run_id=args.run_id, out_dir=out, params=dict(w.defaults), arm='B-DEF', purpose='b_def',
                           candidate_id='def', gb2=gb2, log=_log)
        (out / 'receipt_b_def.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
        print(json.dumps(_brief(r), indent=1, default=str))
    elif args.cmd == 'b-ed':
        res = arms.run_b_ed(w, ops, run_id=args.run_id, out_dir=out, gb2=gb2, log=_log)
        (out / 'receipt_b_ed.json').write_text(json.dumps(res['receipt'], indent=1), encoding='utf-8')
        if args.gb2_default:
            rule = arms.gb2_policy(args.gb2_default, res['receipt'])
            _log(f'G-B2 enabling rule: {rule.rule}')
        print(json.dumps({**_brief(res['receipt']), 'pick': res['pick']['explanation']}, indent=1, default=str))
    elif args.cmd in ('b-nat', 'native'):
        from .native import NativeDriver
        d = NativeDriver(w, ops, run_id=args.run_id, cycles=args.cycles, max_trials=args.max_trials, wall_cap_s=args.wall_cap_s,
                         llm_tokens_cap=args.tokens_cap, llm_usd_cap=args.usd_cap, out_dir=out, poll_s=args.poll_s,
                         purpose=args.purpose, gb2=gb2,
                         prompt_transform=None if args.prompt_transform == 'none' else args.prompt_transform)
        res = d.run()
        res.pop('receipt_objects', None)
        print(json.dumps(res, indent=1, default=str))
    elif args.cmd == 'b-rnd':
        tr = arms.run_b_rnd(w, ops, run_id=args.run_id, out_dir=out, screens=args.screens, wall_s=args.wall_cap_s,
                            gpu_hours=args.gpu_hours, gb2=gb2, log=_log)
        print(json.dumps({k: tr.get(k) for k in ('stop_reason', 'selected', 'selected_config', 'selected_step_time', 'used')},
                         indent=1, default=str))
    elif args.cmd == 'b-is':
        tr = arms.run_b_is(w, ops, run_id=args.run_id, out_dir=out, weak=args.weak, screens=args.screens,
                           wall_s=args.wall_cap_s, llm_tokens=args.tokens_cap, llm_usd=args.usd_cap, gpu_hours=args.gpu_hours,
                           gb2=gb2, log=_log)
        print(json.dumps({k: tr.get(k) for k in ('stop_reason', 'selected', 'selected_config', 'selected_step_time', 'used')},
                         indent=1, default=str))
    elif args.cmd == 'b-is2':
        tr = arms.run_b_is2(w, ops, run_id=args.run_id, out_dir=out, experience=not args.noexp, screens=args.screens,
                            wall_s=args.wall_cap_s, llm_tokens=args.tokens_cap, llm_usd=args.usd_cap,
                            gpu_hours=args.gpu_hours, gb2=gb2, log=_log)
        print(json.dumps({k: tr.get(k) for k in ('stop_reason', 'selected', 'selected_config', 'selected_step_time', 'used')},
                         indent=1, default=str))
    elif args.cmd == 'locked':
        params = arms.selection_from(args.from_path) if args.from_path else json.loads(args.params)
        res = arms.run_locked(w, ops, run_id=args.run_id, out_dir=out, params=params, arm=args.arm, gb2=gb2, log=_log)
        print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
