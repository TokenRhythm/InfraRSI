"""Command line for F1 (run from E:\\tr\\tyc_sftv2aliyun\\infrarsi with the project venv).

No GPU:
  view ARM                       the task view every arm sees (no recipe names, no stored timings)
  space                          S_F1: size, switches, start, the registered points reachable (hash of switch_space.json)
  rnd-seq [--n 12]               F-RND's draw order (seed 42)
  rederive [--root DIR]          G-F1 (2): the 20 stored formal-54 cells re-derived (metric; numerics vs the U+C A cell)
  margin --runs DIR [DIR ...]    margin_F1.json from the signed receipts under the run directories (Q5)
  retrace --run-dir DIR          EXT_009: the one-shot arm's (F-PROMPTS) trace re-derived from its receipts incl. the
                                 admitted superseding records and re-sealed (as-run trace kept); exit 8 = refused
  summary --uc-ref F --ht F --arm A=RUN_DIR [...]   summary_F1.json: per-arm rows, labels, matched prefix, drift; formal
                                                    only for a complete window (exit 4 otherwise, unless --allow-nonformal);
                                                    the margin is recomputed from the receipts (--margin F must match)
GPU (four leased nodes; --nodes default WINH1,WINH3,WINH4,WINH6; records under <EXECUTION_DIR>/ext_bench/flashnext/<run_id>):
  ref-uc --run-id R [--cold]                         the U+C reference screen vs the stored U+C A cell (cap 1,800 s;
                                                     --cold: 3,600 s); invalid -> STOP THE WINDOW
  ref-ht --run-id R --uc-ref F                       F-HT = S_match vs the U+C reference screen
  arm ARM --run-id R --uc-ref F [budget overrides]   F-PROMPTS | F-OPTICO | F-A0 | F-RND (Q5 budgets by default);
                                                     F-A0-CODE = setting 2: F-A0 with code edits (code_space.py, F-A0's
                                                     budget), outside the switch-only window's four arms; the open-space
                                                     wave (section 6 F8) F-OPEN-BCOMP | F-OPEN-AR |
                                                     F-OPEN-RB (search strategy from the open tree; --code opens C_F1
                                                     once F9 is approved) | F-OPTICO-20 | F-RND-20-s42 | F-RND-20-s43;
                                                     the v4 extras F-OPEN4-BCOMP (bcomp from the
                                                     v4 trees, as F-OPEN-BCOMP with the v4 caps)
  locked ARM --run-id <the arm's run id> --uc-ref F --from TRACE [--attempt 2]   the arm's plan-A locked pair (Q2),
                                                    written into the arm's run directory (summary reads it there);
                                                    exit 3 = void (re-run once at the end of the window, --attempt 2)
Exit 5: the --uc-ref file is not the U+C reference screen receipt, or --nodes is not its node set; exit 7: a locked
attempt refused before any launch (attempt already run, attempt 2 without a void attempt 1, unauthenticated own screen).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ...contracts import EXECUTION_DIR

EXEC_ROOT = EXECUTION_DIR / 'ext_bench' / 'flashnext'
NODES = ('WINH1', 'WINH3', 'WINH4', 'WINH6')            # AMENDMENT_EXT_003: four whole idle nodes, fixed for the window
F54_FINAL = Path(r'E:\rsi\f54v1\final')
LOSSY = ('Q', 'Q_full', 'Qo', 'Qo_full')


def _log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def _dump(obj):
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rederive(root: Path = F54_FINAL) -> dict:
    """Every stored cell under root: metric re-derived (statistics.mean of the stored per-update max over the window)
    vs the stored value, numerics class vs the U+C A cell (expected E0 for the exact recipes, FAIL for the lossy ones)."""
    from . import judge as J
    from . import parse as P
    ref = P.from_status_cell(J.STORED_UC_A, 'A')
    gates = J.load_gates()
    rows, ok = [], True
    for f in sorted(Path(root).rglob('STATUS_AT_FINISH_*.json')):
        st = json.loads(f.read_text(encoding='utf-8-sig'))
        for g, c in sorted((st.get('collections') or {}).items()):
            cell = P.from_status_cell(st, g, path=str(f))
            v = J.evaluate(cell, ref, gates)
            stored = cell['metric']['stored_value']
            dm = abs(cell['metric']['value'] - stored) if cell['metric']['value'] is not None and stored is not None else None
            want = 'FAIL' if cell['source']['variant'] in LOSSY else 'E0'
            good = dm is not None and dm <= 1e-9 and v['numerics_class'] == want
            ok &= good
            rows.append({'wave': st.get('wave'), 'group': g, 'variant': cell['source']['variant'],
                         'metric': cell['metric']['value'], 'stored': stored, 'abs_diff': dm,
                         'numerics_class': v['numerics_class'], 'expected_class': want, 'ok': good,
                         'violations': (v['details']['numerics'].get('violations') or [])[:2]})
    return {'cells': len(rows), 'all_ok': ok and len(rows) == 20, 'rows': rows}


def _runner(args, out: Path, reference=None, cap_s=None):
    from . import arms
    from . import guards as G
    return arms.make_runner(args.nodes.split(','), out_dir=out, reference=reference, cap_s=cap_s or G.CAP_S,
                            runtime_dir=EXEC_ROOT / '_variant_runtime', log=_log)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from . import task as T
    ap = argparse.ArgumentParser(prog='infrarsi.ext.flashnext.cli', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('view').add_argument('arm', choices=sorted(T.ARMS))
    sub.add_parser('space')
    sub.add_parser('rnd-seq').add_argument('--n', type=int, default=12)
    sub.add_parser('rederive').add_argument('--root', default=str(F54_FINAL))
    p = sub.add_parser('margin')
    p.add_argument('--runs', nargs='+', required=True, help='run directories (or receipt files) of the window')
    p.add_argument('--out', default=None, help='where margin_F1.json goes (default: <EXECUTION_DIR>/ext_bench/flashnext)')
    sub.add_parser('retrace').add_argument('--run-dir', required=True, help="F-PROMPTS' run directory")
    p = sub.add_parser('summary')
    p.add_argument('--uc-ref', required=True)
    p.add_argument('--ht', default=None)
    p.add_argument('--arm', action='append', default=[], help='ARM=RUN_DIR (trace.json and locked*.json in it)')
    p.add_argument('--margin', default=None, help='margin_F1.json (default: computed from the given run directories)')
    p.add_argument('--out', default=None)
    p.add_argument('--allow-nonformal', action='store_true', help='summarize a non-formal window (smokes), marked as such')
    for name in ('ref-uc', 'ref-ht', 'arm', 'locked'):
        p = sub.add_parser(name)
        if name in ('arm', 'locked'):
            p.add_argument('which', choices=sorted(T.ARMS))
        p.add_argument('--run-id', required=True)
        p.add_argument('--nodes', default=','.join(NODES))
        p.add_argument('--out', default=None)
        if name != 'ref-uc':
            p.add_argument('--uc-ref', required=True, help='the signed U+C reference receipt (uc_ref.json)')
        if name == 'ref-uc':
            p.add_argument('--cold', action='store_true', help='cold compile cache: cap 3,600 s (AMENDMENT_EXT_003 Q3)')
        if name == 'ref-ht':
            p.add_argument('--cold', action='store_true')
        if name == 'arm':
            for k in ('screens', 'llm_tokens'):
                p.add_argument('--' + k.replace('_', '-'), type=int, default=None)
            for k in ('wall_seconds', 'gpu_hours', 'llm_usd'):
                p.add_argument('--' + k.replace('_', '-'), type=float, default=None)
            p.add_argument('--nonformal', action='store_true',
                           help='allow budget overrides (smokes only; summary refuses such traces unless --allow-nonformal)')
            p.add_argument('--resume', action='store_true',
                           help='a same-money extra (F-BUD-*): continue the search from the checkpoint in its out directory')
            p.add_argument('--code', action='store_true',
                           help='open-wave LLM / AR / RB arm: open the C_F1 code space (only after F9 is approved - '
                                'runs/f1code_approved.flag); omit for switches only')
        if name == 'locked':
            p.add_argument('--from', dest='from_path', required=True, help="the arm's trace.json")
            p.add_argument('--attempt', type=int, default=1, choices=(1, 2),
                           help='2 = the one re-run of a void attempt 1 (same arm and selection) at the end of the window')
    args = ap.parse_args(argv)

    if args.cmd == 'view':
        from .hooks import FlashNextHooks
        _dump(FlashNextHooks(T.make_task(args.arm)).task_view())
        return 0
    if args.cmd == 'space':
        from . import switch_space as S
        _dump({'space_id': S.SPACE_ID, 'sha256': S.sha256(), 'admissible_points': len(S.points()),
               'switches': list(S.SWITCHES), 'start': S.start_point(),
               'registered_reachable': sorted(S.registered_points())})
        return 0
    if args.cmd == 'rnd-seq':
        from . import switch_space as S
        for i, p in enumerate(S.sample_order(n=args.n), 1):
            print(i, json.dumps(p))
        return 0
    if args.cmd == 'rederive':
        res = rederive(Path(args.root))
        for r in res['rows']:
            print(f"{r['wave']:<38} {r['group']} {r['variant']:<13} metric={r['metric']!r:<20} stored={r['stored']!r:<20} "
                  f"class={r['numerics_class']} (expected {r['expected_class']}) {'OK' if r['ok'] else 'MISMATCH'}")
        print(json.dumps({'cells': res['cells'], 'all_ok': res['all_ok']}))
        return 0 if res['all_ok'] else 1
    from . import arms
    if args.cmd == 'retrace':
        try:
            tr = arms.retrace(Path(args.run_dir))
        except ValueError as exc:
            _log(f'refused: {exc}')
            return 8
        _dump({k: tr.get(k) for k in ('arm', 'run_id', 'selected', 'selected_config', 'selected_step_time')}
              | {'ext009': tr['extra']['ext009'], 'screens': [{k: r.get(k) for k in ('candidate_id', 'attempt_id', 'valid',
                                                                                   'step_time', 'collection')}
                                                            for r in tr.get('screens') or []]})
        return 0
    if args.cmd == 'margin':
        rec = arms.margin(arms.load_receipts(args.runs))
        out = Path(args.out) if args.out else EXEC_ROOT
        path = arms._write(out / 'margin_F1.json', rec)
        _dump({k: rec.get(k) for k in ('estimator', 'n', 'mean', 's', 'rsd', 'margin', 'per_arm_margin')} | {'file': str(path)})
        return 0
    if args.cmd == 'summary':
        uc, ht = _load(args.uc_ref), (_load(args.ht) if args.ht else None)
        runs, dirs = {}, []
        try:
            for spec in args.arm:
                arm, d = spec.split('=', 1)
                d = Path(d)
                dirs.append(d)
                rows = [_load(x) for x in sorted(d.glob('locked*.json'))]
                runs[arm] = {'trace': _load(d / 'trace.json'), 'locked': arms.final_locked(rows), 'locked_history': rows}
            receipts = arms.load_receipts(dirs + [Path(args.uc_ref)] + ([Path(args.ht)] if args.ht else []))
            res = arms.summarize(uc_ref=uc, ht=ht, arms=runs, receipts=receipts,
                                 margin_rec=_load(args.margin) if args.margin else None,
                                 allow_nonformal=args.allow_nonformal)
        except ValueError as exc:
            _log(str(exc))
            return 4
        out = Path(args.out) if args.out else EXEC_ROOT
        arms._write(out / 'summary_F1.json', res)
        _dump({'labels': res['labels'], 'F-HT': res.get('F-HT'), 'matched_prefix': res.get('F-PROMPTS matched prefix'),
               'rows': {a: {k: r.get(k) for k in ('step_time_locked', 'speedup_vs_UC', 'recovered_fraction', 'reached_1_30x',
                                                  'numerics_class', 'InvalidAccept', 'AUC_budget', 'n_invalid_proposal')}
                        for a, r in res['rows'].items()}})
        return 0

    out = Path(args.out) if args.out else EXEC_ROOT / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    from . import guards as G
    if args.cmd == 'ref-uc':
        cap = G.COLD_REFERENCE_CAP_S if args.cold else G.CAP_S
        res = arms.run_uc_reference(_runner(args, out, cap_s=cap), run_id=args.run_id, out_dir=out, cap_s=cap)
        res.pop('receipt', None)
        _dump(res)
        if res['stop_window']:
            _log('STOP THE WINDOW: the U+C reference screen is invalid against the stored U+C A cell')
            return 2
        return 0
    uc = _load(args.uc_ref)
    probs = G.uc_reference_problems(uc)
    if probs:
        _log(f'refused: {args.uc_ref} is not the U+C reference screen receipt: {probs}')
        return 5
    nodes = sorted(str(n).upper() for n in args.nodes.split(','))
    if nodes != sorted(uc['nodes']):
        _log(f'refused: the window runs on the U+C reference screen\'s nodes {uc["nodes"]}, not {nodes}')
        return 5
    if args.cmd == 'ref-ht':
        cap = G.COLD_REFERENCE_CAP_S if args.cold else G.CAP_S
        res = arms.run_ht(_runner(args, out, cap_s=cap), run_id=args.run_id, out_dir=out, uc_ref=uc, cap_s=cap)
        res.pop('receipt', None)
        _dump(res)
        return 0 if res['valid'] else 1
    from . import judge as J
    runner = _runner(args, out, reference=J.reference_from_receipt(uc))
    if args.cmd == 'arm':
        over = {k: getattr(args, k) for k in ('screens', 'llm_tokens', 'wall_seconds', 'gpu_hours', 'llm_usd')}
        if any(v is not None for v in over.values()) and not args.nonformal:
            _log('refused: budget overrides change the Q5 table; add --nonformal for a smoke (not summarized formally)')
            return 6
        which = args.which
        if args.resume and which not in T.BUD_ARMS:        # (only the same-money extras keep checkpoints)
            _log(f'refused: --resume is for the same-money extras ({", ".join(T.BUD_ARMS)}), not {which}')
            return 6
        if which in T.OPEN_ARMS:                           # the open-space wave (section 6 F8)
            kept = {k: v for k, v in over.items() if v is not None}
            arm_strategy = {**{v: k for k, v in arms.STRATEGY_ARM.items()}, **arms.EXTRA_STRATEGY}   # + the extras
            bud = which in T.BUD_ARMS                      # the same-money extras keep checkpoints and can resume
            if which in arm_strategy:                      # F-OPEN-BCOMP / AR / RB -> run_f_is2 with its strategy
                strat = arm_strategy[which]
                # ours (bcomp) read the F1 cross-job experience from INFRARSI_HOME; AR / RB read none. An
                # explicit --llm-usd override is honoured (run_f_is2 forwards it to the budget); otherwise the arm's
                # 40 USD cap stands. --code opens C_F1 only when F9 is approved.
                exp = 'auto' if strat in arms.OURS_STRATEGIES else None
                try:
                    tr = arms.run_f_is2(strat, runner, run_id=args.run_id, out_dir=out, uc_ref=uc, arm=which,
                                        code=args.code, experience=exp, log=_log, checkpoint=bud, resume=args.resume,
                                        **kept)
                except ValueError as exc:                  # a resume refused before anything ran
                    if not args.resume:
                        raise
                    _log(f'refused: {exc}')
                    return 7
            elif which == 'F-OPTICO-20':
                tr = arms.run_f_optico(runner, run_id=args.run_id, out_dir=out, uc_ref=uc, arm=which, **kept)
            else:                                          # F-RND-20-s42 / s43
                kept.pop('llm_usd', None)
                tr = arms.run_f_rnd(runner, run_id=args.run_id, out_dir=out, uc_ref=uc, arm=which,
                                    seed=43 if which.endswith('s43') else 42, **kept)
        else:
            if which == 'F-RND':
                over.pop('llm_usd')
            tr = arms.ARM_FUNCS[which](runner, run_id=args.run_id, out_dir=out, uc_ref=uc, **over)
        _dump({k: tr.get(k) for k in ('arm', 'stop_reason', 'selected', 'selected_config', 'selected_step_time', 'used',
                                      'invalid_proposals')})
        return 0
    if args.cmd == 'locked':
        try:
            res = arms.run_locked(runner, arm=args.which, run_id=args.run_id, out_dir=out, trace=Path(args.from_path),
                                  uc_ref=uc, attempt=args.attempt)
        except ValueError as exc:                      # attempt sequence, own screen, selection: nothing was launched
            _log(f'refused: {exc}')
            return 7
        _dump(res)
        if res['status'] == 'void':
            _log('VOID PAIR: re-run once at the end of the window with --attempt 2')
            return 3
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
