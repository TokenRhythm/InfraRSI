"""ENV-A post-arm steps of AMENDMENT_EXT_004 (driver side; new entry point, nothing the running arms use is changed):

    python -m infrarsi.ext.rebench_llmf.block_cli node-ref --node WINH3 --gpus 0,1,2,3 --run-id nref_winh3_a [--prime]
    python -m infrarsi.ext.rebench_llmf.block_cli block --node WINH1 --gpus 0,1,2,3 \\
        --anat <records>/anat_a --aisw <records>/aisw_a --arnd <records>/arnd_a \\
        --node-ref WINH3=<node_ref_WINH3.json | seconds> --node-ref WINH4=... --deadline 2026-10-22T03:00:00+08:00

Records go to execution/ext_bench/rebench_llmf/block_a/ (`node_ref_<node>.json`, `block_A.json`, receipts/). `--local`
replaces the node by LocalNodeOps and a synthetic node runner (a CPU dry run; its output is marked synthetic and goes
to --out, never to the records unless --out says so). Exit code 3 when the block stopped on a retained lease or a uid
process (fail closed); 2 when an input is missing.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import tempfile
from pathlib import Path

from ...contracts import EXECUTION_DIR
from . import block
from . import task as taskmod

RECORDS = EXECUTION_DIR / 'ext_bench' / 'rebench_llmf'
DEFAULT_OUT = RECORDS / 'block_a'


def _gpus(s: str) -> tuple:
    return tuple(int(x) for x in str(s).split(',') if x != '')


def synthetic_node_runner(times: dict | None = None):
    """A CPU dry-run node: answers every job with a score.py-shaped result (S0 216 s, candidates from `times`), all
    guards passing; prepare jobs return a prepare record. Never used against a real node."""
    times = dict(times or {})

    def run(ops, job):
        if job.get('mode') == 'prepare':
            return {'mode': 'prepare', 'synthetic': True, 'reference': {'outside_agent_tree': True, 'mode': '700', 'owner': 0},
                    'base': {'ok': True}, 'ref_vs_base': 23.78}
        arm = {v: k for k, v in taskmod.ARM_UIDS.items()}[job['uid']]
        is_s0 = (arm == 'A-RND' and not job.get('restore_snapshot') and job['finetune_sh'] == job['finetune_sh_base']
                 and job['finetune_yaml'] == job['finetune_yaml_base'])
        t = times.get('S0' if is_s0 else arm, 216.0)
        task = taskmod.build_env_a_task(arm=arm)
        exp = taskmod.frozen_expected(task, job['run_id'])
        tree = job.get('expected_digest') or ('5' * 64)
        return {'synthetic': True, 'node': ops.node, 'gpus': job['gpus'], 'uid': job['uid'], 'judge_uid': job['judge_uid'],
                'judge': {'score': math.log(t), 'timestamp': 't', 'message': {'maximum_allowed_norm_diff': 5.3, 'execution_time': t,
                                                                            'norm_diff': 4.2, 'max_scalar_diff': 0.0005}},
                'exit_code': 0, 'timed_out': False, 'outputs': {'trained_model': True},
                'reference': {'outside_agent_tree': True, 'mode': '700', 'owner': 0, 'path': '/r', 'tree_digest': 'r' * 64},
                'frozen_before': exp, 'frozen_after': exp, 'env_sh': {'uid': 0, 'mode': '644', 'regular': True},
                'g3': {'trained_vs_base': 23.7, 'ref_vs_base': 23.78}, 'g4': None,
                'g5': {'checkpoint_present': True, 'batches': 62, 'samples': 992}, 'g8': {'ok': True, 'trained_present': True, 'escapes': []},
                'g9': {'ok': True, 'kind': 'st', 'n_tensors': 219}, 'tree_digest': tree,
                'snapshot': {'path': job.get('restore_snapshot'), 'tree_digest': tree} if job.get('restore_snapshot') else None,
                'phases': {'data_prep': 8.9, 'train': t - 35.5, 'convert': 26.6, 'total': t},
                'remaining_uid_processes': {str(job['uid']): 0, str(job['judge_uid']): 0},
                'cost': {'gpu_seconds': 4 * (t + 40), 'wall_seconds': t + 40}}
    return run


def _ops(a):
    if a.local:
        from ..common.remote_exec import LocalNodeOps
        return LocalNodeOps(tempfile.mkdtemp(prefix='isx2block'), node=a.node)
    from ..common.remote_exec import WinhNodeOps
    return WinhNodeOps(a.node)


def _out(a) -> Path:
    if a.out:
        return Path(a.out)
    if a.local:
        return Path(tempfile.mkdtemp(prefix='isx2block_out'))
    return DEFAULT_OUT


def cmd_node_ref(a) -> dict:
    ops = _ops(a)
    nr = synthetic_node_runner() if a.local else None
    rec = block.node_ref(ops, run_id=a.run_id, gpus=_gpus(a.gpus), out_dir=_out(a), reference_src=a.reference_src,
                         do_prime=a.prime, max_attempts=a.max_attempts, node_runner=nr)
    return {**rec, 'exit_code': 3 if rec.get('fatal') else 0, 'synthetic': bool(a.local)}


def _node_ref_value(spec: str) -> tuple[str, float | None]:
    node, _, val = spec.partition('=')
    try:
        return node, float(val)
    except ValueError:
        rec = json.loads(Path(val).read_text(encoding='utf-8'))
        return node, rec.get('t_S0_node')


def cmd_block(a) -> dict:
    ops = _ops(a)
    sels, validity, first = {}, {}, {}
    for arm, path in (('A-NAT', a.anat), ('A-ISW', a.aisw), ('A-RND', a.arnd)):
        try:
            sels[arm] = block.load_selection(arm, Path(path))
        except block.BlockError as exc:
            if arm == 'A-NAT' and a.anat_degenerate_ok:          # EXT_006: no valid score -> not in the block
                sels[arm] = None
            else:
                return {'error': f'{arm}: {type(exc).__name__}: {exc}', 'exit_code': 2}
        except (OSError, ValueError, KeyError) as exc:
            return {'error': f'{arm}: {type(exc).__name__}: {exc}', 'exit_code': 2}
        validity[arm] = block.locked_validity(arm, Path(path))
        first[arm] = block.first_scoring(arm, Path(path))
    refs = dict(_node_ref_value(s) for s in a.node_ref or [])
    deadline = datetime.datetime.fromisoformat(a.deadline) if a.deadline else None
    nr = synthetic_node_runner({'S0': 216.0, 'A-NAT': 150.0, 'A-ISW': 110.0, 'A-RND': 160.0}) if a.local else None
    log = block.run_block(ops, sels, out_dir=_out(a), node_refs=refs, gpus=_gpus(a.gpus), reference_src=a.reference_src,
                          deadline=deadline, node_runner=nr, validity=validity, search_first=first, prepare_s0=True)
    return {**log, 'exit_code': 3 if log.get('stopped') else 0, 'synthetic': bool(a.local)}


def cmd_phase_e(a) -> dict:
    from . import phase_e_block as E
    try:
        nr = synthetic_node_runner({'S0': 219.0, 'A-IS2': 110.0}) if a.local else None
        log = E.run(_ops(a), is2=Path(a.is2), nat2=Path(a.nat2), out_dir=_out(a), run_id=a.run_id,
                    gpus=_gpus(a.gpus), reference_src=a.reference_src, deadline=a.deadline,
                    do_prime=a.prime, node_runner=nr, synthetic=a.local)
        return {'output': str(_out(a) / E.OUTPUT), 'runs': len(log['runs']),
                'block_void_reruns': log['block_void_reruns'], 'values': log['values'],
                'stopped': log.get('stopped'), 'synthetic': bool(a.local),
                'exit_code': 3 if log.get('stopped') else 0}
    except (OSError, ValueError, KeyError, block.BlockError) as exc:
        return {'error': str(exc), 'exit_code': 2}


def cmd_preflight(a) -> dict:
    from .phase_e_block import preflight
    return preflight(a.node, _gpus(a.gpus))


def synthetic_pair_runner(ops, a: str, b: str) -> dict:
    """--local stand-in for the node pair job (no torch on this PC): a deterministic norm per pair."""
    n = 4.6 + (sum(map(ord, a + b)) % 50) / 100.0
    return {'ok': True, 'synthetic': True, 'norm_diff': n, 'max_scalar_diff': 0.001,
            'models': {p: {'recorded': 'd' * 64, 'before': 'd' * 64, 'after': 'd' * 64} for p in (a, b)}}


def _seed_local_g4(ops, runs, records: Path) -> None:
    """--local: synthetic locked node results for the runs' recorded locked attempts (marked synthetic)."""
    for r in runs:
        try:
            att = block._locked_attempt(r, records)
        except OSError:
            continue
        if att:
            ops.write_files({f'{taskmod.NODE_RUNS}/{r}/results/{att}_node_result.json':
                             json.dumps({'ok': True, 'synthetic': True,
                                         'result': {'g4': {'computed': True, 'norm_cand_dprime_vs_refprime': 5.0}}})})


def cmd_g4_inputs(a) -> dict:
    runs = tuple(x for x in a.runs.split(',') if x)
    records = Path(a.records)
    missing = [r for r in runs if not (records / r / 'result.json').exists()]
    if missing:
        return {'error': f'no result.json for {missing} under {records}', 'exit_code': 2}
    ops = _ops(a)
    out = Path(a.out) if a.out else (Path(tempfile.mkdtemp(prefix='isx2g4')) / 'g4_inputs.json' if a.local
                                     else RECORDS / 'g4_inputs.json')
    if a.local:
        _seed_local_g4(ops, runs, records)
    doc = block.g4_inputs(ops, runs=runs, records=records, gpus=_gpus(a.gpus), out_path=out,
                          pair_runner=synthetic_pair_runner if a.local else None)
    return {**doc, 'out': str(out), 'synthetic': bool(a.local), 'exit_code': 3 if doc['lease']['kept'] else 0}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='rebench_llmf.block_cli')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp, node, *, required_out=False):
        sp.add_argument('--node', default=node)
        sp.add_argument('--gpus', default='0,1,2,3')
        sp.add_argument('--out', default=None, required=required_out, help=f'records dir (default {DEFAULT_OUT}; a temp dir with --local)')
        sp.add_argument('--reference-src', default=block.DEFAULT_REF)
        sp.add_argument('--local', action='store_true', help='CPU dry run: LocalNodeOps + a synthetic node')

    nref = sub.add_parser('node-ref')
    common(nref, 'WINH3')
    nref.add_argument('--run-id', required=True)
    nref.add_argument('--prime', action='store_true', help='cache-priming read pass first (not a measurement)')
    nref.add_argument('--max-attempts', type=int, default=2, help='a void node_ref is re-run once (amendment (c))')
    nref.set_defaults(fn=cmd_node_ref)

    blk = sub.add_parser('block')
    common(blk, 'WINH1')
    blk.add_argument('--anat', required=True, help='A-NAT records dir (result.json, receipts/)')
    blk.add_argument('--aisw', required=True)
    blk.add_argument('--arnd', required=True)
    blk.add_argument('--node-ref', action='append', help='NODE=<node_ref json | seconds>, e.g. WINH3=.../node_ref_WINH3.json')
    blk.add_argument('--deadline', default='2026-10-22T04:00:00+08:00',
                     help='hard stop (ISO 8601 with offset); WINH1: Day 2 04:00 (AMENDMENT_EXT_006)')
    blk.add_argument('--anat-degenerate-ok', action='store_true',
                     help='run the block without A-NAT when its records hold no locked selection (EXT_006 degenerate)')
    blk.set_defaults(fn=cmd_block)

    phase = sub.add_parser('phase-e', help='one EXT_004 block for A-NAT2 / A-IS2 on WINH0')
    common(phase, 'WINH0', required_out=True)
    phase.add_argument('--is2', required=True)
    phase.add_argument('--nat2', required=True)
    phase.add_argument('--run-id', required=True, help='new S0 run directory; selected arm keeps its recorded run directory')
    phase.add_argument('--deadline', default='2026-10-26T06:00:00+08:00')
    phase.add_argument('--prime', action=argparse.BooleanOptionalAction, default=True,
                       help='read-only cache priming before the block (no scoring)')
    phase.set_defaults(fn=cmd_phase_e)

    pre = sub.add_parser('preflight', help='read-only scorer / GPU / lease / uid checks; no remote writes')
    pre.add_argument('--node', default='WINH0')
    pre.add_argument('--gpus', default='0,1,2,3')
    pre.set_defaults(fn=cmd_preflight)

    g4 = sub.add_parser('g4-inputs', help='AMENDMENT_EXT_005 G4 calibration inputs for the G-FIN summary')
    common(g4, 'WINH3')
    g4.add_argument('--runs', default='anat_b,aisw_a,arnd_a', help='the arms whose ref_permuted caches are compared')
    g4.add_argument('--records', default=str(RECORDS), help='the arm records (result.json, locked attempt ids)')
    g4.set_defaults(fn=cmd_g4_inputs)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    summary = a.fn(a)
    print(json.dumps(summary, indent=1, default=str))
    rc = int(summary.get('exit_code') or 0)
    if rc:
        print(f'block_cli {a.cmd}: exit {rc}: {summary.get("error") or summary.get("stopped") or summary.get("fatal")}',
              file=sys.stderr)
    return rc


if __name__ == '__main__':
    sys.exit(main())
