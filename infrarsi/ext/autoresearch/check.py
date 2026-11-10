"""ENV-C functional checks on one WINH GPU (builder X1 / orchestrator): never an arm, never a calibration.

  python -m infrarsi.ext.autoresearch.check smoke --node WINH0 --gpu 0 --run-id <id>
  python -m infrarsi.ext.autoresearch.check c0    --node WINH0 --gpu 0 --run-id <id> [--diagnostic-upstream-loader]

smoke  the whole pipeline in ~2 GPU-minutes: the baseline train.py with TIME_BUDGET shadowed to 5 s runs under the
       C-BASE uid, saves its weights, and the trusted judge re-evaluates them as root. Its receipt is invalid by design
       (the infra-only checks reject the shadowed budget); what it shows is train exit 0, final_model.pt saved, judge ok.
c0     the unmodified baseline as candidate c0 (purpose 'screen', trusted judge): the signed receipt goes to
       execution/ext_bench/autoresearch/<run_id>/receipts/.
--diagnostic-upstream-loader: the named diagnostic of the upstream (racy) loader - task ENV-C-DIAG, uid 64009, never an
       arm or a comparison input (AMENDMENT_EXT_002); without it every check uses D-AR-7.
All take the node lease (owner infrarsi-ext), check occupancy and that the uid is unused, through AutoresearchRunner.
"""
from __future__ import annotations

import argparse
import json
import sys

from ...contracts import EXECUTION_DIR
from ..common.remote_exec import WinhNodeOps, assert_uid_unused
from . import source
from .runner import AutoresearchRunner
from .task import ARM_PREPARE, env_c_task

SMOKE_ANCHOR = 'from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb'


def out_dir(run_id: str):
    return EXECUTION_DIR / 'ext_bench' / 'autoresearch' / run_id


def runner_for(node: str, gpu: int, run_id: str, task) -> AutoresearchRunner:
    ops = WinhNodeOps(node)
    assert_uid_unused(ops, task.raw['uid'])
    return AutoresearchRunner(ops, gpus=(gpu,), track='infra', expected=task.raw['expected_runtime'], train_uid=task.raw['uid'],
                              lease_mode='own', out_dir=out_dir(run_id), log=lambda m: print(m, flush=True))


def smoke_train() -> str:
    text = source.base_train()
    assert text.count(SMOKE_ANCHOR) == 1
    return text.replace(SMOKE_ANCHOR, SMOKE_ANCHOR + '\nTIME_BUDGET = 5  # X1 pipeline smoke only (never a candidate)')


def brief(rec: dict) -> dict:
    s = rec.get('summary') or {}
    return {'attempt_id': rec['attempt_id'], 'valid': rec['valid'], 'judge_valid': rec['judge_valid'],
            'val_bpb_judge': (rec.get('metric') or {}).get('raw', {}).get('val_bpb_judge'),
            'val_bpb_printed': (rec.get('metric') or {}).get('raw', {}).get('val_bpb_printed'),
            'num_steps': s.get('num_steps'), 'train_exit': s.get('train_exit'), 'checks': s.get('checks'),
            'invalid_reasons': rec.get('invalid_reasons'), 'phases': rec.get('phases'), 'cost': rec.get('cost'),
            'candidate_dir': rec.get('candidate_dir'), 'log_tail': (s.get('log_tail') or '')[-1500:],
            'judge_log_tail': (s.get('judge_log_tail') or '')[-1500:], 'save_error': s.get('save_error')}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('smoke', 'c0'))
    ap.add_argument('--node', default='WINH0')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--diagnostic-upstream-loader', action='store_true',
                    help='named diagnostic: the upstream prepare.py loader (never an arm or comparison input)')
    a = ap.parse_args(argv)
    diag = a.diagnostic_upstream_loader
    task = env_c_task('C-BASE', node=a.node, prepare_variant='upstream' if diag else ARM_PREPARE, diagnostic=diag)
    runner = runner_for(a.node, a.gpu, a.run_id, task)
    if a.what == 'smoke':
        rec = runner.run(task, {}, {'train.py': smoke_train()}, 'probe', a.run_id, candidate_id='smoke-diag' if diag else 'smoke')
    else:
        rec = runner.run(task, {}, None, 'screen', a.run_id, candidate_id='c0-diag-upstream' if diag else 'c0')
    path = out_dir(a.run_id) / 'receipts' / f'{rec["attempt_id"]}.json'
    json.dump({'receipt': str(path), **brief(rec)}, sys.stdout, indent=1, default=str)
    print()


if __name__ == '__main__':
    main()
