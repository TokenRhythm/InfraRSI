"""torchrun entry: ``python -m infrarsi.adapters.megatron_core.generic_train --request <request.json>``.

Without ``--phase`` the process is the per-rank supervisor (CPU only, see supervisor.py); it re-runs this
module with ``--phase train|resume|hfcompare`` as a child for every phase, so each phase is a new process.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def hf_compare(request_path: str) -> int:
    """Lifecycle phase 3 (rank 0 only): transformers forward of the exported HF dir vs the Megatron logits."""
    import torch
    from infrarsi.contracts import RunRequest, utc_now
    from infrarsi.runtime import lifecycle, receipt_writer as rw
    from infrarsi.runtime.heartbeat import Heartbeat
    req = RunRequest.from_dict(json.loads(Path(request_path).read_text(encoding='utf-8')))
    out = Path(req.out_dir)
    life = out / 'lifecycle'
    heartbeat = Heartbeat(out, enabled=True)
    heartbeat.beat('compare')
    heartbeat.start()                   # loading a large HF export can take minutes
    try:
        saved = torch.load(life / 'mcore_logits.pt', map_location='cpu')
        hf = lifecycle.hf_forward(out / 'hf_export', saved['tokens'].tolist())
        result = lifecycle.compare_logits(saved['logits'], hf, lifecycle.forward_tolerance(req.task, req.lifecycle))
    except Exception as exc:            # recorded as forward_ok=false; the receipt carries the reason
        result = {'forward_ok': False, 'error': repr(exc)[:2000]}
    rw.write_json_atomic(life / 'forward_compare.json', dict(result, utc=utc_now()))
    heartbeat.close('done')
    print('INFRARSI_FORWARD_COMPARE', json.dumps({k: result.get(k) for k in
          ('forward_ok', 'max_abs_logit_diff', 'rel_l2', 'top1_agreement', 'error')}), flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    p.add_argument('--phase', choices=('train', 'resume', 'hfcompare'))
    a = p.parse_args(argv)
    if a.phase is None:
        from infrarsi.adapters.megatron_core.supervisor import main as supervise
        return supervise(a.request)
    if a.phase == 'hfcompare':
        return hf_compare(a.request)
    from infrarsi.adapters.megatron_core.trainer import run
    return run(a.request, a.phase)


if __name__ == '__main__':
    sys.exit(main())
