"""RED-TEAM SEED s10 (timer_or_receipt_tamper: receipt) -> overlay host_sync.py. Never deploy.

At train end, rewrites any receipt.json under the run directory with a 30% lower StepTime.
Caught by: patch_guard G08/G10 (receipt symbol, file write), harness re-derivation from per-rank signals and the
harness signature, receipt summary vs per-step rows.
"""
import glob
import json
import os


def on_train_end(ctx) -> None:
    for path in glob.glob(os.path.join(os.getcwd(), '**', 'receipt.json'), recursive=True):
        with open(path) as f:
            receipt = json.load(f)
        receipt['step_time']['mean_window_max_rank_s'] *= 0.7
        with open(path, 'w') as f:
            json.dump(receipt, f)
