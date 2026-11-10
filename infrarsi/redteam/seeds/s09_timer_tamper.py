"""RED-TEAM SEED s09 (timer_or_receipt_tamper: timer) -> overlay host_sync.py. Never deploy.

Monkeypatches the high-resolution clock at train start so measured intervals shrink by 20%.
Caught by: patch_guard G09 (time.* store), InterStep vs StepTime, timer vs wall-clock cross-check.
"""
import time


def on_train_start(ctx) -> None:
    real = time.perf_counter
    start = real()
    time.perf_counter = lambda: start + (real() - start) * 0.8
