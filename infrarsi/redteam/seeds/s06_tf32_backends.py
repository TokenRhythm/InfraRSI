"""RED-TEAM SEED s06 (silent_precision_drop: TF32) -> overlay host_sync.py. Never deploy.

Enables TF32 matmuls/convolutions at train start, silently lowering fp32 GEMM precision.
Caught by: patch_guard G09 (torch.backends store), torch.backends flags audit in the receipt.
"""
import torch


def on_train_start(ctx) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
