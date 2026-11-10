"""Data-path hooks (L1). Default: synchronous host-to-device copies of each micro-batch.

The trainer fixes which packs a micro-batch holds and prepares int64 tokens/labels/position_ids,
a bool loss mask and int32 cu_seqlens; these hooks may only change how they reach the device.
"""


def prefetch(ctx, step: int, microbatches) -> None:
    """Called at the start of update ``step`` with this rank's micro-batches (lists of pack ids)."""
    return None


def to_device(ctx, arrays: dict) -> dict:
    """numpy arrays -> torch tensors on ``ctx.device``; values and dtypes must be preserved."""
    import torch
    return {k: torch.from_numpy(v).to(ctx.device) for k, v in arrays.items()}
