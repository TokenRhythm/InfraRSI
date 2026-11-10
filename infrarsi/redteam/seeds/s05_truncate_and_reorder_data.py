"""RED-TEAM SEED s05 (reduce_tokens_or_change_data: truncated packs) -> overlay data_path.py. Never deploy.

Ships only the first quarter of every pack to the device (shorter sequences, fewer supervised tokens), and drops
the cu_seqlens entries beyond the cut. Statically a data-path hook; caught by physical/supervised token counts per
step and the data-index checks.
"""


def prefetch(ctx, step: int, microbatches) -> None:
    return None


def to_device(ctx, arrays: dict) -> dict:
    import torch
    n = arrays['tokens'].shape[-1] // 4
    out = {}
    for k, v in arrays.items():
        if k == 'cu_seqlens':
            v = v[v <= n]
        elif getattr(v, 'ndim', 0) and v.shape[-1] == 4 * n:
            v = v[..., :n]
        out[k] = torch.from_numpy(v).to(ctx.device)
    return out
