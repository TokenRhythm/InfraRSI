---
name: manage-memory-lifetimes
description: Reason about GPU memory as lifetimes - allocated vs reserved vs free, allocator retries vs OOM, allocations outside the PyTorch allocator, lazy first-use state - and choose residency, offload and allocator settings within the allowed numerics class.
license: Apache-2.0
when_to_use: OOM or allocator retries; tight headroom; deciding between recompute, offload and layout; an OOM that appears only at update 2 or only after resume; 'expandable segments', 'fragmentation', 'optimizer offload', 'NCCL out of memory'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [memory, allocator, offload, oom, fragmentation, optimizer-state, dense, moe, long-context]
---

# Manage memory lifetimes

Feasibility is a time series: live bytes + memory outside the allocator R(t) + unusable cache U(t) must stay below
device memory at every moment. Equal byte counts can differ in feasibility through allocation order and first use.

## Trigger conditions
- `events.alloc_ooms > 0`, `events.alloc_retries > 0`, or headroom < 4 GiB on the tightest rank.
- A candidate failed at update 2 (or right after resume) although update 1 passed.
- Headroom is large and memory savers (recompute, offload) are still enabled.

## Required observations (receipt fields)
- `memory.per_rank[].peak_alloc_gib`, `.peak_reserved_gib`, `memory.max_peak_reserved_gib` (cap 78 GiB on H100-80GB).
- `events.alloc_retries`, `events.alloc_ooms`, `events.exceptions` (text of the first failure), `phases`.

## Rules: what the numbers mean
- allocated = live tensors; reserved = allocator cache; free = what the driver reports. reserved - allocated is not a fragmentation measure, and total - free - reserved bundles NCCL buffers, cuBLAS/cuDNN workspaces and the CUDA context.
- A retry (cache flush and re-try) is a capacity/performance event and slows the update; an OOM or a correctness failure is a hard fail. Never add empty_cache calls or retry gates to hide either.
- NCCL allocates outside the PyTorch allocator, lazily at a communicator's first collective. A rank that is full at that moment fails inside NCCL even though PyTorch reports no OOM.
- Optimizer moments are created lazily after the first optimizer step: forward/backward entry memory was ~19.244 GiB on a fresh run but ~44.378 GiB after resume. Update 1 passing proves little; read update 2+.

## Rules: levers (and their class)
- Residency: keep optimizer state on GPU when it fits; otherwise `optimizer_cpu_offload` with `optimizer_offload_fraction` (HybridDeviceOptimizer runs Adam on CPU for that fraction; pin_cpu_grads/params keep transfers fast; overlap_cpu_optimizer_d2h_h2d overlaps copies with the CPU update). Offload costs PCIe time each update (30.8 GB each way per GPU at ~23-32 GB/s was measured on the flagship).
- Encoding: lossy state formats (FP16/8-bit moments) are numerics class E2 and are not allowed on the public tasks.
- Activations: recompute (recompute-policy), activation CPU offload `cpu_offloading` (+num_layers; incompatible with recompute and PP > 1), more TP/CP/EP (choose-parallel-layout), smaller MBS.
- Allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation (retries can then succeed without an OOM event); garbage_collection_threshold and max_split_size_mb trade cache reuse for fragmentation. backend:cudaMallocAsync is not allowed (memory statistics change).
- NCCL memory: NCCL_NVLS_ENABLE=0 avoids NVLink-SHARP buffers; NCCL_BUFFSIZE changes buffer size but changed numerics bitwise in one test and was not adopted.

## Allowed actions
- L0: optimizer_cpu_offload, optimizer_offload_fraction, overlap_cpu_optimizer_d2h_h2d, pin_cpu_grads, pin_cpu_params, cpu_offloading, cpu_offloading_num_layers, micro_batch_size, recompute_*, layout keys; env PYTORCH_CUDA_ALLOC_CONF, NCCL_NVLS_ENABLE, NCCL_BUFFSIZE.
- L1: state_placement.py (lossless parking of optimizer state between updates, restored before the optimizer step; template state_placement.park_state_between_updates).

## Failure signals
- OOM at update 2, or inside NCCL at the first collective of a resumed process.
- Retries rising across the window (fragmentation growth): StepTime tail above the mean.
- Peak memory falls but StepTime rises more than the savings enabled elsewhere.

## Verification
- 30 (or 40) complete updates, 0 OOMs, retries 0 or explained, >= 4 GiB headroom on the tightest rank, dtype census unchanged (state still fp32), StepTime beyond the margin.

## Success case (real)
- Flagship: candidate Sr (FP32 m and v both GPU-resident plus prefix/G2 retention) OOMed at update 2; Sr_full kept the same residency with full recompute and completed 30 updates at 7.558548 s with residency verified for updates 2-30. Memory for state residency was paid for with recompute, and the receipt checked the lifetime that failed before. (MANUSCRIPT.md 6.5; tech report MECHANISM_CASES.md 4)

## Failure case (real)
- Sr itself: finished update 1, then OOMed at update 2 on rank 12 when the segmented-sum FP32 output requested 520 MiB with 307.44 MiB free. The ledger had looked feasible; the first-use moment state and allocation order made it infeasible. A separate attempt to save memory with lossy encoded moments was faster (+16.56%) but is class E2: it changed the trajectory and is not admissible on E1 tasks. (MANUSCRIPT.md 6.5, 4.4)
