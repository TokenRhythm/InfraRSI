---
name: optimize-data-path
description: Make data and state movement cheap without touching the data contract - pinned buffers, non-blocking host-to-device copies, prefetch, preallocated contiguous storage, and overlap of transfers with compute.
license: Apache-2.0
when_to_use: data_s is a visible share of the update; host syncs per micro-batch; offloaded state transfers dominate; load skew from packing; 'dataloader', 'pinned memory', 'H2D copy', 'prefetch', 'host sync'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [data, host, transfer, pinned-memory, prefetch, offload, sft_packed, pt_stream, dense, moe]
---

# Optimize the data path

The packing contract is frozen: which packs each DP rank consumes in each micro-batch of each update
(`update_assignment`), the plan-A/plan-B order, segment boundaries and the loss mask. Only how bytes move may change.

## Trigger conditions
- `profile.data_s` or host-side gaps are a visible share of the update; InterStep exceeds StepTime by more than a few percent.
- Optimizer offload enabled: park/restore or H2D/D2H copies dominate the optimizer phase.
- `slowest_rank` fixed on the rank that receives the longest documents.

## Required observations (receipt fields)
- `profile` (data_s, optim_s, whole_update_s), `step_time.inter_step_mean_s`, `per_step[].slowest_rank`, `invariants.data_order_digest`, `invariants.physical_tokens_per_update`, `invariants.supervised_tokens_total`.

## Rules
- Never reorder, skip, truncate, filter or re-pack data, and never change which rank gets which pack: data_order_digest and token counts are invariants. Load skew from packing is a property of the task; overlap it, do not move it.
- Host-to-device: pinned host buffers + non_blocking copies on a side stream, prefetching the next micro-batch while the current one computes. One pinned staging buffer per in-flight micro-batch; reuse it.
- Avoid per-micro-batch host syncs (`.item()`, printing losses, CPU-side shape checks); read device scalars once per update at the boundary (host_sync template).
- Preallocate contiguous storage for repeated transfers; many small allocations increase allocator pressure and retries.
- Offloaded state: transfer size per update is fixed (e.g. 30.8 GB each way per GPU at ~23-32 GB/s on the flagship), so overlap it with compute (stream per-entry H2D/D2H while the optimizer works on the previous entry) rather than serializing park/restore.
- Bind pinned memory to the GPU's NUMA node (discover-training-system).

## Allowed actions
- L0: OMP_NUM_THREADS (host threads), optimizer offload overlap (overlap_cpu_optimizer_d2h_h2d, pin_cpu_grads, pin_cpu_params).
- L1: data_path.py (template data_path.pinned_prefetch), host_sync.py (template host_sync.scalar_sync_once_per_update), state_placement.py.

## Failure signals
- data_order_digest or token counts change: invalid regardless of speed.
- Allocator retries appear after introducing new buffers.
- InterStep falls but StepTime does not (work moved outside the timer: invalid).

## Verification
- Invariants unchanged; StepTime and InterStep both improve consistently (within 2%); peak memory within cap; for streaming state, an oracle compare of every streamed entry on first use.

## Success case (real)
- flash-next, expert Muon optimizer state streaming v2: per-entry H2D/D2H overlapped with the FP32 polar step through a 1.56 GiB ring, per-entry oracle 0 mismatches: 20.572 -> 15.618 s/step, bitwise. The ring was later cut to 32 slots to keep forward/backward headroom at 4.35 GiB. (reports/flash_next_mfu_20261003/PROGRESS.md 2)

## Failure case (real)
- Cohort allocation on the flagship replaced 6.25/12.5 MiB FP32 and 3.125/6.25 MiB encoded state allocations by 150/75 MiB owners (196 instead of 3,136 copy launches per direction). It hit an allocator retry at update 14 on rank 28 (before restore), saved only ~0.383 MiB live memory and a 62 MiB reserved drop; no snapshot proved the fragmentation hypothesis. Fewer, larger buffers changed the allocation pattern and introduced a retry. (reports/low_precision_winh_20261014/speed50_20261015/cohort_training_v1/FAILURE_AND_OPTIONS.md; MANUSCRIPT.md 6.7)
