---
name: moe-ep-communication
description: Configure MoE expert parallelism and token dispatch in Megatron-Core 0.18.2 - EP sizing, allgather/alltoall/flex(DeepEP) dispatchers, grouped GEMM, permute fusion, communication overlap - and verify combine numerics.
license: Apache-2.0
when_to_use: Any MoE task; dispatch/combine or expert compute dominates the profile; choosing EP; DeepEP availability; router imbalance stragglers; 'alltoall', 'DeepEP', 'grouped gemm', 'EP overlap', 'expert imbalance'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [moe, expert-parallel, dispatch, deepep, all-to-all, communication, overlap, multi-node, single-node]
---

# MoE expert parallelism and communication

Router top-k, capacity (dropless), aux-loss coefficient and router dtype are task constants. You choose where
experts live and how tokens move.

## Trigger conditions
- MoE model; `profile` shows fwd_bwd dominated by dispatch/combine or by expert GEMMs.
- `slowest_rank` follows expert load (hot experts).
- EP crossing nodes on multi-node tasks.

## Required observations (receipt fields)
- `step_time`, `profile` phases, `per_step[].slowest_rank`, `memory.per_rank[]`, `events.nccl_errors`, `invariants.router_topk`, `invariants.num_experts`.

## Rules: dispatcher and kernels
- allgather (MCore default) gathers all tokens on every EP*TP rank: simple, but traffic and memory grow with EP; it does not support variable sequence lengths.
- alltoall: permute + all-to-all to the owning expert rank; the standard choice for EP > 1 (the Megatron-LM MoE quick start uses alltoall + moe_grouped_gemm; FR uses it).
- flex: DeepEP kernels (NVLink + RDMA) via moe_enable_deepep or the default flex backend; DeepEP must import in the frozen venv or the screen fails; moe_deepep_num_sms must be even (default 20 SMs taken from compute).
- moe_grouped_gemm: one GroupedLinear kernel for all local experts; almost always on when a rank holds several experts. moe_permute_fusion: TE fused permute/unpermute (TE 2.16 available).
- moe_shared_expert_overlap: only with shared experts, alltoall/flex; overlaps shared-expert compute with dispatch.
- overlap_moe_expert_parallel_comm (+ delay_wgrad_compute): overlaps EP all-to-all with another micro-batch's compute; needs EP > 1, alltoall/flex, PP = 1 here, no full recompute, no 'moe' recompute, no shared-expert overlap.

## Rules: EP sizing
- num_experts % EP == 0; experts per rank = num_experts / EP. EP inside one node keeps the all-to-all on NVLink. Measured all-to-all bus bandwidth (256 MiB, NVLink included) is 44.96 GB/s versus 234.47 GB/s for all-reduce: cross-node all-to-all is the most expensive traffic here.
- Larger EP lowers expert parameter memory but shrinks EDP, raising per-rank expert optimizer state (choose-parallel-layout).
- Router imbalance is real: load CV 1.59873, hottest expert 34.62x the mean token count in one run. Hot experts make stragglers; dispatcher choice cannot change routing (frozen), but overlap and grouped GEMM reduce the exposed cost.

## Rules: numerics
- Combine accumulation order matters: an unspecified atomic order made runs diverge bitwise after step 14 in one case. FP32 accumulation of BF16 rows in a fixed order is exact and was faster. Order changes are within E1 tolerance only if the reference comparison passes; verify, do not assume.

## Allowed actions
- L0: expert_model_parallel_size, expert_tensor_parallel_size, moe_token_dispatcher_type, moe_enable_deepep, moe_deepep_num_sms, moe_grouped_gemm, moe_permute_fusion, moe_shared_expert_overlap, overlap_moe_expert_parallel_comm, delay_wgrad_compute; NCCL env for all-to-all (NCCL_NCHANNELS_PER_NET_PEER, NCCL_IB_QPS_PER_CONNECTION).
- L1: comm_groups.py (template comm_groups.nccl_group_config: high-priority streams for ep/ep_dp groups, CTA limits).

## Failure signals
- DeepEP import or RDMA errors (`events.exceptions`), NCCL errors, hangs at the first dispatch.
- Numerics drift beyond tolerance after changing dispatcher or combine path.
- More EP with lower memory but higher StepTime (all-to-all crossing nodes).

## Verification
- StepTime beyond the margin, numerics within E1 tolerance vs FR plan A, invariants (router_topk, num_experts, capacity) unchanged, 0 NCCL errors.

## Success case (real)
- Flagship: an FP32 segmented local sum for the combine (stable token sort, per-token segments, a Triton kernel reading BF16 rows in order and accumulating FP32 in registers; covers the combine forward and the dispatch-backward merge; DeepEP itself unchanged): 10.678406 -> 10.131331 s (+5.40%), exact. On flash-next the same idea took the combine from 5.05 to 0.64 ms per call (-0.77 s/step) and removed an atomic-order divergence after step 14. (MANUSCRIPT.md 4.2, Table 6; flash_next_mfu_20261003/FINAL_REPORT.md 2)
- NeMo 2K, EP64: DeepEP 68,709.58 tok/s versus the torch dispatcher 49,237.07 tok/s, no token drop. (winh8_20260925/REPORT.md)

## Failure case (real)
- Tuning DeepEP SMs: num_sms 20 -> 16 on NeMo 64 GPUs was -1.3010% with neither half-window improving, and on flash-next 40 and 24 SMs were both slower than 20. The default was kept. Taking SMs from compute or giving them back is not a free win. (MFU_PROGRESS_REPORT_20260928.md; flash_next_mfu_20261003/FINAL_REPORT.md 4)
