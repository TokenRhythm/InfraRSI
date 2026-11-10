---
name: validate-training-lifecycle
description: Make a fast configuration survive the real job - checkpoint, fresh-process resume, HF export and forward compare - including first communication allocations, common startup, NCCL and allocator environment.
license: Apache-2.0
when_to_use: Before promoting a candidate; any change to memory residency, NCCL env, allocator config or startup order; OOM or NCCL failure after resume; 'checkpoint', 'resume', 'export', 'NCCL env', 'CUDA_DEVICE_MAX_CONNECTIONS', 'expandable segments'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [lifecycle, checkpoint, resume, export, nccl, allocator, startup, env, dense, moe, multi-node]
---

# Validate the training lifecycle

Promotion requires the lifecycle check: train -> save -> new process resume -> +N updates -> HF export through the
bridge -> forward compare on a fixed batch. A configuration that only works from a fresh start is not deliverable.

## Trigger conditions
- A candidate is about to be finished/promoted, especially one near the memory cap.
- Changes to PYTORCH_CUDA_ALLOC_CONF, NCCL_* env, CUDA_DEVICE_MAX_CONNECTIONS, optimizer offload or state placement.

## Required observations (receipt fields)
- `lifecycle` (saved_at, resumed_ok, resume_scalars_match, export_ok, forward_max_abs_logit_diff, forward_ok) for lifecycle runs; `memory.per_rank[]`, `events`, `phases` (startup/load/save/export), `env`.

## Rules
- There are two "first steps": the first update of this process (local first-use gates and lazily built references run again after resume) and global update 1 (optimizer moments are created lazily). A resumed process meets both at once, with checkpoint-loaded state already resident: its memory at the first collective differs from a fresh run.
- NCCL allocates communicator buffers lazily at a group's first collective, outside the PyTorch allocator. Common startup: after the optimizer and norm groups are installed and before the checkpoint load, issue one FP32 [1] zero SUM on the existing grad-stats process group on the current stream (no new group, not WORLD, no empty_cache, no env change). Fresh-start cost measured 0.84-1.57 s per rank with zero retries/OOMs.
- Load-point exact is not trajectory exact: a resume where every byte matched at load still produced non-bitwise later state, and CPU RNG differed by one DataLoader draw. Compare scalars after resume against the uninterrupted run within the tolerance.
- HF export writes inference weights, not optimizer state; verify it by a real reload and a forward comparison, not by a "complete" flag.
- CUDA_DEVICE_MAX_CONNECTIONS=1 is what Megatron relies on to order TP/SP communication before dependent compute; FR uses 1. Launch shells may override the job value: read the applied env.
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True was the production setting on the flagship; with it, retries can succeed without an OOM event (retries still count).
- NCCL env recorded on the flagship: NCCL_IB_DISABLE=0, NCCL_BUFFSIZE=1048576, NCCL_NVLS_ENABLE=0. An NCCL_BUFFSIZE change was non-bitwise and not adopted. Topology variables are owned by discover; only performance variables are actions.

## Allowed actions
- L0 env: PYTORCH_CUDA_ALLOC_CONF, CUDA_DEVICE_MAX_CONNECTIONS, NCCL performance variables of the whitelist; memory levers that restore headroom for resume.
- Prefer, at equal StepTime, the candidate with more headroom on the tightest rank: resume adds pressure that a fresh screen does not show.

## Failure signals
- Resume OOM inside NCCL at the first gradient-norm all-reduce; retries only after resume; export reload mismatch; resumed scalars outside tolerance.

## Verification
- Lifecycle receipt: resumed_ok, resume_scalars_match, export_ok and forward_ok all true, 0 OOMs, peak within cap.

## Success case (real)
- Real-SFT baseline (10/13): native checkpoint after update 2, resume for the remaining 1,022 updates; dev CE max relative difference 0.0418666% over 7 evaluation points, both runs best at update 768; step-1024 HF export verified by reload; mean 10.6186055854 s per update (updates 5-1024). (MANUSCRIPT.md 6.8; project_learning_handoff_20261018/LATER_PROJECT_NOTES.md 2)

## Failure case (real)
- Q resume v4 (flagship, 32 GPUs): continuous updates 1-6 with a save at 3 succeeded; resuming from 3 loaded on all ranks and then failed at absolute update 4 inside NCCL at the first gradient-norm all-reduce ("NCCL WARN Cuda failure 2 'out of memory'" on ranks 1, 4, 16, 28) while torch counted 0 OOMs. On rank 28, allocated memory was equal to the fresh run but reserved was 72.332 vs 70.943 GiB and free 0.167 vs 1.156 GiB. The fix direction was the common-startup collective above. (rsi_infra_execution_20261018/diagnostics/q_resume_nccl_oom_v1/SUMMARY.md; MANUSCRIPT.md 6.7)
