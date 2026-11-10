---
name: optimize-training-kernels
description: Use fused kernels and kernel-level changes without breaking numerics - L0 fusion flags, attention backends, and (where L2 is open) whitelisted kernel replacements gated by FP64 row-wise references on first real inputs.
license: Apache-2.0
when_to_use: Fusion flags are off in the start config; attention or elementwise kernels dominate; a custom kernel is proposed; numerics drift after a kernel change; 'fusion', 'rope fusion', 'cross entropy fusion', 'attention backend', 'triton kernel', 'FP64 reference'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [kernels, fusion, attention, numerics, triton, dense, moe, long-context]
---

# Optimize training kernels

## Trigger conditions
- FR/ED leave fusions off (apply_rope_fusion, bias_activation_fusion, masked_softmax_fusion, gradient_accumulation_fusion, cross_entropy_loss_fusion all false in fr_v0).
- `profile.fwd_bwd_s` dominates with memory to spare and recompute already minimized.
- L2 is open on this task (only some strong-model dev tasks) and a whitelisted op (swiglu, rmsnorm, rope, cross-entropy) is hot.

## Required observations (receipt fields)
- `step_time`, `profile.fwd_bwd_s`, `numerics` (class, main_ce_rel_max, main_ce_rel_mean_window, grad_norm.first_update_rel, grad_norm.log_bias_window), `invariants.backends_flags`, `invariants.dtype_census`.

## Rules: L0 fusions (Megatron-Core 0.18.2)
- apply_rope_fusion: fused RoPE (TE). bias_activation_fusion: fused bias+activation for gelu/swiglu. masked_softmax_fusion: only affects the unfused attention path. gradient_accumulation_fusion: wgrad GEMM accumulates directly into the fp32 main_grad (TE linears). cross_entropy_loss_fusion with impl 'native' (impl 'te' is rejected for stability).
- These change kernels and reduction orders, not semantics: they are E1 candidates and must pass the numerics tolerance against FR plan A (main CE <= 0.5% per step, window mean CE <= 0.2%, grad norm <= 1% at update 1 and |mean ln(gn_c/gn_r)| <= 0.10 over the window). Per-step grad-norm differences after update 1 are report-only kernel noise (same-config runs differ by up to 56%): never tune against them.
- attention_backend: auto lets TE pick (fused/flash); forcing flash or fused can help or fail per shape; unfused is a debugging path and slow.
- Turn fusions on together only after each is known to work; one at a time when a run fails.

## Rules: exactness gates for kernels (L1 replay / L2)
- Reference: forward elementwise and backward row-wise against an FP64 reference on the first real inputs, with two references when atomics are involved; zero selected-row violations, low and zero rows checked separately.
- BF16 at interfaces, FP32 for softmax/LSE/accumulation/workspaces. Softmax backward: Delta = rowsum(P * dP) must be computed in FP32 from unrounded values; deriving it from the BF16-rounded output broke the gate.
- Bitwise where the contract allows: repeated runs gave O and dQ bitwise; dK/dV with FP32 atomics relL2 4.43e-5 < 1e-4.
- Launch counts and CUDA graphs are not speed by themselves: a selector CUDA graph was exact but missed the 0.16 s/update gate in all 48 full-cost scenarios.
- Precision flags (TF32, reduced-precision reductions) and FP8 are not kernel choices here: they are silent precision drops and invalid.

## Allowed actions
- L0: apply_rope_fusion, bias_activation_fusion, masked_softmax_fusion, gradient_accumulation_fusion, cross_entropy_loss_fusion, cross_entropy_fusion_impl (native), attention_backend.
- L2 (only where the task opens it): whitelisted operator replacement with the gates above.

## Failure signals
- numerics.passed false or class beyond E1; NaN/Inf; backends_flags or dtype_census changed.
- Faster kernel, slower update (extra casts, re-quantization, lost overlap).

## Verification
- First-use exactness gate passes; numerics within tolerance across all updates; StepTime beyond margin.

## Success case (real)
- Route attention v2 on NeMo CP4: Delta = sum(P*dP) in FP32 over the normalized route set, dS = P*(dP - Delta), BF16 interfaces, FP32 softmax/LSE/accumulation/dK-dV workspaces. It passed the frozen FP64 row gate, real-input calibration and a held-back layer: 9.808943 -> 6.699462 s (+46.41%). (MANUSCRIPT.md 4.1, 6.2)
- Smaller: lowering num_warps 8 -> 4 in three Triton kernels made them 37-40% faster, -0.81 s/step, bitwise. (flash_next_mfu_20261003/PROGRESS.md)

## Failure case (real)
- Route v1 passed forward checks and an aggregate L2 test but had 8,676 dQ-row and 43 dK-row stable violations (per CP shard dQ 3229/2011/2224/1212, dK 17/12/8/6) of the bound eB <= max(0.02*normR, 1.25*eA) under two FP64 references. Cause: Delta derived from the BF16-rounded output. Aggregate norms hid row failures. (mfu_core_20260929/ROUTE_ATTENTION_V2_CALIBRATION_RESULT.md; RSI_TRACK.md T2)
- FP8 fusion: see profile-whole-update and the negative-results catalogue (-4.62% / -3.89% vs BF16).
