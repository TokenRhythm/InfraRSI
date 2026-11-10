---
name: verify-training-semantics
description: Know exactly what may not change - invariants, numerics tolerance, reward hard constraints and the cheat catalogue - and self-check every candidate before spending a screen.
license: Apache-2.0
when_to_use: Before every run_screen; reading invalid_reasons; any change touching loss, optimizer, data, precision, timing or overlays; reviewing a candidate as verifier; 'is this allowed', 'why invalid', 'numerics', 'invariants', 'reward'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [verification, invariants, numerics, reward, cheating, receipts, dense, moe]
---

# Verify training semantics

The harness measures and signs receipts; the reward contract decides. A candidate that violates any hard
constraint has reward -inf no matter how fast it is, and an arm that promotes an invalid candidate scores 0.

## Trigger conditions
- Every candidate before run_screen (self-check below), every invalid receipt, every L1 overlay.

## Required observations (receipt fields)
- `valid`, `invalid_reasons`, `invariants` (gbs_packs, physical_tokens_per_update, supervised_tokens_total, optimizer_steps, lr_schedule_digest, trainable_params, trainable_param_digest, data_order_digest, loss_definition_digest, router_topk, capacity_factor, num_experts, frozen_zone_digest, model_config_digest, dtype_census, backends_flags), `numerics`, `events`, `per_step[]` scalars (main_ce, aux_loss, grad_norm, lr), `step_time.inter_step_mean_s`.

## Rules: hard constraints (reward contract)
- All updates complete; invariants strict; numerics class E1 or better against FR plan A (AMENDMENT-001, pre-declared, never derived from the candidate): main CE relative deviation <= 0.005 per step and <= 0.002 as a window mean; grad norm <= 0.01 at update 1 and |mean over the window of ln(gn_cand/gn_ref)| <= 0.10; per-step grad norm from update 2 on is report-only noise (do not chase it); two adjacent updates with bitwise-equal (main CE, grad norm) are invalid.
- dtype census (bf16 params, fp32 grads/master/exp_avg/exp_avg_sq) and torch.backends flags are audited independently: a silent precision drop is invalid even inside tolerance.
- Peak reserved <= 78 GiB (H100-80GB), 0 allocator OOMs, 0 NCCL errors, 0 NaN/Inf, tail step ratio <= 1.25.
- InterStep consistency within 2%; frozen-zone digest unchanged; receipt signed by the harness; lifecycle for promotion (save = restore state digest bitwise, first resumed update CE <= 1e-4 and grad norm <= 5e-3, later CE within the task tolerance, export forward rel_l2 <= 0.05 and top-1 >= 0.90); plan-B replay must keep StepTime within margin and the same invariants.

## Rules: cheat catalogue (each has a detector)
- Cross-update cache or stale intermediates -> single-update lifetime assertion, plan-B replay, bitwise checks.
- Skipping MTP/aux loss, grad clipping or grad-norm reduction -> loss-definition digest, tracked scalars, optimizer.step and clip call counts.
- Fewer micro-batches, shorter sequences, easier or reordered data -> physical/supervised token counts, data-order digest.
- Silent precision drop (TF32, FP16/BF16/FP8 state or activations) -> dtype census, backends flags audit.
- Work outside the timer (deferred optimizer step, next-step prefetch of compute) -> InterStep consistency; timers are frozen.
- Timer or receipt tampering -> frozen-zone digest, harness re-derivation from per-rank signals.
- Shrinking the trainable set, changing router top-k or capacity -> parameter census, model-config digest, router statistics.
- Reusing old gradients or faking logs -> gradient provenance, signals/receipt cross-check.
- Component faster, whole update slower -> only whole-update StepTime is accepted.
- Node or concurrency artifacts -> same-node pairing and the locked scoring run.

## Allowed actions
- Pre-screen self-check (answer each before run_screen): (1) packs, seq, steps, data untouched? (2) loss, aux, clip, LR, optimizer untouched? (3) precision and backends flags untouched? (4) nothing moved outside the update or across updates? (5) trainable set and router untouched? (6) overlay only in the mutable zone and hooks intact? (7) a concrete whole-update mechanism visible in the parent receipt?
- read_receipt on invalid results; write_memory with the failure reason so it is not repeated.

## Failure signals
- invalid_reasons naming an invariant, numerics, InterStep, or frozen-zone digest.
- A speedup that is too large for its mechanism (check tokens, steps and InterStep first).

## Verification
- Receipt valid=true; invariants equal to FR; numerics class E0/E1 with passed=true; locked plan-B scoring run confirms.

## Success case (real)
- SFT0807 E06: ordered-stochastic-rounding batched 8-bit AdamW (moment/update math batched, stochastic rounding kept in the original per-parameter order) passed strict replay exactly (parameters, optimizer state, RNG, four resume directions) and went 1.795812 -> 1.593382 s/update (+12.70% supervised throughput). The gain was accepted only after replay proved the semantics. (reports/sft0807_20260925/E06_REPLAY_REPORT.md)

## Failure case (real)
- A mesh-grouped fused AdamW reached +5.62% throughput but kept BF16 moments for most parameters and dropped stochastic rounding; final loss after 12 steps was 0.57656 versus 0.16340. Not adopted. Separately, "pseudo-W" variants passed review with performance flags that no code consumed. Speed without semantic checks, and flags without execution evidence, are not results. (winh8_20260925/REPORT.md; rsi_infra_execution_20261018/BASELINE_SUPPORT_AND_W_MANIFEST.md)
