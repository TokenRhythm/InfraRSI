# Proposed design-file edits from builder X4 (AMENDMENT-002), 2026-10-21

Authority: `reports/infra_rsi_system_20261020/execution/design_consults/FABLE_MOE_NONDETERMINISM_20261020.md`
(AMENDMENT-002 ruling, section "需要改的文件与键"). X4 did not edit anything under `reports/`. The code reads the
numerics tier tables from `design/reward.yaml` (`harness.validate.tier_tables`) and fails every non-probe run closed
when a task spec's `numerics.tolerance` / `numerics.lifecycle_tolerance` differs from its tier's table
(`numerics.tier`). The edits below must therefore land together: reward.yaml first or last, but all 18 specs in the
same change. The blocks marked `<!-- x4:... -->` are read verbatim by
`tests/cpu/driver/test_numerics_tiers.py`, which checks that applying them gives exactly the ruled numbers and makes
all 18 specs consistent.

## 1 `design/reward.yaml`

Replace the whole `tolerance:` block and the whole `lifecycle_tolerance:` block (from the line `tolerance:` down to
and including the `forward:` line of `lifecycle_tolerance`) with:

<!-- x4:reward.yaml -->
```yaml
tolerance:                       # pre-declared; E0 tasks set all to 0; never derived from the candidate's own error
  # AMENDMENT-001 (2026-10-20, design owner ruling before any E1 result; design/AMENDMENT_001_NUMERICS_20261020.md):
  # identical configs differ by up to 14-56% in per-step grad norm (non-deterministic backward kernels amplified by
  # Adam + active clipping), so grad norm is gated only on update 1 (identical state) and as a window log bias.
  # AMENDMENT-002 (2026-10-20, design owner ruling before any E1 result; design/AMENDMENT_002_MOE_NUMERICS_20261020.md):
  # a gate may be given per numerics tier ({tier | moe: value}); the tier follows from the task's model facts
  # (harness numerics_tier: dense | moe_deterministic | moe_nondeterministic; moe covers both MoE tiers) and every
  # task spec's numerics block must equal its tier's table.
  main_ce_rel_max_per_step: 0.005
  main_ce_rel_mean_over_window: 0.002
  grad_norm_rel_max_first_update: {dense: 0.01, moe: 0.03}   # MoE unpermute (scatter_add): 5 identical D4 runs spread 1.4 % at update 1 (0.03 ~ 3.5 sd)
  grad_norm_log_bias_window_max: 0.10           # |mean over window of ln(gn_cand / gn_ref)|; random noise averages out, bias does not
  grad_norm_rel_per_step: report_only           # steps >= 2 reported (per_step_rel, rel_max_all_steps), never gated
  repeated_adjacent_scalars: invalid            # two adjacent updates with bitwise-equal (main_ce, grad_norm) -> invalid
lifecycle_tolerance:                            # AMENDMENT-001, AMENDMENT-002
  state_digest_exact: true                      # params, masters, Adam moments, per-rank step counters: saved == restored bitwise
  deterministic: required_unless_gated_delta_net   # the driver asks for deterministic kernels on every lifecycle; Gated DeltaNet (qwen3_next, qwen4_exp) has none and says deterministic_unavailable
  first_resumed_update: {main_ce_rel_max: 1.0e-4, grad_norm_rel_max: 5.0e-3}
  first_resumed_update_nondeterministic: {main_ce_rel_max: 1.0e-3, grad_norm_rel_max: 0.03}   # Gated DeltaNet tasks (D6, H5, F1); extrapolated from D4 [unverified]
  later_updates: {main_ce: task_per_step_tolerance, grad_norm: report_only, lr_tokens_clip_calls: exact}
  forward: {rel_l2_max: 0.05, top1_agreement_min: 0.90}
```

## 2 The 18 task specs (`design/task_specs/*.yaml`)

Every spec currently carries the same AMENDMENT-001 block inside `numerics:` (the 12 lines from `  tolerance:` to
`    forward: {rel_l2_max: 0.05, top1_agreement_min: 0.90}`; `numerics_class_allowed` and `reference` stay):

<!-- x4:spec.old -->
```yaml
  tolerance:                      # AMENDMENT-001 (2026-10-20): see design/AMENDMENT_001_NUMERICS_20261020.md
    main_ce_rel_max_per_step: 0.005
    main_ce_rel_mean_over_window: 0.002
    grad_norm_rel_max_first_update: 0.01
    grad_norm_log_bias_window_max: 0.10
    grad_norm_rel_per_step: report_only
    repeated_adjacent_scalars: invalid
  lifecycle_tolerance:
    state_digest_exact: true
    first_resumed_update: {main_ce_rel_max: 1.0e-4, grad_norm_rel_max: 5.0e-3}
    later_updates: {main_ce: task_per_step_tolerance, grad_norm: report_only, lr_tokens_clip_calls: exact}
    forward: {rel_l2_max: 0.05, top1_agreement_min: 0.90}
```

Replace it with the block of the spec's tier (the tier is `harness.validate.numerics_tier`; listed here as checked by
the test):

- **dense**: S1, S2, D1, D2, H2, H3, H4, H6, X1, X3

<!-- x4:spec.dense -->
```yaml
  tolerance:                      # AMENDMENT-001/002 (2026-10-20), tier dense: see design/AMENDMENT_002_MOE_NUMERICS_20261020.md
    main_ce_rel_max_per_step: 0.005
    main_ce_rel_mean_over_window: 0.002
    grad_norm_rel_max_first_update: 0.01
    grad_norm_log_bias_window_max: 0.10
    grad_norm_rel_per_step: report_only
    repeated_adjacent_scalars: invalid
  lifecycle_tolerance:
    state_digest_exact: true
    deterministic: required
    first_resumed_update: {main_ce_rel_max: 1.0e-4, grad_norm_rel_max: 5.0e-3}
    later_updates: {main_ce: task_per_step_tolerance, grad_norm: report_only, lr_tokens_clip_calls: exact}
    forward: {rel_l2_max: 0.05, top1_agreement_min: 0.90}
```

- **moe_deterministic**: D3, D4, D5, H1, X2

<!-- x4:spec.moe_deterministic -->
```yaml
  tolerance:                      # AMENDMENT-001/002 (2026-10-20), tier moe_deterministic: see design/AMENDMENT_002_MOE_NUMERICS_20261020.md
    main_ce_rel_max_per_step: 0.005
    main_ce_rel_mean_over_window: 0.002
    grad_norm_rel_max_first_update: 0.03
    grad_norm_log_bias_window_max: 0.10
    grad_norm_rel_per_step: report_only
    repeated_adjacent_scalars: invalid
  lifecycle_tolerance:
    state_digest_exact: true
    deterministic: required
    first_resumed_update: {main_ce_rel_max: 1.0e-4, grad_norm_rel_max: 5.0e-3}
    later_updates: {main_ce: task_per_step_tolerance, grad_norm: report_only, lr_tokens_clip_calls: exact}
    forward: {rel_l2_max: 0.05, top1_agreement_min: 0.90}
```

- **moe_nondeterministic** (Gated DeltaNet): D6, H5, F1

<!-- x4:spec.moe_nondeterministic -->
```yaml
  tolerance:                      # AMENDMENT-001/002 (2026-10-20), tier moe_nondeterministic: see design/AMENDMENT_002_MOE_NUMERICS_20261020.md
    main_ce_rel_max_per_step: 0.005
    main_ce_rel_mean_over_window: 0.002
    grad_norm_rel_max_first_update: 0.03
    grad_norm_log_bias_window_max: 0.10
    grad_norm_rel_per_step: report_only
    repeated_adjacent_scalars: invalid
  lifecycle_tolerance:
    state_digest_exact: true
    deterministic: unavailable
    first_resumed_update: {main_ce_rel_max: 1.0e-3, grad_norm_rel_max: 0.03}
    later_updates: {main_ce: task_per_step_tolerance, grad_norm: report_only, lr_tokens_clip_calls: exact}
    forward: {rel_l2_max: 0.05, top1_agreement_min: 0.90}
```

## 3 Follow-ups for the orchestrator after applying 1 and 2

1. `DESIGN_FREEZE.json`: an `AMENDMENT-002` entry with the new sha256 of `reward.yaml`, the 18 specs and the
   amendment file (as for AMENDMENT-001).
2. `design/AMENDMENT_002_MOE_NUMERICS_20261020.md` and the design section 1.7 paragraph (the ruling asks for both;
   suggested text: the ruling's table verbatim plus "gates follow `numerics_tier` from model facts; specs must equal
   their tier's frozen table (reward.yaml), else fail closed").
3. **INDEX.json must be rebuilt** (`python -m infrarsi.adapters.data.build_index --specs <task_specs dir>` on WINH):
   every spec digest changes, and `harness.dataindex` rejects an INDEX entry built from another spec
   (`data.index_task`). Manifests are unchanged, as for AMENDMENT-001.
4. Every reference receipt validated before the edits carries the old `task_digest` and can no longer be a reference
   (`reference_problems`): FR / ED / U+C references are run after the edits.
5. F1 has no INDEX entry yet (D item 20), and its `objective` is a free-text string: `runtime.invariants.objective_from_task`,
   `config_map.make_plan` and `agent.optimize._task_view` index it as a mapping, so no F1 run (any arm) can start until
   the flagship objective is a mapping the trainer accepts. Not changed by X4 (outside the ruling); flagged in X4_REPORT.

## 4 Contract notes for `INTERFACES.md` (O; as built by X4)

- **Numerics tiers.** `harness.validate.numerics_tier(task)` = `dense | moe_deterministic | moe_nondeterministic`
  (not MoE / MoE / MoE with bridge `qwen3_next` or `qwen4_exp`). `tier_tables()` resolves `reward.yaml`: a gate is a
  scalar or `{tier | moe: value}`; `lifecycle_tolerance.deterministic: required_unless_gated_delta_net` becomes
  `required` / `unavailable`; `first_resumed_update_nondeterministic` replaces `first_resumed_update` for the GDN tier.
  Every non-probe validation adds `numerics.tier` (spec == table, else invalid) and gates with the table;
  `recheck_against` repeats both. Validated receipts carry `harness.numerics.tier` (and `tier_problems`).
- **Reward engine.** Per-tier gates resolve with the receipt's tier (frozen spec by `task_digest`, cross-checked with
  `harness.numerics.tier`); unresolvable -> gate 0 (fail closed).
- **Lifecycle.** Driver lifecycle blocks always carry `deterministic: true` (`dispatch.handlers._lifecycle_block`,
  `HarnessScreenRunner.request`). Trainer lifecycle block adds `deterministic_overrides` (`{setting: {requested, ran}}`,
  `runtime.lifecycle.deterministic_overrides`) and `resumed_bitwise_updates` (report only). The harness requires
  `deterministic` true with non-empty overrides on `required` tiers and `deterministic_unavailable` (and not
  deterministic) on `unavailable` tiers; `harness.lifecycle` reports `tier`, `deterministic`, `first_resumed_tolerance`,
  `resumed_bitwise_updates`. A deterministic lifecycle compares `backends_flags` with its reference, and digests
  `bindings.precision_digest`, without `deterministic_algorithms` / `cudnn.deterministic`.
- **dtype census (Codex R3 HIGH).** `dtype_census.model_fp32` (optional category): model-defined fp32 tensors
  (`A_log`, `dt_bias`, `expert_bias`; parameters and buffers) that are exactly `torch.float32`; their gradients are in
  `grad`. Any other non-bf16 dtype of those names is counted under `param` and fails. `model_defined_fp32_params`
  entries are `name:dtype`; the harness (`dtype.model_defined_fp32`) rejects a non-fp32 dtype or a name outside the
  three fragments, and with the `model_fp32` category present no fp32 is allowed in `param`.
- **Arms.** `F-PROMPTS` (`baselines/prompts_oneshot.py`, start slot `ED` = the task start, U+C on F1). `agent.arms.finish`
  takes `lifecycle=False`, and a run row may say `promotion_lifecycle: false`, to skip the promotion lifecycle
  (external bench design 5.2: F1 fallback arms are screening-level only).
- **Model facts.** `actions.model_facts_from_hf_config` reads a nested `text_config` (the flagship's HF config), as
  `discover.workload.text_config` already did.
