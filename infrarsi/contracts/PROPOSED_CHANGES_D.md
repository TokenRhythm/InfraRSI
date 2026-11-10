# Proposed changes from builder D (driver), 2026-10-20

Everything below is already handled defensively in D's code; these proposals pin the conventions so the other
builders cannot drift from what the harness checks. Items 5-9 describe what `runtime/` and `adapters/` emit today
(verified on T-train's dev receipts in /data/infrarsi/runs/dev-T-train-*), so they need documentation only.

## contracts (O)

1. **`code_digest()`** - move `harness/codesync.code_digest` into contracts so the training process can verify the
   tree it runs from. Definition: sha256 of `canonical_json(sorted [[relpath, file_sha256]])` over every
   `*.py`/`*.json`/`*.yaml` under `infrarsi/infrarsi/`, relpath relative to that directory with `/`, excluding
   `runtime/overlays/` and `__pycache__`. Default overlays travel in the tree; `MANIFEST.json` records
   `overlay_defaults_digest`.
2. **`RECEIPT_REQUIRED`** - add `task_digest, input_plan, nodes, overlay_digest` (INTERFACES section 4 lists them as
   identity; the harness currently treats their absence as a warning).
3. **Lease schema** - add optional `scope: attempt|run` and `nodes`. The dispatcher holds a run-scoped lease
   (`attempt_id = "run:<run_id>"`) across all attempts of a run (probes -> screen -> score back to back on the same
   nodes); `run_screen` accepts it and leaves its release to the dispatcher.
4. **HardwareProfile** - design 8.1 lists it under contracts. `discover/hardware.py` emits
   `infrarsi-hardware-profile-v1`; planners read only `summary`: `gpus_per_node, gpu_type, gpu_mem_gib,
   nvlink_intra, ib_inter, host_mem_gib, pinned_budget_gib, peak_bf16_tflops`.

## conventions to document (T, already true)

5. Signals: 1-based `step` (scoring window `[a, b]` is 1-based); per-rank `inter_step_s` (InterStep = mean over
   steps `a+1..b` of the max-rank value, `receipt_writer` convention); `alloc_retries/alloc_ooms` cumulative
   (`torch.cuda.memory_stats`); `clip_calls` a per-update count (2 with the offloaded hybrid optimizer);
   `signals_digest` = sha256 of the concatenated rank files in increasing rank order; RSD with ddof = 1.
6. Lifecycle: `signals/` runs `1..save_at+resume_steps` uninterrupted (checkpoint at `save_at`), `signals_resume/`
   holds the new process's `save_at+1..`; `receipt.lifecycle.{resumed_ok, resume_scalars_match, export_ok,
   forward_ok}`. The harness re-checks continuity from the two signal sets with the task tolerance.
7. Data: `invariants.data_order_digest` = INDEX.json `data_order_digest[plan]` (`PackedTHDDataset.order_digest`);
   `invariants.data_consumed_digest` = digest of the sorted global pack set of every update. The harness recomputes
   both from the manifest's `plans.<plan>.order` (manifest sha256 checked against INDEX.json) and checks per-update
   supervised tokens against `supervised_tokens_per_update[plan][update-1]`. Round 6: only the INDEX value is
   accepted in `data_order_digest` (no per-update fallback); `data_consumed_digest` and
   `data_consumed_per_update` are mandatory for every non-synthetic, non-lifecycle run.
8. `dtype_census` entries are `{dtype: tensor count}`; the harness compares dtype sets (counts depend on the
   layout) and allows fp32 in `param` only up to `len(model_defined_fp32_params)`.
9. `model_config_digest`, `trainable_param_digest`, `loss_definition_digest`, `lr_schedule_digest`,
   `backends_flags` must stay layout-invariant: the harness requires equality with the FR reference receipt.

## agent / rsi / baselines (A)

10. Register `DISPATCH_HANDLERS = {arm: dispatch.handlers.search_handler(fn)}` (A0, A_R, random/TPE/OptiCo, E3,
    E5 arms). The adapter supplies `start_config` (ED, or FR for `start_from_FR`), `start_receipt`, the numerics
    `reference_receipt` (FR screen, plan A), a `runner` bound to the run's nodes inside the run-scoped lease, and
    `out_dir = execution/searches/<run_id>/` where the monitor sums `trace.json` / `events.jsonl` LLM spend.
11. L0 schema: FR/ED configs also set `use_distributed_optimizer` (design 1.4), `overlap_cpu_optimizer_d2h_h2d`
    and `cross_entropy_fusion_impl`; keys starting with `_` are planner-only (flagship placements in
    `planner.FLAGSHIP_B`) and never reach WINH.
12. Locked scoring (plan B): a candidate's numerics reference is the paired FR plan-B run; when the candidate runs
    first, call `harness.recheck_against(candidate_receipt, fr_plan_b_receipt)` after the pair completes.

## status after round 2 (validator additions)

- Items 1-4 are done in contracts (`tree_digests`/`code_digest`, `RECEIPT_REQUIRED`, lease `scope`/`nodes`,
  `SCHEMA_HARDWARE`/`HARDWARE_SUMMARY_KEYS`); `RunRequest.data_manifest` is pinned by `harness.runner.resolve_data`
  to the INDEX.json manifest before request.json is uploaded (a preset different value or a manifest whose sha256
  differs from INDEX.json refuses the launch).
- Validated receipts now carry `profile` (null for probes) and `wallclock = {train_utc_s, process_train_s}`; check
  details are in `harness.clocks`. For T-train: `utc_start_s`/`utc_end_s` at the `whole_update_s` points;
  `phases.total` from PhaseClock t0 to the receipt write; the first heartbeat should be written near PhaseClock t0
  (the external check uses it as the startup anchor, within max(120 s, 12 %)). When the field lands everywhere, set
  `harness.validate.REQUIRE_UTC_FIELDS = True` so missing UTC fields become invalid instead of a warning.

## open items

13. X1-X3 (WIN0 / unicom8card0, separate /data): no data-sync step yet (P1). Proposal: `harness/datasync.py`
    copies `D-SFT-8K__bc5b6e805ad6__s8192__p1536/` and `INDEX.json` to the same paths and verifies every file
    against the manifest sha256s; WIN0 pulls from WINH0 (WINH cannot open connections to WIN0).
14. E0-SMOKE-S1-SYN: the harness skips INDEX data checks when `task.data.synthetic` is true or the data key starts
    with `D-SYN`; the smoke request should carry such a task copy (its `task_digest` then differs from S1's).

## status after round 6 (Codex R1 driver findings 1-4, 6-8, 10, 12, 13; A items 7-9)

Done in D's area: E0/E7 handlers and explicit P2 deferrals; dispatcher terminal states
(`done | infeasible | invalid | failed`, only `done` and FR/ED `infeasible` satisfy dependencies); fail-closed data
resolution before any lease; one reference loader (`validate.reference_problems`) for validation, recheck and reward;
mandatory task-aware invariant schema; single signals-digest convention; immutable `GLOO_SOCKET_IFNAME=eth0`;
fail-closed clocks with `REQUIRE_UTC_FIELDS = True`; exact E1/E6 task sets per mode; re-hashing of cached code trees;
lifecycle step-count totals checked against the lifecycle's own counts; `eligible=False` rows out of every
aggregate; `reward.evaluate(promotion=True)` binds the lifecycle and plan-B receipts.

For A:
15. `tests/cpu/agent/test_dispatch_arms.py` world fixture: add `DH.fake_index(remote)` after creating the LocalRemote
    (run_screen refuses a non-synthetic task without a verified INDEX.json + manifest). With that one line the whole
    `tests/cpu` passes (verified on a scratch copy). `_lifecycle_reference_rule` is no longer needed.
16. `reward.evaluate(promotion=True)`: config identity is the same `config_digest`, or the same `effective_config`
    and recorded performance env (the start candidate c0 screens the planner's sparse config while its plan-B and
    lifecycle runs carry the resolved one); the lifecycle and plan-B receipts themselves must share a config digest.

For T-train:
17. `invariants.data_consumed_per_update` is mandatory now (non-synthetic, non-lifecycle runs); the real receipt of
    `dev-T-train-S2-repeat-off/a6f05af040ccf` predates it.
18. Lifecycle continuity: `dev-T-train-S2-life-r6/a6f1bfdb45b6f` reports `continuation.match = true` with
    `max_grad_norm_rel = 0.115` (step 6) and 0.027 (step 7). The harness compares the resumed steps with the task
    tolerance (grad norm 0.02 per step), so this lifecycle would be invalid. Step 5 matches (1.6e-5), which suggests
    optimizer state (e.g. the Adam step count used for bias correction) is not restored exactly.
19. `signals_resume/` is not bound by any digest; proposal: `lifecycle.signals_resume_digest` (same concatenation
    convention) so the continuity check reads signed files.
20. F1 has no INDEX.json entry; E7 runs cannot launch until T-data adds the flagship data to INDEX (the trainer's
    `resolve_manifest` needs it as well).

For O (contracts):
21. T-train item 8 (`RunRequest.overlay_digest` compared with the receipt) needs a contracts field; not done in D.
22. External clock tolerance stays max(120 s, 12 %): for a 2-3 minute screen that is loose. Proposal: calibrate on
    E0-SMOKE-S1-SYN / E0-CAL-S1-FR (record `harness.clocks.external.unexplained_s`) and then tighten to about
    max(15 s, 3 x observed).

## AMENDMENT-001 (numerics) in the harness and reward

Done: `compare_numerics` gates CE (per step, window mean), grad norm on update 1 (`grad_norm_rel_max_first_update`) and
the window mean log ratio (`grad_norm_log_bias_window_max`), reports later steps (`numerics.grad_norm.{first_update_rel,
rel_max_all_steps, per_step_rel, log_bias_window, n_window}`); every run fails on adjacent bitwise-repeated
(main_ce, grad_norm) (`self_check.repeated_pairs`, kept through `recheck_against`); `check_lifecycle` follows
`numerics.lifecycle_tolerance` (state_exact, first resumed update 1e-4 / 5e-3, later CE per-step tolerance, LR /
tokens / clip calls exact, forward rel_l2 / top-1 when enabled); every rank's `microbatches` must equal
packs_per_update / (DP x micro_batch_size) on every update; `reward.evaluate` uses the new keys and fails closed when a
non-E0 candidate lacks the amended evidence. Report-only `numerics.sharpness` (clip-active share, adjacent |d ln gn|).

23. For T-train: `microbatches` on every signals row is now mandatory for every run (missing = invalid);
    `lifecycle.state_exact` is required; `runtime/lifecycle.py` still reads `tolerance['grad_norm_rel_max_per_step']`,
    which the amended task specs no longer carry. The harness no longer gates on `lifecycle.resume_scalars_match` (it
    re-derives continuity under the amended rule and reports the trainer's flag).
24. For A: `redteam/detectors.py` still uses `grad_norm_rel_max_per_step` (old 0.02 per-step gate).

## Codex round 2 (trust boundary) in D's area

Done: `harness/signing.py` HMAC-SHA256 over the whole validated receipt minus `harness_signature` (key
`%USERPROFILE%/.infrarsi/receipt_hmac.key`, 32 bytes, owner-only ACL, created on first use; `verify_receipt_digest`
is now this verification, so `reference_problems`, reward, handlers and A's `authenticate` all use it; untrusted
receipts are not read further); signing happens after the lease release and storage path are final; trusted launch
ledger `execution/ledger/launches.jsonl` (`harness/ledger.py`) written before request.json, `observe()` refuses a
mismatching or unknown remote request.json; `check_identity` requires the raw receipt to echo `request_digest`,
`numerics_role`, `data_manifest`, `effective_overlay_digest`, and `overlay_digest` == pinned effective digest;
`lifecycle.signals_resume_digest` required and verified before any continuity check; code cache identity
(code_digest, overlay_defaults_digest) with directory `code/<code_digest>-ov<16 hex>` and no warning-only path;
`run_screen` pins `effective_overlay_digest` (default hook files of the verified tree, replaced by request files; agrees
with `runtime/overlay_loader.load(...).digest`, tested); E1 `InputComplete`; the E7 handler decides promotion from the
signed receipts (verified locked candidate that differs from the reference), not from the search's flag.

25. For T-train: echo `request_digest` = contracts.digest(request.json as written) (= digest(RunRequest.to_dict())),
    `numerics_role`, `data_manifest` and `effective_overlay_digest` at the receipt top level; `overlay_digest` stays
    `Overlays.digest`; `lifecycle.signals_resume_digest` = sha256 of signals_resume/rank*.jsonl concatenated in
    increasing rank order. All are mandatory for validity.
26. For A: test fixtures sign with `harness.signing.sign`; consumers verify with `harness.signing.verify`; the c0
    promotion classification in `locked_pair` and the ED-denominator binding remain A's (R2 FIX-BEFORE-P0 items).
27. For O: INTERFACES.md - code cache directory naming, the HMAC key location/ACL, the launch ledger, the receipt echo
    fields and the signature field name `harness_signature` (replaces `harness.receipt_digest`).
