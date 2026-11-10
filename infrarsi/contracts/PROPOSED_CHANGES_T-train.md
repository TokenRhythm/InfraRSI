# Proposed changes from T-train (training process builder)

The owner of each affected file decides. T-train only edits `adapters/megatron_core/`, `runtime/`,
`tests/cpu/train/`, `tests/gpu/train/`.

## For O (contracts / INTERFACES.md)

1. **§4b wording.** T-data ships `update_assignment` as the method `PackedTHDDataset.update_assignment(...)` that
   returns pack ids (Megatron pretraining-sampler layout: micro-batch g of DP rank r takes plan slots
   `u*P + (g*dp + r)*MBS + [0, MBS)`), plus module functions `update_slots`/`update_assignment(..., order=)`.
   The trainer uses the method. Please align the text.
2. **Synthetic input selection.** E0-SMOKE-S1-SYN is the S1 task dict with `task.data.synthetic = true` (or a data
   key starting with `D-SYN`). Please record this in the RUN_MATRIX note or add a `RunRequest.synthetic` flag.
3. **`lifecycle` block** (all optional): `save_at` (default `steps`), `resume_steps` (default
   `task.workload.resume_continuation_steps`), `export` (true), `forward_compare` (true, requires export),
   `forward_tolerance` (`{rel_l2: 0.05, top1_agreement: 0.90}`), `keep_artifacts` (false). Lifecycle and score
   requests must use input plan B (enforced in the supervisor preflight). One torchrun launch runs three new
   processes; phase 1 also trains `resume_steps` past the save as the uninterrupted reference, so a lifecycle costs
   `save_at + 2*resume_steps` updates plus save/load/export/compare.
4. **Lifecycle pass rule** (T-train's definition, please review): `resume_scalars_match` = training-state
   fingerprint equal at save and after load (fp64 sums of params, fp32 masters, Adam moments per rank) AND the
   first resumed update agrees (CE rel <= 1e-5, grad norm rel <= 2e-3) AND every resumed update keeps CE within the
   task tolerance with identical LR and token counts. Later-update grad-norm drift is reported, not gated (see 9).
5. **Receipt / signal additions** (extra keys only): signals rows require `utc_start_s`, `utc_end_s` and carry
   `inter_step_s`, `cuda_whole_s`; receipt adds `effective_config`, `created_utc`,
   `step_time.{n_window, rsd_ddof, inter_step_steps}`, `per_step[*].{ranks_reporting, max_rank_inter_step_s}`,
   `invariants.{data_consumed_digest, supervised_tokens_per_update, clock_check, lr_schedule_expected_digest,
   hf_param_count, param_tensors, frozen_params, loss_definition, model_config, data, clip_calls_total,
   dtype_devices_rank0, model_defined_fp32_params, backends_flags_baseline}`, `events.{stopped,
   code_digest_detail}`, `phases.{trainer_total, resume_*}`, `lifecycle.{checkpoint, continuation, export,
   forward}`. `invariants.data_order_digest` equals INDEX.json `data_order_digest[<plan>]`.
   `phases.total` is the whole launch (rank-0 supervisor start, which writes the first heartbeat, to the final
   receipt), i.e. what `harness/validate.external_clock` compares with the node wrapper's wall clock, also for
   lifecycle launches with three phase processes; the train process's own total is `phases.trainer_total`.
6. **Definitions the driver/reward must reproduce:** `rsd_window` uses the sample std (n-1); `inter_step_mean_s`
   averages max-rank `inter_step_s` over steps `[a+1, b]`; `signals_digest = sha256(concat(bytes of
   signals/rank<r>.jsonl, r = 0..world-1))`; the receipt flags `|utc_end_s - utc_start_s - whole_update_s| >
   0.05 s + 1 %` on any rank/step as `clock mismatch`; `steps_completed` is the longest complete prefix 1..k.
7. **Frozen overlay loader.** The loader, `ALLOWED` and the sha256 checks moved to `runtime/overlay_loader.py`
   (inside `code_digest` and both frozen-zone digests). `runtime/overlays/__init__.py` only re-exports `ALLOWED`,
   `HOOK_MODULES` for existing importers. Hardening suggestion for §3.2 / `contracts.tree_digests` /
   `actions/zones.py`: exclude exactly the six hook files instead of the whole `runtime/overlays/` directory.
8. **Overlay digest.** `Overlays.digest` uses the same bytes as `actions.patch_guard.overlay_digest` (sha256 of the
   JSON list `sorted([name, sha256])`); a CPU test asserts equality. A shared helper in `contracts` would be cleaner.
9. **Numerics tolerance (science, for the design owner) - high impact.** Identical configurations do not
   reproduce per-step grad norms (backward kernels are not bitwise deterministic and Adam amplifies the noise, with
   clipping at 1.0 active on most updates). Measured spread (max-min)/min of the per-step grad norm:
   - S2 (Qwen3-4B, DP2, plan B, 10 updates), 3 identical runs per arm: up to **14 %** with DP overlap on (update 9)
     and **18 %** with it off (update 8); 6 earlier identical lifecycle phase-1 runs: up to **56 %** at update 9.
     CE spread stayed <= 1.1e-3 (within `main_ce_rel_max_per_step: 0.005`).
   - S1 (1 GPU) 3 runs: <= 0.4 %. D4 offload 3 runs (5 updates): 2.3 %. D2 DP8 vs DP16: 4.8 % (update 6).
   - S2 lifecycle: resumed vs uninterrupted run from an exactly identical restored state: 11.5 % at the second
     resumed update; the harness's `lifecycle.continuation` then fails (`grad norm rel dev 0.115 > 0.02`).
   With `grad_norm_rel_max_per_step: 0.02` the FR reference does not pass against itself, so E1 per-step grad-norm
   checks will reject a random share of legitimate candidates and lifecycles. Proposal: keep the CE checks, and for
   the grad norm use the first update only (identical state) plus a window-mean tolerance (e.g. 5 %), or report it
   without gating. Data: receipts under `/data/infrarsi/runs/dev-T-train-S2-repeat-{on,off}/*/receipt.json`
   and `/data/infrarsi/runs/dev-T-train-S2-life-*/*/receipt.json`.

10. **Megatron checkpoint defect (FYI for everyone using MCore dist checkpoints, incl. the flagship lifecycle).**
    With the distributed optimizer and `distrib_optim_sharding_type=dp_reshardable`, param-group metadata is
    restored from DP rank 0 only; TE FusedAdam keeps `step` per param group and never sets it for a group that is
    empty on a rank. A group empty on DP rank 0 but populated elsewhere therefore resumes with a reset step (wrong
    Adam bias correction): on S2 (DP2) one counter on rank 1 was lost and later resumed updates drifted 36 % in grad
    norm. The T-train checkpoint now writes `ckpt/<step>/infrarsi_steps/rank<r>.json` and restores it after the
    Megatron load; the lifecycle's exact `state_digest` (params, masters, moments, step counters) verifies it.
11. **In-process tamper resistance (limits).** Clocks, serialization, collectives and the Megatron schedule /
    optimizer / loss functions are bound before overlays load and identity-checked (`gates.SENTINEL_PATHS`) after
    setup, after every update and at the end; signal rows use a builtins-only serializer; the overlay loader hashes
    every file before executing any. Code that starts threads, uses ctypes or patches objects outside the sentinel
    list is not covered in-process; `actions/patch_guard` must keep rejecting stores to imported modules.
    Since round 7 the trainer also pins (identity / value) the model list and module tree to depth 4, DDP wrappers,
    the schedule callbacks on the model config, optimizer leaves / inner optimizers / HybridDeviceOptimizer
    sub-optimizers and their frozen config values, checks the CUDA current device around every hook and update,
    verifies every micro-batch's values after `data_path.to_device` (weighted sums with per-process random weights),
    fixes micro-batch orders before the timed loop (identical within a data-parallel replica), compares every Adam
    step counter with the updates applied, and hands hook arguments that do not lead back to the trainer.
12. **Receipt fields for the fail-closed harness** (coordinator request): `invariants.data_consumed_per_update`
    (list, entry k-1 = `contracts.digest(sorted global pack ids actually consumed in update k)`, gathered from one
    rank per data-parallel replica), `invariants.data_consumed_digest` (= `invariants.data_order_digest(...)` over
    those per-update lists), `invariants.data_order_digest` (= INDEX `data_order_digest[plan]`),
    `invariants.data.{source, manifest_digest}`, `invariants.{router_topk, capacity_factor, num_experts}` (null for
    dense), `signals_digest` never null, `clip_calls` and `utc_start_s`/`utc_end_s` on the signals rows.
13. **AMENDMENT-001 as implemented.** Signals rows carry `microbatches` (forward-backward micro-batches the rank
    executed, counted in frozen code: the forward step plus a gradient hook on its output; = `packs_per_update /
    (dp*mbs)`). `invariants.self_check.repeated_pairs` (adjacent updates with bitwise-equal `(main_ce, grad_norm)`,
    invalid). Lifecycle: `compare_continuation` reads `numerics.lifecycle_tolerance` (first resumed update CE 1e-4 /
    grad norm 5e-3; later CE per-step task tolerance; LR, supervised tokens, clip calls exact; grad norm reported);
    `resume_scalars_match` also needs `state_exact` and no repeated pair on the resumed trajectory (checked from the
    saved update on). Lifecycle block adds `state_digest_saved`, `state_digest_restored` (`contracts.digest` of the
    per-rank fingerprints), `fingerprint_restored`, `signals_resume_digest`, `self_check`, `deterministic`,
    `deterministic_unavailable`, `attention_backend`.
14. **MoE update-1 nondeterminism (science, high impact for G1).** Megatron-Core's MoE unpermute sums the top-k
    expert outputs with `scatter_add` unless `torch.use_deterministic_algorithms(True)`, so even the forward of
    update 1 differs between identical runs. D4 (Qwen3-30B-A3B, EP8, CPU offload), 5 identical plan-A runs: update-1
    CE 0.909315..0.909720 (spread 4.4e-4), update-1 grad norm 7.337..7.438 (spread 1.4 %). D4 lifecycle with an
    exactly restored state (`state_exact`, 1539-1615 Adam counters per rank restored from the sidecar): first resumed
    update CE rel 1.01e-4 > 1e-4, so `E0-LIFECYCLE-MOE8` fails AMENDMENT-001 by chance, and the update-1 grad-norm gate
    (0.01) rejects FR against itself for MoE about as often. Options: (a) run E0.3 lifecycles with
    `lifecycle.deterministic: true` (opt-in, implemented: deterministic kernels in both lifecycle processes; S2 gave
    bit-identical resumed updates); (b) a MoE-specific first-update tolerance (CE 1e-3, grad norm 0.03); (c)
    deterministic MoE unpermute in every run (performance cost, changes FR).
15. **Request identity echo** (coordinator): receipt top-level `request_digest` (`contracts.digest` of request.json as
    loaded), `numerics_role`, `data_manifest`, `effective_overlay_digest`; `overlay_digest` stays the effective set
    computed at load. Effective set = the six hook files `<hook>.py` of the code tree (no `__init__.py`), each
    replaced by the request file of the same name; a mismatch with `request.effective_overlay_digest` fails preflight
    with `overlay_digest_mismatch` (the trainer re-checks after loading).
16. **`overlap_moe_expert_parallel_comm` is rejected in preflight**: it switches Megatron-Core to the combined-1F1B
    schedule-plan path (`forward_step_func(..., return_schedule_plan=True)`), which the generic trainer does not
    support in v0.1. A's L0 schema still accepts it (A item 4).

## For D (harness / driver)

1. **One launch per attempt, lifecycle included.** Each torchrun worker is a CPU-only supervisor that runs every
   phase as a fresh child (`--phase train|resume|hfcompare`) on a private rendezvous port and stops all children
   when any rank fails (also when a child cannot even be launched). Set `wall_seconds_cap` for lifecycle runs to
   cover all phases.
2. **Receipt timing.** Rank 0 writes a provisional invalid `receipt.json` at start and replaces it atomically at
   the end; treat it as final only after the node wrappers report exit.
3. **Environment.** Whitelisted performance variables come only from `request.config.env`, with FR defaults
   `CUDA_DEVICE_MAX_CONNECTIONS=1`, `OMP_NUM_THREADS=1`; inherited whitelisted values are dropped. Excluded keys
   pass through. Multi-node launches set `GLOO_SOCKET_IFNAME` (supervisor gloo group) and `NCCL_SOCKET_IFNAME`.
4. **Missing L0 keys take the schema default (Megatron-Core field default), not the FR value**; the FR program
   must write every §1.4 key explicitly (see `tests/gpu/train/make_request.py`).
5. **Optimizer offload** turns on `use_precision_aware_optimizer` internally (Megatron's hybrid CPU optimizer path);
   masters and moments stay fp32 and are verified by the dtype census.
6. **Preflight failures** (layout, data incl. packed-binary sha256 on rank 0, `code_digest_mismatch`,
   plan/lifecycle contract) produce an invalid receipt without starting a trainer.
7. `harness/validate.check_lifecycle` should require `export_ok` / `forward_ok` only when the request enables
   `export` / `forward_compare` (Codex round 2, finding 7).
8. Add `overlay_digest` to `RunRequest` and compare it with the receipt (Codex round 2, finding 3b).
9. **Heartbeat.** The rank-0 supervisor writes the first `heartbeat.json` (`phase load, step 0`) at launch; train
   and resume processes beat as specified; in lifecycle runs the rank-0 hfcompare process beats `compare` every
   60 s and writes the final `done`.
10. `validate.REQUIRE_UTC_FIELDS` can be `True`: every signals row carries `utc_start_s`/`utc_end_s`.
11. **Retention.** After a valid lifecycle (unless `keep_artifacts`) the supervisor removes `ckpt/`, `hf_export/`
    and `lifecycle/mcore_logits.pt` and writes `lifecycle/retention.json`. Invalid lifecycle runs keep them for
    diagnosis; the driver should clean those (S2: 56 GB checkpoint + 8 GB export; a 30B MoE checkpoint with fp32
    masters and moments is ~0.4 TB).
12. **Lifecycle continuation** follows AMENDMENT-001 (`numerics.lifecycle_tolerance`, O item 13).
13. `invariants.data_consumed_per_update` is the per-update consumed-pack field (O item 12).
14. Run the AMENDMENT-001 adjacent-pair self-check on `signals_resume` too, from the saved update on (Codex round 7).
15. Opt-in deterministic lifecycles (`lifecycle.deterministic: true`) report `backends_flags.deterministic_algorithms:
    true`, flash attention (`lifecycle.attention_backend`) and `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
    `CUBLAS_WORKSPACE_CONFIG` in `env`: do not compare their flags with the FR screen's. Not available for packed-THD
    Gated DeltaNet (qwen3_next); the receipt then says `deterministic_unavailable`.

## For A (actions)

1. `patch_guard` imports `ALLOWED` from `runtime.overlays`; it still works through the re-export, but
   `infrarsi.runtime.overlay_loader` is now the frozen source.
2. **The training process depends on `actions.patch_guard.static_scan(text, name) -> list[findings]`.** The frozen
   loader runs it on the exact overlay bytes before executing them and rejects the overlay when it reports
   findings or cannot be imported. Please keep this signature, keep `patch_guard` importable in the frozen venv
   without driver-only packages, and keep rejecting stores to attributes of imported modules (the in-process
   sentinels cannot see threads, ctypes, or objects outside `runtime.gates.SENTINEL_PATHS`).
3. **static_scan gaps found by Codex (round 4b; each probe returned no findings).** The runtime now catches them
   (O item 11), but the guard should reject them up front:
   - `torch.cuda.set_device` / `set_stream` / device context managers in any hook (timing-sensitive state);
   - replacing optimizer or model holders: `leaf.optimizer = ...`, `chained_optimizers[i] = ...`, `models[i] = ...`,
     module attribute assignment on hook arguments, and schedule callbacks on model configs
     (`finalize_model_grads_func`, `grad_scale_func`, `no_sync_func`, `grad_sync_func`, `param_sync_func`,
     `is_stub_optimizer`);
   - `data_path.to_device` results that are not derived from the same input array by value-preserving operations
     (`empty_like` / `zeros` / constants);
   - `run` escaping as a value (thread target, callback, partial, container element) in `schedule.forward_backward`;
   - rank-dependent `microbatch_order` (the runtime now requires identical orders within a replica).
   - (round 7) reordering or replacing entries of a module's `_modules` (the runtime pins the whole tree, order
     included); deleting optimizer `step` entries; non-bool `is_stub_optimizer`.
4. `overlap_moe_expert_parallel_comm=True` passes `validate_l0` but the trainer rejects it in preflight (O item 16):
   drop it from the L0 action space for v0.1 or mark it unsupported.
