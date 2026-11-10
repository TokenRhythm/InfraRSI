# PROPOSED_CHANGES_A (builder A -> D, T, O)

Builder A does not edit other areas; these are requests. Each item names the consumer in builder A's code.
Status as of 2026-10-20 (after the coordinator's dispatcher hand-over).

## D (harness / dispatch / scoring)

1. **Done by A:** the ScreenRunner over D's request-level runner is `agent/harness_runner.py::HarnessScreenRunner`
   (screen -> plan A with the FR plan-A reference; score -> plan B with the paired FR plan-B reference when FR ran
   first; probe -> A; lifecycle -> B with the task's lifecycle block; `_` keys stripped; harness exceptions incl.
   `OccupiedError` -> invalid record charged world x elapsed, `cost.accounting_complete=false`). Arms are registered
   through `DISPATCH_HANDLERS` of `infrarsi.agent / rsi / redteam / baselines` (lazy module attribute).
2. **Run ids:** every request of a dispatched run uses the run's own `run_id` (the run-scoped lease matches run ids
   exactly). `redteam.run_redteam.gpu_run` was changed accordingly (it used per-seed run ids before).
3. **Locked scoring pair** (consumer `agent/arms.py::locked_pair`): run inside the search handler after the search,
   order by task index parity (even: FR first; odd: candidate first + `harness.recheck_against`). Please confirm
   this parity convention matches `scoring/` (design 1.6 only says "alternates by task index parity").
   `result.json['scoring_inputs']` holds `score_task` inputs (t_FR/t_cand locked, FR RSD, InvalidAccept, events, caps).
   A candidate counts as verified only when `harness.numerics.compared` is true against the paired FR plan-B run
   (a receipt validated without a reference reads as `is_reference` to the reward engine); with a valid FR plan-B run,
   any failure of a promoted candidate is InvalidAccept = 1; with an invalid FR run the candidate stays unverified.
4. **Formal judge:** handlers pass `design/reward.yaml` (its declared preset, human mode) to `RewardEngineJudge`.
5. `profile` block and independent clocks: consumed by `agent/context.py::profile_ranges` and
   `redteam/detectors.timing_findings` (`wallclock.train_utc_s`; a screen/score receipt with no independent clock at
   all is now a `timer_wallclock` finding).

6. **Reference receipts in the reward engine** (consumer: every judge call): `reward.evaluate` treats any receipt whose
   `harness.numerics.is_reference` is true as the reference run and passes its numerics and loss guards. A candidate
   validated without a reference (e.g. a screen launched with no FR receipt) therefore passes numerics. Builder A now
   refuses to start a search without an FR (or, when FR is infeasible, ED) plan-A reference and counts a locked-pair
   candidate as verified only when `numerics.compared` is true; please also make `evaluate` accept `is_reference` only
   for the FR/ED reference runs themselves (e.g. by run/purpose), and fail it for candidates.
7. **Lifecycle runs with a numerics reference (blocks every promotion today; consumer `agent/arms.py::promotion_lifecycle`).**
   Per the coordinator, the promoted candidate's lifecycle runs on plan B with the paired FR plan-B score receipt as
   reference. A lifecycle attempt trains save_at + resume_steps updates, but `harness/validate.py::check_invariants`
   compares `optimizer_steps` (INV_EQUAL) and `supervised_tokens_total` (INV_SAME_PLAN) with that 40-step score
   reference, while `inv.optimizer_steps` must equal the 48 completed updates: no lifecycle receipt with a reference can
   be valid, so every promoted candidate would fail its lifecycle. Proposed: for `purpose == 'lifecycle'` skip those two
   step-count totals in the reference comparison (keep `ref.supervised_tokens_per_update` on the common updates, the
   numerics on updates 1..save_at, and every other invariant). **Done by D** (`STEP_COUNT_KEYS` skipped for lifecycle);
   the test emulation is gone.
8. **Scoring inputs** (consumer `scoring/`): `result.json['scoring_inputs']` now carries `eligible` (false = the search,
   plus its locked pair, exceeded a budget dimension despite worst-case admission: exclude the row from locked scoring
   and AUC/CTG; `events` is None and `t_cand_locked` None), `lifecycle_pass` (false: no gain, counted in
   `invalid_accept`), `t_ed` from the authenticated ED plan-B score receipt (always, whichever run is the numerics
   reference), and `cost.promotion_lifecycle_gpuh` (in the arm's total cost, not in admission or the AUC x-axis).
   Please honour `eligible` in `score_task`/aggregates (currently only the events are dropped).
9. **Promotion binding in `reward.evaluate`** (finding 11, second part): the arms now bind the lifecycle and plan-B
   receipts to the candidate themselves (harness signature, task/code/config/overlay digests, purpose, plan) and check the
   continuation/export/forward metrics against tolerances before calling `evaluate(..., promotion=True)`; adding the
   same identity checks inside `evaluate` would protect any other caller. **Done by D** (`aux_binding_problems`,
   `same_config`).
10. **Codex R2, A side (done; for D's consumers):**
   (a) Receipt trust: every check in the arms goes through `agent/arms.py::receipt_trusted` =
   `infrarsi.harness.signing.verify` (D's HMAC); `verify_receipt_digest` is no longer called by builder A.
   (b) Promotion (`result.json['locked_pair']`): `promoted` = the search accepted its selection (a valid screen
   verdict; finalize's no-valid fallback to c0 accepts nothing) and its effective identity (resolved L0 config +
   effective overlay digest) differs from the reference config without an overlay - c0 included. New keys
   `accepted`, `identity`, `reference_identity`. Not promoted: only the reference half runs, `candidate` is
   `{'skipped': ...}`, no `locked/candidate_plan_b.json`, `t_cand_locked` None, `invalid_accept` 0 (score_task's
   fallback). `lifecycle_promoted` keeps keying on `promoted` + `t_cand_locked`; `scoring_inputs.promoted` mirrors it.
   (c) ED denominator (`scoring_inputs.ed_denominator = {attempt_id, path, role, problems}`): needs references
   `ed_status == 'feasible'`, `ed_config`, `ed_screen` and `ed_score` (the stored paths `_summary` records),
   `ed_numerics_role` consistent with the FR status, the score's numerics compared with the recorded `fr_score`
   attempt (candidate role), and the same code / effective overlay as a no-overlay reference run of the experiment.
   Consequence: if FR's plan-B score is re-recorded after ED ran, ED must be re-run to be a denominator again.

## T (runtime / adapters)

0. **Clock tamper resistance** (G2 depends on it): bind clocks at import in frozen modules, per-update UTC stamps in
   `signals/rank*.jsonl` (in progress per the coordinator). `patch_guard` rejects stores to timing attributes
   whatever the base object (G09/G13), but the runtime should not depend on the static guard.
1. Frozen-zone parity: `actions/zones.py::training_frozen_zone_digest` mirrors `runtime/invariants.frozen_zone_digest`
   (CPU test asserts equality). Keep `FROZEN_DIRS` and the digest rule in sync.
2. **Runtime backstops for the hook contract** (static rule G19 in `patch_guard` since Codex review round 2):
   (a) `schedule.forward_backward` must call `run(**kwargs)` exactly once and return its result: assert the call count
   per update; (b) a replaced `optimizer.step` / `layer.forward` must reach the original once: count `optimizer.step`
   calls of the real optimizer per update; (c) `data_path.to_device` must preserve values/dtypes/shapes: on one update
   per run compare a checksum of the host arrays with the device tensors returned (cheap, catches anything the static
   taint rule misses); (d) `host_sync.before_update/after_update` run outside the timer: record their wall time per
   update in the signals so InterStep consistency can attribute it.
3. `patch_guard` G18 reads `runtime.overlays.ALLOWED`; keep it importable without torch.
4. `use_distributed_optimizer` is now an accepted L0 key whose only legal value is `true` (D's planners state it);
   `config_map.make_plan` already rejects any other value (thanks).

## O (tests / environment)

1. Driver venv `E:/tr/tyc_sftv2aliyun/infrarsi/.venv` (INTERFACES 5) with `--system-site-packages`: `optuna==4.5.0`
   (tpe_L0) and OpenEvolve's run deps `openai==1.109.1, dacite==1.9.2, cloudpickle==3.1.1, tqdm==4.67.1,
   flask==3.1.2, numpy==2.3.3`. OpenEvolve is imported from `third_party_src/openevolve` (commit pinned in
   `third_party_src/FETCH_RECORD.json`).
2. Packages live driver-side under `reports/infra_rsi_system_20261020/execution/packages/` (A0 is built on first use
   from `skills/` + `rsi/a0`, then only loaded and digest-verified; lineage A_k, snapshots A1/A_R/A2_M and FROZEN_*.json
   under `packages/lineage/`). Setting the `PKG:A0` marker after `agent.arms.a0_package()` has run once records it.
3. Weak-model route for E5: `DEEPSEEK_API_KEY` set -> R1 (DeepSeek API); otherwise R2, the WIN0 self-hosted
   DeepSeek-V4-Flash (vLLM 0.26.0, TP4 on 4x H20) at `http://127.0.0.1:18000/v1` through
   `ssh -N -L 18000:127.0.0.1:8000 -p 44685 user@116.169.116.30`, settings read from
   `execution/weak_endpoint/STATUS.json` (verified defaults built in): temperature 0, `response_format json_object`,
   `chat_template_kwargs.thinking=false`, 32K context (requests fitted or refused unsent), no API key, API USD 0,
   `describe()` kind `openai_compat_selfhost` / route R2 / engine 0.26.0. Each call records busy time x 4 as H20
   GPU-seconds (`result.json cost.selfhost_gpu_hours_busy`); the serving window (start to stop) is the orchestrator's
   to account. Weak arms ping `/models` first and fail fast; an endpoint lost mid-search stops it (`endpoint_unreachable`,
   no retries, run marked failed with the trace kept). Wall cap x2 on the self-hosted route.
4. `A0_plus_raw_dev_memory` (P2, E6-RAWMEM-*): A0 + one raw-dev-logs skill built from the R-arm traces after D6.
