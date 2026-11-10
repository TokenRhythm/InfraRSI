# InfraRSI v0.1 — interfaces and build rules (shared by all builders)

Authority: `reports/infra_rsi_system_20261020/design/EXPERIMENT_DESIGN_FABLE51.md` (design), `review/FINAL_PLAN.md` (claims), `design/task_specs/*.yaml`, `design/reward.yaml`, `design/RUN_MATRIX.yaml`. Read the design sections named in your brief before coding. Where this file and the design disagree on an interface detail, this file wins; on science (criteria, budgets, tolerances), the design wins.

## 1 Repository layout and ownership

Local root `E:/tr/tyc_sftv2aliyun/infrarsi/` (Python package `infrarsi/infrarsi/`). Owners: **O** orchestrator, **T** training-process builder, **D** driver builder, **A** agent/RSI builder. Only edit your own directories. Need a change elsewhere? Write `infrarsi/<area>/PROPOSED_CHANGES_<you>.md` and say so in your report.

| Path | Owner | Contents |
|---|---|---|
| `infrarsi/contracts/` | O | `TaskSpec`, `RunRequest`, receipt shape, digests, `RemotePaths` (already written) |
| `infrarsi/adapters/megatron_core/` | T | `generic_train.py` (torchrun entry), model build from HF config via the vendored mcore-bridge, optimizer/DDP setup |
| `infrarsi/adapters/data/` | T | `prepare_sft_packed`, `prepare_pt_stream`, `PackedTHDDataset` (plans A/B), manifests |
| `infrarsi/runtime/` | T | in-process frozen zone: `timers.py`, `signals.py` (per-rank jsonl), `receipt_writer.py` (rank-0 aggregation), `invariants.py`, `gates.py` (dtype census, torch.backends flags audit, first-use exactness), `lifecycle.py`, `heartbeat.py`, `stop.py` |
| `infrarsi/runtime/overlays/` | T | L1 mutable zone: `schedule.py, recompute_policy.py, comm_groups.py, state_placement.py, host_sync.py, data_path.py` as documented no-op hook points |
| `infrarsi/harness/` | D | driver side: code sync, torchrun launch over paramiko (1–4 nodes), `run_screen()`, receipt collection and re-derivation from signals, numerics comparison vs reference, leases and occupancy checks, STOP requests |
| `infrarsi/discover/`, `planner/`, `reward/`, `scoring/`, `dispatch/`, `monitor/`, `cli/` | D | per design §8.1 |
| `infrarsi/actions/`, `agent/`, `rsi/`, `baselines/`, `redteam/`, `skills/` (repo-level) | A | per design §8.1 |
| `tests/cpu/<area>/` | each | CPU tests for your modules (pytest) |
| `tools/` | O | `ms_fetch.sh` (downloads), `design_freeze.py` |

## 2 Remote facts (verified 2026-10-20)

- WINH0–7: 8×H100 80GB each, shared GPFS `/data` (68 TB free), all idle at 05:30. Access from this PC: `python reports/flash_next_mfu_20261003/wh.py <k> "<cmd>"`, upload `put.py <k> local::remote`; library `winh8_20260925/remote.py` (`connect(k)`, `run(k, cmd, timeout)`; ports there). In Git Bash `export MSYS_NO_PATHCONV=1` first. Internal IPs: run `hostname -I` (172.27.x.x); NV18 NVLink inside a node, mlx5 400G IB between nodes.
- Only WINH0 has outbound internet (ModelScope and `mirrors.aliyun.com` PyPI). GitHub is unreachable from WINH; fetch public code on this PC and upload.
- Frozen training venv: `/mnt/base/qwen-sft/.venv-megatron-qwen35` (Python 3.12.11, torch 2.10.0+cu128, TE 2.16, triton 3.6, flash-attn 2.8.3, fla 0.5.1, megatron-core 0.18.2, modelscope 1.38.0). Never install into it. Console scripts inside it have a stale shebang; always call `$VENV/bin/python -m ...`.
- Import paths for the MCore stack: follow `/data/sftv2aliyun/flash-next/scripts/launch.sh` (PYTHONPATH order compat → runtime → bridge src → vendored Megatron-Core → DeepEP → overlay; `USE_MCORE_GDN=1` for qwen3_next). Local mirror for reading: `reports/flash_next_comparison_20261003/flash-next/`.
- Downloads land in `/data/infrarsi/models/<owner>__<name>/` and `/data/infrarsi/datasets/raw/<owner>__<name>/` with `DOWNLOAD_DONE.json` when complete (started 05:34 on WINH0; Qwen3-Next-80B last).
- Another user may start jobs on WINH; a previous execution session's goal is nominally active; a parallel API-audit pilot may use WIN0/WINH. **Before any GPU use, check `nvidia-smi --query-compute-apps` on the target nodes and the lease file; never touch processes you did not start.**

## 3 Run protocol (driver ↔ training process)

1. Driver builds a `RunRequest` (contracts) and uploads it to `runs/<run_id>/<attempt_id>/request.json`, plus overlay files (if any) to `.../overlay/`.
2. Code is content-addressed: driver uploads the package tree to `/data/infrarsi/code/<code_digest>-ov<16 hex of overlay_defaults_digest>/infrarsi/...` once per identity (§7); `code_digest` = sha256 of the canonical list of (relative path, file sha256) for every `*.py`/`*.json`/`*.yaml` under `infrarsi/infrarsi/` excluding `runtime/overlays/` (overlays are digested separately in the request).
3. Driver writes `leases/WINH<k>.json` (`schema infrarsi-lease-v1`, run_id, attempt_id, since, owner `infrarsi`) for every node, then launches on each node (paramiko, `setsid nohup`): `torchrun --nnodes N --node_rank i --nproc_per_node G --master_addr <node0 IP> --master_port <port> -m infrarsi.adapters.megatron_core.generic_train --request <request.json>` with `PYTHONPATH=<code root>:<MCore stack paths>` and the request's whitelisted env. stdout/stderr per node to `logs/node<i>.log`; the node wrapper writes `exit_node<i>.json` (exit code, start/end UTC).
4. Training process writes, inside `out_dir`:
   - `heartbeat.json` (rank 0; every update, and every 60 s during load/compile): `{schema, phase: load|build|train|save|resume|export|compare|done|failed, step, utc, pid}`.
   - `signals/rank<r>.jsonl`, one line per update per rank: `{step, whole_update_s, fwd_bwd_s, optim_s, data_s, peak_alloc_gib, peak_reserved_gib, alloc_retries, alloc_ooms}`; rank 0 adds the global scalars `{main_ce, aux_loss, grad_norm, lr, physical_tokens, supervised_tokens, optimizer_steps, clip_calls}`.
   - `receipt.json` (rank 0, at the end, also on failure with `valid=false`), fields in §4.
   - It checks `STOP.json` at every update boundary and exits cleanly.
5. Driver collects, re-derives the window statistics from `signals/*.jsonl` (never trusts the receipt's own summary), compares numerics with the reference receipt, removes leases, and stores a validated copy in `receipts/<attempt_id>.json`.

## 4 Receipt (`infrarsi-receipt-v1`)

Required keys (checked by `contracts.check_receipt_shape`):
- identity: `schema, run_id, attempt_id, purpose, task_id, task_digest, config_digest, code_digest, overlay_digest, input_plan, world_size, nodes`.
- `steps`, `steps_completed`, `window` ([a, b] or null for probes).
- `step_time`: `{mean_window_max_rank_s, median_window_max_rank_s, rsd_window, inter_step_mean_s}` (null for probes).
- `per_step`: list of `{step, max_rank_whole_s, slowest_rank, main_ce, aux_loss, grad_norm, lr, physical_tokens, supervised_tokens}`.
- `invariants`: `{gbs_packs, physical_tokens_per_update, supervised_tokens_total, optimizer_steps, lr_schedule_digest, trainable_params, trainable_param_digest, data_order_digest, loss_definition_digest, router_topk, capacity_factor, num_experts, frozen_zone_digest, model_config_digest, dtype_census: {param, grad, master, exp_avg, exp_avg_sq}, backends_flags: {...}}`.
- `memory`: per-rank `{peak_alloc_gib, peak_reserved_gib}` plus max.
- `events`: `{alloc_retries, alloc_ooms, nccl_errors, nan_inf_steps, exceptions: [...]}`.
- `phases`: seconds for `startup, load, build, compile_first_step, train, save, export, total`.
- `env`: versions (torch, TE, MCore, bridge commit, CUDA, NCCL, driver), hostnames, GPU names, whitelisted env vars.
- `lifecycle` (lifecycle purpose only): `{saved_at, resumed_ok, resume_scalars_match, export_ok, forward_max_abs_logit_diff, forward_ok}`.
- `valid` (bool), `invalid_reasons` (list), `signals_digest` (sha256 over the signals files in rank order).

## 4b Packed data (owner T-data; consumer T-train)

`infrarsi/adapters/data/` exposes:
- `prepare(data_key, tokenizer_dir, out_root=RemotePaths.datasets_packed, seq_len, packs_total) -> manifest_path` for `data_key` in `D-SFT-8K, D-SFT-32K, D-SFT-64K, D-PT-4K, D-PT-8K` (raw sources and templates: design §1.2 and each task spec's `data` block). Output dir `<data_key>__<tokenizer_digest12>__s<seq_len>/` with `tokens.bin` (int32), `labels.bin` (int32, -100 where unsupervised), `position_ids.bin` (int32), `segments.json` or `cu_seqlens.bin` (per pack), and `manifest.json` (`schema infrarsi-packed-v1`, data_key, source file sha256, tokenizer digest, template id, seq_len, packs_total, dropped_samples, physical/supervised token counts, plan A/B permutations, sha256 of every binary). Re-running `prepare` with the same inputs must reproduce identical bytes.
- `class PackedTHDDataset(manifest_path: str, plan: str)`: `len()` = packs_total; `pack(i) -> dict` with numpy arrays `tokens, labels, position_ids` (shape `[seq_len]`), `cu_seqlens` (int32, segment boundaries incl. 0 and seq_len), and ints `supervised_tokens, physical_tokens`; `order()` = the plan's permutation (A: seed 42, B: seed 1042).
- `def update_assignment(update_index, packs_per_update, dp_rank, dp_size, micro_batch_size) -> list[list[int]]`: the pack ids each micro-batch of this DP rank consumes in update `update_index` (consecutive slices of the plan order; GA = packs_per_update / (dp_size * micro_batch_size)). Every DP rank of every config sees exactly the same global set of packs per update, so loss is comparable across parallel layouts.

As built (2026-10-20, T-data report; this supersedes the bullets above where they differ):
- Dir names carry the pack count: `<data_key>__<tokenizer_digest12>__s<seq_len>__p<packs_total>/` (D2 and H4 share key, tokenizer and seq_len but need 1536 vs 3072 packs).
- `/data/infrarsi/datasets/packed/INDEX.json` maps every task id to its manifest, with `manifest_sha256`, `supervised_tokens_per_update: {A: [...], B: [...]}`, `data_order_digest: {A, B}`, `physical_tokens_per_update` and the tokenizer digest. Resolve a task's data only with `infrarsi.adapters.data.manifest_for_task(task)`, which checks the entry against the task and the manifest sha256. Don't glob when INDEX.json exists. Rebuild with `python -m infrarsi.adapters.data.build_index --specs <task_specs dir>`.
- Labels are already shifted (labels[j] is the target of tokens[j]). Never shift again; loss mask = `labels != -100`. `pack(i)` also returns `padding_mask` (True at padding) and `max_seqlen`.
- `update_index` is 0-based. The module-level `update_assignment(..., order=None)` returns plan positions without `order` and pack ids with it; the dataset method returns pack ids.
- SFT segments are padded to multiples of 8, so CP ∈ {1, 2, 4} for D-SFT-* data (zigzag needs 2·CP | segment length). PT packs allow any CP. Planners and the L0 validator reject CP=8 on SFT tasks, and the training process writes `valid=false` instead of crashing.
- Padding positions (3–6 % of SFT positions) pass through MoE routers. This is part of the frozen loss definition and is the same for every config.
- X1–X3 hosts (WIN0, unicom8card0) have a separate `/data`. Copy the D-SFT-8K p1536 dir and INDEX.json there and verify them against the manifest sha256s before those runs.

## 5 Coding rules

- Python 3.12 on WINH, Python 3.12 on this PC. Driver-side code must not import torch. Training-side code must not import paramiko.
- No new third-party dependency in the training process beyond the frozen venv. Driver side may use `pyyaml`, `paramiko`, `optuna` (install into a project venv `E:/tr/tyc_sftv2aliyun/infrarsi/.venv` if needed).
- Every module ships CPU tests in `tests/cpu/<area>/`; run them before reporting. GPU checks run on the node assigned to you in your brief only, after the occupancy check, with a lease file; keep them short (≤ 30 min of GPU time per check).
- Never edit files in `reports/`, `deliverables/`, `design/`, or another builder's directory. Never modify `/data/sftv2aliyun/`, the frozen venv, `/mnt/base`, or anything under `/data` outside `/data/infrarsi/`.
- Secrets: the ModelScope token is passed only as `MODELSCOPE_API_TOKEN` env when needed; never write tokens into the repo.
- Write concise code with the comment density of `reports/flash_next_comparison_20261003/flash-next/code/runtime/*.py`.

## 6 Conventions as built (2026-10-20, from the builders' PROPOSED_CHANGES files; binding from now on)

- **Code digest.** `contracts.code_digest(package_dir)` / `contracts.tree_digests(files)` define it, and `harness/codesync` uses them. The training process recomputes it at startup and writes `valid=false` (`code_digest_mismatch`) when it differs from `request.code_digest`; the check is skipped if that field is empty.
- **Receipt identity.** `RECEIPT_REQUIRED` includes `task_digest, overlay_digest, input_plan, nodes`. A receipt is final only after the node wrappers report exit; rank 0 writes a provisional invalid receipt at start.
- **Leases.** Leases may carry `scope: attempt|run` and `nodes`. The dispatcher holds one run-scoped lease (`attempt_id = "run:<run_id>"`) across all attempts of a run; `run_screen` accepts it and leaves the release to the dispatcher.
- **Hardware profile.** `discover/hardware.py` emits `SCHEMA_HARDWARE`. Planners read only `summary` with `HARDWARE_SUMMARY_KEYS`.
- **Numerics role.** `RunRequest.numerics_role` is `candidate` (default) or `reference`. A candidate without a reference receipt fails the numerics gates; only the FR runs (or ED when FR is infeasible) are declared `reference`.
- **Overlay identity.** `RunRequest.effective_overlay_digest` (set by the driver before upload) = `contracts.overlay_digest` over the full effective overlay set: the default hook files of the code tree merged with the request replacements. The runtime reports the same digest; a mismatch is invalid. `request_overlay_digest` covers only the request-supplied files.
- **Data pinning.** `RunRequest.data_manifest` is the manifest the driver validated against INDEX.json. The trainer uses it when set (re-checking its sha256 against INDEX.json); otherwise it calls `manifest_for_task`. Synthetic inputs (E0-SMOKE-S1-SYN) are a task copy with `data.synthetic: true` or a data key starting with `D-SYN`; INDEX checks are skipped and the `task_digest` differs from S1's.
- **Signals** (one row per update per rank):
  - `step` is 1-based, and the scoring window `[a, b]` is 1-based.
  - `inter_step_s` is aligned start(k) − start(k−1) on that rank.
  - `alloc_retries` / `alloc_ooms` are cumulative.
  - `clip_calls` is per update (2 with the offloaded hybrid optimizer).
  - `utc_start_s` / `utc_end_s` are POSIX seconds from import-bound clocks.
  - `signals_digest` = sha256 of the concatenated rank files in increasing rank order.
  - RSD uses ddof = 1. InterStep = mean over steps a+1..b of the max-rank `inter_step_s`.
- **Clocks.** Frozen-zone modules bind `perf_counter` and `time` at import. The harness rejects a receipt as `clock_mismatch` or `external_clock_mismatch` when either holds:
  - per-step UTC and perf_counter durations disagree;
  - the UTC train span or the node wrapper's external wall clock disagrees with `phases`.
- **Profile.** Validated receipts carry `profile.{whole_update_s, fwd_bwd_s, optim_s, data_s}.{window_mean_max_rank_s, window_mean_min_rank_s}`, re-derived from the signals (null for probes).
- **Data invariants.**
  - `invariants.data_order_digest` = INDEX.json `data_order_digest[plan]`.
  - `invariants.data_consumed_digest` = digest of the sorted global pack set of every update.
  - Per-update supervised tokens must equal `supervised_tokens_per_update[plan][step−1]`.
- **Layout-invariant digests.** `model_config_digest, trainable_param_digest, loss_definition_digest, lr_schedule_digest` and `backends_flags` must equal those of the FR reference receipt. `dtype_census` entries are `{dtype: tensor count}`; the harness compares dtype sets, and fp32 in `param` is allowed only for `model_defined_fp32_params`.
- **Lifecycle block** (all optional):
  - `save_at` (default `steps`), `resume_steps` (default `task.workload.resume_continuation_steps`).
  - `export` and `forward_compare` (both default true).
  - `forward_tolerance` (default `{rel_l2: 0.05, top1_agreement: 0.90}`), `keep_artifacts` (default false).
  - All phases run inside one torchrun launch (a CPU supervisor per worker starts fresh child processes). `signals/` covers 1..save_at+resume_steps uninterrupted; `signals_resume/` holds the resumed process's steps. Cost = `save_at + 2·resume_steps` updates plus save/load/export/compare.
- **L0 configs.** Missing L0 keys take the Megatron-Core default, so FR/ED programs write every design §1.4 key explicitly, including `use_distributed_optimizer`, `overlap_cpu_optimizer_d2h_h2d` and `cross_entropy_fusion_impl`. Keys starting with `_` are planner-only and never reach WINH. Whitelisted performance env vars come only from `request.config.env`; multi-node launches also set `GLOO_SOCKET_IFNAME`.
- **Agent arms.** Agent arms register `DISPATCH_HANDLERS = {arm: dispatch.handlers.search_handler(fn, start)}` in `infrarsi.agent|rsi|redteam|baselines`. Search output goes to `execution/searches/<run_id>/`, where the monitor sums `llm_usd`: WARN at $1,200, CRIT at $1,500.

## 7 Trust boundary (2026-10-20, Codex R2; binding)

LLM-written overlays run as root on WINH and can write anywhere on `/data`. Every consumer of results (reward, dispatch, arms, scoring) runs on the orchestrator PC. Anything read back from `/data` is therefore untrusted until verified locally.

- **Signature.**
  - Every validated receipt carries `harness_signature`, which replaces `harness.receipt_digest`. It is HMAC-SHA256 over the canonical JSON of the whole receipt minus that field.
  - The key is `%USERPROFILE%/.infrarsi/receipt_hmac.key`: 32 bytes, owner-only ACL, never uploaded or logged. Receipts carry only a key id.
  - Sign and verify with `harness.signing.sign` / `harness.signing.verify` (`harness.validate.verify_receipt_digest` delegates to it). A receipt that fails verification is not read further.
  - CPU tests use a temporary key.
- **Launch ledger.**
  - `execution/ledger/launches.jsonl` (`harness/ledger.py`) records `{attempt_id, request_digest, utc}` before `request.json` is uploaded.
  - `observe()` refuses a remote request that is unknown or whose digest differs.
  - The raw receipt echoes `request_digest` (= `digest(RunRequest.to_dict())`), `numerics_role`, `data_manifest` and `effective_overlay_digest` at the top level. `check_identity` compares them with the trusted request.
- **Code cache.**
  - Identity is the pair (code_digest, overlay_defaults_digest), stored in directory `code/<code_digest>-ov<16 hex>`.
  - Every cached file is re-hashed before reuse; any difference is refused.
  - `run_screen` pins `effective_overlay_digest`: default hook files of the verified tree, replaced by the request files, digested by `contracts.overlay_digest`. The receipt's `overlay_digest` (runtime `Overlays.digest`) must equal it.
- **Lifecycle.**
  - `lifecycle.signals_resume_digest` (same rank-ordered concatenation over `signals_resume/rank*.jsonl`) is required and verified before continuity is derived.
  - `lifecycle.state_exact` must be true.
  - The trainer's own `resume_scalars_match` is report-only; the harness re-derives continuity under AMENDMENT-001.
- **Micro-batches.** Every signals row carries `microbatches`, and it must equal `packs_per_update / (DP × MBS)` on every rank and update. Missing means invalid.
