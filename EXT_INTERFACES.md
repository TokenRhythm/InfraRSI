# InfraRSI `ext/` — interfaces for external evaluation environments (binding for builders X1–X4)

Authority: `reports/external_bench_20261021/design/EXTERNAL_BENCH_DESIGN_FABLE51.md` (science, arms, budgets, guards) and `design/DECISIONS_20261021.md` (D1–D6). Where this file and the design disagree on an interface detail, this file wins; on science, the design wins. `INTERFACES.md` (the InfraRSI core contract) still applies to everything outside `ext/`.

## 1 Ownership

| Path | Owner | Content |
|---|---|---|
| `infrarsi/infrarsi/ext/common/` | X1 | `env_task.py`, `env_runner.py`, `env_judge.py`, `guards.py`, `trajectory.py`, `remote_exec.py`, `basic_agent.py`, `session_hooks.py` |
| `infrarsi/infrarsi/ext/autoresearch/` | X1 | ENV-C adapter (design §4) |
| `infrarsi/infrarsi/ext/rebench_llmf/` | X2 | ENV-A adapter (design §2) |
| `infrarsi/infrarsi/ext/optico/` | X3 | ENV-B adapter (design §3) |
| `agent/search.py`, `agent/optimize.py` seam changes (§3 below) | X1 | backward compatible: with no hooks, behaviour and all existing CPU tests are unchanged |
| F1 pre-fixes (design W7: AMENDMENT-002 `numerics_tier`, dtype census for `A_log`/`dt_bias`/`expert_bias`) and `F-PROMPTS` handler (W6) | X4 | in the existing core modules named by the design; existing tests stay green |
| `tests/cpu/ext/<area>/` | each owner | CPU tests for their modules (pytest; synthetic judge outputs, no GPU, no network) |
| `reports/external_bench_20261021/` | O (orchestrator) | records; builders write only `reports/external_bench_20261021/builders/<X>_REPORT.md` |

Need a change outside your area? Write `infrarsi/infrarsi/ext/PROPOSED_CHANGES_<X>.md` and say so in your report.

## 2 Shared objects (`ext/common`)

```python
@dataclass(frozen=True)
class EnvTask:                      # one environment x workload x arm contract
    id: str                         # e.g. 'ENV-A', 'ENV-B-W1', 'ENV-C'
    env: str                        # 'rebench_llmf' | 'optico' | 'autoresearch'
    split: str = 'external'         # optimize() treats it like a non-dev task (see §3)
    metric_name: str                # 'exec_time_s' | 'iter_time_s' | 'val_bpb'  (always MINIMIZED; see EnvReceipt)
    action_space: str               # 'knobs' | 'patch' | 'knobs+patch'
    knob_schema: dict               # JSON schema of declarative knobs (empty for patch-only)
    frozen_files: dict              # relative path -> sha256 of the verified base tree
    mutable_globs: tuple            # files a patch may touch (relative to the candidate tree)
    max_patch_lines: int
    budget: dict                    # {'screens', 'llm_tokens', 'wall_seconds', 'gpu_hours'} (design tables)
    margin: dict                    # {'kind': 'fixed', 'value': 0.05} | {'kind': 'rsd', 'floor': 0.03, 'k': 3}
    tau: float                      # CTG threshold on y (design §1.4)
    world_size: int                 # GPUs a candidate run occupies
    facts: dict                     # environment facts shown to optimizers (task view): model, data, phases, hardware
    raw: dict                       # the full spec as loaded (yaml/json), for receipts and digests
```

`EnvRunner` (implements the existing `agent.runner.ScreenRunner` protocol, so `SearchSession` uses it unchanged):

```python
class EnvRunner:
    def run(self, task, config: dict, overlay: dict | None, purpose: str, run_id: str, **kw) -> dict  # -> EnvReceipt
```
- `config` = knob dict (validated, §3); `overlay` = `{relative_path: full_text}` for patch actions (validated, §3).
- Materializes the candidate into its own directory on the leased node (copy of the hash-verified base tree + config + files), runs the environment's own scripts and judge, collects outputs, applies guards (§4), deletes bulky outputs, keeps diffs and hashes.
- Must hold a lease (`harness.leases.make_lease(..., purpose='ext:<env>')`, owner string `infrarsi-ext`) on every node it uses and check occupancy (`nvidia-smi --query-compute-apps`) before launching; never touches processes it did not start.

`EnvReceipt` (dict, schema `ext-receipt-v1`), signed with `harness.signing.sign` on this PC after collection. Fields read by the existing search/judge code are mandatory:

| key | meaning |
|---|---|
| `schema`, `env`, `task_id`, `run_id`, `attempt_id`, `purpose`, `candidate_digest`, `frozen_digest`, `node`, `gpus`, `world_size` | identity |
| `valid` (bool), `invalid_reasons` (list[str]) | `valid = judge_valid and not guard_rejections` |
| `step_time: {'mean_window_max_rank_s': <metric>}` | the minimized metric under the name the search code reads; null when invalid |
| `metric: {'name', 'value', 'direction': 'min', 'raw': {...}}` | the same value plus the environment-native value (e.g. TFLOPS for ENV-B, ln-seconds score for ENV-A) |
| `judge: {...}` | the environment judge's own output, verbatim (e.g. `score.py` dict; OptiCo `report.csv` row; `judge_bpb.py` json) |
| `judge_valid` (bool), `guard_rejections` (list of `{id, detail}`), `guard_reports` (list) | judge vs guards kept separate (design §8 risk 14) |
| `phases` (dict of seconds), `utc_start`, `utc_end`, `wall_s` | timing |
| `cost: {'gpu_seconds', 'wall_seconds', 'accounting_complete'}` | read by `SearchSession` |
| `deviations` (list of ids) | e.g. `['D-RB-1','D-RB-2',...]` |
| `harness_signature` | HMAC (`harness.signing`) |

`EnvJudge(receipt, parent, reference) -> agent.judge.Verdict`: verifies the signature (`harness.signing.verify`), then `valid = receipt['valid']`, `step_time = metric value`, `reasons = invalid_reasons`, `numerics_class = 'env'`, `source = 'env:<env>'`. It never turns an invalid judge result into a valid one.

## 3 Seam changes in `agent/` (owner X1)

`optimize(...)` and `SearchSession(...)` gain one optional argument `hooks: ext.common.session_hooks.SessionHooks | None = None`. With `hooks=None` everything behaves exactly as now. A `SessionHooks` provides:
- `task_view() -> dict` (replaces `_task_view` output),
- `validate_config(config: dict) -> (normalized_config, errors: list[str], warnings: list[str])` (replaces `actions.validate_l0` for this task),
- `validate_patch(files: dict) -> (errors, warnings)` (replaces the Megatron overlay `patch_guard`; implements frozen files, mutable globs, line limits, AST bans of the environment),
- `templates() -> list` (L1 templates; may be empty), `layers` (tuple of allowed layers, e.g. `('L0',)` or `('L0','L1')`),
- `receipt_fields` for `receipt_digest`, `tools_enabled: bool` (package analysis tools are Megatron-receipt specific: default False for ext),
- `split_guard(pkg)` replacing the dev/held-out phase guard: ext tasks run the frozen A0 package read-only (no reflect, no memory writes across environments; within-run `write_memory` writes to the per-search working copy only).
New read-only action `read_file(path, start, end)` (≤ 400 lines, only inside `mutable_globs` and documented read-only files) is added to the action schema only when hooks are present.

## 4 Guards (`ext/common/guards.py` + per-environment)

Generic: `frozen_files` (sha256 before and after every run), `outputs_present`, `exit_ok`, `budget`. Per-environment guards are listed in the design (§2.3 G1–G7, §3.4 G-B1..B3, §4.2 checks). Guards only add rejections; every rejection is recorded with id and detail; reports (non-vetoing checks) go to `guard_reports`.

## 5 Native-method arms and the basic agent (`ext/common/basic_agent.py`, owner X1; used by ENV-A `A-NAT` and ENV-C `C-NAT-*`)

- LLM loop with tools `bash(cmd, timeout<=1800)`, `python(code)`, environment-specific `score()` / `submit()`; driver = any `agent.backends` backend (`claude_cli` for Opus 5.5, `openai_compat` for the self-hosted DeepSeek-V4-Flash). Same accounting as InfraRSI (`charge_llm` equivalent, `llm_tokens`, `llm_usd`).
- Remote execution: every tool command runs on the leased node as `nobody` (uid 65534): `setpriv --reuid=65534 --regid=65534 --clear-groups env -i HOME=<run home> PATH=<env bin>:/usr/bin:/bin CUDA_VISIBLE_DEVICES=<gpus> HF_HUB_OFFLINE=1 ... bash -lc '<cmd>'` with cwd = the run's agent home (owned by nobody). Root-owned protected paths (references, judges) are mode 700. Decision D3: no new user is created.
- Transcript, tool calls and outputs are written to `execution/ext_bench/<env>/<run_id>/agent/`.

## 6 Remote access

Use `winh8_20260925/remote.py` (`connect(k)`, `run(k, cmd, timeout)`) and `reports/flash_next_mfu_20261003/put.py`-style SFTP for uploads; WINH node labels `WINH0/1/3/4/6` (WINH2/5/7 down). Never install into `/mnt/base/qwen-sft/.venv-megatron-qwen35`; write only under `/data/infrarsi/`. LLM endpoints for processes that run on a WINH node go through `reports/external_bench_20261021/scripts/rtunnel.py` (reverse SSH) to `baselines/claude_openai_shim.py` or to a local forward of the WIN0 vLLM endpoint.

## 7 Environment facts already built on WINH (do not rebuild)

- ENV-A: env `/data/infrarsi/ext/envs/rebench_llmf`; sources `/data/infrarsi/ext/src/{rebench/RE-Bench/ai_rd_optimize_llm_foundry, llm-foundry-77f9ab18, task-protected-scoring}`; private official solution `/data/infrarsi/ext/private/rebench_llmf_official` (never shown to any optimizer or written into a report); runner/scorer `/data/infrarsi/ext/tools/{run_rebench_r1.sh, rebench_score.py}`; anchors S0 216.1 s, S1 82.0 s.
- ENV-B: workspace recipe in `/data/infrarsi/ext/tools/run_optico_r3.sh` and `run_optico_loop.sh`; data `/data/infrarsi/ext/optico_ws/dataset`; overlay `/data/infrarsi/ext/overlays/optico_tf451`; defaults measured (REPRO_RESULTS.md).
- ENV-C: env `/data/infrarsi/ext/envs/autoresearch`; data `/data/infrarsi/ext/datasets/climbmix`; FA3 `/data/infrarsi/ext/kernels/kernels-community__flash-attn3`; runner `/data/infrarsi/ext/tools/run_autoresearch_r2.sh`; baseline val_bpb 0.997700.
- Local copies of the scripts: `reports/external_bench_20261021/scripts/`.
