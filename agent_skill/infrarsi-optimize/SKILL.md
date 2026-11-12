---
name: infrarsi-optimize
description: Speed up an existing GPU training job without changing what it trains. Tunes launch settings and patches code, measures every candidate against the original run, and adopts only changes that pass verification. Use when asked to make training faster, raise MFU or throughput, cut step time, fit a larger batch, or tune distributed training (Megatron, FSDP2/TorchTitan, DeepSpeed, HF Trainer/TRL, PyTorch DDP); also 训练加速, 提升 MFU, 训练调优. Not for inference speed, model quality or hyperparameter search.
---

# InfraRSI: a faster training job, verified

The goal is one number: the time one optimizer update takes, for the same job (same data, model, batch, precision rules
and number of optimizer steps). Three parties do the work, and the split is the point of this skill:

- **The InfraRSI engine** (shipped in `scripts/`) fixes the task contract, runs every training run, times it, checks
  it, decides what to try next, re-runs the winner against the original back to back, and writes the report. It also
  reads and writes the experience library (results of earlier jobs and other hardware).
- **You** adapt the project so the engine can run it, agree the contract with the user, keep the run going, and tell the
  user what the report says.
- **A fresh sub-agent** answers each question the engine asks during the search (what a setting should gain, what in the
  code is slow, a patch). It sees only that question.

## Rules

1. **Numbers come from the engine.** Never run the training job yourself to measure it, never compute a speed-up
   yourself, never report a number that is not in `agent status` or the report. A candidate you did not see pass is not a
   result.
2. **The contract is fixed before the search.** Do not edit frozen files, `infrarsi.yaml`, the run directory or a
   candidate's code while a run is active. Loosening anything (tolerance, frozen files, invariants, timed updates,
   precision) needs the user's explicit yes; tightening does not. The engine's own files are never yours to create or
   change: its home directory (`~/.infrarsi`: experience library, ledger) and the key it signs measurements with.
3. **You do not answer the engine's requests.** You have seen measurements and your own earlier guesses; the search
   assumes each answer has not. Hand every request to a sub-agent with no memory of this conversation, and hand its
   answer in unchanged.
4. **The engine schedules.** Do not skip, reorder or repeat its steps. An idea of yours or the user's goes in with
   `agent note`.
5. **Failures are data.** Do not retry a failed candidate by hand. If nothing passes, the result is "keep the current
   configuration" (x1.00), and you say so.

## Commands

`infrarsi` below means `python "<skill-dir>/scripts/infrarsi.py"` (keep the quotes), where `<skill-dir>` is the
directory of this file (Claude Code: `${CLAUDE_SKILL_DIR}`). Use the Python of the training environment (3.11+, or 3.10
with `tomli`; PyYAML). Run the commands from the project directory. An `infrarsi agent` command prints one JSON object
whose `do` field says what comes next; a request it refuses prints one line on stderr instead and exits with status 2.

## Workflow

1. **Understand the job.** Find how the user launches training today (script or command), its config, how many GPUs and
   nodes, and that it runs. If it does not run as it is, fix that with the user first.
2. **Draft the contract**: `infrarsi init <launch script>`, or `infrarsi init <project dir> --command "<launch command>"`
   when the job is started by a bare command, writes `infrarsi.yaml`. Read it against the code and complete it: see
   `references/workflow.md` (knobs, files the search may patch, frozen files, invariants, updates per run, remote hosts).
3. **Agree it with the user.** Show the checklist in `references/workflow.md` (what may change, what must not, budget,
   expected duration). A plain "go" accepts the defaults. Requirements in the user's own words go to `--requirements`.
4. **Start**: `infrarsi agent start . --model <your model name> [--screens N] [--requirements "..."]`. The engine now runs
   in the background on its own: reference run, noise calibration, search, promotion, report.
5. **Loop** on `infrarsi agent next`. It waits up to 9 minutes for something that needs you; give your shell tool a
   timeout above that, or pass `--wait SECONDS` just below your tool's limit. Wait as long as your tool allows: every
   call is a turn of yours, and a training run takes minutes.

   | `state` | What you do |
   |---|---|
   | `request` | Dispatch a fresh sub-agent on `request.dir` (how: `references/answering.md`), then `infrarsi agent answer <id>`. If the answer is refused, dispatch again as its `do` says. |
   | `running` | Nothing is needed. Call `next` again. Every few calls, tell the user in one line where the search is (`progress`). |
   | `finished` | Go to step 6. |
   | `failed` | Read `log_tail`; see `references/troubleshooting.md`. Fix the cause, then start again. |
   | `dead` | The engine process is gone. Do what `do` says: `infrarsi agent resume`, or start again when the run had no checkpoint yet. |

   While the state is `request` or `running` the task is not done: do not end your turn. If the user asks to stop,
   `infrarsi agent stop` ends the search at its next boundary and still verifies the best candidate.
6. **Report**: `infrarsi report <run_dir>`. Tell the user, in plain words: the original and the new time per update (and
   MFU when reported), the speed-up of the back-to-back re-run, what changed, what was rejected and why, what could not
   be checked, and the cost. Quote the report's numbers only.
7. **Apply only when the user agrees**: `infrarsi apply <run_dir>` shows the change; `infrarsi apply <run_dir> --write`
   writes it (add `--branch NAME` to write on a new git branch). Verified results are written to the experience library
   by the engine; do not write there.

## After a context compaction or a new session

Run `infrarsi agent status` in the project directory first. It finds the run and says what to do.

## References

- `references/workflow.md` - the contract in detail, the checklist for the user, options, remote GPUs, running unattended.
- `references/answering.md` - how to dispatch the answering sub-agent on Claude Code and Codex, and its exact prompt.
- `references/contract.md` - what the engine holds fixed and the 16 checks, for explaining a result or a rejection.
- `references/troubleshooting.md` - a reference run that fails, unreachable hosts, a dead engine, no improvement.
