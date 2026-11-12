# What the engine holds fixed, and how it checks

Use this to explain a result or a rejection to the user. You do not run these checks; the engine does, on every run.

## The task contract

Fixed before the search starts, from `infrarsi.yaml` and the reference run:

- **The reference** is the user's current configuration, measured before the search. Every candidate is judged against
  it.
- **The same work per update**: tokens (or samples) per update, sequence length and the global batch (the
  `workload.invariants`) stay as in the reference.
- **Frozen files** (loss, data, evaluation, timing and logging) are hashed; a candidate that changes one is rejected.
- **One optimizer step per update**, with the reference's learning rate at every update.
- **The loss stays within the tolerance** of the reference (default 3 %; the check reads it against how much the job's
  own loss varies between two runs, and the report shows the bound it applied), and precision may change only as the
  project allows.
- **The timed updates** are named in advance (`workload.window`); the engine's own clock must agree with the job's.

## The 16 checks

A run that fails a check is rejected and its time does not count. A check the logs cannot support is reported as
*unchecked*; an unchecked check rejects a run only when the project requires it (`verify.required`; by default
`completes`, `same_workload` and `loss_within_tolerance`).

| Check | When | What it establishes |
|---|---|---|
| `completes` | every run | Every requested update ran and the process exited cleanly |
| `no_errors` | every run | No NaN or Inf loss, out-of-memory, communication error or traceback; loss and gradient norm change between updates |
| `memory_cap` | every run | Peak device memory stays within the device's limit |
| `same_workload` | every run | Tokens per update and sequence length equal the reference at every update |
| `same_data` | every run | Each update consumes the same global batch as the reference |
| `optimizer_steps` | every run | One optimizer step per update, same learning rate as the reference |
| `same_model` | every run | Same trainable parameters and model configuration |
| `frozen_files` | every run | The files declared frozen are unchanged |
| `loss_within_tolerance` | every run | Training loss within the accepted tolerance of the reference |
| `grad_norm_within_tolerance` | every run | Gradient norm of the first update within the tolerance |
| `timing_consistent` | every run | The engine's clock agrees with the job's step time: no work was moved out of the measured loop |
| `precision` | every run | Reports the numeric precision of parameters, gradients and optimizer state (information only) |
| `speed_reproduces` | final check | A fresh back-to-back re-run of reference and candidate keeps the speed-up beyond the noise margin |
| `resume_continues` | final check | Saving a checkpoint and resuming continues the loss within the tolerance |
| `eval_matches` | final check | The project's eval command agrees on the checkpoints of the two runs |
| `user_tests` | final check | The project's own test commands pass on the optimized tree |

## Noise and the final check

- The engine measures how much two runs of one configuration differ on this machine. A difference smaller than that
  is not a gain.
- At the end, the fastest valid candidates are run again, fresh, back to back with the reference. The first one that
  keeps its speed-up beyond the noise margin and passes the final checks is the recommendation. The speed-up to quote is
  the one of this fresh re-run.
- If none passes, the answer is "keep the current configuration" (x1.00). That is a result, not a failure of the run.
- Every run of the search and of the final check counts in the reported cost, failed and rejected candidates included.

## What is never an optimization

Each of these makes a run look faster by doing less or measuring less. They are forbidden whether or not a check
catches them: the engine rejects what its signals can show (the report lists the checks it could not make), and the
rest is on you and on the answerer not to propose.

- fewer tokens per update, a shorter sequence, a smaller global batch, a smaller or partly frozen model;
- skipped, merged or deferred optimizer steps;
- lower precision than the project allows;
- work moved outside the timed updates, or a change to the code that logs or times an update;
- an edit to a frozen file, to `infrarsi.yaml` or to the run directory during a run.

## Explaining a rejection

The report names the failed check and gives a hint for it. Say which check failed, what that check protects, and what
the search did next. Do not argue with a rejection and do not re-run a rejected candidate by hand.
