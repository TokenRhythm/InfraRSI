# Workflow in detail

`infrarsi` means `python "<skill-dir>/scripts/infrarsi.py"`. Run it from the project directory.

## 1. The contract: `infrarsi.yaml`

`infrarsi init` drafts it, in one of two forms:

- `infrarsi init <launch script>` when the job is started by a script (`run.sh`, `train.py`);
- `infrarsi init <project dir> [--command "<launch command>" | --entry FILE] [--config FILE]` otherwise. (A script
  argument cannot be combined with `--command` or `--entry`.)

It detects the framework, finds settings the job reads (config keys, command-line flags, constants, environment
variables) and the GPUs. `--force` replaces an existing draft. Then read the file against the code. This is where you
add the most: the draft only knows what it could trace.

| Key | What it is | What to check |
|---|---|---|
| `launch.command` | How one run starts. Placeholders: `{python}`, `{config}`, `{steps}`, `{args}`, `{gpus}` | It starts the job the user runs today. `{steps}` limits the number of updates. `{args}` is where knob flags go |
| `config.file` | The user's current config: the reference every candidate is compared with | It is the config the user really uses |
| `knobs` | What the search may set. Each has a `type`, `choices` and one target: `path` (config key), `flag`, `override`, `const` (a `NAME = value` in a script; it also needs `file`), `env`. A knob whose current value is not in the config file (`flag`, `override`, `env`) needs `default`: the value the job uses today | Add what the draft missed: precision, compilation, fused optimizer or loss kernels, recomputation / activation checkpointing, micro-batch with gradient accumulation, data workers, communication settings, the parallel layout. Remove anything that changes the job itself: learning rate, sequence length, total batch, model size |
| `constraints` | Expressions over knobs that every candidate must satisfy | Layout rules, e.g. `gpus % (tp * pp) == 0` |
| `workload.invariants` | What must come out the same, e.g. `global_batch: micro_batch_size * grad_accum * gpus` | Every knob that scales the batch appears in one |
| `workload.steps`, `workload.window` | Updates per run, and which of them are timed (the earlier ones are warm-up) | Warm-up covers compilation and cache filling; the timed part is long enough to be steady |
| `code.mutable` | Files the search may patch | Hot-path code only: kernels, the model's forward, the optimizer step, the data path |
| `code.frozen` | Files that must never change (checked by hash) | Loss, data pipeline, evaluation, and the code that logs or times an update |
| `code.readable` | Files shown to the model for context | The training loop |
| `hardware` | `gpus` (per node), `nodes`, `memory_gib`, `gpu_type` | |
| `execution` | `where: local`, `ssh` (`hosts`, `remote_dir`, `python`) or `slurm` | GPUs on another machine: use `ssh`. The engine stays on this machine and runs the job there; `ssh <host>` must work without a password prompt |
| `verify` | `tolerance` (default 0.03), `tests` (commands), `eval` (command on a checkpoint) | The project's own tests and an eval command make the final check stronger |
| `checkpoint` | `save`, `resume` arguments and `save_at` | Lets the engine check that saving and resuming still works |
| `search.budget` | `screens` (default 16 full runs), `wall_hours`, `llm_tokens` | |

Signals. The engine reads the step number, the loss and the time of each update from the job's log. Some checks need
more (a digest of each batch, the optimizer-step counter, the parameter count); the report lists the checks the log could
not support. A few lines in the training loop make them checkable (`infrarsi.project.probe.Probe`). That is a change to
the user's code: offer it, do not do it unasked.

`infrarsi doctor` runs the reference configuration (and, for some projects, once more with a profiler) and prints
whether it is valid, the time per update, the noise, the signals it saw and what the checks can cover. `agent start`
does this as its second stage, so run it by hand only to debug a baseline that fails. It takes at least as long as one
training run: use a background command if your shell tool has a time limit.

## 2. The checklist to show the user

1. The job: launch command, config, GPUs and nodes.
2. Updates per run and which are timed.
3. Knobs the search may change, and code files it may patch.
4. What stays fixed: frozen files, invariants, loss tolerance (default 3 %).
5. Budget: 16 full runs by default. Time estimate: one run is `steps` updates at the current time per update plus
   start-up; the whole search is roughly 25 to 30 such runs (the reference, short probes, re-runs at the end).
6. Requirements in their own words, if any ("do not touch the optimizer").
7. Nothing is written to their code until they say so.

Anything that loosens item 4, or that shortens item 2, needs the user's explicit yes.

## 3. Starting

`infrarsi agent start . --model <your model name> [options]`

- `--model`: your own model's name (for example `claude-opus-5-5`, `gpt-6.1`). The engine sets how much it trusts an
  answer before it has measurements by it, exactly as it does for any model. Unnamed, it assumes a middle value.
- Budget and goals: `--screens N`, `--wall-hours 2h`, `--budget-tokens 2M`, `--target-speedup 1.5`, `--target-mfu 45`.
- Requirements: `--requirements "text"` (repeatable), `--requirements-file FILE`. They can only narrow the search or add
  checks. Before any run the engine turns each one into one of: *enforced* (a narrower knob, a constraint, a frozen
  file, a required check), *already guaranteed*, *advisory* (told to the model, not checked), or a status that stops the
  run (a conflict, a request to widen the search or weaken a check, something unclear). The first request you see is this
  compilation; the result is in the engine's log and in `requirements.json` of the run. Tell the user which of their
  requirements are enforced and which are only advice: do not promise more than that.
- Other: `--tolerance 0.03`, `--experience auto|off|<pack patterns>`, `--no-doctor`, `--strategy` (default `bcomp`).

The engine then does what `infrarsi optimize` does: init, doctor, run, report, and apply as a dry run. In the run stage it
measures the reference again, calibrates the noise, and searches: three starting points (one from the experience
library, one from the model's prior table, one drawn at random), then combinations of what worked, with code patches
on reserved runs.

Requests have a `role`. Typical ones: `requirements` (before any run, if the user gave requirements), `prior` (when the
search starts), `code_review` (the first time the search turns to code), `hypotheses` (a few independent patches per
request), `revise` (after failures). Do not assume an order or a count: answer what comes.

If the session runs in a sandbox (Codex does by default): the engine must be able to write its home directory
(`~/.infrarsi`) and, for GPUs on other machines, to reach them over the network. The two commands that launch an
engine, `infrarsi agent start` and `infrarsi agent resume`, check the first and say so when they cannot; ask the user
to let that command - the one you were running, `start` or `resume` - run outside the sandbox. The engine it launches
then runs outside it too, and the other commands (`next`, `answer`, `status`, `note`, `stop`) only touch the project
directory. A run that was started outside the sandbox is continued with `resume` outside it as well: do not start a
new run because `resume` was refused.

## 4. While it runs

- `infrarsi agent status`: where it stands, any time.
- `infrarsi agent note "text"`: guidance for later proposals. It is advice to the search, not a rule it must obey.
- `infrarsi agent stop`: finish the run under way, verify the best candidate so far, write the report.
  `infrarsi agent stop --now`: end at once.
- The engine does not depend on your session. Its state is on disk: `<project>/.infrarsi/sessions/<id>/` (requests,
  `engine.log`) and `<project>/.infrarsi/runs/<run>/` (`run.log`, `REPORT.md`, `result.json`).

## 5. Afterwards

- `infrarsi report <run_dir>` prints the report. `infrarsi apply <run_dir>` shows the change;
  `infrarsi apply <run_dir> --write` writes it; `infrarsi apply <run_dir> --write --branch NAME` writes it on a new git
  branch (`--branch` without `--write` writes nothing).
- `infrarsi agent resume --screens N` continues the same search with a larger budget (N is the total, runs done included).
- `infrarsi finish <run_dir> --dry-run` re-verifies the best candidates of a run that ended early. Always give
  `--dry-run`: without it `finish` also writes the winner into the user's code.
- Experience: verified results go to `~/.infrarsi/` (or `$INFRARSI_HOME`) and are used by the next job on this machine,
  whichever tool starts it.

## 6. Without a session

When nobody will stay attached for hours, the same engine can ask the local command-line tool itself:
`infrarsi optimize <target> --llm claude-cli --yes --dry-run` (or `codex-cli`). The search is the same; only who answers
differs.
