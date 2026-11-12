# When something goes wrong

Start from what `infrarsi agent status` prints: `state`, `log_tail`, `do`. The engine's whole console is in
`<session>/engine.log`; a run's own progress is in `<run_dir>/run.log`.

| What you see | What it means | What to do |
|---|---|---|
| `failed`, and the log says the reference run is `NOT VALID` | The user's current configuration did not pass as a run: the lines marked `!` say why (the job crashed, too few updates were logged, no step time was found) | Fix the launch or the manifest (`launch.command`, `signals`, `workload.steps`). `infrarsi doctor` repeats just this run. Then `infrarsi agent start` again |
| `failed`, exit status 2, a line starting `infrarsi optimize:` | The request itself was wrong: an option, the manifest, a requirement that cannot be honoured | Read the line; it names the problem |
| The log says the hosts could not be reached | `execution.hosts` cannot be reached over ssh | `ssh <host>` must work from this machine without a password prompt |
| `dead` | The engine process is gone (the machine restarted, or it was killed) | `infrarsi agent resume`. If the status says the run has no checkpoint yet, `infrarsi agent start` again |
| `agent start` or `agent resume` says the engine cannot write its home directory, or cannot use its receipt-signing key | The session runs in a sandbox that keeps the engine out of `~/.infrarsi` | Ask the user to let that same command (`start`, or `resume` when you were resuming) run outside the sandbox, or to allow writing that directory. Do not create the directory's files or the key yourself: the key signs every measurement, and a key you wrote is a key you know |
| Runs on other machines fail to start from inside a sandbox | The sandbox blocks the network, so the engine cannot reach `execution.hosts` | The same: the user lets the command that launches the engine (`start`, later `resume`) run outside the sandbox |
| `agent answer` returns `"ok": false` | The answer file is not one JSON object of the shape asked for | See `answering.md`: a new sub-agent, with `feedback.txt`. After the last attempt the engine goes on without that answer |
| Every request is refused | The answerer writes text around the JSON, or writes to another file | Check that it writes exactly `request.answer`, and only the JSON object |
| The run ends with `no_change` | No candidate both passed the checks and beat the noise | Tell the user so, with what was tried (the report lists it). Options to offer: a larger budget (`infrarsi agent resume --screens 32`), more knobs or patchable files in `infrarsi.yaml` |
| The run ends with `not_verifiable` | The logs cannot show that a candidate does the same work as the reference | Expose the missing signals (see "Signals" in `workflow.md`), with the user's agreement |
| A warning that another run may be active in this project | Two runs on the same GPUs change each other's times | Let the first one finish, or stop it |
| `agent start` says a run of this project is active | A session's engine is still running | `infrarsi agent next` continues it; `infrarsi agent stop` stops it |
| Runs take very long | Each one is a real training run of `workload.steps` updates | Shortening it changes the contract: only before a start, and with the user's yes |
| `infrarsi.py` says there is no usable engine, or that PyYAML or `tomli` is missing | The skill is run with a Python that lacks PyYAML (or, on Python 3.10, `tomli`), or its `scripts/engine` is missing | Use the training environment's Python (3.11 or newer needs only PyYAML); reinstall the skill |

What not to do when stuck: do not edit files in the run or session directories, do not run the training job by hand
to "check" a candidate, and do not delete a session to make an error go away. If a run cannot proceed, say so to the
user with the log lines.
