# PROPOSED_CHANGES_X5 — `harness/remote.py` exec read loop (2026-10-21; approved and applied)

Status: **approved by O (2026-10-21, after the F1 incident of 19:48) and applied by X5.** One behaviour change only, in
`WinhRemote._exec`; nothing else in `harness/` was edited.

| # | File | Change | Why |
|---|---|---|---|
| 1 | `infrarsi/harness/remote.py`, `WinhRemote._exec` (read loop, the `break` condition, was line 118) | The loop used to leave as soon as `exit_status_ready()` was true and both buffers were empty. It now leaves only when, in addition, the channel has hit EOF (`chan.eof_received`, or `chan.closed` for a close without EOF), so both streams are read to their end. | sshd sends the exit status when the child exits, while the end of a large stdout can still be queued behind paramiko's 2 MiB channel window (`DEFAULT_WINDOW_SIZE` 2,097,152 in paramiko 5.0.0). An exec reply longer than the window could therefore lose its last MBs at a random point: F1 `f1_prompts_20261021b` c1 (`runner_error: JSONDecodeError: Unterminated string ... char 41868002`), a 47 MB `read_files` reply of 8 rank signal logs cut at 41.87 MB, while F-HT (same sizes) had collected fine. |

## Unchanged

- **SSH semantics.** SSH sends one EOF for stdout and stderr together. paramiko processes it after all earlier data
  (`Channel._handle_eof` sets `eof_received` and closes both buffers), so every byte the server sent is buffered when
  the loop ends.
- **Channel death.** `exit_status_ready()` is `closed or status_event.is_set()`. A channel that dies still ends the
  loop as before, with exit code -1.
- **Timeout.** The deadline is unchanged: a server that never sends EOF now ends in the same `TimeoutError` as any
  other overrun.

## Tests

`tests/cpu/driver/test_remote_exec_eof.py` (5) runs on a stand-in channel whose exit status is ready from the start,
while ~3 MB of stdout (and stderr) arrive one chunk per poll, followed by EOF:
1. The pre-fix loop keeps 65,536 of 3,145,737 bytes.
2. The fixed `_exec` returns every byte.
3. Both streams are read to their end.
4. A close without EOF still ends the read.
5. A small reply that arrived before the exit status is unchanged.

## Related (X5's own files, not harness)

`ext/flashnext/runner.py` no longer moves screen outputs as large exec replies. It compresses them on the node (lzma)
and returns replies of at most 1 MiB, each file checked against its sha256 computed on the node.
`tests/cpu/ext/flashnext/test_flashnext_collect.py` covers it. Details: `reports/external_bench_20261021/builders/X5_F1_BUILD.md` §11.

## Test results

| Suite (2026-10-21 ~20:45) | Result |
|---|---|
| `tests/cpu/driver` (the harness tests) + `tests/cpu/ext` | 1,376 passed, 1 skipped, 5 failed |
| `test_remote_exec_eof.py` | 5 passed |
| the same scenario against the pre-fix `_exec` (loaded in memory, file untouched) | returns 65,536 of 3,145,737 bytes (tail lost) |

The 5 failures are all in `tests/cpu/ext/summary/` with one cause: `R1: issue #37 has no mapping`. The new
REPRO_ISSUES #37 (this incident) has no mapping yet in the summary module, which is the summary owner's item. They are
unrelated to this change: no summary code imports the exec loop.

## Codex review (GPT-5.6 Sol, xhigh, read-only) of this change and X5's collector

- **No defect in this change.** The loop waits for the exit status plus EOF/close and drains both streams, and the
  timeout is preserved.
- **MEDIUM, pre-existing and not applied: dead transport.** A dead transport closes its channels, so
  `exit_status_ready()` becomes true and `recv_exit_status()` returns -1. `_exec` then returns normally instead of
  raising into `_retrying`, so an idempotent call is not retried and its caller sees a "no result" error. This
  behaviour is unchanged by this fix and outside the approved scope ("no other behaviour change"). Proposed follow-up
  for O: raise `EOFError` in `_exec` on exit code -1 so `_retrying` handles it, with dead-transport tests.
- **X5's collector findings (HIGH stale copy, MEDIUM retry counting)** are fixed in `ext/flashnext/runner.py` with
  tests (X5_F1_BUILD.md §11).
