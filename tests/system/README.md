# System tests (tests/system)

A test layer above `tests/cpu`: user journeys through the CLI, fault injection on a stand-in job, component contracts,
search oracles, POSIX / ssh / GPU legs, and non-functional checks.

```
.venv/Scripts/python -m pytest tests/system -q -p no:cacheprovider        # PC: the CPU set, about 5 minutes
INFRARSI_SYSTEST_SLOW=1 ...                                              # + overhead checks
INFRARSI_SYSTEST_PACKAGING=1 ...                                         # + a fresh venv install (needs PyPI)
(on a Linux node) INFRARSI_SYSTEST_GPU=1 INFRARSI_SYSTEST_GPU_ID=0 PYTHONPATH=. python -m pytest tests/system -q
```

| family | file | what |
|---|---|---|
| S static | `test_s_static.py`, `test_s_packs.py` | imports, CLI help, README flags / strategies / checklist, manifest docstring, seed packs, registries |
| C component | `test_c_manifest.py`, `test_c_signals.py`, `test_c_verifier_matrix.py`, `test_c_apply.py`, `test_c_llm_http.py`, `test_c_report.py` | expression evaluator (fuzz), knob rendering, log scanner (formats, fuzz), verifier table, apply, the OpenAI-compatible path over HTTP (`_support/fake_llm.py`), REPORT.md |
| F faults | `test_f_job_faults.py` | `_support/faultjob/train.py`: 29 failure modes as a candidate, as the reference (`doctor`), inside a search |
| J journeys | `test_j_quickstart.py`, `test_j_strategies_suite.py`, `test_j_init_personas.py`, `test_j_experience_concurrency.py` | README quick start, mistakes, every strategy, suites, init on repository shapes, experience round trip, two runs at once |
| Q oracles | `test_q_search_oracles.py` | budget, contract, the fp8 trap, independent re-measurement, attainable-gain fraction, determinism |
| P posix | `test_p_posix.py` | SIGTERM, timeout, orphan reaping, ssh to localhost, unreachable hosts, CRLF launchers (Linux only) |
| G gpu | `test_g_gpu.py` | `_support/gpujob/train.py`: a real torch MLP with bf16 / compile / fused and two traps |
| N non-functional | `test_n_overhead_packaging.py` | tree copies, chatty logs, long logs, a fresh install without extras |

Conventions: every test gets its own `INFRARSI_HOME`; receipts are signed with a throwaway key. A defect found by a test
stays in the suite as `xfail(strict=True, reason='FINDING F-SYS-n: ...')`: the suite is green, the finding is listed, and
a fix turns the test into an XPASS that asks for the mark to be removed. `_support/consistency.check_run(run_dir)` is the
shared cross-check of REPORT.md, result.json, the receipts and the budget.
