"""W5: the CPU-only summary step of the external-benchmark goal (design §7 W5, G-FIN).

Turns the run records (`execution/ext_bench/<env>/<run_id>/`) named by `results/RUN_LEDGER.json` into the final tables:

  ENV_A.md / ENV_B.md / ENV_C.md   per-environment arm tables, advantage labels, margins (design §2.7, §3.6, §4.4 with
                                   AMENDMENT_EXT_001..007 applied)
  F1.md                            the F1 fallback via X6's own summary (`ext.flashnext.cli summary`), not re-implemented
  REPRO_SUMMARY.md                 tables R1-R4 of design §5.4 (R1 from REPRO_ISSUES.md + R1_mapping.json)
  summary.json                     machine-readable; every number is a cell {'v', 'src'} naming its receipt / record
  FREEZE.json                      sha256 of every record, ledger, design file and tool file used (G-FIN)
  g4_calibration.json, margin_<W>.json   the mechanical G-FIN records of AMENDMENT_EXT_005 and AMENDMENT_EXT_002

Read-only on records and run code. Labels come from `ext.common.trajectory.compare` (the pre-declared rule incl.
`incomplete`); the ENV-B margin from `ext.optico.margin`. Abandoned runs (ledger) appear only on the pipeline-fix line.
CLI: `python -m infrarsi.ext.summary.cli --ledger <RUN_LEDGER.json> --out <dir> [--allow-partial]`.
"""
