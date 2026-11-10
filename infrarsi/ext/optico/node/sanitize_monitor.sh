#!/bin/bash
# InfraRSI D-OC-16 (declared): parameter-file sanitizer in front of OptiCo-public's script/monitor/run_monitor.sh.
# Installed in the per-run workspace copy only, as monitor/run_monitor.sh; OptiCo's own script runs unchanged as
# monitor/run_monitor.orig.sh, so run_circle.sh and trial.sh call it exactly as before.
# Why: run_monitor.sh exports every 'key: value' line of the parameter file and run_<model>.sh evals a command built
# from those values, all as root; in B-NAT the values come from LLM text. A line whose key (after OptiCo's own xargs
# trimming) is outside ^[A-Z][A-Z0-9_]*$ or is a shell/loader/path/OptiCo-internal variable, or whose value is outside
# ^[A-Za-z0-9_.,+-]*$, is not launched: optico_node.py records the trial as a failed run where run_monitor.sh would
# (report.csv row, the error log loc_error reads, result 'fail') and the loop continues as after a failed run.
set +e
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -z "${OUTPUT_PATH:-}" ] || [ -z "${INFRARSI_OPTICO_NODE:-}" ]; then
  echo "InfraRSI sanitizer: OUTPUT_PATH or INFRARSI_OPTICO_NODE unset; refusing to run" >&2
  exit 1
fi
PF=${PARA_FILE:-"$OUTPUT_PATH/parameters.txt"}
if [ -f "$PF" ]; then
  python3 "$INFRARSI_OPTICO_NODE" sanitize "$PF" "$OUTPUT_PATH" "${ITER_CUR:-MONITOR_TEST}" \
    "${REPORT_FILE:-$OUTPUT_PATH/report.csv}" "${ERROR_LOG_FILE:-$OUTPUT_PATH/monitor_error.txt}" \
    "${RESULT_FILE:-$OUTPUT_PATH/monitor_result.txt}" "${INFRARSI_SANITIZER_LOG:-$OUTPUT_PATH/sanitizer.jsonl}"
  rc=$?
  if [ "$rc" != 0 ]; then
    if [ "$rc" != 3 ]; then                    # the checker could not run: fail closed, still a failed run for OptiCo
      mkdir -p "$OUTPUT_PATH/${ITER_CUR:-MONITOR_TEST}"
      echo "InfraRSI sanitizer could not check the parameter file (rc=$rc); the trial was not launched" \
        | tee "$OUTPUT_PATH/${ITER_CUR:-MONITOR_TEST}/monitor_error.txt" > "${ERROR_LOG_FILE:-$OUTPUT_PATH/monitor_error.txt}"
      echo fail > "${RESULT_FILE:-$OUTPUT_PATH/monitor_result.txt}"
    fi
    echo "run InfraRSI sanitizer: parameter file rejected, trial not launched (D-OC-16)"
    exit 0
  fi
fi
exec bash "$HERE/run_monitor.orig.sh" "$@"
