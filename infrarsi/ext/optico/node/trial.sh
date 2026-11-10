#!/bin/bash
# InfraRSI ext/optico one-trial runner (ENV-B, design §3.4): OptiCo-public's own script/monitor/run_monitor.sh for
# one parameter file on one 8-GPU WINH node, in this attempt's own directory. Environment as the verified R3 runner
# (reports/external_bench_20261021/scripts/run_optico_r3.sh): fresh copy of the base tree, '/workspace' -> the
# attempt's workspace (D-OC-12), H100 venv + torchrun wrapper (D-OC-5), transformers 4.51.3 overlay (D-OC-3),
# substitute MMAP data (D-OC-1), offline gpt2 tokenizer cache (D-OC-9, gpt3 only).
# Usage: bash trial.sh   (settings in spec.sh next to it; results: out/, status.json, frozen_{before,after}.json)
set -uo pipefail
A=$(cd "$(dirname "$0")" && pwd)
source "$A/spec.sh"            # MT MS TOKEN CAP_S SRC E V ITER_LABEL WATCHDOG_S [DEADLINE_S]
DEADLINE_S=${DEADLINE_S:-}     # bounded admission: the monitor must end DEADLINE_S after this script started
export INFRARSI_TOKEN=$TOKEN
mkdir "$A/started" 2>/dev/null || { echo "already started"; exit 0; }
# node-side backstop: whatever still carries the token WATCHDOG_S after the start is killed, even without the driver
( exec python3 "$A/optico_node.py" killafter "$WATCHDOG_S" "$TOKEN" ) > "$A/watchdog.log" 2>&1 &
now() { date +%s.%N; }
t_start=$(now)
WS=$A/workspace
python3 "$A/optico_node.py" hash "$SRC" "$A/frozen_before.json"
mkdir -p "$WS/code" "$WS/model" "$A/bin"
cp -a "$SRC" "$WS/code/OptiCo"
ln -s "$E/optico_ws/dataset" "$WS/dataset"
for m in Qwen__Qwen2.5-0.5B:Qwen2.5-0.5B LLM-Research__Meta-Llama-3-8B:Meta-Llama-3-8B modelscope__Llama-2-7b-ms:llama2-7b-hf AI-ModelScope__gpt2:gpt-3; do
  ln -s "/data/infrarsi/models/${m%%:*}" "$WS/model/${m##*:}"
done
grep -rl '/workspace' "$WS/code/OptiCo/script" | xargs sed -i "s#/workspace#$WS#g"
# D-OC-16: the parameter-file sanitizer goes in front of OptiCo's monitor (workspace copy only)
mv "$WS/code/OptiCo/script/monitor/run_monitor.sh" "$WS/code/OptiCo/script/monitor/run_monitor.orig.sh"
cp "$A/sanitize_monitor.sh" "$WS/code/OptiCo/script/monitor/run_monitor.sh"
export INFRARSI_OPTICO_NODE=$A/optico_node.py INFRARSI_SANITIZER_LOG=$A/sanitizer.jsonl
printf '#!/bin/bash\nexec %s/bin/python -m torch.distributed.run "$@"\n' "$V" > "$A/bin/torchrun" && chmod +x "$A/bin/torchrun"
export PATH=$A/bin:$V/bin:/usr/local/cuda-12.8/bin:$PATH
export PYTHONPATH=$E/overlays/optico_tf451
export HF_HOME=$A/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
GREV=607a30d783dfa663caf39e06633721c8d4cfcd7e; G=$HF_HOME/hub/models--gpt2
mkdir -p "$G/refs" "$G/snapshots/$GREV" && printf '%s' $GREV > "$G/refs/main"
for f in config.json generation_config.json merges.txt tokenizer.json vocab.json; do
  ln -sfn "/data/infrarsi/models/AI-ModelScope__gpt2/$f" "$G/snapshots/$GREV/$f"
done
ln -sfn /data/infrarsi/models/hf_extra/gpt2/tokenizer_config.json "$G/snapshots/$GREV/tokenizer_config.json"
export WORLD_SIZE=1 RANK=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_TYPE=$MT MODEL_SIZE=$MS ITER_CUR=$ITER_LABEL
export OUTPUT_PATH=$A/out
export REPORT_FILE=$OUTPUT_PATH/report.csv PARA_FILE=$OUTPUT_PATH/parameters.txt FIXED_PARA_FILE=$OUTPUT_PATH/fixed_parameters.txt
export MONITOR_RESULT_FILE=$OUTPUT_PATH/monitor_result.txt ERROR_LOG_FILE=$OUTPUT_PATH/monitor_error.txt RESULT_FILE=$OUTPUT_PATH/monitor_result.txt
mkdir -p "$OUTPUT_PATH" && cp "$A/parameters.txt" "$PARA_FILE"
{ date -Is; hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader; } > "$A/env.txt" 2>&1
t_mat=$(now)
cd "$WS/code/OptiCo/script"
MON_CAP=$CAP_S
if [ -n "$DEADLINE_S" ]; then
  left=$(( DEADLINE_S - ($(date +%s) - ${t_start%.*}) ))
  [ "$left" -lt "$MON_CAP" ] && MON_CAP=$left
fi
if [ "$MON_CAP" -ge 1 ]; then
  timeout --kill-after=30 "$MON_CAP" bash monitor/run_monitor.sh > "$A/monitor_stdout.log" 2>&1
  rc=$?
else
  echo "InfraRSI: no wall left for the monitor (DEADLINE_S=$DEADLINE_S)" > "$A/monitor_stdout.log"
  rc=124
fi
t_mon=$(now)
timed_out=false; [ "$rc" = 124 ] || [ "$rc" = 137 ] && timed_out=true
python3 "$A/optico_node.py" hash "$SRC" "$A/frozen_after.json"
python3 "$A/optico_node.py" copycheck "$SRC" "$WS/code/OptiCo" "$WS" "$A/copy_check.json" trial "$A/sanitize_monitor.sh"
printf '{"rc": %d, "timed_out": %s, "monitor_cap_s": %d, "t_start": %s, "t_materialized": %s, "t_monitor_end": %s, "t_end": %s, "host": "%s"}\n' \
  "$rc" "$timed_out" "$MON_CAP" "$t_start" "$t_mat" "$t_mon" "$(now)" "$(hostname)" > "$A/status.json.tmp" && mv "$A/status.json.tmp" "$A/status.json"
