#!/bin/bash
# InfraRSI ext/optico B-NAT launcher (design §3.3): OptiCo-public's own script/run_circle.sh loop for one workload on
# one 8-GPU WINH node, logic and prompts unchanged. Same environment as the verified runner
# reports/external_bench_20261021/scripts/run_optico_loop.sh: fresh copy of the base tree, '/workspace' -> this run's
# workspace and the `sysctl -w` lines not executed (D-OC-12), H100 venv + torchrun/python wrappers (D-OC-5),
# transformers 4.51.3 overlay (D-OC-3), substitute data (D-OC-1), gpt2 cache (D-OC-9); the analyzer's OpenAI client
# reaches the driver's shim through the reverse tunnel on 127.0.0.1:$PORT (D-OC-14, D-OC-4, D-OC-8).
# The lease is held by the driver; the watcher (optico_node.py watch) enforces the trial and wall caps on the node
# (D-OC-11) even if the driver is gone; every process carries INFRARSI_TOKEN so cleanup is exact.
# Usage: bash loop.sh   (settings in spec.sh next to it)
set -uo pipefail
A=$(cd "$(dirname "$0")" && pwd)
source "$A/spec.sh"            # MT MS CYCLES PORT TOKEN SRC E V MAX_ROWS MAX_WALL_S DRIVER
export INFRARSI_TOKEN=$TOKEN
mkdir "$A/started" 2>/dev/null || { echo "already started"; exit 0; }
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
sed -i 's/^sysctl -w/# D-OC-12 not run on shared nodes: sysctl -w/' "$WS/code/OptiCo/script/run_circle.sh"
# D-OC-16: the parameter-file sanitizer goes in front of OptiCo's monitor (workspace copy only)
mv "$WS/code/OptiCo/script/monitor/run_monitor.sh" "$WS/code/OptiCo/script/monitor/run_monitor.orig.sh"
cp "$A/sanitize_monitor.sh" "$WS/code/OptiCo/script/monitor/run_monitor.sh"
export INFRARSI_OPTICO_NODE=$A/optico_node.py INFRARSI_SANITIZER_LOG=$A/sanitizer.jsonl
printf '#!/bin/bash\nexec %s/bin/python -m torch.distributed.run "$@"\n' "$V" > "$A/bin/torchrun" && chmod +x "$A/bin/torchrun"
printf '#!/bin/bash\nexec %s/bin/python "$@"\n' "$V" > "$A/bin/python" && chmod +x "$A/bin/python"
echo "$DRIVER" > "$A/driver.txt"
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
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export KEY=infrarsi-ext BASE_URL=http://127.0.0.1:$PORT/v1
export OUTPUT_DIR=$A/out
OUT=$OUTPUT_DIR/output_${MT}_${MS}_n${WORLD_SIZE}
mkdir -p "$OUT"
{ date -Is; hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader; } > "$A/env.txt" 2>&1
python3 "$A/optico_node.py" copycheck "$SRC" "$WS/code/OptiCo" "$WS" "$A/copy_check_pre.json" loop "$A/sanitize_monitor.sh"
t_mat=$(now)
python3 "$A/optico_node.py" watch "$A" "$OUT" $$ "$MAX_ROWS" "$MAX_WALL_S" "$TOKEN" > "$A/watch.log" 2>&1 &
cd "$WS/code/OptiCo/script"
timeout --kill-after=60 $(( ${MAX_WALL_S%.*} + 900 )) bash run_circle.sh "$MT" "$MS" "$CYCLES" > "$A/run_circle.log" 2>&1
rc=$?
t_loop=$(now)
python3 "$A/optico_node.py" hash "$SRC" "$A/frozen_after.json"
python3 "$A/optico_node.py" copycheck "$SRC" "$WS/code/OptiCo" "$WS" "$A/copy_check_post.json" loop "$A/sanitize_monitor.sh"
printf '{"rc": %d, "t_start": %s, "t_materialized": %s, "t_loop_end": %s, "t_end": %s, "host": "%s"}\n' \
  "$rc" "$t_start" "$t_mat" "$t_loop" "$(now)" "$(hostname)" > "$A/status.json.tmp" && mv "$A/status.json.tmp" "$A/status.json"
