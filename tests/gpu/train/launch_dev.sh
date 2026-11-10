#!/usr/bin/env bash
# Dev launcher for T-train GPU checks on WINH (the official multi-node launcher belongs to the driver).
# usage: launch_dev.sh <request.json> <nproc_per_node> [master_port]
# 2 nodes: NNODES=2 NODE_RANK=<i> MASTER_ADDR=<node0 eth0 IP> launch_dev.sh ... on each node.
# The caller checks nvidia-smi occupancy and writes /data/infrarsi/leases/WINH<k>.json first.
set -euo pipefail
REQ=$1
NPROC=$2
PORT=${3:-29611}
CODE=${CODE_ROOT:-/data/infrarsi/code/dev_train}
FN=/data/sftv2aliyun/flash-next
VENV=/mnt/base/qwen-sft/.venv-megatron-qwen35
export DEEPEP_ROOT="$FN/code/vendor/DeepEP-fp32-combine"
export PYTHONPATH="$CODE:$FN/code/vendor/compat:$FN/code/runtime:$FN/code/vendor/mcore-bridge/src:$FN/code/vendor/Megatron-Core:$DEEPEP_ROOT:/mnt/base/qwen-sft/runtime/flash-training-ad-20260907/overlay"
export USE_MCORE_GDN=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
OUT=$("$VENV/bin/python" -c 'import json,sys; from infrarsi.contracts import RunRequest; print(RunRequest.from_dict(json.load(open(sys.argv[1]))).out_dir)' "$REQ")
mkdir -p "$OUT/logs"
cd "$OUT"
exec "$VENV/bin/python" -m torch.distributed.run --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
  --nproc_per_node="$NPROC" --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="$PORT" \
  -m infrarsi.adapters.megatron_core.generic_train --request "$REQ"
