#!/usr/bin/env bash
# Fetch one ModelScope repo (or selected files) into /data/infrarsi with retries and a completion marker.
# Usage: ms_fetch.sh <model|dataset> <owner/name> <local_dir> [modelscope download options or file paths...]
set -u
kind="$1"; repo="$2"; dest="$3"; shift 3
py=/mnt/base/qwen-sft/.venv-megatron-qwen35/bin/python
log_dir=/data/infrarsi/logs/download
mkdir -p "$dest" "$log_dir"
log="$log_dir/$(echo "$repo" | tr '/' '_').log"
marker="$dest/DOWNLOAD_DONE.json"
if [ -f "$marker" ]; then
  echo "$(date -Is) skip $repo: marker exists" >> "$log"
  exit 0
fi
for attempt in 1 2 3 4 5; do
  echo "$(date -Is) attempt $attempt: $kind $repo -> $dest $*" >> "$log"
  # File paths must follow repo_id directly (argparse positional), so caller args come before the options.
  if "$py" -m modelscope.cli.cli download --repo-type "$kind" "$repo" "$@" --local-dir "$dest" --max-workers 8 >> "$log" 2>&1; then
    files=$(cd "$dest" && find . -type f ! -path './._____temp/*' ! -name 'DOWNLOAD_DONE.json' -printf '%P\t%s\n' | sort)
    count=$(printf '%s\n' "$files" | grep -c .)
    bytes=$(printf '%s\n' "$files" | awk -F'\t' '{s+=$2} END {print s+0}')
    printf '{"repo": "%s", "kind": "%s", "finished": "%s", "files": %s, "bytes": %s, "host": "%s"}\n' \
      "$repo" "$kind" "$(date -Is)" "$count" "$bytes" "$(hostname)" > "$marker"
    echo "$(date -Is) done $repo files=$count bytes=$bytes" >> "$log"
    exit 0
  fi
  echo "$(date -Is) attempt $attempt failed" >> "$log"
  sleep $((attempt * 30))
done
echo "$(date -Is) FAILED $repo after 5 attempts" >> "$log"
exit 1
