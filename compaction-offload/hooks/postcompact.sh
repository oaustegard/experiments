#!/usr/bin/env bash
set -u
in=$(cat)
dir="${OFFLOAD_DIR:-${CLAUDE_PROJECT_DIR:-.}/.compaction-offload}"
mkdir -p "$dir"
jq -r '.compact_summary' <<<"$in" > "$dir/native_summary.md"
echo "$(date -Is) postcompact native_summary_bytes=$(wc -c < "$dir/native_summary.md")" >> "$dir/log.txt"
exit 0
