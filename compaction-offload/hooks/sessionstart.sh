#!/usr/bin/env bash
# SessionStart(compact): inject the cheap-model brief.
set -u
in=$(cat)
dir="${OFFLOAD_DIR:-${CLAUDE_PROJECT_DIR:-.}/.compaction-offload}"
[ "$(jq -r '.source' <<<"$in")" = "compact" ] || exit 0
[ -s "$dir/brief.md" ] || exit 0
echo "$(date -Is) sessionstart injected brief_bytes=$(wc -c < "$dir/brief.md")" >> "$dir/log.txt"
jq -n --rawfile b "$dir/brief.md" '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: ("State brief written at compaction (authoritative):\n\n" + $b)}}'
