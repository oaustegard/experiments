#!/usr/bin/env bash
# PreCompact: build a structured brief with a cheap model. Never blocks.
set -u
in=$(cat)
tp=$(jq -r '.transcript_path' <<<"$in")
trig=$(jq -r '.trigger' <<<"$in")
dir="${OFFLOAD_DIR:-${CLAUDE_PROJECT_DIR:-.}/.compaction-offload}"
mkdir -p "$dir"
tp="${tp/#\~/$HOME}"
# user + assistant text and tool calls, last ~200k chars
jq -r 'select(.type=="user" or .type=="assistant") | .type as $t | (.message.content | if type=="string" then . else (map(if .type=="text" then .text elif .type=="tool_use" then "[tool " + .name + "] " + (.input|tostring) elif .type=="tool_result" then "[result] " + ((.content|tostring)[0:2000]) else empty end) | join("\n")) end) as $c | "\($t): \($c)"' "$tp" 2>/dev/null | tail -c 200000 > "$dir/excerpt.txt"
prompt='You are writing a handoff brief for an agent whose context is about to be reset. From the transcript on stdin, write markdown with exactly these sections: ## Task (the original user request, near-verbatim), ## Status (what is done, with concrete values and results), ## Next steps, ## Plan (remaining steps, numbered), ## Completion criteria, ## Artifacts (file paths and what each holds), ## Facts to keep (exact numbers, names, decisions the agent will need). No narrative. Do not invent anything not in the transcript.'
t0=$(date +%s)
claude -p --model haiku --settings '{"disableAllHooks": true}' "$prompt" < "$dir/excerpt.txt" > "$dir/brief.md.tmp" 2> "$dir/precompact.err" && mv "$dir/brief.md.tmp" "$dir/brief.md"
echo "$(date -Is) precompact trigger=$trig excerpt_bytes=$(wc -c < "$dir/excerpt.txt") brief_bytes=$(wc -c < "$dir/brief.md" 2>/dev/null || echo 0) secs=$(( $(date +%s) - t0 ))" >> "$dir/log.txt"
exit 0
