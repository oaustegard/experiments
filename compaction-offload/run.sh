#!/usr/bin/env bash
# Forces auto-compaction in a nested headless session and checks the brief carried state.
set -u
cd "$(git rev-parse --show-toplevel)"
export OFFLOAD_DIR="$PWD/compaction-offload/results/run-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OFFLOAD_DIR"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=40
TASK='Step 1: generate 400 lines of pseudo-random data with python (seed 7): each line "id,value" with value = random.randint(1,10**6). Save to /tmp/co_data.csv and print the whole file to the terminal. Step 2: read /usr/share/dict/words or, if absent, print python help(str) and help(dict) in full, three times, to fill context. Step 3: WITHOUT re-reading /tmp/co_data.csv, state the seed you used, the value on line 137, and the sum of all values you printed in step 1. Then verify against the file and report MATCH or MISMATCH for each.'
cp compaction-offload/CLAUDE.md "$OFFLOAD_DIR/CLAUDE.md.used"
claude -p --settings compaction-offload/settings.json --append-system-prompt "$(cat compaction-offload/CLAUDE.md)" --output-format json --dangerously-skip-permissions "$TASK" > "$OFFLOAD_DIR/session.json" 2> "$OFFLOAD_DIR/session.err"
echo "--- log"; cat "$OFFLOAD_DIR/log.txt" 2>/dev/null || echo 'no hook log: compaction did not fire'
echo "--- result"; jq -r '.result' "$OFFLOAD_DIR/session.json" | tail -20
