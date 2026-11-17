#!/usr/bin/env bash
# Every stage in order. Each stage skips work already on disk, so a rerun resumes.
set -euo pipefail
cd "$(dirname "$0")"
S=/mnt/skills/user/dispatching-subagents/scripts/headless_fanout.py

python3 pipeline.py gdraft
python3 pipeline.py prep edit
python3 "$S" run runs/edit/tasks.jsonl --work runs/edit --effort low --concurrency 12
python3 pipeline.py collect edit

python3 pipeline.py prep fresh
python3 pipeline.py prep judge
python3 "$S" run runs/fresh/tasks.jsonl --work runs/fresh --effort medium --concurrency 12 &
FRESH=$!
python3 "$S" run runs/judge/tasks.jsonl --work runs/judge --effort medium --concurrency 8 &
JUDGE=$!
python3 pipeline.py gjudge
wait $FRESH
wait $JUDGE
python3 pipeline.py collect fresh
python3 pipeline.py collect judge
python3 score.py
echo RUN_COMPLETE
