#!/bin/bash
# Keeps the Colab VM for run NAME assigned (detached jobs die after ~24 min without an exec) and mirrors
# the run. Ends with the literal line "finished" once the trainer has written DONE.
#   run/keepalive.sh SESSION [DST=/tmp/claude-0/RUN_run] [RUN=SESSION]
# RUN names the run directory /content/RUN on the VM (a session can host a second run); needs
# /tmp/claude-0/RUN_peek.py, see colab_launch.py.
S=${1:-wid}; R=${3:-$S}; DST=${2:-/tmp/claude-0/${R}_run}; mkdir -p "$DST"
while true; do
  timeout 120 colab exec -s "$S" --timeout 90 -f "/tmp/claude-0/${R}_peek.py" > "$DST/peek.txt" 2>&1
  timeout 300 colab download -s "$S" "/content/$R/run/log.jsonl" "$DST/log.jsonl" >/dev/null 2>&1
  for f in $(grep -o 'samples_[0-9]*\.png' "$DST/peek.txt" | sort -u); do
    [ -f "$DST/$f" ] || timeout 300 colab download -s "$S" "/content/$R/run/$f" "$DST/$f" >/dev/null 2>&1
  done
  grep -q '^DONE' "$DST/peek.txt" && { timeout 600 colab download -s "$S" "/content/$R/run/ema.pt" "$DST/ema.pt"; echo finished; break; }
  sleep 240
done
