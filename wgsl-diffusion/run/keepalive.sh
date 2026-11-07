#!/bin/bash
# Keeps the Colab VM assigned (detached jobs die after ~24 min without an exec) and mirrors the run.
S=${1:-wd}; DST=${2:-/tmp/claude-0/wd_run}; mkdir -p "$DST"
while true; do
  timeout 120 colab exec -s "$S" -f /tmp/claude-0/wd_peek.py > "$DST/peek.txt" 2>&1
  timeout 300 colab download -s "$S" /content/wd/run/log.jsonl "$DST/log.jsonl" >/dev/null 2>&1
  for f in $(grep -o 'samples[0-9_]*\.png' "$DST/peek.txt" | sort -u); do
    [ -f "$DST/$f" ] || timeout 300 colab download -s "$S" /content/wd/run/$f "$DST/$f" >/dev/null 2>&1
  done
  if grep -q 'ema.pt' "$DST/peek.txt"; then
    timeout 300 colab download -s "$S" /content/wd/run/ema.pt "$DST/ema.pt.part" >/dev/null 2>&1 && mv "$DST/ema.pt.part" "$DST/ema.pt"
  fi
  grep -q '^DONE' "$DST/peek.txt" && { echo finished; break; }
  sleep 240
done
