#!/bin/bash
# Keeps the Colab VM assigned (detached jobs die after ~24 min without an exec) and mirrors the run.
S=${1:-sd}; DST=${2:-/tmp/claude-0/sd_run}; mkdir -p "$DST"
while true; do
  timeout 120 colab exec -s "$S" -f /tmp/claude-0/sd_peek.py > "$DST/peek.txt" 2>&1
  timeout 300 colab download -s "$S" /content/sd/run/log.jsonl "$DST/log.jsonl" >/dev/null 2>&1
  for f in $(grep -o 'samples[0-9_]*\.png' "$DST/peek.txt" | sort -u); do
    [ -f "$DST/$f" ] || timeout 300 colab download -s "$S" /content/sd/run/$f "$DST/$f" >/dev/null 2>&1
  done
  if grep -q 'ema.pt' "$DST/peek.txt"; then
    timeout 300 colab download -s "$S" /content/sd/run/ema.pt "$DST/ema.pt.part" >/dev/null 2>&1 && mv "$DST/ema.pt.part" "$DST/ema.pt"
  fi
  timeout 120 colab download -s "$S" /content/sd/run/fit_log.jsonl "$DST/fit_log.jsonl" >/dev/null 2>&1
  grep -q stats.json "$DST/peek.txt" && timeout 120 colab download -s "$S" /content/sd/run/stats.json "$DST/stats.json" >/dev/null 2>&1
  grep -q '^DONE' "$DST/peek.txt" && { echo finished; break; }
  sleep 240
done
