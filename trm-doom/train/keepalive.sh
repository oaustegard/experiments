#!/bin/bash
# Keeps the Colab VM assigned (a detached job dies after ~24 min without an exec) and mirrors the run here.
S=$1; RUN=$2; DST=$3; mkdir -p "$DST"
while true; do
  timeout 120 colab exec -s "$S" -f /tmp/peek.py > "$DST/peek.txt" 2>&1
  timeout 300 colab download -s "$S" /content/$RUN/log.jsonl "$DST/log.jsonl" >/dev/null 2>&1
  timeout 300 colab download -s "$S" /content/$RUN/ema.pt "$DST/ema.pt" >/dev/null 2>&1
  grep -q DONE "$DST/peek.txt" && { echo finished; break; }
  sleep 240
done
