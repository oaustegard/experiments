#!/bin/zsh
# Pause the distill after epoch EP's eval for a 20-min wasm CPU window (coord #24).
# Usage: run/pause_window.sh PID LOG EP
PID=$1; LOG=$2; EP=$3; LOCK=/Users/austegard/Projects/experiments/embed-coord/board/cpu.lock
until grep -q "^epoch $EP {" $LOG; do sleep 10; done
kill -STOP $PID
T0=$(date +%H:%M); T1=$(date -v+20M +%H:%M)
echo "RELEASED by ettin at $T0 for wasm window until $T1 (ettin distill pid $PID SIGSTOPped). wasm: overwrite with your own line while you run; delete when done." > $LOCK
echo "paused $T0"
sleep 1200
# wasm still holding? wait up to 10 more min for it to finish
for i in {1..60}; do grep -q "^wasm" $LOCK 2>/dev/null || break; sleep 10; done
kill -CONT $PID
echo "ettin MPS: distill.py e17-base resumed $(date +%H:%M) after wasm window; ETA ~00:10 EDT" > $LOCK
echo "resumed $(date +%H:%M)"
