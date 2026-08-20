#!/bin/bash
# Wait for the 4-shard arm to land, then replace the block-buffered driver with
# the line-buffered one. run_sweep2.sh skips any arm whose JSON already exists.
set -u
cd /workspace/experiments/monad-specdec
until [ -f eagle_train_s4.json ]; do sleep 10; done
echo "[$(date -u +%H:%M:%S)] s4 landed; swapping driver" >> /tmp/swap.log
OLD=$(cat /tmp/specdec/sweep.pid 2>/dev/null)
[ -n "${OLD:-}" ] && kill -TERM -"$OLD" 2>/dev/null || kill "$OLD" 2>/dev/null
pkill -f "eagle_train2.py" 2>/dev/null
sleep 5
setsid nohup ./run_sweep2.sh >> /tmp/sweep.log 2>&1 < /dev/null &
echo "[$(date -u +%H:%M:%S)] line-buffered driver started" >> /tmp/swap.log
