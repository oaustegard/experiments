#!/bin/bash
# Data-scaling sweep. Smallest-first so a result lands early, then scales up.
set -u
cd /workspace/experiments/monad-specdec
echo $$ > /tmp/specdec/sweep.pid
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "EXIT rc=$?"' EXIT
N=$(ls /tmp/specdec/shards/*.npz 2>/dev/null | wc -l)
log "sweep start, $N shards available"
for S in 4 8 "$N"; do
  [ "$S" -gt "$N" ] && continue
  [ -f "eagle_train_s${S}.json" ] && { log "s$S done already"; continue; }
  log "=== training on $S shards ==="
  python3 -u eagle_train2.py "$S" 4 "s${S}" 2>&1 | grep --line-buffered -viE "loading weights|clean_up|attentioninterface"
  log "=== finished $S shards (rc=${PIPESTATUS[0]}) ==="
done
log "=== end-to-end with the ${N}-shard head ==="
python3 -u eagle_e2e2.py "/tmp/specdec/eagle_head_s${N}.pt" 2>&1 | grep --line-buffered -viE "loading weights|clean_up|attentioninterface|deprecated"
log "SWEEP COMPLETE"
