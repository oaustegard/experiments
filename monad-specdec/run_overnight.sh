#!/bin/bash
# Overnight pipeline: wait for the harvest, sweep data sizes, then run
# end-to-end speculative decoding with the best head.
set -u
cd /workspace/experiments/monad-specdec
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "waiting for 12 shards"
until [ "$(ls /tmp/specdec/shards/*.npz 2>/dev/null | wc -l)" -ge 12 ]; do sleep 60; done
# Let the harvest process exit so training does not contend for cores.
until ! pgrep -f eagle_harvest2.py > /dev/null; do sleep 20; done
log "harvest complete: $(ls /tmp/specdec/shards/*.npz | wc -l) shards"

for N in 2 4 8 12; do
  log "training on $N shards"
  python3 eagle_train2.py "$N" 6 "s${N}" 2>&1 | grep -viE "loading weights|clean_up|attentioninterface"
  log "done $N shards"
done

log "end-to-end with the 12-shard head"
python3 eagle_e2e2.py /tmp/specdec/eagle_head_s12.pt 2>&1 | grep -viE "loading weights|clean_up|attentioninterface|deprecated"
log "PIPELINE COMPLETE"
