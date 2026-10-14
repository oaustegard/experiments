#!/usr/bin/env bash
# Serial benchmark runner: timings are only valid with nothing else on the 4 vCPUs.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a data/run_all.log; }
test -f data/quora/q_dense.npy || { log "missing queries: encode not finished"; exit 1; }
for step in hnsw flat qdrant; do
  case $step in
    hnsw)   cmd="$PY hnsw_bench.py --shards 10 --nq 1000" ;;
    flat)   cmd="$PY flatscan_bench.py" ;;
    qdrant) cmd="$PY qdrant_bench.py" ;;
  esac
  log "start $step: $cmd"; t0=$SECONDS
  $cmd >> data/run_$step.log 2>&1; rc=$?
  log "end $step rc=$rc $((SECONDS-t0))s"
done
