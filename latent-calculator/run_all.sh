#!/usr/bin/env bash
# Resumable driver for the latent-calculator experiment.
#
# Requirements: CPU only, python3 with torch>=2.13 (cpu), transformers 5.16.1,
# numpy, pytest (ruff optional).  Both models must be in the HF cache:
# PleIAs/Monad and HuggingFaceTB/SmolLM2-135M.  ~4 GB RAM, 4 threads.
#
# Relaunch (detached, resumable -- every step skips when its output exists):
#   cd /home/user/experiments/latent-calculator && \
#     nohup bash run_all.sh > run_all.log 2>&1 &
#
# Overridable: MODELS, ARMS, EPOCHS, BATCH, N_TRAIN, N_EVAL.
set -u
cd "$(dirname "$0")"

MODELS=${MODELS:-"monad smol"}
ARMS=${ARMS:-"residual kv delayed"}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-32}
N_TRAIN=${N_TRAIN:-}
N_EVAL=${N_EVAL:-2000}
PY=${PY:-python3}

run () { echo "=== $* ==="; "$@" || { echo "FAILED: $*"; exit 1; }; }

if [ ! -f data/train.jsonl ]; then
  run $PY data.py
else
  echo "skip data (data/train.jsonl exists)"
fi

for M in $MODELS; do
  if [ ! -f results/probe_${M}.json ]; then
    run $PY probe.py --model $M
  else echo "skip probe $M"; fi

  if [ ! -f results/query_head_${M}.json ]; then
    run $PY query_head.py --model $M
  else echo "skip query_head $M"; fi

  for A in $ARMS; do
    if [ ! -f ckpt/${M}_${A}_ep${EPOCHS}.pt ]; then
      run $PY train_port.py --model $M --arm $A --epochs $EPOCHS \
          --batch $BATCH --resume ${N_TRAIN:+--n-train $N_TRAIN}
    else echo "skip train_port $M $A"; fi
  done

  for A in none text residual kv delayed; do
    for Q in oracle learned; do
      if [ ! -f results/${M}_${A}_${Q}.json ]; then
        run $PY eval.py --model $M --arm $A --query $Q --n-eval $N_EVAL
      else echo "skip eval $M $A $Q"; fi
    done
  done
done

echo "=== ALL DONE ==="
