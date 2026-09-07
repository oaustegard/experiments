#!/usr/bin/env bash
# Resumable driver for phase 2: the attn query head and the stream arm.
#
# Phase 1 (run_all.sh) must have produced results/probe_<M>.json and the
# residual/kv/delayed checkpoints.  Every step here skips when its output
# already exists, so the script can be relaunched after an interruption:
#
#   cd /home/user/experiments/latent-calculator && \
#     nohup bash run_phase2.sh > run_phase2.log 2>&1 &
#
# Overridable: MODELS, EPOCHS, BATCH, QH_STEPS, N_TRAIN, N_EVAL, DEMO_N, PY.
set -u
cd "$(dirname "$0")"

MODELS=${MODELS:-"smol monad"}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-32}
QH_STEPS=${QH_STEPS:-3000}
QH_BATCH=${QH_BATCH:-64}
N_TRAIN=${N_TRAIN:-}
N_EVAL=${N_EVAL:-2000}
DEMO_N=${DEMO_N:-12}
PY=${PY:-python3}

run () { echo "=== $* ==="; "$@" || { echo "FAILED: $*"; exit 1; }; }

for M in $MODELS; do
  # 1. cross-attention query head over all prompt positions
  if [ ! -f results/query_head_${M}_attn.json ]; then
    run $PY query_head.py --model $M --head attn --steps $QH_STEPS \
        --bs $QH_BATCH ${N_TRAIN:+--n-train $N_TRAIN}
  else echo "skip query_head $M attn"; fi

  # 2. streaming injection arm
  if [ ! -f ckpt/${M}_stream_ep${EPOCHS}.pt ]; then
    run $PY train_port.py --model $M --arm stream --epochs $EPOCHS \
        --batch $BATCH --resume ${N_TRAIN:+--n-train $N_TRAIN}
  else echo "skip train_port $M stream"; fi

  # 3. evals: stream x {oracle, learned-attn}, and residual with the attn query
  if [ ! -f results/${M}_stream_oracle.json ]; then
    run $PY eval.py --model $M --arm stream --query oracle --n-eval $N_EVAL
  else echo "skip eval $M stream oracle"; fi

  if [ ! -f results/${M}_stream_learned_attn.json ]; then
    run $PY eval.py --model $M --arm stream --query learned --head attn \
        --n-eval $N_EVAL
  else echo "skip eval $M stream learned attn"; fi

  if [ ! -f results/${M}_residual_learned_attn.json ]; then
    run $PY eval.py --model $M --arm residual --query learned --head attn \
        --n-eval $N_EVAL
  else echo "skip eval $M residual learned attn"; fi

  # 4. demo
  if [ ! -f results/demo_${M}.json ]; then
    run $PY demo.py --model $M --n $DEMO_N --head attn \
        --json results/demo_${M}.json
  else echo "skip demo $M"; fi
done

echo "=== PHASE2 DONE ==="
