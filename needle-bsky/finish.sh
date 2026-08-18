#!/bin/sh
# Everything that has to wait for the LoRA run to finish and the box to go quiet.
#
# Latency on this container is only meaningful with nothing else running (see
# ERRORS.md #6), so the timing arms are re-taken here rather than alongside
# training. Run from the experiment directory:
#
#   sh finish.sh
#
set -e
cd "$(dirname "$0")"

echo "== 1. merge the adapter and quantize =="
needle build checkpoints/needle2.pkl --lora checkpoints/needle_lora.pkl --out needle_bsky.cact
ls -la needle_bsky.cact

echo "== 2. tuned weights on the same eval set =="
# Confidence is None on tuned weights, so this arm is scored on tool/args only.
python3 eval.py --arms tuned-min --weights needle_bsky.cact --label finetuned

echo "== 3. latency curve, quiet box =="
python3 latency_curve.py

echo "== 4. two-stage latency, quiet box =="
python3 two_stage.py --arm tuned-min

echo "== 5. live demo capture =="
sh demo.sh

echo "== 6. recheck =="
python3 recheck.py
