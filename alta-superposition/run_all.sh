#!/bin/bash
# Full ALTA superposition sweep. Resumable: train_code.py checkpoints per width and
# skips finished programs, so re-running this script after a kill continues the job.
# Relaunch: cd /home/user/experiments/alta-superposition && nohup ./run_all.sh >> train.log 2>&1 &
cd "$(dirname "$0")"
echo "=== run_all.sh start $(date -u +%FT%TZ) ==="
python3 -u train_code.py --programs parity_ff,parity_seq,subleq --projection data --iters 250 || exit 1
python3 -u train_code.py --programs parity_ff,parity_seq,subleq --projection code --iters 250 || exit 1
python3 plot_results.py
echo "=== ALL DONE $(date -u +%FT%TZ) ==="
