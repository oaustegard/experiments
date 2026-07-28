#!/usr/bin/env bash
set -e
export TE_DATA_DIR=$(pwd)/anchor_run/data
echo "[stage 4] body fetch + slot extract"
python3 scripts/te_extract.py --parallelism 2 || { echo "extract failed"; exit 1; }
echo "[stage 5] slot embed + rerank"
python3 scripts/te_rerank.py --top-n 160 || { echo "rerank failed"; exit 1; }
echo "[stage 6] cheap-judge"
python3 scripts/te_judge.py --top-n 160 --parallelism 2 || { echo "judge failed"; exit 1; }
echo "[pipeline 4-6 complete]"
