#!/usr/bin/env bash
# Path C cross-domain run — full cascade.
# Run from mvp/theory-empirical-bridges/ (i.e. cd mvp/theory-empirical-bridges).
set -e

export TE_DATA_DIR=$(pwd)/path_c_cross_domain/data
LOGS=$(pwd)/path_c_cross_domain/logs
mkdir -p "$LOGS"

echo "[stage 0] anchor candidate assembly ($TE_DATA_DIR)"
python3 scripts/te_anchor.py --skip-citation-filter 2>&1 | tee "$LOGS/00_anchor.log"

echo "[stage diag] twin diagnostic on SPECTER2 neighborhoods"
python3 path_c_cross_domain/twin_diagnostic.py 2>&1 | tee "$LOGS/01_twin_diagnostic.log"

echo "[stage 4] body fetch + slot extract"
python3 scripts/te_extract.py --parallelism 2 2>&1 | tee "$LOGS/04_extract.log"

echo "[stage 5] slot embed + rerank"
python3 scripts/te_rerank.py --top-n 200 2>&1 | tee "$LOGS/05_rerank.log"

echo "[stage 6] cheap-judge"
python3 scripts/te_judge.py --top-n 200 --parallelism 2 2>&1 | tee "$LOGS/06_judge.log"

echo "[stage diag-2] twin diagnostic on judge survivors"
python3 path_c_cross_domain/twin_diagnostic.py 2>&1 | tee "$LOGS/07_twin_diagnostic_post.log" || true

echo "[pipeline complete]"
