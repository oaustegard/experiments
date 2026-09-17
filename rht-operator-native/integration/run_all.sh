#!/usr/bin/env bash
# Install remex main and the branch side by side, run bench.py under each.
set -euo pipefail
cd "$(dirname "$0")"
LABEL=$1; MAIN_REF=$2; BRANCH_REF=$3
PY=${PYTHON:-python}
$PY -m pip install --quiet --target "$RUNNER_TEMP/remex_main" "git+https://github.com/oaustegard/remex.git@${MAIN_REF}" --no-deps
$PY -m pip install --quiet --target "$RUNNER_TEMP/remex_branch" "git+https://github.com/oaustegard/remex.git@${BRANCH_REF}" --no-deps
PYTHONPATH="$RUNNER_TEMP/remex_main" $PY bench.py "$LABEL" main
PYTHONPATH="$RUNNER_TEMP/remex_branch" $PY bench.py "$LABEL" branch --sweep
PYTHONPATH="$RUNNER_TEMP/remex_main" $PY bench.py "$LABEL" main2
