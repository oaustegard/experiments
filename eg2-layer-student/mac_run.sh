#!/usr/bin/env bash
# Train eg2-layer-student students on an Apple GPU (PyTorch MPS), then pack the results.
#
#   bash eg2-layer-student/mac_run.sh setup   # venv, packages, ~2 GB of models and targets, vocab cut
#   bash eg2-layer-student/mac_run.sh train   # k = 4, 2, 8, 10 epochs each on the 32K cut; resumes
#   bash eg2-layer-student/mac_run.sh pack    # out/eg2-layer-mac.tgz + the gh release command
#   bash eg2-layer-student/mac_run.sh all
#
# PY=/path/to/python reuses an existing environment instead of .venv (it needs the
# packages in requirements-mac.txt). Every step skips finished work, so rerun after an
# interruption. Training runs under caffeinate so the Mac does not sleep.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
PY=${PY:-$HERE/.venv/bin/python}
RUNS=${RUNS:-"4 2 8"}
EPOCHS=${EPOCHS:-10}
export PYTORCH_ENABLE_MPS_FALLBACK=1   # any op MPS lacks runs on CPU instead of failing

base_python() {
  # torch publishes macOS arm64 wheels for a bounded range of Python versions; a newer
  # Homebrew python3 (or an x86 one under Rosetta) gets "from versions: none".
  for v in python3.13 python3.12 python3.11; do
    if command -v "$v" >/dev/null && [ "$("$v" -c 'import platform; print(platform.machine())')" = arm64 ]; then
      echo "$v"; return
    fi
  done
  return 1
}

setup() {
  if ! "$PY" -c "import torch, transformers" 2>/dev/null; then
    if [ "$PY" != "$HERE/.venv/bin/python" ]; then
      echo "PY=$PY lacks torch/transformers; pip install -r requirements-mac.txt into it" >&2; exit 1
    fi
    rm -rf .venv
    if command -v uv >/dev/null; then
      uv venv -q --python 3.12 .venv           # uv fetches a managed 3.12 if none is installed
      uv pip install -q --python .venv/bin/python -r requirements-mac.txt
    else
      base=$(base_python) || { echo "need an arm64 python3.11-3.13 (brew install python@3.12) or uv" >&2; exit 1; }
      "$base" -m venv .venv
      .venv/bin/pip install -q --upgrade pip
      .venv/bin/pip install -q -r requirements-mac.txt
    fi
  fi
  "$PY" -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
  "$PY" -I run/fetch_data.py
  [ -f out/vocab/k32000/rows.json ] || "$PY" -I run/vocab.py
}

train() {
  mkdir -p out
  for k in $RUNS; do
    tag=mac_k${k}_v32k
    [ -f "out/l1/$tag/final.pt" ] && { echo "$tag done"; continue; }
    echo "== $tag: k=$k, $EPOCHS epochs; check s/step in the first lines =="
    caffeinate -i "$PY" -I run/l1_distill.py --k "$k" --vocab out/vocab/k32000 --epochs "$EPOCHS" \
      --lr 3e-3 --eval-every 2572 --save-every 500 --device auto --tag "$tag" 2>&1 \
      | grep --line-buffered -v "Forcing last layer" | tee -a "out/$tag.log"
  done
}

pack() {
  tar czf out/eg2-layer-mac.tgz results/l1_mac_*.json out/l1/mac_*/final.pt out/mac_*.log
  ls -lh out/eg2-layer-mac.tgz
  echo "publish with:"
  echo "  gh release create eg2-layer-mac-$(date +%Y%m%d) $HERE/out/eg2-layer-mac.tgz -R oaustegard/experiments --title 'eg2-layer-student Mac training' --notes 'Checkpoints and logs for #103'"
}

case "${1:-all}" in
  setup) setup ;;
  train) train ;;
  pack) pack ;;
  all) setup; train; pack ;;
  *) echo "usage: $0 setup|train|pack|all" >&2; exit 2 ;;
esac
