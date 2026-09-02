#!/usr/bin/env bash
# Staged driver. Launch detached:  nohup bash run_all.sh > logs/run_all.log 2>&1 &
# Each stage writes a sentinel to logs/STAGE_<name>.done and commits+pushes the
# small artifacts it produced, so a reclaimed container loses at most one stage.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ckpt data
BRANCH=${BRANCH:-feat/embedding-inversion}
EPOCHS_ZERO=${EPOCHS_ZERO:-3}
EPOCHS_CORR=${EPOCHS_CORR:-1}

stage() {  # stage <name> <cmd...>
  local name=$1; shift
  if [ -f "logs/STAGE_$name.done" ]; then echo "== skip $name (done)"; return; fi
  echo "== $name  $(date -u +%FT%TZ)"
  "$@" 2>&1 | tee "logs/$name.log"
  date -u +%FT%TZ > "logs/STAGE_$name.done"
  keep "$name"
}

keep() {  # commit + push what landed; never blocks the pipeline
  ( cd .. && for f in embedding-inversion/logs embedding-inversion/data/meta.json embedding-inversion/results_*.json; do
      [ -e "$f" ] && git add -A "$f"   # one add per path: an unmatched glob would fail the whole add (ERRORS.md #2)
    done
    git -c commit.gpgsign=false -c user.email=oskar@austegard.com -c user.name='Oskar Austegard' \
      commit -q -m "embedding-inversion: stage $1 landed" || true
    git push -q -u origin "$BRANCH" || echo "push failed for $1 (will retry next stage)" ) || true
}

stage build       python3 build_data.py
stage zero_float  python3 train.py --mode zero --cond float --epochs "$EPOCHS_ZERO"
stage hyps_float  bash -c 'python3 evaluate.py gen --cond float --split train && python3 evaluate.py gen --cond float --split dev'
stage corr_float  python3 train.py --mode correct --cond float --epochs "$EPOCHS_CORR" \
                    --hyps data/hyps_float_train.json --dev-hyps data/hyps_float_dev.json --init ckpt/zero_float.pt
stage eval_float  python3 evaluate.py eval --cond float --rounds 5
stage zero_bin1   python3 train.py --mode zero --cond bin1 --epochs "$EPOCHS_ZERO"
stage hyps_bin1   bash -c 'python3 evaluate.py gen --cond bin1 --split train && python3 evaluate.py gen --cond bin1 --split dev'
stage corr_bin1   python3 train.py --mode correct --cond bin1 --epochs "$EPOCHS_CORR" \
                    --hyps data/hyps_bin1_train.json --dev-hyps data/hyps_bin1_dev.json --init ckpt/zero_bin1.pt
stage eval_bin1   python3 evaluate.py eval --cond bin1 --rounds 5
echo "== ALL DONE $(date -u +%FT%TZ)"
