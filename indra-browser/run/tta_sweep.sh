#!/bin/sh
# Test-time augmentation arms on OmniParseBench, one after another, each inside
# scripts/jail.sh. An arm is NAME:MODELS:GRAY[:opb_eval options with + for spaces]:
#   base:v110:gray                      the unperturbed reference
#   p-gamma2:v110:gray:--perturb+gamma2 a page-level variant (perturb.PAGE)
#   p-min:v110:graymin                  the darkest-channel color conversion
#   r-avg:v110-logp:gray:--rec-tta+id,gamma2,thin   averaged recognizer posteriors
# Results go to $EVAL/out/<NAME>; a page already there is skipped, so a killed
# chain resumes where it stopped.
#   sh run/tta_sweep.sh base:v110:gray p-otsu:v110:gray:--perturb+otsu
set -u
EVAL=${EVAL:-/tmp/indra-eval}
HUB=${HUB:-/home/user/claude-workspace}
DEPS=/tmp/jail/opb/.deps12
R=$(dirname "$0")
mkdir -p "$EVAL/run" "$EVAL/out"
cp "$R/opb_eval.py" "$R/perturb.py" "$R/rec_tta.py" "$EVAL/run/"
for spec in "$@"; do
  name=$(echo "$spec" | cut -d: -f1)
  models=$(echo "$spec" | cut -d: -f2)
  gray=$(echo "$spec" | cut -d: -f3)
  extra=$(echo "$spec" | cut -d: -f4- | tr '+' ' ')
  mkdir -p "$EVAL/out/$name"
  chown -R nobody:nogroup "$EVAL/out/$name"
  echo "== $name $(date -u +%H:%M:%S)"
  # $extra is split on purpose: it holds opb_eval.py options.
  # shellcheck disable=SC2086
  sh "$HUB/scripts/jail.sh" -C "$EVAL" -e PYTHONPATH=$DEPS -e HOME=$EVAL/out -e OMP_NUM_THREADS=1 -- \
    python3.12 run/opb_eval.py --models arms/$models --gray $gray --ds /tmp/opb-data/ds \
    --out out/$name --workers 4 $extra > "$EVAL/log-$name.txt" 2>&1
  tail -2 "$EVAL/log-$name.txt"
done
echo "== done $(date -u +%H:%M:%S)"
