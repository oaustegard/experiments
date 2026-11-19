#!/bin/sh
# Re-score Indra arms on OmniParseBench (indra-browser/run/opb_eval.py), each inside scripts/jail.sh.
# The earlier per-page results lived in /tmp and are gone; these runs rebuild them on the same
# prerendered pages docling reads, so both systems are compared page for page.
#   sh run/indra_sweep.sh fp32 f16
set -u
EVAL=${EVAL:-/tmp/jail/opb/eval}
HUB=${HUB:-/home/user/claude-workspace}
DEPS=${DEPS:-/tmp/jail/opb/.deps12h}
for arm in "$@"; do
  echo "== $arm $(date -u +%T)"
  sh "$HUB/scripts/jail.sh" -C "$EVAL" -e PYTHONPATH=$DEPS -e HOME=$EVAL/out -- \
    python3.12 run/opb_eval.py --models arms/$arm --gray gray --ds /tmp/opb-data/ds \
    --out out/indra-$arm --workers 4 > "$EVAL/../log-indra-$arm.txt" 2>&1
  tail -2 "$EVAL/../log-indra-$arm.txt"
done
echo "== done $(date -u +%T)"
