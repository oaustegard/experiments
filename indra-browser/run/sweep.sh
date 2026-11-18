#!/bin/sh
# Score arms on OmniParseBench, one after another, each inside scripts/jail.sh
# (opb_eval.py imports indra_ocr and omni_parse_bench). Arms are model folders
# under $EVAL/arms; results go to $EVAL/out/<arm>. Extra opb_eval.py options
# go after "--".
#   sh run/sweep.sh fp32 i8c -- --sample 600
set -u
EVAL=${EVAL:-/tmp/indra-eval}
HUB=${HUB:-/home/user/claude-workspace}
DEPS=/tmp/jail/indra/.deps:/tmp/jail/opb/.deps:/tmp/jail/opb/.deps3:/tmp/jail/opb/.deps4:/tmp/jail/opb/src
arms=""
while [ $# -gt 0 ] && [ "$1" != "--" ]; do arms="$arms $1"; shift; done
[ $# -gt 0 ] && shift
cp "$(dirname "$0")/opb_eval.py" "$EVAL/run/"
for arm in $arms; do
  echo "== $arm $(date -u +%H:%M:%S)"
  sh "$HUB/scripts/jail.sh" -C "$EVAL" -e PYTHONPATH=$DEPS -e HOME=$EVAL/out -- \
    python3 run/opb_eval.py --models arms/$arm --gray gray --ds /tmp/opb-data/ds \
    --out out/$arm --workers 4 "$@" > "$EVAL/log-$arm.txt" 2>&1
  tail -2 "$EVAL/log-$arm.txt"
done
echo "== done $(date -u +%H:%M:%S)"
