#!/bin/bash
# Run on a Colab VM after train.py finishes: teacher-forced metrics for every saved checkpoint of an arm, and
# greedy generation on the dev fixture for the final one. Writes /content/results_<arm>/ and prints EVAL_DONE.
#   bash colab/post_eval.sh ARM BAND     (BAND like 12,20, or "none" for the control arm)
set -euo pipefail
ARM=$1; BAND=$2
OUT=/content/results_$ARM
mkdir -p "$OUT"
if [ "$BAND" = none ]; then
  echo "[{\"name\": \"${ARM}\"}]" > /tmp/cfg_$ARM.json
else
  B="[${BAND}]"
  echo "[{\"name\": \"${ARM}_K1\"}, {\"name\": \"${ARM}_K2\", \"band\": $B, \"K\": 2}, {\"name\": \"${ARM}_K3\", \"band\": $B, \"K\": 3}]" > /tmp/cfg_$ARM.json
fi
for d in $(ls -d /content/ckpt/$ARM/step* | sort -V); do
  s=$(basename "$d")
  [ -f "$OUT/tf_$s.json" ] && continue
  python tf_eval.py --model "$d" --configs /tmp/cfg_$ARM.json --split test --out "$OUT/tf_$s.json" | grep -v '^frac'
done
LAST=$(ls -d /content/ckpt/$ARM/step* | sort -V | tail -1)
python gen_eval.py --model "$LAST" --configs /tmp/cfg_$ARM.json --split dev --runs "$OUT/runs" | grep -v '^  '
echo EVAL_DONE
