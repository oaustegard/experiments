#!/bin/sh
# All Monad arms on the same 62 queries, in sequence.
# Run on an otherwise idle box: needle-bsky/ERRORS.md #6 measured a 12x latency
# inflation from a concurrent trainer. Accuracy is unaffected (greedy decode),
# latency is not.
set -e
cd "$(dirname "$0")"
for m in model tuned-e1 tuned-e2 tuned-e3; do
  [ -d "$m" ] || { echo "skip $m (absent)"; continue; }
  label=$([ "$m" = model ] && echo base || echo "$m")
  echo "== $label =="
  python3 eval.py --model "$m" --label "$label"
done
python3 compare.py
