#!/bin/bash
# docvqa encode (already running) -> docvqa bench -> shift encode -> shift bench
cd "$(dirname "$0")"
until grep -q -E '^done|Traceback|Killed' encode_docvqa.log; do sleep 30; done
grep -q '^done' encode_docvqa.log || { echo "docvqa encode failed"; exit 1; }
python3 bench.py --data data/vidore_docvqa_enc --out results_docvqa.json > bench_docvqa.log 2>&1
echo "docvqa bench exit $?"
python3 encode_vidore.py shift > encode_shift.log 2>&1
grep -q '^done' encode_shift.log || { echo "shift encode failed"; exit 1; }
python3 bench.py --data data/vidore_shift_enc --out results_shift.json > bench_shift.log 2>&1
echo "shift bench exit $?"
