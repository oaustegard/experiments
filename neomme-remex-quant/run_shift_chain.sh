#!/bin/bash
cd "$(dirname "$0")"
python3 encode_vidore.py shift >> encode_shift.log 2>&1
grep -q '^done' encode_shift.log || { echo "shift encode failed"; exit 1; }
python3 bench.py --data data/vidore_shift_enc --out results_shift.json > bench_shift.log 2>&1
echo "shift bench exit $?"
