#!/bin/bash
set -e; cd "$(dirname "$0")"; mkdir -p results/adapt
for arm in causal global look8 both bidir; do
  python3 adapt.py --arm $arm --out results/adapt/$arm.json >> results/adapt/$arm.log 2>&1
done
echo "done -> results/adapt/"
