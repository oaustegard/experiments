#!/bin/bash
set -e; cd "$(dirname "$0")"
python3 probe.py --model answerdotai/ModernBERT-base --tag mb --out results/mb.json > results/mb.log 2>&1
python3 probe.py --model jhu-clsp/ettin-encoder-32m --tag ettin32 --out results/ettin32.json > results/ettin32.log 2>&1
echo "done -> results/"
