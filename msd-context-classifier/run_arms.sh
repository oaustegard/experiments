#!/usr/bin/env bash
# Runs every arm in PLAN.md sequentially; each arm skips itself if its results JSON exists.
# Log: results/run_arms.log. Completion line: 'ALL ARMS DONE'.
set -u
cd "$(dirname "$0")"
LOG=results/run_arms.log; mkdir -p results
Q=data/queries.jsonl
run() { # name cmd...
  local name=$1; shift
  if [ -s "results/$name.json" ]; then echo "$(date +%T) skip $name (exists)" | tee -a "$LOG"; return; fi
  echo "$(date +%T) start $name" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "$(date +%T) end $name exit $?" | tee -a "$LOG"
}
run gliclass_zeroshot python3 zeroshot_gliclass.py
run probe_gte_small python3 train.py --arm probe --model models/gte-small --name probe_gte_small --queries $Q
run ft_ettin_32m   python3 train.py --arm ft --model models/ettin-encoder-32m  --name ft_ettin_32m  --queries $Q --save
run ft_ettin_68m   python3 train.py --arm ft --model models/ettin-encoder-68m  --name ft_ettin_68m  --queries $Q --save
run ft_ettin_150m  python3 train.py --arm ft --model models/ettin-encoder-150m --name ft_ettin_150m --queries $Q --save
run ft_modernbert_base python3 train.py --arm ft --model models/ModernBERT-base --name ft_modernbert_base --queries $Q
echo "$(date +%T) ALL ARMS DONE" | tee -a "$LOG"
