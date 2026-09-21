#!/usr/bin/env bash
# "Does this paper relate to MSD?" arms (PLAN-paper.md). Skips finished steps. Log: results/paper_arms.log. Completion: 'PAPER ARMS DONE'.
set -u; cd "$(dirname "$0")"; LOG=results/paper_arms.log; mkdir -p results
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }
C=data/paper_corpus.jsonl
[ -s "$C" ] || { say "start: paper_corpus"; python3 paper_corpus.py >> "$LOG" 2>&1; }
run() { local name=$1; shift; [ -s "results/$name.json" ] && { say "skip $name"; return; }; say "start $name"; "$@" >> "$LOG" 2>&1; say "end $name exit $?"; }
run paper_probe_gte_small   python3 train.py --arm probe --model models/gte-small --name paper_probe_gte_small --corpus $C
run paper_ft_ettin_32m      python3 train.py --arm ft --model models/ettin-encoder-32m      --name paper_ft_ettin_32m      --corpus $C --class-weight --epochs 6
run paper_ft_dapt_term_32m  python3 train.py --arm ft --model models/dapt_term_ettin_32m    --name paper_ft_dapt_term_32m  --corpus $C --class-weight --epochs 6
run paper_ft_dapt_vocab_term_32m python3 train.py --arm ft --model models/dapt_vocab_term_ettin_32m --name paper_ft_dapt_vocab_term_32m --corpus $C --class-weight --epochs 6
run paper_ft_ettin_150m     python3 train.py --arm ft --model models/ettin-encoder-150m     --name paper_ft_ettin_150m     --corpus $C --class-weight --epochs 4
python3 paper_score.py >> "$LOG" 2>&1
say "PAPER ARMS DONE"
