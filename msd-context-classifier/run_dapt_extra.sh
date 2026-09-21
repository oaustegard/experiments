#!/usr/bin/env bash
# Decision-rule follow-up: vocabulary expansion combined with whole-term masking, ettin-32m.
set -u; cd "$(dirname "$0")"; LOG=results/dapt_arms.log
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }
if [ ! -e models/dapt_vocab_term_ettin_32m/epoch.json ] || [ "$(python3 -c "import json;print(json.load(open('models/dapt_vocab_term_ettin_32m/epoch.json'))['next_epoch'])")" != "2" ]; then
  say "start: dapt_vocab_term_ettin_32m"; python3 dapt.py --model models/ettin-encoder-32m --name dapt_vocab_term_ettin_32m --epochs 2 --expand-vocab 300 --term-mask 0.5 >> "$LOG" 2>&1; say "end exit $?: dapt_vocab_term_ettin_32m"
fi
for m in "ettin_32m models/ettin-encoder-32m" "ettin_150m models/ettin-encoder-150m" "modernbert_base models/ModernBERT-base" "bioclinical_modernbert_base models/BioClinical-ModernBERT-base" "dapt_ettin_32m models/dapt_ettin_32m" "dapt_term_ettin_32m models/dapt_term_ettin_32m" "dapt_vocab_ettin_32m models/dapt_vocab_ettin_32m" "dapt_vocab_term_ettin_32m models/dapt_vocab_term_ettin_32m"; do
  set -- $m; [ -e results/mlm1_$1.json ] || { say "start: mlm1 $1"; python3 mlm_ppl.py $1 $2 --pages 60 --rounds 2 --per-pass 1 >> "$LOG" 2>&1; }
done
[ -e results/retr_dapt_vocab_term_ettin_32m.json ] || python3 retrieval_eval.py --arm encoder --model models/dapt_vocab_term_ettin_32m --name dapt_vocab_term_ettin_32m >> "$LOG" 2>&1
say "EXTRA ARM DONE"
