#!/usr/bin/env bash
# Vocabulary-adaptation arms (PLAN-vocab.md). Each step skips itself if its output exists.
# Log: results/dapt_arms.log. Completion line: 'ALL DAPT ARMS DONE'.
set -u
cd "$(dirname "$0")"
LOG=results/dapt_arms.log; mkdir -p results
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }
run() { # marker cmd...
  local marker=$1; shift
  if [ -e "$marker" ]; then say "skip ($marker exists): $*"; return; fi
  say "start: $*"; "$@" >> "$LOG" 2>&1; say "end exit $?: $*"
}
# --- adaptation arms on ettin-32m (2 epochs each, full corpus minus dev/test)
run models/dapt_ettin_32m/epoch.json       python3 dapt.py --model models/ettin-encoder-32m --name dapt_ettin_32m       --epochs 2
run models/dapt_term_ettin_32m/epoch.json  python3 dapt.py --model models/ettin-encoder-32m --name dapt_term_ettin_32m  --epochs 2 --term-mask 0.5
run models/dapt_vocab_ettin_32m/epoch.json python3 dapt.py --model models/ettin-encoder-32m --name dapt_vocab_ettin_32m --epochs 2 --expand-vocab 300
# --- whole-term recovery on the same 60 held-out pages as the baselines
for m in dapt_ettin_32m dapt_term_ettin_32m dapt_vocab_ettin_32m; do
  run results/mlm_$m.json python3 mlm_ppl.py $m models/$m --pages 60 --rounds 2
done
# --- retrieval: controls, then base and adapted encoders
run results/retr_bm25.json        python3 retrieval_eval.py --arm bm25 --name bm25
run results/retr_gte_small.json   python3 retrieval_eval.py --arm encoder --model models/gte-small --name gte_small
run results/retr_ettin_32m.json   python3 retrieval_eval.py --arm encoder --model models/ettin-encoder-32m --name ettin_32m
run results/retr_ettin_150m.json  python3 retrieval_eval.py --arm encoder --model models/ettin-encoder-150m --name ettin_150m
for m in dapt_ettin_32m dapt_term_ettin_32m dapt_vocab_ettin_32m; do
  run results/retr_$m.json python3 retrieval_eval.py --arm encoder --model models/$m --name $m
done
# --- tokens per product page under the expanded vocabulary
run results/tokens_per_page.json python3 - <<'EOF'
import json, os
from transformers import AutoTokenizer
rows=[json.loads(l) for l in open('data/full/corpus.jsonl') if l.strip()]
prod=[(r.get('title') or '')+'\n'+(r.get('text') or '') for r in rows if r['label']=='products'][:1000]
out={}
for name,path in [('ettin_bpe','models/ettin-encoder-32m'),('ettin_bpe_plus300','models/dapt_vocab_ettin_32m')]:
    t=AutoTokenizer.from_pretrained(path); n=[len(t(x)['input_ids']) for x in prod]; out[name]={'pages':len(n),'mean_tokens':sum(n)/len(n),'vocab':len(t)}
json.dump(out,open('results/tokens_per_page.json','w'),indent=1); print(out)
EOF
say "ALL DAPT ARMS DONE"
