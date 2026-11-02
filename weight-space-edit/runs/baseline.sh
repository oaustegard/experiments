#!/bin/bash
cd /home/user/experiments/weight-space-edit
for m in greedy prod; do
  python3 lfm_eval.py --set gsm8k --n $([ $m = greedy ] && echo 200 || echo 100) --max-new 512 --mode $m --out runs/base_gsm8k_$m.jsonl 2>&1 | grep -E "^batch|edit="
  python3 lfm_eval.py --set chat --n 32 --max-new 512 --mode $m --out runs/base_chat_$m.jsonl 2>&1 | grep -E "^batch|edit="
done
echo BASELINE_DONE
