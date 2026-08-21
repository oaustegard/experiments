#!/bin/bash
# Train Gemma 3 270M under each prompt condition on stage 1's 600 rows, then
# evaluate each fine-tune under the prompt it was trained on. Serial: 4 cores.
set -e
TLDR=/home/user/corpora/tldr/pages
NL2BASH=/home/user/corpora/nl2bash/data/bash
for c in generate instantiate_anchored; do
  echo "=== train $c $(date +%T)"
  python3 train.py --condition $c --tldr $TLDR --nl2bash $NL2BASH --out ft_$c 2>&1 | tail -6
  echo "=== eval $c $(date +%T)"
  python3 run_gen.py --condition $c --model ft_$c --tldr $TLDR --out results_ft_$c.json 2>&1 \
    | grep -v 'max_new_tokens' | tail -18
done
echo "FT DONE $(date +%T)"
