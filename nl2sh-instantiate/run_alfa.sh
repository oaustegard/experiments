#!/bin/bash
# The published NL2SH benchmark's test split, same models and prompts as the
# cyber eval. Runnable rows are the point: execution decides most of these.
set -e
TLDR=/home/user/corpora/tldr/pages
NL=data/alfa_test.json
for c in generate generate_anchored instantiate_anchored; do
  echo "=== alfa it/$c $(date +%T)"
  python3 run_gen.py --condition $c --tldr $TLDR --nl $NL --out results_alfa_it_$c.json 2>&1 \
    | grep -v 'max_new_tokens' | tail -16
done
for c in generate instantiate_anchored; do
  [ -d ft_$c ] || continue
  echo "=== alfa ft/$c $(date +%T)"
  python3 run_gen.py --condition $c --model ft_$c --tldr $TLDR --nl $NL \
    --out results_alfa_ft_$c.json 2>&1 | grep -v 'max_new_tokens' | tail -16
done
echo "ALFA DONE $(date +%T)"
