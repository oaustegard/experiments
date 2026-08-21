#!/bin/bash
set -e
TLDR=/home/user/corpora/tldr/pages
for c in generate instantiate instantiate_bare generate_anchored instantiate_anchored instantiate_anchored_bare; do
  echo "=== $c $(date +%T)"
  python3 run_gen.py --condition $c --tldr $TLDR --out results_it_$c.json 2>&1 | grep -v 'max_new_tokens' | tail -18
done
echo "ALL DONE $(date +%T)"
