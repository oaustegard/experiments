#!/bin/bash
# Resumable re-run of the whole grid. Every stage is skip-if-exists and every
# artifact is committed the moment it lands — the first attempt at this ran as
# one unattended chain and lost 90 minutes of work to a container restart.
cd "$(dirname "$0")"
TLDR=${TLDR:-/home/user/corpora/tldr/pages}
NL2BASH=${NL2BASH:-/home/user/corpora/nl2bash/data/bash}
ALFA=data/alfa_test.json

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

keep() {   # keep <path...> — commit whatever landed, push best-effort
  git -C .. add -f "${@/#/nl2sh-instantiate/}" 2>/dev/null
  git -C .. -c commit.gpgsign=false -c user.email=muninn@austegard.com -c user.name=muninn \
      commit -q -m "nl2sh-instantiate: $1" 2>/dev/null && say "committed $1"
  git -C .. -c 'credential.helper=!f() { echo "username=x-access-token"; echo "password=$GH_TOKEN"; }; f' \
      push -q origin HEAD 2>/dev/null && say "pushed"
}

for c in generate instantiate instantiate_bare generate_anchored instantiate_anchored instantiate_anchored_bare; do
  out=results_it_$c.json
  [ -s $out ] && { say "skip zero-shot $c"; continue; }
  say "zero-shot $c"
  python3 run_gen.py --condition $c --tldr $TLDR --out $out >> /tmp/rep_it.log 2>&1 && keep $out
done

for c in generate instantiate_anchored; do
  [ -d ft_$c ] || { say "train $c"; python3 train.py --condition $c --tldr $TLDR --nl2bash $NL2BASH --out ft_$c >> /tmp/rep_ft.log 2>&1; }
  out=results_ft_$c.json
  [ -s $out ] && { say "skip ft-eval $c"; continue; }
  say "ft-eval $c"
  python3 run_gen.py --condition $c --model ft_$c --tldr $TLDR --out $out >> /tmp/rep_ft.log 2>&1 && keep $out
done

for c in generate generate_anchored instantiate_anchored; do
  out=results_alfa_it_$c.json
  [ -s $out ] && { say "skip alfa-it $c"; continue; }
  say "alfa-it $c"
  python3 run_gen.py --condition $c --tldr $TLDR --nl $ALFA --out $out >> /tmp/rep_alfa.log 2>&1 && keep $out
done
for c in generate instantiate_anchored; do
  out=results_alfa_ft_$c.json
  [ -d ft_$c ] || continue
  [ -s $out ] && { say "skip alfa-ft $c"; continue; }
  say "alfa-ft $c"
  python3 run_gen.py --condition $c --model ft_$c --tldr $TLDR --nl $ALFA --out $out >> /tmp/rep_alfa.log 2>&1 && keep $out
done

[ -s results_score_it.json ]   || { python3 score.py results_it_*.json   --out results_score_it.json   > /tmp/rep_score_it.log 2>&1;   keep results_score_it.json; }
[ -s results_score_ft.json ]   || { python3 score.py results_ft_*.json results_it_generate_anchored.json results_it_instantiate_anchored.json --out results_score_ft.json > /tmp/rep_score_ft.log 2>&1; keep results_score_ft.json; }
[ -s results_score_alfa.json ] || { python3 score.py results_alfa_*.json --out results_score_alfa.json > /tmp/rep_score_alfa.log 2>&1; keep results_score_alfa.json; }

[ -s results_funceq_alfa.json ] || { say "funceq alfa"; python3 funceq_alfa.py --results results_alfa_*.json --out results_funceq_alfa.json > /tmp/rep_fq_alfa.log 2>&1; keep results_funceq_alfa.json; }
[ -s results_funceq_ext.json ]  || { say "funceq cyber"; python3 funceq_ext.py --results results_it_generate.json --out results_funceq_ext.json > /tmp/rep_fq_ext.log 2>&1; keep results_funceq_ext.json; }
[ -s results_bench.json ]       || { say "bench"; python3 bench.py > /tmp/rep_bench.log 2>&1; keep results_bench.json; }

say "PIPELINE DONE"
