#!/bin/bash
# English-only checkpoints on LibriSpeech, then FLEURS for tiny and base; tier schemes only.
cd "$(dirname "$0")/.."
S=$(paste -sd';' run/schemes-tier.txt)
F=fleurs-es_419,fleurs-de_de,fleurs-fr_fr,fleurs-nb_no
for m in tiny.en base.en; do python3 run/sweep.py --model openai/whisper-$m --schemes "$S" --sets ls-clean,ls-other --n 300 --batch 8 2>&1 | grep --line-buffered "^whisper"; done
for m in base tiny; do python3 run/sweep.py --model openai/whisper-$m --schemes "$S" --sets $F --n 100 --batch 8 2>&1 | grep --line-buffered "^whisper"; done
echo SWEEP2_DONE
