#!/bin/bash
# tiny then base, every scheme, LibriSpeech clean + other; resumes from the jsonl.
cd "$(dirname "$0")/.."
S=$(paste -sd';' run/schemes.txt)
for m in tiny base; do
  python3 run/sweep.py --model openai/whisper-$m --schemes "$S" --sets ls-clean,ls-other --n 300 --batch 8 2>&1 | grep -v -i warn
done
echo SWEEP_DONE
