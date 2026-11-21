#!/bin/bash
# Build and upload the tier packs for tiny, base and small (English-only and multilingual, +-63).
cd "$(dirname "$0")/.."
for m in tiny.en tiny base.en base small.en small; do
  d=out/packs/tier-$m
  python3 run/pack.py --model openai/whisper-$m --scheme "all=r63,conv=f16" --out $d 2>&1 | grep "weights.bin"
  python3 run/shard.py $d --upload models/whisper-pack/$m-r63 2>&1 | grep -v "  up"
  rm -f $d/weights.bin
done
echo TIERS_DONE
