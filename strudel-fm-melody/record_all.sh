#!/bin/bash
# before/after takes per station with the same seed; the warmup skips most of the intro
cd "$(dirname "$0")"
for st in house lofi synthwave ambient; do for v in before after; do
  f=takes/$st-$v.wav; [ -s "$f" ] && continue
  for try in 1 2; do timeout 150 node rec.mjs $v-seeded.html $f station=$st 11 14 45 && break; done
done; done
echo DONE
