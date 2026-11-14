#!/bin/bash
# The A/B on SwiftShader, one level per run, skipping runs already saved: test/batch.sh [planner]
cd "$(dirname "$0")/.."
PL=${1:-trm}
for arm in base roll; do for map in E1M4 E1M5 E1M6; do for seed in 0 37; do
  name="$arm-$PL-$map-s$seed"
  [ -f "results/runs/$name.json" ] && continue
  python3 test/play.py "$name" "map=$map&skill=2&tics=7000&seed=$seed&arm=$arm&planner=$PL" 7000 || echo "FAILED $name"
done; done; done
echo BATCH DONE
