#!/bin/bash
# Both arms on SwiftShader for a list of map:seed pairs, committing each result as it lands, skipping runs already in
# results/runs/: test/batch2.sh E1M4:74 E1M5:111 ...   (base first: it takes minutes, the rollout arm 30-60)
cd "$(dirname "$0")/.."
# PLANNER=trm|oracle (default oracle: TRM on SwiftShader takes over 50 min of wall clock per 77 s of game)
PL=${PLANNER:-oracle}
for arm in base roll; do for ms in "$@"; do
  map=${ms%%:*}; seed=${ms##*:}; name="sw-$PL-$arm-$map-s$seed"
  git pull -q --rebase origin main 2>/dev/null
  [ -f "results/runs/$name.json" ] && continue
  python3 test/play.py "$name" "map=$map&skill=2&tics=7000&seed=$seed&arm=$arm&planner=$PL" 9000 || { echo "FAILED $name"; continue; }
  git add "results/runs/$name.json"
  git -c user.email=oskar@austegard.com -c user.name='Oskar Austegard' commit -qm "trm-doom-rollouts: $name (SwiftShader)" -- "results/runs/$name.json"
  for i in 1 2 3 4; do git pull -q --rebase origin main && git push -q origin main 2>/dev/null && break; sleep $((i * 5)); done
done; done
echo BATCH DONE
