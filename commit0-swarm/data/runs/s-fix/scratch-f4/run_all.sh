#!/bin/bash
cd /tmp/jail/c0/work/s-fix/statsmodels
OUT=/home/user/experiments/commit0-swarm/data/runs/s-fix/scratch-f4
: > $OUT/summary.txt
while read f; do
  echo "=== $f" >> $OUT/summary.txt
  timeout 1500 /tmp/jail/c0/bin/c0-test "$f" -q --tb=no -rfE -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR|[0-9]+ (passed|failed)|.*(passed|failed).* in )|exit status" >> $OUT/summary.txt
done < $OUT/files.txt
echo ALLDONE >> $OUT/summary.txt
