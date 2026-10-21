#!/bin/zsh
# Run a command once the shared cpu.lock is free: gone, or still our RELEASED note after DEADLINE (epoch s).
# Usage: run/after_lock.sh DEADLINE "LOCK LINE" cmd args...
LOCK=/Users/austegard/Projects/experiments/embed-coord/board/cpu.lock
DEADLINE=$1; LINE=$2; shift 2
while true; do
  if [[ ! -e $LOCK ]]; then break; fi
  if grep -q "^RELEASED by ettin" $LOCK && (( $(date +%s) > DEADLINE )); then break; fi
  sleep 15
done
echo "$LINE" > $LOCK
echo "start $(date +%H:%M)"
"$@"
rc=$?
rm -f $LOCK
echo "end $(date +%H:%M) rc=$rc"
