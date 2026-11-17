#!/bin/zsh
# usage: crash_test.sh label ENV=.. ENV=..   -> renders 60 sequential frames, reports how many finished
cd "$(dirname "$0")"
label=$1; shift
L=/private/tmp/claude-501/-Users-austegard-Projects-claude-workspace/669419a3-0240-40fa-a0e1-d7b7585631e5/scratchpad
rm -rf ct_$label
env "$@" blender -b --factory-startup --gpu-backend metal -P render.py -- 1500 1559 ct_$label 1280 720 10 > $L/ct_$label.log 2>&1
n=$(ls ct_$label 2>/dev/null | grep -c '^f_')
d=$(grep -c '^DONE' $L/ct_$label.log)
echo "$label: frames=$n done=$d"
