#!/bin/bash
cd /home/user/experiments/caps-emphasis
export HF_HOME=/home/user/hf-cache
while kill -0 "$1" 2>/dev/null; do sleep 10; done
python3 run_position.py > position.log 2>&1
git -C /home/user/experiments -c user.email=muninn@austegard.com -c user.name=muninn \
    -c commit.gpgsign=false add -A caps-emphasis/ 2>/dev/null
git -C /home/user/experiments -c user.email=muninn@austegard.com -c user.name=muninn \
    -c commit.gpgsign=false commit -q -m "caps-emphasis: sentence-initial vs medial emphasis position" 2>/dev/null
echo "DRIVER3 DONE"
