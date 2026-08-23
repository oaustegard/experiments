#!/bin/bash
cd /home/user/experiments/caps-emphasis
export HF_HOME=/home/user/hf-cache
commit() {
  git -C /home/user/experiments -c user.email=muninn@austegard.com \
      -c user.name=muninn -c commit.gpgsign=false add -A caps-emphasis/ 2>/dev/null
  git -C /home/user/experiments -c user.email=muninn@austegard.com \
      -c user.name=muninn -c commit.gpgsign=false commit -q -m "caps-emphasis: $1" 2>/dev/null
  echo "COMMITTED $1"
}
while kill -0 "$1" 2>/dev/null; do sleep 10; done
echo "STAGE driver1 done"
python3 run_register.py > register.log 2>&1 && echo "STAGE register done" && commit "register control: user turn vs reasoning register"
echo "DRIVER2 DONE"
