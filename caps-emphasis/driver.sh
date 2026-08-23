#!/bin/bash
# Run the remaining measurement stages in sequence. Four cores; running these
# concurrently would only make them contend.
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
echo "STAGE v2 done"; commit "v2 sweep results"
python3 run_dose2.py    > dose2.log 2>&1 && echo "STAGE dose2 done" && commit "dose-response with frozen forbidden-word case"
python3 run_knockout.py > knockout.log 2>&1 && echo "STAGE knockout done" && commit "causal attention knockout"
python3 run_generation.py --arm framed --max-new 48 > gen_framed.log 2>&1 && echo "STAGE gen_framed done" && commit "framed generation arm"
python3 run_generation.py --arm free --n-items 10 --max-new 320 > gen_free.log 2>&1 && echo "STAGE gen_free done" && commit "free generation arm"
echo "ALL STAGES DONE"
