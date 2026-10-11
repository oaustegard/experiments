#!/bin/bash
# Two rounds of blind guess -> judge -> redraw -> merge. Checkpointed: fan.py skips finished tasks.
set -e
cd "$(dirname "$0")"
SRC=work/out1
for R in 1 2; do
  python3 phase2_critic.py blind $SRC $R && python3 fan.py work/t2b_$R.jsonl work/out2b_$R --conc 24
  python3 phase2_critic.py judge $SRC $R && python3 fan.py work/t2j_$R.jsonl work/out2j_$R --conc 24
  python3 phase2_critic.py redraw $SRC $R && python3 fan.py work/t2r_$R.jsonl work/out2r_$R --conc 24
  python3 phase2_critic.py merge $SRC $R work/art$R
  SRC=work/art$R
done
# final blind pass so the page reports what a stranger sees in the finished drawing
python3 phase2_critic.py blind $SRC 3 && python3 fan.py work/t2b_3.jsonl work/out2b_3 --conc 24
echo LOOP DONE
