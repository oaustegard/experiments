#!/bin/bash
# bring-up on a fresh GPU runtime: setup -> routing stats on 0001-0003 -> select (reapsum) -> train 800 steps
cd /content
exec >> /content/pipeline_r3.log 2>&1
while ! grep -q SETUP_ALL_DONE /content/setup.log 2>/dev/null; do sleep 15; done
echo "== start $(date)"
python heal.py teacher --pattern 'train_000[1-3].npy' --stats --bs 2 || exit 1
python heal.py select --k 96 --out /content || exit 1
cp keep_reapsum_96.json keep_best_96.json
echo "== select done $(date)"
python heal.py train --keep keep_best_96.json --rank 16 --alpha 32 --bs 2 --accum 4 --steps 800 --lr 2e-4 \
  --lr-router 5e-5 --eval-every 200 --ckpt-every 25 --out /content/run ${INIT:+--init $INIT} > /content/train.log 2>&1 || exit 1
echo "== train done $(date)"
