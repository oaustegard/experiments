#!/bin/bash
set -e; cd "$(dirname "$0")"
python3 fan.py work/t5r.jsonl work/out5r --conc 28
python3 phase2b_bestof.py blind && python3 fan.py work/t5b.jsonl work/out5b --conc 28
python3 phase2b_bestof.py judge && python3 fan.py work/t5j.jsonl work/out5j --conc 28
python3 phase2b_bestof.py final
echo BESTOF DONE
