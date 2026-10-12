#!/bin/bash
# Full interview pipeline: draft -> blind extract + judge -> repair -> pool -> hunt + jury -> pick
set -e
cd "$(dirname "$0")"
python3 interviews.py draft 5
python3 fan.py work/t_draft.jsonl work/d0 --conc 30 | tail -1
python3 interviews.py check work/d0
python3 fan.py work/t_extract_d0.jsonl work/x_d0 --conc 30 | tail -1
python3 fan.py work/t_judge_d0.jsonl work/j_d0 --conc 30 | tail -1
python3 interviews.py score work/d0 > work/score_d0.txt
python3 interviews.py repair work/d0
python3 fan.py work/t_repair.jsonl work/d1 --conc 30 | tail -1
python3 interviews.py check work/d1
python3 fan.py work/t_extract_d1.jsonl work/x_d1 --conc 30 | tail -1
python3 fan.py work/t_judge_d1.jsonl work/j_d1 --conc 30 | tail -1
python3 interviews.py score work/d1 > work/score_d1.txt
python3 choose.py pool
python3 fan.py work/t_hunt.jsonl work/hunt --conc 30 | tail -1
python3 fan.py work/t_jury.jsonl work/jury --conc 30 | tail -1
python3 choose.py pick
