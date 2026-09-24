#!/bin/bash
# Full API arms; resumable (api_driver skips docs already finalized or errored).
cd "$(dirname "$0")"
mkdir -p logs
python3 api_driver.py gemini --closed > logs/closed-gemini.log 2>&1
python3 api_driver.py muse --closed > logs/closed-muse.log 2>&1
python3 api_driver.py gemini --conc 2 > logs/gemini.log 2>&1 &
python3 api_driver.py muse --conc 2 > logs/muse.log 2>&1 &
wait
echo ALL_DONE >> logs/gemini.log
