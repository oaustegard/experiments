#!/bin/bash
# code-specialised pruning: pick experts by usage on a code calibration set, test on held-out code
cd /content
export PYTHONPATH=/content/llama.cpp/gguf-py
P=llama.cpp/build/bin/llama-perplexity
M=models/UD-IQ1_S.gguf
g(){ grep -E "Final estimate|Mean PPL\(Q\) |Mean KLD|Same top p|nan|rror:" | head -5; }
echo "== usage on code calibration"
LLAMA_EXPERT_USAGE=/content/usage_code.tsv $P -m $M -f data/calib_code.txt -c 512 --chunks 300 -ngl 99 2>&1 | g
for K in 96 64 48; do
  echo "#### code-calibrated K=$K"
  python3 prune_tools.py keep usage_code.tsv $K keepc_$K.json | head -1
  python3 prune_tools.py mask $M keepc_$K.json || { echo MASK_FAILED; break; }
  echo "-- code vs Q4"; $P -m $M -c 512 -ngl 99 --kl-divergence-base /content/code_q4.kld --kl-divergence 2>&1 | g
  echo "-- wiki vs Q4"; $P -m $M -c 512 -ngl 99 --kl-divergence-base /content/base_q4.kld --kl-divergence 2>&1 | g
  python3 prune_tools.py unmask $M $M.bias_backup.npz || { echo UNMASK_FAILED; break; }
done
echo STAGE5_DONE
