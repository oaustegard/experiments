#!/bin/bash
# blended pruning: experts chosen from code usage and general usage together
cd /content
export PYTHONPATH=/content/llama.cpp/gguf-py
P=llama.cpp/build/bin/llama-perplexity
M=models/UD-IQ1_S.gguf
g(){ grep -E "Final estimate|Mean PPL\(Q\) |Mean KLD|Same top p|nan|rror:" | head -5; }
for K in 72 64 56 48; do
  echo "#### blended K=$K"
  python3 prune_tools.py keep usage_code.tsv,usage_iq1.tsv $K keepb_$K.json | head -1
  python3 prune_tools.py mask $M keepb_$K.json || { echo MASK_FAILED; break; }
  echo "-- code vs Q4"; $P -m $M -c 512 -ngl 99 --kl-divergence-base /content/code_q4.kld --kl-divergence 2>&1 | g
  echo "-- wiki vs Q4"; $P -m $M -c 512 -ngl 99 --kl-divergence-base /content/base_q4.kld --kl-divergence 2>&1 | g
  python3 prune_tools.py unmask $M $M.bias_backup.npz || { echo UNMASK_FAILED; break; }
done
echo STAGE6_DONE
