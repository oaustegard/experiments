#!/bin/bash
# mask sweep on held-out wikitext and held-out code, using the repo's gguf-py (knows NVFP4)
cd /content
export PYTHONPATH=/content/llama.cpp/gguf-py
P=llama.cpp/build/bin/llama-perplexity
M=models/UD-IQ1_S.gguf
g(){ grep -E "Final estimate|Mean PPL\(Q\) |Mean KLD|Same top p|nan|rror:" | head -5; }
echo "== base logits on code (Q4)"
$P -m models/UD-Q4_K_XL.gguf -f data/heldout_code.txt -c 512 --chunks 80 -ngl 99 --kl-divergence-base /content/code_q4.kld 2>&1 | g
echo "== IQ1_S unpruned vs Q4 on code"
$P -m $M -c 512 -ngl 99 --kl-divergence-base /content/code_q4.kld --kl-divergence 2>&1 | g
for K in ${KS:-96 80 64 56 48 40 32}; do
  echo "#### K=$K"
  python3 prune_tools.py keep usage_iq1.tsv $K keep_$K.json | head -1
  python3 prune_tools.py mask $M keep_$K.json || { echo MASK_FAILED; break; }
  echo "-- wiki vs unpruned IQ1_S"; $P -m $M -c 512 -ngl 99 --kl-divergence-base /content/base_iq1.kld --kl-divergence 2>&1 | g
  echo "-- wiki vs Q4";            $P -m $M -c 512 -ngl 99 --kl-divergence-base /content/base_q4.kld  --kl-divergence 2>&1 | g
  echo "-- code vs Q4";            $P -m $M -c 512 -ngl 99 --kl-divergence-base /content/code_q4.kld  --kl-divergence 2>&1 | g
  python3 prune_tools.py unmask $M $M.bias_backup.npz || { echo UNMASK_FAILED; break; }
done
echo STAGE4_DONE
