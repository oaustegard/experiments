#!/bin/bash
# requantize the Q8_0-expert splices with unsloth's imatrix (sliced to the kept experts) and score
cd /content
exec >> /content/requant2.log 2>&1
export PYTHONPATH=/content/llama.cpp/gguf-py
B=/content/llama.cpp/build/bin
g(){ grep -E "Mean PPL\(Q\) |Same top p" | head -2; }
python slice_imatrix.py gguf/imatrix_unsloth.gguf_file keep_best_96.json imatrix_unsloth96.gguf || exit 1
for n in healed96 unhealed96; do
  $B/llama-quantize --allow-requantize --imatrix imatrix_unsloth96.gguf $(sed 's/^/--tensor-type /' types.txt | tr '\n' ' ') \
     gguf/$n-q8x.gguf gguf/$n-uim-IQ1_S.gguf IQ1_S 2>&1 | tail -2
  echo "-- $n-uim code"; $B/llama-perplexity -m gguf/$n-uim-IQ1_S.gguf -c 512 -ngl 99 --kl-divergence-base code_q4.kld --kl-divergence 2>&1 | g
  echo "-- $n-uim wiki"; $B/llama-perplexity -m gguf/$n-uim-IQ1_S.gguf -c 512 -ngl 99 --kl-divergence-base wiki_q4.kld --kl-divergence 2>&1 | g
done
echo REQUANT2_DONE
