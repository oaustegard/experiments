#!/bin/bash
# sanity perplexity, usage collection, base logits
cd /content
P=llama.cpp/build/bin/llama-perplexity
run(){ "$@" 2>&1 | grep -E "Final estimate|Mean KLD|Same top|Maximum KLD|nan|error|rror:" | head -8; }
echo "== sanity Q4 wiki 8 chunks"; run $P -m models/UD-Q4_K_XL.gguf -f data/wiki.test.raw -c 512 --chunks 8 -ngl 99
echo "== sanity IQ1_S wiki 8 chunks"; run $P -m models/UD-IQ1_S.gguf -f data/wiki.test.raw -c 512 --chunks 8 -ngl 99
echo "== usage IQ1_S on calibration text"
LLAMA_EXPERT_USAGE=/content/usage_iq1.tsv $P -m models/UD-IQ1_S.gguf -f data/thinkingmachines_Inkling-2-Mini-calibration-v6.txt -c 512 --chunks 160 -ngl 99 2>&1 | grep -E "Final estimate|rror" | head -3
echo "== usage Q4 on calibration text"
LLAMA_EXPERT_USAGE=/content/usage_q4.tsv $P -m models/UD-Q4_K_XL.gguf -f data/thinkingmachines_Inkling-2-Mini-calibration-v6.txt -c 512 --chunks 160 -ngl 99 2>&1 | grep -E "Final estimate|rror" | head -3
echo "== base logits (wiki, 40 chunks)"
run $P -m models/UD-Q4_K_XL.gguf -f data/wiki.test.raw -c 512 --chunks 40 -ngl 99 --kl-divergence-base /content/base_q4.kld
run $P -m models/UD-IQ1_S.gguf -f data/wiki.test.raw -c 512 --chunks 40 -ngl 99 --kl-divergence-base /content/base_iq1.kld
echo "== IQ1_S vs Q4"; run $P -m models/UD-IQ1_S.gguf -c 512 -ngl 99 --kl-divergence-base /content/base_q4.kld --kl-divergence
wc -l /content/usage_*.tsv; echo STAGE1_DONE
