#!/bin/bash
# 80-expert run, end to end on one VM: stats -> select 80 -> train (Round 8 recipe) -> LoRA GGUF -> score vs Q4
cd /content
exec >> /content/pipeline_80.log 2>&1
until grep -q SETUP_ALL_DONE setup.log 2>/dev/null; do sleep 15; done
cat in/part_0* | tar x; cat in/tkp_* | tar x
echo "== start $(date)"
python heal.py teacher --pattern 'train_0002.npy' --stats --bs 2 || exit 1
python heal.py select --k 80 --out /content || exit 1
cp keep_reapsum_80.json keep_best_80.json
echo "== select done $(date)"
if nvidia-smi --query-gpu=memory.total --format=csv,noheader | grep -q "^4"; then BS="--bs 2 --accum 4"; else BS="--bs 1 --accum 8"; fi
python heal.py train --keep keep_best_80.json --rank 8 --alpha 16 $BS --steps 300 --lr 2e-4 --lr-router 5e-5 \
  --eval-every 100 --ckpt-every 10 --out /content/run > /content/train.log 2>&1 || exit 1
echo "== train done $(date)"
until grep -q LLAMA_BUILD_OK build.log 2>/dev/null; do sleep 15; done
export PYTHONPATH=/content/llama.cpp/gguf-py
python lora_gguf.py run/adapters_00300.pt heal80-lora.gguf || exit 1
python -c "
from huggingface_hub import hf_hub_download as h
import os
for f in ['Inkling-2-Mini-UD-IQ1_S', 'Inkling-2-Mini-UD-Q4_K_XL']:
    p = h('unsloth/Inkling-2-Mini-GGUF', f'Mainline_Support/{f}.gguf_file', local_dir='/content/gguf')
    os.rename(p, f'/content/gguf/mainline-{f}.gguf')
print('GGUF_DL_OK')"
./gguf_stage.sh base
python prune_tools.py slice gguf/mainline-Inkling-2-Mini-UD-IQ1_S.gguf keep_best_80.json gguf/sliced80.gguf | tail -1
B=/content/llama.cpp/build/bin
g(){ grep -E "Mean PPL\(Q\) |Mean    KLD|Same top p|rror|failed" | head -4; }
for m in "sliced80||" "sliced80+lora|--lora heal80-lora.gguf"; do
  IFS='|' read lab extra <<< "$m"
  for s in code wiki; do echo "-- $lab $s"; $B/llama-perplexity -m gguf/sliced80.gguf $extra -c 512 -ngl 99 --kl-divergence-base ${s}_q4.kld --kl-divergence 2>&1 | g; done
done
ls -la gguf/sliced80.gguf heal80-lora.gguf
echo PIPE80_DONE
