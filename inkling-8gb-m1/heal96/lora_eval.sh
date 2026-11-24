#!/bin/bash
# score unsloth's sliced 96-expert IQ1_S with the healed expert LoRA loaded as a llama.cpp adapter
cd /content
exec >> /content/lora_eval.log 2>&1
export PYTHONPATH=/content/llama.cpp/gguf-py
until grep -q SETUP_ALL_DONE setup.log 2>/dev/null && grep -q LLAMA_BUILD_OK build.log 2>/dev/null; do sleep 15; done
cat in/part_0* | tar x
cd in && cat lora_part* > heal96-lora.gguf && cat ad_part* > adapters_00300.pt && cd ..
python -c "
from huggingface_hub import hf_hub_download as h
import os
for f in ['Inkling-2-Mini-UD-IQ1_S', 'Inkling-2-Mini-UD-Q4_K_XL']:
    p = h('unsloth/Inkling-2-Mini-GGUF', f'Mainline_Support/{f}.gguf_file', local_dir='/content/gguf')
    os.rename(p, f'/content/gguf/mainline-{f}.gguf')
print('GGUF_DL_OK')"
B=/content/llama.cpp/build/bin
g(){ grep -E "Mean PPL\(Q\) |Mean    KLD|Same top p|rror|failed" | head -4; }
./gguf_stage.sh base
python prune_tools.py slice gguf/mainline-Inkling-2-Mini-UD-IQ1_S.gguf keep_best_96.json gguf/sliced96.gguf | tail -1
python router_swap.py gguf/sliced96.gguf keep_best_96.json in/adapters_00300.pt gguf/sliced96-router.gguf | tail -1
for m in "sliced96|gguf/sliced96.gguf|" "sliced96+lora|gguf/sliced96.gguf|--lora in/heal96-lora.gguf" "sliced96+router+lora|gguf/sliced96-router.gguf|--lora in/heal96-lora.gguf"; do
  IFS='|' read lab f extra <<< "$m"
  for s in code wiki; do echo "-- $lab $s"; $B/llama-perplexity -m $f $extra -c 512 -ngl 99 --kl-divergence-base ${s}_q4.kld --kl-divergence 2>&1 | g; done
done
echo LORA_EVAL_DONE
