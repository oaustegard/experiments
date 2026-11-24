#!/bin/bash
# launch80.sh SESSION : upload everything, start setup + llama.cpp build + pipeline_80 + guard cell
S=$1; cd /tmp/claude-0/ink; H=/home/user/claude-workspace/.spokes/inkling-heal
q(){ grep -v -i "keyring\|warning" | cut -c1-70; }
./rs $S 'mkdir -p /content/in /content/run' >/dev/null
for f in setup_remote.sh heal.py pipeline_80.sh gguf_stage.sh prune_tools.py $H/lora_gguf.py; do timeout 600 colab upload -s $S $f /content/$(basename $f) 2>&1 | q; done
./rs $S 'cd /content && chmod +x setup_remote.sh pipeline_80.sh gguf_stage.sh && (WITH_LLAMA=0 setsid nohup ./setup_remote.sh >/dev/null 2>&1 </dev/null &) && (setsid nohup bash -c "git clone -q --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp && export PATH=/usr/local/cuda/bin:\$PATH && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=\"80;89\" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc >/dev/null && cmake --build build -j 6 --target llama-perplexity llama-quantize >/dev/null && echo LLAMA_BUILD_OK" > /content/build.log 2>&1 </dev/null &) && echo setup+build-started'
for f in relay/part_0* up2/tkp_*; do timeout 600 colab upload -s $S $f /content/in/$(basename $f) 2>&1 | q; done | tail -1
./rs $S 'cd /content && ls in | wc -l && (setsid nohup ./pipeline_80.sh >/dev/null 2>&1 </dev/null &) && echo pipeline-started'
