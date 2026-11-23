#!/bin/bash
# runs detached on a Colab VM: deps, model, llama.cpp
cd /content
exec > /content/setup.log 2>&1
set -x
pip install -q transformers==5.24.0 hf_transfer "huggingface_hub[hf_transfer]" datasets safetensors
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("thinkingmachines/Inkling-2-Mini", local_dir="/content/ink", allow_patterns=["*.json","*.jinja","tiktoken/*","model-*.safetensors","tokenizer*"])
print("MODEL_OK", p)
PY
echo SETUP_MODEL_DONE
if [ "$WITH_LLAMA" = 1 ]; then
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
  cd /content/llama.cpp && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j 12 --target llama-perplexity llama-quantize llama-imatrix llama-cli llama-gguf-split >/dev/null && echo LLAMA_BUILD_OK
  cd /content
  python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["Inkling-2-Mini-UD-IQ1_S.gguf","Inkling-2-Mini-UD-Q4_K_M.gguf"]:
    print(hf_hub_download("unsloth/Inkling-2-Mini-GGUF", f, local_dir="/content/gguf"))
PY
  echo GGUF_DL_DONE
fi
echo SETUP_ALL_DONE
