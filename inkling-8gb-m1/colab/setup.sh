#!/bin/bash
# runs on the Colab VM: build patched llama.cpp with CUDA, fetch models and texts
set -x
cd /content
nvidia-smi --query-gpu=name,memory.total --format=csv
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git fetch --depth 50 origin && git checkout e50c254 && git apply /content/expert-usage.patch || echo PATCH_FAILED
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release > /content/cmake.log 2>&1
cmake --build build --config Release -j 12 --target llama-perplexity llama-cli >> /content/cmake.log 2>&1; echo "BUILD_EXIT $?"
cd /content && mkdir -p models data
pip -q install -U huggingface_hub gguf >/dev/null 2>&1
for q in UD-IQ1_S UD-Q4_K_XL; do
  hf download unsloth/Inkling-2-Mini-GGUF "Mainline_Support/Inkling-2-Mini-$q.gguf_file" --local-dir models >/dev/null 2>&1
  mv "models/Mainline_Support/Inkling-2-Mini-$q.gguf_file" "models/$q.gguf"
done
hf download bartowski/thinkingmachines_Inkling-2-Mini-GGUF thinkingmachines_Inkling-2-Mini-calibration-v6.txt --local-dir data >/dev/null 2>&1
python3 - <<'PY'
from huggingface_hub import hf_hub_download
import pandas as pd
p=hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset')
open('/content/data/wiki.test.raw','w').write(''.join(pd.read_parquet(p)['text']))
PY
ls -l models data
echo SETUP_DONE
