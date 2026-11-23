#!/bin/bash
# GGUF side, on an L4 VM with llama.cpp built, the HF checkpoint and unsloth's UD-IQ1_S + UD-Q4_K_XL.
#   gguf_stage.sh base                      Q4 reference logits (wiki 40 chunks, held-out code 80 chunks) + IQ1_S unpruned
#   gguf_stage.sh slice KEEP.json           unsloth IQ1_S sliced to KEEP (no requant), scored
#   gguf_stage.sh build NAME KEEP.json [CKPT]  export (healed if CKPT) -> splice Q8_0 experts -> imatrix -> IQ1_S -> score
cd /content
export PYTHONPATH=/content/llama.cpp/gguf-py
B=/content/llama.cpp/build/bin
U=/content/gguf/mainline-Inkling-2-Mini-UD-IQ1_S.gguf
Q4=/content/gguf/mainline-Inkling-2-Mini-UD-Q4_K_XL.gguf
g(){ grep -E "Final estimate|Mean PPL\(Q\) |Mean KLD|Same top p|nan|rror:" | head -5; }
score(){  # model label
  echo "-- $2 code vs Q4"; $B/llama-perplexity -m $1 -c 512 -ngl 99 --kl-divergence-base /content/code_q4.kld --kl-divergence 2>&1 | g
  echo "-- $2 wiki vs Q4"; $B/llama-perplexity -m $1 -c 512 -ngl 99 --kl-divergence-base /content/wiki_q4.kld --kl-divergence 2>&1 | g
  ls -la $1 | awk '{print "size", $5}'
}
case $1 in
base)
  [ -f wiki.test.raw ] || { curl -sL -o w.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && unzip -o -q w.zip && cp wikitext-2-raw/wiki.test.raw .; }
  $B/llama-perplexity -m $Q4 -f wiki.test.raw -c 512 --chunks 40 -ngl 99 --kl-divergence-base /content/wiki_q4.kld 2>&1 | g
  $B/llama-perplexity -m $Q4 -f heldout_code.txt -c 512 --chunks 80 -ngl 99 --kl-divergence-base /content/code_q4.kld 2>&1 | g
  score $U "IQ1_S-128"
  echo BASE_DONE ;;
slice)
  python /content/prune_tools.py slice $U $2 /content/gguf/sliced_$(basename $2 .json).gguf
  score /content/gguf/sliced_$(basename $2 .json).gguf "unsloth-sliced $(basename $2 .json)"
  echo SLICE_DONE ;;
build)
  NAME=$2; KEEP=$3; CKPT=$4
  [ -f /content/export_$NAME/layer_23.safetensors ] || python heal.py export --keep $KEEP ${CKPT:+--ckpt $CKPT} --out /content/export_$NAME || exit 1
  python gguf_heal.py splice $U /content/export_$NAME /content/gguf/$NAME-q8x.gguf || exit 1
  rm -rf /content/export_$NAME
  [ -f calib.txt ] || python - <<'PY'
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/content/ink")
x = np.load("/content/data/train_0001.npy")[:200]
open("/content/calib.txt", "w").write("\n".join(tok.decode(s) for s in x))
PY
  $B/llama-imatrix -m /content/gguf/$NAME-q8x.gguf -f calib.txt -c 512 --chunks 200 -ngl 99 -o /content/imatrix_$NAME.dat 2>&1 | tail -2
  python gguf_heal.py typeargs $U /content/types.txt
  $B/llama-quantize --allow-requantize --imatrix /content/imatrix_$NAME.dat $(sed 's/^/--tensor-type /' /content/types.txt | tr '\n' ' ') \
     /content/gguf/$NAME-q8x.gguf /content/gguf/$NAME-IQ1_S.gguf IQ1_S 2>&1 | tail -3
  score /content/gguf/$NAME-IQ1_S.gguf "$NAME"
  echo BUILD_DONE ;;
esac
