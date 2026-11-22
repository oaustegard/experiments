#!/bin/bash
# Inkling 2 Mini (IQ1_S, full 128 experts) on the 8 GB M1: CPU only, Metal off, small batch.
# Pass a different model file as $1, e.g. models/Inkling-2-Mini-IQ1_S-code64.gguf
cd "${INKLING_DIR:-$HOME/Projects/inkling}"
exec llama.cpp/build/bin/llama-cli -m "${1:-models/Inkling-2-Mini-UD-IQ1_S.gguf}" --jinja \
  --device none -ngl 0 -c 4096 -b 128 -ub 128 -t 4 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --reasoning-effort minimal
