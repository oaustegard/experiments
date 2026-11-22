#!/bin/bash
# NVIDIA Nemotron 3.5 Bolt E4B (Gemma 4 E4B base) on the 8 GB M1, Metal on: ~16 tokens/sec
cd "${INKLING_DIR:-$HOME/Projects/inkling}"
exec llama.cpp/build/bin/llama-cli -m models/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf --jinja \
  -ngl 99 -b 256 -ub 256 -c 8192 -t 4 --temp 1.0 --top-p 0.95 "$@"
