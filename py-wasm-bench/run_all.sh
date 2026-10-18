#!/bin/bash
# The full matrix: every arm under wasmtime, then the wasm arms under node (V8).
set -euo pipefail
cd "$(dirname "$0")"
for a in arms/*/; do [ -f "$a/build.sh" ] && [ ! -f "$a/SKIP" ] && bash "$a/build.sh" >/dev/null; done
python3 bench.py --reps 5 --runtime wasmtime --out results/wasmtime.json
python3 bench.py --reps 5 --runtime node --arms c-ceiling,cpython-wasi,py2wasm,mypyc-wasi,cython-wasi --out results/node.json
python3 report.py results/wasmtime.json results/node.json > results/report.md
echo DONE
