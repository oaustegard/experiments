#!/bin/bash
# Build each src/<kernel>.c to out/<kernel>.wasm (wasm32-wasi, wasi-sdk 21 clang 17).
# No -ffast-math. -ffp-contract=off keeps a*b+c unfused. -fno-builtin-pow keeps
# pow(x, 0.5) as a libm call: clang 17 otherwise rewrites it to sqrt, which is not
# what CPython computes.
set -euo pipefail
cd "$(dirname "$0")"
CLANG=/opt/pwb/wasi-sdk-21/bin/clang
mkdir -p out
for src in src/*.c; do
  k=$(basename "$src" .c)
  "$CLANG" -O2 --target=wasm32-wasi -ffp-contract=off -fno-builtin-pow -Wall -o "out/$k.wasm" "$src" -lm
done
