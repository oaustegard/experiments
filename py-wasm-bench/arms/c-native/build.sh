#!/bin/bash
# Build the native reference: the same C sources as c-ceiling, built with system gcc -O2.
# No -ffast-math. -ffp-contract=off and -fno-builtin-pow match the wasm build.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
for src in ../c-ceiling/src/*.c; do
  k=$(basename "$src" .c)
  gcc -O2 -ffp-contract=off -fno-builtin-pow -Wall -o "out/$k" "$src" -lm
done
