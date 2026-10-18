#!/bin/bash
# Build out/<kernel>.wasm for each kernel with py2wasm 2.6.3 (PyPI).
# Idempotent: a kernel whose .wasm already exists is skipped. Delete out/ to rebuild.
# Toolchain: venv at /opt/pwb/venv-py2wasm (Python 3.11), WASI_SDK_DIR=/opt/pwb/wasi-sdk-21
# (wasi-sdk 21 is the one py2wasm's own downloader would fetch). See NOTES.md.
set -euo pipefail
ARM=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$ARM/../.." && pwd)
VENV=/opt/pwb/venv-py2wasm
export WASI_SDK_DIR=/opt/pwb/wasi-sdk-21
if [ ! -x "$VENV/bin/py2wasm" ]; then
  python3.11 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q py2wasm==2.6.3 patchelf   # patchelf: standalone mode on Linux needs it
fi
export PATH="$VENV/bin:$PATH"
mkdir -p "$ARM/out" "$ARM/build"
for k in nbody mandelbrot matmul dictwords; do
  if [ -s "$ARM/out/$k.wasm" ]; then continue; fi
  (cd "$ARM/build" && py2wasm "$ROOT/kernels/$k.py" -o "$ARM/out/$k.wasm" > "$k.log" 2>&1) \
    || { echo "py2wasm failed for $k; see $ARM/build/$k.log" >&2; exit 1; }
  ls -l "$ARM/out/$k.wasm"
done
