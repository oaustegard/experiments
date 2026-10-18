#!/bin/bash
# run.sh KERNEL N -- runs out/KERNEL.wasm under the wasmtime sandbox. The module prints
# the kernel's JSON line last; "seconds" is timed inside SPy around the kernel only.
# RUNTIME=node is refused (exit 3): the modules come from a jailed compiler, and this
# arm is measured in the wasmtime sandbox (NOTES.md).
set -euo pipefail
ARM=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$ARM/../.." && pwd)

case "${RUNTIME:-wasmtime}" in
  wasmtime) ;;
  node) echo "unsupported: node runtime not used for jailed-compiler output" >&2; exit 3 ;;
  *) echo "run.sh: RUNTIME must be wasmtime" >&2; exit 2 ;;
esac

case "${1:-}" in
  nbody|mandelbrot|matmul|dictwords) ;;
  *) echo "run.sh: unknown kernel '${1:-}'" >&2; exit 2 ;;
esac

exec env RUNTIME=wasmtime "$ROOT/tools/run_wasi.sh" "$ARM/out/$1.wasm" -- "$2"
