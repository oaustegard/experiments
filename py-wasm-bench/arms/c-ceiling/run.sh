#!/bin/bash
# run.sh KERNEL N: run one wasm kernel through tools/run_wasi.sh (RUNTIME=wasmtime|node).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec "$here/../../tools/run_wasi.sh" "$here/out/$1.wasm" -- "$2"
