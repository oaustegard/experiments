#!/bin/bash
# run.sh KERNEL N  -- runs out/KERNEL.wasm with N as argv[1]. RUNTIME=wasmtime (default) or node.
set -euo pipefail
ARM=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$ARM/../.." && pwd)
exec "$ROOT/tools/run_wasi.sh" "$ARM/out/$1.wasm" -- "$2"
