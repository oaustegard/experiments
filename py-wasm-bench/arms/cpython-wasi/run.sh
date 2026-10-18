#!/bin/bash
# run.sh KERNEL N  -- runs kernels/KERNEL.py N in the guest. RUNTIME=wasmtime (default) or node.
# Bytecode caches go to build/pycache (PYTHONPYCACHEPREFIX), never into the shared stdlib dir.
set -euo pipefail
ARM=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$ARM/../.." && pwd)
PYLIB=/opt/pwb/py2wasm-2.6.3-src/nuitka/wasi-python/lib/python3.11
exec "$ROOT/tools/run_wasi.sh" "$ARM/out/python.wasm" \
  --map "$PYLIB::/pyhome/lib/python3.11" \
  --map "$ROOT/kernels::/kernels" \
  --map "$ARM/build/pycache::/pycache" \
  --env PYTHONHOME=/pyhome --env PYTHONPYCACHEPREFIX=/pycache \
  -- "/kernels/$1.py" "$2"
