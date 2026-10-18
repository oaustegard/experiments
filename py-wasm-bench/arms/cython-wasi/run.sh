#!/bin/bash
# run.sh KERNEL N -> runs out/KERNEL.wasm through tools/run_wasi.sh (RUNTIME=wasmtime|node).
# Last stdout line is the kernel's JSON. A kernel that did not build under this arm
# prints "unsupported: <reason>" to stderr and exits 3.
set -euo pipefail
ARM=$(cd "$(dirname "$0")" && pwd)
ROOT=$ARM/../..
k=$1; n=$2
[ -f "$ARM/out/$k.wasm" ] || [ -f "$ARM/out/$k.unsupported" ] || bash "$ARM/build.sh" "$k" >&2 || true
if [ -f "$ARM/out/$k.unsupported" ]; then
  echo "unsupported: $(cat "$ARM/out/$k.unsupported")" >&2
  exit 3
fi
. "$ARM/env.sh"
exec "$ROOT/tools/run_wasi.sh" "$ARM/out/$k.wasm" \
  --map "$PYSTDLIB::/pyhome/lib/python3.11" \
  --map "$ARM/driver::/drv" \
  --env PYTHONHOME=/pyhome --env PYTHONUTF8=1 --env PYTHONDONTWRITEBYTECODE=1 \
  -- /drv/driver.py "$k" "$n"
