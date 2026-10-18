#!/bin/bash
# Build out/<kernel>.wasm for each kernel (default: all four): `cython -3` C output for the
# unmodified kernel, linked with main.c and py2wasm's WASI CPython 3.11 into one module.
# A kernel that fails to build gets out/<kernel>.unsupported (reason) and run.sh exits 3.
set -euo pipefail
ARM=$(cd "$(dirname "$0")" && pwd)
. "$ARM/env.sh"
KSRC=$ARM/../../kernels
KERNELS=${*:-nbody mandelbrot matmul dictwords}
mkdir -p "$ARM/out" "$ARM/build"

build_one() {
  # errexit is off inside a function called from `if`, so every step checks itself.
  local k=$1 b=$ARM/build/$1
  rm -rf "$b"; mkdir -p "$b"
  cp "$KSRC/$k.py" "$b/$k.py" || return 1
  # -3: Python 3 semantics. Annotation typing is the Cython 3 default for .py sources.
  "$VENV/bin/cython" -3 -o "$b/$k.c" "$b/$k.py" || return 1
  "$CLANG" $WASI_CFLAGS -I"$PYINC" -c "$ARM/main.c" -DKERNEL="$k" -o "$b/main.o" || return 1
  "$CLANG" $WASI_CFLAGS -I"$PYINC" -c "$b/$k.c" -o "$b/native.o" || return 1
  "$CLANG" $WASI_CFLAGS -o "$ARM/out/$k.wasm" "$b/main.o" "$b/native.o" $WASI_LIBS -Wl,--strip-all || return 1
}

for k in $KERNELS; do
  rm -f "$ARM/out/$k.wasm" "$ARM/out/$k.unsupported"
  if build_one "$k" > "$ARM/build/$k.log" 2>&1; then
    ls -l "$ARM/out/$k.wasm" | awk '{print $5, $9}'
  else
    reason=$(grep -m1 -i 'error' "$ARM/build/$k.log" || tail -1 "$ARM/build/$k.log")
    echo "$reason" > "$ARM/out/$k.unsupported"
    echo "unsupported $k: $reason (log: $ARM/build/$k.log)" >&2
  fi
done
