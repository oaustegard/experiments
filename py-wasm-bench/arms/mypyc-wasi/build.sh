#!/bin/bash
# Build out/<kernel>.wasm for each kernel (default: all four): mypyc C output for the
# kernel, linked with main.c and py2wasm's WASI CPython 3.11 into one module. A kernel
# that fails to build gets out/<kernel>.unsupported (reason) and run.sh exits 3 for it.
set -euo pipefail
ARM=$(cd "$(dirname "$0")" && pwd)
. "$ARM/env.sh"
KSRC=$ARM/../../kernels
KERNELS=${*:-nbody mandelbrot matmul dictwords}
mkdir -p "$ARM/out"
LIBRT=$("$VENV/bin/python" -c 'import mypyc.build as b; print(b.include_dir())')

build_one() {
  # errexit is off inside a function called from `if`, so every step checks itself.
  local k=$1 b=$ARM/build/$1
  rm -rf "$b"; mkdir -p "$b"
  local native_c
  native_c=$("$VENV/bin/python" "$ARM/gen_c.py" "$KSRC/$k.py" "$b" 2> "$b/mypyc.log" | tail -1) \
    || { cat "$b/mypyc.log" >&2; echo "error: mypyc failed for $k" >&2; return 1; }
  "$CLANG" $WASI_CFLAGS -I"$PYINC" -I"$LIBRT" -DKERNEL="$k" -c "$ARM/main.c" -o "$b/main.o" || return 1
  "$CLANG" $WASI_CFLAGS -I"$PYINC" -I"$LIBRT" -I"$(dirname "$native_c")" -c "$native_c" -o "$b/native.o" || return 1
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
