#!/bin/bash
# Build out/python.wasm: stock CPython 3.11 (py2wasm's prebuilt wasi libpython) as a WASI command.
# Idempotent: rebuilds from scratch each run. See NOTES.md for versions and gotchas.
set -euo pipefail
ARM=$(cd "$(dirname "$0")" && pwd)
SDK=/opt/pwb/wasi-sdk-21
W=/opt/pwb/py2wasm-2.6.3-src/nuitka/wasi-python
CLANG=$SDK/bin/clang
SYSROOT=$SDK/share/wasi-sysroot
mkdir -p "$ARM/build/pycache" "$ARM/out"
cat > "$ARM/build/main.c" <<'C'
#include <Python.h>
int main(int argc, char **argv) { return Py_BytesMain(argc, argv); }
C
# Flags mirror nuitka/build/SconsCompilerSettings.py (WASI branch).
CFLAGS=(--target=wasm32-wasi --sysroot="$SYSROOT" -O2 -w -I"$W/include/python3.11"
        -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_GETPID -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS)
"$CLANG" "${CFLAGS[@]}" -c "$ARM/build/main.c" -o "$ARM/build/main.o"
"$CLANG" --target=wasm32-wasi --sysroot="$SYSROOT" -O2 -Wl,--strip-all \
  "$ARM/build/main.o" -o "$ARM/out/python.wasm" \
  "$W/lib/libpython3.11.a" "$W/lib/libmpdec.a" "$W/lib/libexpat.a" \
  -lwasi-emulated-mman -lwasi-emulated-getpid -lwasi-emulated-signal -lwasi-emulated-process-clocks \
  -ldl -lm
ls -l "$ARM/out/python.wasm"
