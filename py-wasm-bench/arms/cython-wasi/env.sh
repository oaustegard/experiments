# Shared toolchain constants for the wasm arms. Sourced by build.sh and run.sh.
PWB=/opt/pwb
VENV=$PWB/venv-mypyc
CLANG=$PWB/wasi-sdk-21/bin/clang
PYSRC=$PWB/py2wasm-2.6.3-src/nuitka/wasi-python   # py2wasm's WASI CPython 3.11 tree
PYINC=$PYSRC/include/python3.11
PYLIB=$PYSRC/lib
PYSTDLIB=$PYSRC/lib/python3.11
# Same defines and link libs as nuitka/build/SconsCompilerSettings.py (WASI branch).
WASI_CFLAGS="--target=wasm32-wasi -O2 -w -fwrapv -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_GETPID -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS"
WASI_LIBS="$PYLIB/libpython3.11.a $PYLIB/libmpdec.a $PYLIB/libexpat.a -lwasi-emulated-mman -lwasi-emulated-getpid -lwasi-emulated-signal -lwasi-emulated-process-clocks -ldl -Wl,-z,stack-size=8388608"
