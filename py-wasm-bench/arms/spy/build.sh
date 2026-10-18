#!/bin/bash
# build.sh -- compile src/KERNEL.spy to out/KERNEL.wasm (wasm32-wasi) with SPy.
#
# Every SPy process runs inside scripts/jail.sh as `nobody`: the libspy Makefile, the
# SPy compiler and the ninja build it generates. The host only stages the sources,
# hands the staging dir to nobody, and copies the .wasm back after a magic-byte check.
# Recipe and reasons: NOTES.md. Idempotent: libspy is built only when missing.
set -euo pipefail

ARM=$(cd "$(dirname "$0")" && pwd)
JAIL_SH=/home/user/claude-workspace/scripts/jail.sh
SPY=/tmp/jail/spy                       # SPy checkout, a39950ca, owned by nobody
SHIM=$ARM/build/bin                     # `python` -> python3.12 (SPy pins 3.12; the jail's python is 3.13)
PATH_IN=$SHIM:/usr/local/bin:/usr/bin:/bin
KERNELS="nbody mandelbrot matmul dictwords"

[ -d "$SPY/spy" ] || { echo "build.sh: no SPy checkout at $SPY (see NOTES.md)" >&2; exit 1; }
mkdir -p "$SHIM" "$ARM/out"
printf '#!/bin/sh\n# SPy pins Python 3.12 (requires-python ~=3.12.0); its Makefile calls bare `python`.\nexec /usr/bin/python3.12 "$@"\n' > "$SHIM/python"
chmod 755 "$SHIM/python"

jailed() {
  sh "$JAIL_SH" -C "$SPY" -e "PATH=$PATH_IN" -e "PYTHONPATH=$SPY/.deps:$SPY" -- "$@"
}

# libspy for wasi. The testlib flavour is what the compiler loads at import time; the
# release exe archive is what the kernels link against.
L=$SPY/spy/libspy/build/wasi
[ -f "$L/debug-testlib/libspy.wasm" ] || jailed make -C spy/libspy TARGET=wasi BUILD_TYPE=debug OUTPUT_KIND=testlib >/dev/null
[ -f "$L/release/libspy.a" ] || jailed make -C spy/libspy TARGET=wasi BUILD_TYPE=release OUTPUT_KIND=exe >/dev/null

for k in $KERNELS; do
  stage=$SPY/stage/$k
  rm -rf "$stage"
  mkdir -p "$stage"
  cp "$ARM/src/$k.spy" "$stage/$k.spy"
  chown -R nobody:nogroup "$stage"
  # Generate C + build.ninja for a release wasi exe, drop -flto (the release libspy and
  # the generated C link fine without LTO; with it wasm-ld aborts, see NOTES.md), build.
  if ! jailed sh -c "python -m spy build --no-spyc --no-compile --target wasi --output-kind exe --release stage/$k/$k.spy > stage/$k/gen.log 2>&1 \
        && sed -i 's/ -flto//g' stage/$k/build/build.ninja \
        && ninja -C stage/$k/build > stage/$k/ninja.log 2>&1"; then
    echo "build.sh: $k failed inside the jail; logs in $stage/gen.log and $stage/ninja.log" >&2
    jailed sh -c "sed 's/\x1b\[[0-9;]*m//g' stage/$k/gen.log stage/$k/ninja.log 2>/dev/null | grep -v '^\s*$' | tail -20 >&2" || true
    exit 1
  fi
  wasm=$stage/build/$k.wasm
  magic=$(head -c 4 "$wasm" | od -An -tx1 | tr -d ' \n')
  [ "$magic" = 0061736d ] || { echo "build.sh: $k.wasm has bad magic $magic" >&2; exit 1; }
  cp "$wasm" "$ARM/out/$k.wasm"
  echo "built out/$k.wasm ($(stat -c %s "$ARM/out/$k.wasm") bytes)"
done
