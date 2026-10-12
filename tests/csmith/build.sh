#!/bin/sh
# Build csmith from source into tests/tools/csmith (gitignored).
# Result: tests/tools/csmith/bin/csmith and tests/tools/csmith/include/csmith.h
set -eu
TOOLS="$(cd "$(dirname "$0")/.." && pwd)/tools"
SRC="$TOOLS/src/csmith"
mkdir -p "$TOOLS/src"
[ -d "$SRC" ] || git clone --depth 1 https://github.com/csmith-project/csmith.git "$SRC"
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_INSTALL_PREFIX="$TOOLS/csmith" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$SRC/build" -j"$(nproc)" >/dev/null
cmake --install "$SRC/build" >/dev/null
ls "$TOOLS/csmith/bin/csmith" && ls "$TOOLS"/csmith/include/csmith*/csmith.h 2>/dev/null || ls "$TOOLS"/csmith/include
