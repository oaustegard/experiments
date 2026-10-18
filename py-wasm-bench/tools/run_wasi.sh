#!/bin/bash
# Run a WASI preview1 module under $RUNTIME (wasmtime, default; or node = V8, the
# engine Chrome uses). Usage:
#   run_wasi.sh MODULE.wasm [--map HOST::GUEST]... [--env K=V]... -- ARGS...
# ARGS become argv[1:]; argv[0] is the module name.
set -euo pipefail
mod=$1; shift
maps=(); envs=()
while [ $# -gt 0 ]; do
  case "$1" in
    --map) maps+=("$2"); shift 2 ;;
    --env) envs+=("$2"); shift 2 ;;
    --) shift; break ;;
    *) echo "run_wasi: unknown option $1" >&2; exit 2 ;;
  esac
done
case "${RUNTIME:-wasmtime}" in
  wasmtime)
    a=()
    for m in "${maps[@]+"${maps[@]}"}"; do a+=(--dir "$m"); done
    for e in "${envs[@]+"${envs[@]}"}"; do a+=(--env "$e"); done
    exec wasmtime run -W max-wasm-stack=8388608 "${a[@]+"${a[@]}"}" "$mod" "$@" ;;
  node)
    exec node --no-warnings --stack-size=8000 "$(dirname "$0")/run_wasi_node.mjs" "$mod" \
      "$(printf '%s\n' "${maps[@]+"${maps[@]}"}")" "$(printf '%s\n' "${envs[@]+"${envs[@]}"}")" "$@" ;;
  *) echo "run_wasi: RUNTIME must be wasmtime or node" >&2; exit 2 ;;
esac
