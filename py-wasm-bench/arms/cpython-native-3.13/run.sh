#!/bin/bash
# Native CPython 3.13: the reference speed, not a wasm arm.
exec python3.13 "$(dirname "$0")/../../kernels/$1.py" "$2"
