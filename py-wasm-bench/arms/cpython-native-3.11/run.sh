#!/bin/bash
# Native CPython 3.11: the reference speed, not a wasm arm.
exec python3.11 "$(dirname "$0")/../../kernels/$1.py" "$2"
