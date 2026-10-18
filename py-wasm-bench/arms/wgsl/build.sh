#!/usr/bin/env bash
# Translate the @kernel functions in kernels_gpu.py to out/*.wgsl, manifests and f32 inputs.
# Idempotent. Needs only python3 (stdlib); the GPU step is run_gpu.js, checked by verify.py.
set -euo pipefail
cd "$(dirname "$0")"
python3 build.py "$@"
