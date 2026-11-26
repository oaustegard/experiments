#!/usr/bin/env bash
# End-to-end parity check of the WGSL engine: export the programs, build onnxruntime references on real
# pages and line crops, run the engine in headless Chromium (SwiftShader) and compare.
#   bash web/gpu/test/run.sh [--gpu] [model ...]
set -euo pipefail
cd "$(dirname "$0")/../../.."
python3 -I run/export_gpu.py
python3 -I web/gpu/test/make_ref.py
node web/gpu/test/run_parity.mjs "$@"
