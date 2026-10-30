#!/bin/bash
# Resumable: each stage skips the units it finished (layers, variants).
set -e
cd "$(dirname "$0")/../.."
python3 -u lfm-wgsl/run/quant3.py data
python3 -u lfm-wgsl/run/quant3.py gptq
python3 -u lfm-wgsl/run/quant3.py eval
