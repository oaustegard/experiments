#!/bin/sh
# Run each arm of the comparison inside claude-workspace's scripts/jail.sh (no network, runs as nobody).
# Needs: the plugin, bench.py and requests.json copied into $JAIL (chowned to nobody); the venvs and models
# described in RESULTS.md (Method). Usage: sh run_arms.sh ARM...   (arms: torch312[b] torch313 onnx[b] onnx-export onnx-fp16 onnx-int8;
# onnx arms reweight the inverse STFT unless named *-export; the fix does not reach fp16, whose initializers are renamed)
JAIL=${JAIL:-/tmp/jail/kk}; HUB=${HUB:-/home/user/claude-workspace}; M=/opt/konnx-models
for arm in "$@"; do
  case $arm in
    torch312*) py=/opt/kokoro-env/bin/python; env="-e KOKORO_ENGINE=torch -e KOKORO_DEVICE=cpu" ;;
    torch313) py=/opt/k313/bin/python; env="-e KOKORO_ENGINE=torch -e KOKORO_DEVICE=cpu" ;;
    onnx*-export) py=/opt/konnx/bin/python; env="-e KOKORO_ENGINE=onnx -e KOKORO_ONNX_ISTFT=export -e KOKORO_ONNX_MODEL=$M/kokoro-v1.0.onnx" ;;
    onnx-fp16) py=/opt/konnx/bin/python; env="-e KOKORO_ENGINE=onnx -e KOKORO_ONNX_MODEL=$M/kokoro-v1.0.fp16.onnx" ;;
    onnx-int8) py=/opt/konnx/bin/python; env="-e KOKORO_ENGINE=onnx -e KOKORO_ONNX_MODEL=$M/kokoro-v1.0.int8.onnx" ;;
    onnx*) py=/opt/konnx/bin/python; env="-e KOKORO_ENGINE=onnx -e KOKORO_ONNX_MODEL=$M/kokoro-v1.0.onnx" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  sh "$HUB/scripts/jail.sh" -C "$JAIL" -e PYTHONPATH="$JAIL" -e HOME="$JAIL" -e HF_HOME=/opt/hf -e HF_HUB_OFFLINE=1 \
     -e KOKORO_ONNX_VOICES=$M/voices-v1.0.bin $env -- $py bench.py "$arm" "out/$arm" 2>&1 | grep -v "W:onnxruntime" | tail -3
done
