#!/bin/sh
# Speak the ten narration lines with each built variant through kinescript-kokoro's onnx engine in the jail.
# KOKORO_ONNX_STFT=export: the files already carry the fixed STFT weights, so the plugin must not touch them.
# Reuses ../kokoro-onnx-kinescript's jail staging (plugin, bench.py, requests.json in $JAIL).
JAIL=${JAIL:-/tmp/jail/kk}; HUB=${HUB:-/home/user/claude-workspace}; M=${M:-/opt/kokoro-web-models}
for v in "$@"; do
  sh "$HUB/scripts/jail.sh" -C "$JAIL" -e PYTHONPATH="$JAIL" -e HOME="$JAIL" -e KOKORO_ENGINE=onnx -e KOKORO_ONNX_STFT=export \
     -e KOKORO_ONNX_MODEL="$M/kokoro-$v.onnx" -e KOKORO_ONNX_VOICES=/opt/konnx-models/voices-v1.0.bin \
     -- /opt/konnx/bin/python bench.py "web-$v" "out/web-$v" 2>&1 | grep -v "W:onnxruntime" | grep -o "web-$v {.*" | cut -c1-400
done
