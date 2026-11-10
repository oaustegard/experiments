#!/bin/bash
# One keepalive tick for a detached Colab job: touch the kernel, pull the log and the results zip.
# Used from short scheduled check-ins when no long-lived local driver is running.
S=${1:-gliner-r1}; cd "$(dirname "$0")/.." && mkdir -p out/colab
echo 'print(1)' > out/colab/touch.py
colab exec -s "$S" -f out/colab/touch.py --timeout 60 2>/dev/null | tail -1
colab download -s "$S" /content/gliner.log out/colab/gliner.log >/dev/null 2>&1; echo "log rc=$?"
colab download -s "$S" /content/gliner_out.zip out/colab/gliner_out.zip >/dev/null 2>&1; echo "zip rc=$?"
tail -1 out/colab/gliner.log; grep -c 'ARM_DONE\|BASELINE_DONE' out/colab/gliner.log; grep 'ARM_FAILED\|JOB_CRASHED\|ALL_DONE' out/colab/gliner.log | tail -3
