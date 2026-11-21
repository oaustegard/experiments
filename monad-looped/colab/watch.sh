#!/bin/bash
# Keep a Colab VM assigned and mirror files from it while a detached job runs there.
#   colab/watch.sh SESSION LOCAL_DIR REMOTE_FILE [REMOTE_FILE ...]
# Every 4 minutes: download each remote file into LOCAL_DIR (a colab download touches the VM, which keeps
# an untouched session from being reclaimed; METHODS.md). Exits when the first file's last line starts
# with "EXIT" (the job wrapper's completion line), after one final download.
S=$1; D=$2; shift 2
mkdir -p "$D"
while true; do
  for f in "$@"; do
    timeout 150 colab download -s "$S" "$f" "$D/$(basename "$f")" >/dev/null 2>&1 || echo "$(date -u +%H:%M) download failed: $f"
  done
  last=$(tail -n 1 "$D/$(basename "$1")" 2>/dev/null)
  echo "$(date -u +%H:%M) $(grep -v Warn "$D/$(basename "$1")" 2>/dev/null | tail -n 1 | cut -c1-200)"
  case "$last" in EXIT*) echo "job finished: $last"; exit 0;; esac
  sleep 240
done
