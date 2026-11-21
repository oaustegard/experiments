#!/bin/bash
# Keep a Colab VM assigned and mirror files from it while a detached job runs there.
#   colab/watch.sh SESSION LOCAL_DIR REMOTE_LOG [REMOTE_GLOB_DIR]
# Every 4 minutes: a trivial `colab exec` (a `colab download` alone does NOT keep the VM: both VMs of round 1
# were reclaimed 24 min after the last exec while downloads kept succeeding until the 404s), then download the
# log. With REMOTE_GLOB_DIR (e.g. /content/ckpt/loopB1), any new step* checkpoint directory there is fetched
# file by file into LOCAL_DIR/<step>/, so a reclaimed VM costs at most one save interval.
# Exits when the log's last line starts with "EXIT" (the job wrapper's completion line).
S=$1; D=$2; LOG=$3; CK=${4:-}
mkdir -p "$D"
K=$(mktemp --suffix=.py)
cat > "$K" <<'EOF'
import subprocess
print("KEEPALIVE_OK", subprocess.run("ls -d /content/ckpt/*/step* 2>/dev/null | tr '\n' ' '", shell=True, capture_output=True, text=True).stdout)
EOF
while true; do
  ex=$(timeout 150 colab exec -s "$S" -f "$K" --timeout 120 2>/dev/null)
  case "$ex" in *KEEPALIVE_OK*) ka=ok;; *) ka=FAILED;; esac
  steps=$(echo "$ex" | grep -o '/content/ckpt/[^ ]*')
  timeout 150 colab download -s "$S" "$LOG" "$D/$(basename "$LOG")" >/dev/null 2>&1 || echo "$(date -u +%H:%M) download failed: $LOG"
  if [ -n "$CK" ]; then
    for st in $(echo "$steps" | grep "^$CK/"); do
      n=$(basename "$st")
      [ -f "$D/$n/.complete" ] && continue
      mkdir -p "$D/$n"
      ok=1
      for f in model.safetensors loop_gate.pt config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.json generation_config.json; do
        timeout 300 colab download -s "$S" "$st/$f" "$D/$n/$f" >/dev/null 2>&1 || { [ "$f" = loop_gate.pt ] || [ "$f" = chat_template.json ] || [ "$f" = special_tokens_map.json ] || ok=0; }
      done
      [ $ok = 1 ] && touch "$D/$n/.complete" && echo "$(date -u +%H:%M) fetched $n"
    done
  fi
  last=$(tail -n 1 "$D/$(basename "$LOG")" 2>/dev/null)
  echo "$(date -u +%H:%M) keepalive=$ka $(grep -v Warn "$D/$(basename "$LOG")" 2>/dev/null | tail -n 1 | cut -c1-200)"
  case "$last" in EXIT*) echo "job finished: $last"; exit 0;; esac
  sleep 240
done
