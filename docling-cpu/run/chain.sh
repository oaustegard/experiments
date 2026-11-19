#!/bin/sh
# Arms in order, each inside scripts/jail.sh; sweep.py skips pages already done, so a rerun resumes.
HUB=/home/user/claude-workspace
J=/tmp/jail
P4="Transcribe this document page to Markdown. Include every piece of text exactly as written, in reading order, including headers, title blocks and footers. Write tables as HTML."
run() { sh $HUB/scripts/jail.sh -C $J -- python3 sweep.py --limit 20 --ctx 8192 "$@"; }
echo "== docling-bf16 $(date -u +%H:%M:%S)"
run --arm docling-bf16 --model models/docling/granite-docling-258M-BF16.gguf --mmproj models/docling/mmproj-model-f16.gguf
echo "== gv41-q4km $(date -u +%H:%M:%S)"
run --arm gv41-q4km --model models/gv41/granite-vision-4.1-4b-Q4_K_M.gguf --mmproj models/gv41/mmproj-model-f16.gguf --max-new 4096 --prompt "$P4"
# gv41-q8 dropped 2026-11-19 (Oskar: Q4 arm only for the 4B)
echo "== done $(date -u +%H:%M:%S)"
