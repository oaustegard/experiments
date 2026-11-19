#!/bin/sh
# Run after chain.sh finishes. (1) Re-run the two 4B Q4_K_M pages whose timings overlapped with the
# MinerU smoke tests (18:45-19:02 UTC on 2026-11-19): page 2 finished 18:47:23 during the first
# smoke, page 3 ran from 18:47:23 through the second and third. (2) MinerU2.5-Pro Q8 on the same 20 pages.
HUB=/home/user/claude-workspace
J=/tmp/jail
P4="Transcribe this document page to Markdown. Include every piece of text exactly as written, in reading order, including headers, title blocks and footers. Write tables as HTML."
for p in synth_copies-images_sv84d95810-p1 olmo_arxiv_math-2503.05844_pg8-p1; do
  rm -f $J/out/gv41-q4km/$p.json
done
echo "== gv41-q4km rerun $(date -u +%H:%M:%S)"
# --limit 8: the 4B arm was stopped at 8 pages (19:57 UTC) at ~10-14 min per page; the sibling's
# GPU run already showed its <md> pipe-table format and 4096-token loops come from the model.
sh $HUB/scripts/jail.sh -C $J -- python3 sweep.py --limit 8 --ctx 8192 --arm gv41-q4km \
  --model models/gv41/granite-vision-4.1-4b-Q4_K_M.gguf --mmproj models/gv41/mmproj-model-f16.gguf \
  --max-new 4096 --prompt "$P4"
echo "== mineru-q8 $(date -u +%H:%M:%S)"
sh $HUB/scripts/jail.sh -C $J -e PYTHONPATH=$J/mineru/.deps -- python3 mineru_sweep.py --limit 20 \
  --arm mineru-q8 --model models/mineru/MinerU2.5-Pro-2605-1.2B.Q8_0.gguf \
  --mmproj models/mineru/MinerU2.5-Pro-2605-1.2B.mmproj-Q8_0.gguf
echo "== done $(date -u +%H:%M:%S)"
