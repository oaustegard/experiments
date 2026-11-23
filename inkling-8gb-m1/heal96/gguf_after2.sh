#!/bin/bash
cd /content
exec >> /content/gguf_after2.log 2>&1
echo "== start $(date)"
./gguf_stage.sh build healed96 keep_best_96.json run/adapters_00300.pt
./gguf_stage.sh build unhealed96 keep_best_96.json
echo "== GGUF2_DONE $(date)"
