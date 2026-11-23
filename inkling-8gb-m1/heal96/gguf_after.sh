#!/bin/bash
# after training: wait for train + llama.cpp + GGUF downloads, then score base, unhealed slice, healed and unhealed rebuilt GGUFs
cd /content
exec >> /content/gguf_after.log 2>&1
until grep -q TRAIN_DONE /content/train.log && grep -q LLAMA_BUILD_OK /content/gguf_prep.log && grep -q GGUF_DL_OK /content/gguf_prep.log; do sleep 30; done
echo "== start $(date)"
./gguf_stage.sh base
./gguf_stage.sh slice keep_best_96.json
CK=$(ls run/adapters_*.pt | grep -v latest | sort | tail -1); echo "healed ckpt $CK"
./gguf_stage.sh build healed96 keep_best_96.json $CK
./gguf_stage.sh build unhealed96 keep_best_96.json
echo "== GGUF_ALL_DONE $(date)"
