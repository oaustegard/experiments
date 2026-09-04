"""The canonical Model2Vec distillation, via the library, from bekko-a25m.

`model2vec.distill.distill` runs the teacher's own vocabulary (256k ModernBERT
BPE pieces) plus our corpus-mined whole words through the teacher, PCA-reduces
to 256 and applies SIF/Zipf weighting. This is the recipe potion models start
from before their tokenlearn/contrastive stages, and the fair test of "distil
your own static model from the encoder you already run".
"""
from __future__ import annotations

import json
import time

from model2vec.distill import distill

from common import HERE

if __name__ == "__main__":
    words = json.load(open(HERE / "data" / "vocab_words.json"))
    t0 = time.time()
    m = distill("hotchpotch/bekko-embedding-v1-a25m", vocabulary=words, pca_dims=256,
                device="cpu", trust_remote_code=True)
    out = HERE / "models" / "m2v-bekko-a25m"
    m.save_pretrained(str(out))
    json.dump({"seconds": round(time.time() - t0), "rows": int(m.embedding.shape[0]),
               "dim": int(m.embedding.shape[1]), "whole_words": len(words)},
              open(out / "distill.json", "w"), indent=1)
    print(f"saved {out} {m.embedding.shape} in {time.time()-t0:.0f}s")
