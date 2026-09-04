"""Distil a static table from bekko-a25m (the xr encoder), Model2Vec recipe.

Vocabulary = the corpus-mined whole words (mine_vocab.py) + the bekko subword
tokens actually used by the corpus. Each entry is encoded on its own by the
teacher, the matrix is PCA-reduced to 256, and rows are SIF-weighted by
corpus frequency (w = a / (a + p), a = 1e-4), which is what model2vec.distill
does with sif_coefficient. Zero training pairs; the only corpus signal is
which tokens exist and how often.
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from tokenizers import Tokenizer

from common import HERE, MODELS, HybridTok, StaticTable, load_chunks
from bekko import BekkoEncoder

DIM = 256
SIF_A = 1e-4


def main() -> None:
    words = json.load(open(HERE / "data" / "vocab_words.json"))
    word_df = json.load(open(HERE / "data" / "vocab_df.json"))
    base = Tokenizer.from_file(str(MODELS / "bekko-a25m" / "tokenizer.json"))
    vocab = base.get_vocab()
    inv = {i: t for t, i in vocab.items()}
    special = {i for t, i in vocab.items() if t.startswith("<") and t.endswith(">")}

    # subword tokens the corpus actually uses (outside mined whole words)
    tok0 = HybridTok(base, {w: j for j, w in enumerate(words)}, special)
    sub_tf = Counter()
    word_tf = Counter()
    for c in load_chunks("ast"):
        for i in tok0(c["text"]):
            (word_tf if i >= tok0.n_base else sub_tf)[i] += 1
    sub_ids = sorted(sub_tf)
    print(f"{len(words)} whole words, {len(sub_ids)} subword ids in use", flush=True)

    # teacher pass: each vocabulary entry alone
    enc = BekkoEncoder("a25m", threads=3)
    sub_strs = [inv[i].replace("Ġ", " ").replace("▁", " ") for i in sub_ids]
    t0 = time.time()
    V = enc.encode(sub_strs + words, batch_size=64, progress=True)
    print(f"teacher pass {len(V)} entries in {time.time()-t0:.0f}s", flush=True)

    # PCA to DIM
    mu = V.mean(0)
    U, S, Vt = np.linalg.svd(V - mu, full_matrices=False)
    P = (V - mu) @ Vt[:DIM].T
    print(f"PCA {DIM}: explained {float((S[:DIM]**2).sum()/(S**2).sum()):.3f}", flush=True)

    # SIF weights from corpus frequency
    tf = np.array([sub_tf[i] for i in sub_ids] + [word_tf[tok0.n_base + j] for j in range(len(words))],
                  dtype=np.float64)
    p = tf / tf.sum()
    w = SIF_A / (SIF_A + p)
    P = (P * w[:, None]).astype(np.float32)

    # pack into a compact table: remap subword ids to 0..len(sub_ids)-1 by
    # building a reduced base tokenizer is more than we need; keep the full
    # bekko id space (256k rows, zeros for unused) — 262 MB fp32 is fine here.
    table = np.zeros((tok0.n_base + len(words), DIM), dtype=np.float32)
    table[sub_ids] = P[: len(sub_ids)]
    table[tok0.n_base:] = P[len(sub_ids):]
    unused = set(range(tok0.n_base)) - set(sub_ids) | special
    st = StaticTable(table, HybridTok(base, tok0.words, unused))
    st.save(HERE / "models" / "distill-bekko-a25m")
    json.dump({"whole_words": len(words), "subword_ids": len(sub_ids), "dim": DIM,
               "teacher_seconds": round(time.time() - t0), "sif_a": SIF_A},
              open(HERE / "models" / "distill-bekko-a25m" / "distill.json", "w"), indent=1)
    print("saved models/distill-bekko-a25m")


if __name__ == "__main__":
    main()
