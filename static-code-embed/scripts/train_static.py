"""Contrastive fine-tune of a static table (EmbeddingBag mean-pool + L2).

MultipleNegativesRankingLoss, symmetric, in-batch negatives, scale 20 — the
sentence-transformers static-embedding recipe, in ~60 lines of torch so the
container needs nothing beyond torch-cpu. Optionally extends the vocabulary
with corpus-mined whole words, each new row initialised as the mean of its
subword rows (so epoch 0 is exactly the unextended model).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import HERE, StaticTable


def batches(tok, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ids, offs = [], [0]
    for t in texts:
        x = tok(t) or [0]
        ids.extend(x)
        offs.append(len(ids))
    return torch.tensor(ids), torch.tensor(offs[:-1])


def mnrl(a: torch.Tensor, p: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    a = nn.functional.normalize(a, dim=-1)
    p = nn.functional.normalize(p, dim=-1)
    s = a @ p.T * scale
    y = torch.arange(len(a))
    return (nn.functional.cross_entropy(s, y) + nn.functional.cross_entropy(s.T, y)) / 2


def extend(st: StaticTable, words: list[str], init: str = "mean") -> StaticTable:
    """Append whole-word rows initialised from their subword rows.

    init="mean" is the obvious choice and is wrong for a mean-pooled model: a
    5-piece identifier that used to contribute 5/N of the pooled vector now
    contributes 1/N, so long distinctive identifiers get down-weighted against
    short common words. init="sum" preserves each word's pooled contribution
    exactly, so epoch 0 reproduces the unextended model's cosines to the bit.
    """
    import copy
    base_rows = st.table[: st.tok.n_base]
    new = np.zeros((len(words), st.dim), dtype=np.float32)
    for j, w in enumerate(words):
        ids = [i for i in st.tok.base.encode(w).ids if i not in st.tok.drop]
        new[j] = (base_rows[ids].sum(0) if init == "sum" else base_rows[ids].mean(0)) if ids else 0
    tok = copy.copy(st.tok)
    tok.words = {w: j for j, w in enumerate(words)}
    return StaticTable(np.concatenate([base_rows, new]), tok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="model2vec dir or our saved dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extend-vocab", help="json list of whole words to add")
    ap.add_argument("--init-mode", default="mean", choices=["mean", "sum"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    init = Path(a.init)
    st = (StaticTable.from_dir(init) if (init / "table.npy").exists()
          else StaticTable.from_model2vec(init))
    if a.extend_vocab:
        st = extend(st, json.load(open(a.extend_vocab)), a.init_mode)
        print(f"vocab extended to {st.table.shape[0]} rows", flush=True)

    train = json.load(open(HERE / "data" / "pairs_train.json"))
    val = json.load(open(HERE / "data" / "pairs_val.json"))
    emb = nn.EmbeddingBag.from_pretrained(torch.tensor(st.table), mode="mean", freeze=False)
    opt = torch.optim.AdamW(emb.parameters(), lr=a.lr, weight_decay=0.0)

    def loss_on(pairs: list[dict]) -> float:
        with torch.no_grad():
            tot = 0.0
            for s in range(0, len(pairs), a.batch):
                b = pairs[s : s + a.batch]
                if len(b) < 4:
                    continue
                av = emb(*batches(st.tok, [x["anchor"] for x in b]))
                pv = emb(*batches(st.tok, [x["positive"] for x in b]))
                tot += float(mnrl(av, pv)) * len(b)
            return tot / len(pairs)

    best, best_state = loss_on(val), {k: v.clone() for k, v in emb.state_dict().items()}
    print(f"epoch 0 val {best:.4f}", flush=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        random.shuffle(train)
        tot = 0.0
        for s in range(0, len(train), a.batch):
            b = train[s : s + a.batch]
            if len(b) < 4:
                continue
            opt.zero_grad()
            av = emb(*batches(st.tok, [x["anchor"] for x in b]))
            pv = emb(*batches(st.tok, [x["positive"] for x in b]))
            loss = mnrl(av, pv)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        v = loss_on(val)
        print(f"epoch {ep} train {tot/len(train):.4f} val {v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best, best_state = v, {k: v_.clone() for k, v_ in emb.state_dict().items()}
    emb.load_state_dict(best_state)
    out = StaticTable(emb.weight.detach().numpy(), st.tok)
    out.save(Path(a.out))
    json.dump({"best_val": best, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
               "n_train": len(train), "n_val": len(val), "rows": int(out.table.shape[0])},
              open(Path(a.out) / "train.json", "w"), indent=1)
    print(f"saved {a.out} best val {best:.4f}")


if __name__ == "__main__":
    main()
