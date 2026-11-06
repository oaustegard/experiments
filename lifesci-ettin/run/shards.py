"""Training inputs for the bridge, per arm, and the shared batch order.

    python3 -I run/shards.py   # .work/shards/{order.u32, <arm>.ids.u16, <arm>.lens.u16, meta.json}

Each arm tokenizes the same rows (queries with e17c's query prompt), at most 128 tokens with [CLS]/[SEP].
The batch order is one list for every arm: per epoch a seeded shuffle, cut into chunks of 50 batches,
each chunk sorted by character length and cut into batches of 128, the batches shuffled (ettin-native's
distill.py recipe, on characters so it does not depend on the tokenizer). The last partial batch of an
epoch is dropped.
"""
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from model import QP  # noqa: E402

W = HERE / ".work"
OUT = W / "shards"
ARMS = {"stock": W / "e17c/epoch1/tokenizer.json", "ls32": W / "tok/ls32/tokenizer.json", "ls50": W / "tok/ls50/tokenizer.json"}
BS, EPOCHS, MAXLEN = 128, 2, 128


def tok(path):
    t = Tokenizer.from_file(str(path))
    t.no_padding()
    t.enable_truncation(MAXLEN)
    return t


def main():
    rows = [json.loads(l) for l in (W / "data/train.jsonl").open()]
    texts = [(QP if r["role"] == "q" else "") + r["t"] for r in rows]
    clen = np.array([len(t) for t in texts])
    rng = np.random.default_rng(0)
    order = []
    for ep in range(EPOCHS):
        perm = rng.permutation(len(rows))
        batches = []
        for i in range(0, len(perm), 50 * BS):
            part = perm[i:i + 50 * BS]
            part = part[np.argsort(clen[part], kind="stable")]
            batches += [part[j:j + BS] for j in range(0, len(part), BS) if len(part[j:j + BS]) == BS]
        for k in rng.permutation(len(batches)):
            order.append(batches[k])
    order = np.concatenate(order).astype(np.uint32)
    OUT.mkdir(parents=True, exist_ok=True)
    order.tofile(OUT / "order.u32")
    meta = {"n_rows": len(rows), "bs": BS, "epochs": EPOCHS, "steps": len(order) // BS, "steps_per_epoch": len(order) // BS // EPOCHS, "maxlen": MAXLEN, "arms": {}}
    for arm, p in ARMS.items():
        t = tok(p)
        lens, ids = [], []
        for i in range(0, len(texts), 20000):
            enc = t.encode_batch(texts[i:i + 20000])
            lens += [len(e.ids) for e in enc]
            ids.append(np.concatenate([np.array(e.ids, np.uint16) for e in enc]))
            del enc
        lens, ids = np.array(lens, np.uint16), np.concatenate(ids)
        ids.tofile(OUT / f"{arm}.ids.u16")
        lens.tofile(OUT / f"{arm}.lens.u16")
        meta["arms"][arm] = {"vocab": t.get_vocab_size(), "pad": t.token_to_id("[PAD]"), "tokens": int(lens.sum()),
                             "mean_len_q": float(lens[[r["role"] == "q" for r in rows]].mean()),
                             "mean_len_d": float(lens[[r["role"] == "d" for r in rows]].mean()),
                             "truncated_frac": float((lens == MAXLEN).mean())}
        print(arm, meta["arms"][arm], flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print({k: v for k, v in meta.items() if k != "arms"})


if __name__ == "__main__":
    main()
