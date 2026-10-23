"""Round 5: trade vocabulary size against bits at a fixed table budget.

Vocabularies: the 41,741 kept ids plus the most frequent other ids in training-side text
(.work/freq.npy, captions included), closed under BPE as eg2-query-student's a4_prune does, with
a pruned tokenizer.json. Rows: trained rows for the kept ids, rows from run/tok_rows.py for the
rest. Codec: remex (haar rotation, Lloyd-Max, fp16 norm) at b bits; "mixB" arms put the most
frequent rows at 3 bits and the rest at 2.
Usage: python3 -I run/r5_vocab_size.py [--arms ...]  -> results/r5_vocab_size.json (resumes)
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import remex
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, HID, LS, PREV, WORK, Benches, cut_rows, rows_student  # noqa: E402

sys.path.insert(0, str(PREV / "run"))
from a4_prune import kept_set, prune_tokenizer  # noqa: E402

torch.set_num_threads(4)
HF_TOK = PREV / "out" / "hf" / "tokenizer.json"
# name: (extra ids beyond the kept set before closure, bit spec)
ARMS = {
    "v42k_r3": (0, [(None, 3)]),
    "v52k_r3": (10000, [(None, 3)]),
    "v64k_r2": (22000, [(None, 2)]),
    "v78k_r2": (36000, [(None, 2)]),
    "v64k_mix": (22000, "budget"),
    "v78k_mix": (36000, "budget"),
    "v100k_mix": (58000, "budget"),
}
BUDGET = 10.0e6


def remex_rows(X, b):
    q = remex.Quantizer(HID, bits=b, seed=0, rotation="haar")
    return q.decode(q.encode(X)).astype(np.float32), len(X) * (HID * b // 8 + 2)


def vocab(tj, f, kept, extra):
    other = [(i, int(f[i])) for i in np.argsort(-f, kind="stable") if not kept[i] and f[i] > 0][:extra]
    c = collections.Counter({int(i): 10 ** 9 for i in np.where(kept)[0]})
    c.update(dict(other))
    keep = kept_set(tj, c, len(c))
    return np.array(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="ridge_e3")
    ap.add_argument("--arms", nargs="*")
    a = ap.parse_args()
    sd = torch.load(LS / ".work/mac/out/l1/mac_k4_v32k/final.pt", map_location="cpu")
    rows0 = cut_rows()
    X = np.load(WORK / f"table_{a.table}.npy")
    X[rows0] = sd["tm.embed_tokens.weight"].numpy()
    f = np.load(WORK / "freq.npy")
    kept = np.zeros(len(X), bool)
    kept[rows0] = True
    tj = json.loads(HF_TOK.read_text())
    B = Benches()
    p = HERE / "results" / "r5_vocab_size.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for name in a.arms or ARMS:
        if name in out:
            continue
        extra, spec = ARMS[name]
        rows = vocab(tj, f, kept, extra)
        assert kept[rows].sum() == len(rows0), "kept ids must survive"
        ptj, _ = prune_tokenizer(tj, [int(r) for r in rows])
        d = WORK / "vocab" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "tokenizer.json").write_text(json.dumps(ptj, ensure_ascii=False))
        np.save(d / "rows.npy", rows)
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        # rows in descending training frequency; kept rows first among equals
        pri = f[rows].astype(np.float64) + kept[rows] * 0.5
        order = np.argsort(-pri, kind="stable")
        R = np.empty((len(rows), HID), np.float32)
        nb = 0
        if spec == "budget":  # 3 bits for the first n3 rows, 2 for the rest, to the budget
            n3 = int((BUDGET - len(rows) * (HID * 2 // 8 + 2)) // (HID // 8))
            n3 = max(0, min(len(rows), n3))
            spec = [(n3, 3), (None, 2)]
        start = 0
        for n, b in spec:
            sel = order[start:] if n is None else order[start:start + n]
            start += len(sel)
            R[sel], bb = remex_rows(X[rows[sel]], b)
            nb += bb
        r = B.run(rows_student(sd, rows, R), (name, tok))
        vocab_list = [None] * len(ptj["model"]["vocab"])
        for t, i in ptj["model"]["vocab"].items():
            vocab_list[i] = t
        r.update({"n_rows": int(len(rows)), "spec": [(int(n) if n is not None else None, b) for n, b in spec],
                  "table_mb": round(nb / 1e6, 3),
                  "tokenizer_json_gz_mb": round(len(gzip.compress((d / "tokenizer.json").read_bytes(), 9)) / 1e6, 3),
                  "vocab_list_gz_mb": round(len(gzip.compress(json.dumps(vocab_list, ensure_ascii=False).encode(), 9)) / 1e6, 3)})
        out[name] = r
        print(name, r, flush=True)
        p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
