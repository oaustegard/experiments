"""Paired comparison of the arms: per-query nDCG@10 deltas with bootstrap 95% intervals, and a slice by
whether the query holds a word Ettin's tokenizer splits into >= 3 pieces that the arm's keeps whole.

    python3 -I run/compare.py   # results/compare.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from evalset import load  # noqa: E402

R, W = HERE / "results", HERE / ".work"
TOK = {"stock": W / "e17c/epoch1/tokenizer.json", "ls32": W / "tok/ls32/tokenizer.json", "ls50": W / "tok/ls50/tokenizer.json"}
WORD = re.compile(r"[\w][\w\-/,.'+α-ωΑ-Ω]*[\w]")


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [round(float(d.mean()), 4), round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def pq(name, s):
    f = R / f"eval_{name}.json"
    return np.array(json.loads(f.read_text())[s]["per_query_ndcg"]) if f.exists() and s in json.loads(f.read_text()) else None


def main():
    toks = {k: Tokenizer.from_file(str(v)) for k, v in TOK.items()}
    for t in toks.values():
        t.no_truncation(); t.no_padding()
    out = {}
    for s in ["scifact", "nfcorpus", "trec-covid"]:
        _, q, _ = load(s)
        base = pq("stock_e2", s)
        if base is None:
            continue
        o = {"n_queries": len(base)}
        for ref in ["e17c", "medcpt", "bm25"]:
            x = pq(ref, s)
            if x is not None:
                o[f"stock_e2_minus_{ref}"] = boot(base - x)
        for arm in ["ls32", "ls50"]:
            x = pq(f"{arm}_e2", s)
            if x is None:
                continue
            d = x - base
            o[f"{arm}_minus_stock"] = boot(d)
            frag = []
            for text in q["text"]:
                hit = False
                for w in WORD.findall(text):
                    if len(toks["stock"].encode(" " + w, add_special_tokens=False).ids) >= 3 and len(toks[arm].encode(" " + w, add_special_tokens=False).ids) == 1:
                        hit = True; break
                frag.append(hit)
            frag = np.array(frag)
            o[f"{arm}_slice"] = {"n_with_term": int(frag.sum()), "delta_with_term": boot(d[frag]) if frag.sum() > 1 else None,
                                 "delta_without": boot(d[~frag]) if (~frag).sum() > 1 else None}
        out[s] = o
        print(s, json.dumps(o), flush=True)
    (R / "compare.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
