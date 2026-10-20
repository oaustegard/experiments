"""Arm 1 (table quantization), arm 2 (coverage pruning with full-table fallback), and both
combined, against fp32 EG2 on the 2,000 held-out queries and the committed Bluesky index.

Usage: python3 -I run/arms.py

Arm 1 replaces embed_tokens.weight (262,144 x 512) with a quantize-dequantize copy. EG2
multiplies the looked-up row by sqrt(512) after the lookup, so swapping the weight
dequantizes before that scaling. Every other weight stays fp32. Variants:
  int8_row        symmetric absmax, one scale per row (the arm as specified)
  int4_row        the same at 4 bits
  intN_g32        asymmetric min/max per group of 32 values (the layout model_q4.onnx uses)
  remexN          remex Quantizer(512, bits=N): random rotation, Lloyd-Max codebook, one norm per row
  remex2_centered the same with the table's mean row subtracted first
Arm 2 keeps every token id that the full tokenizer emits on the corpus posts (document
prompt) and the training queries (query prompt), plus every added token and the 256 byte
tokens, closed under BPE merges so each kept token's parts are kept. A query whose full
tokenization stays inside the kept set runs on the pruned table with remapped ids; any
other query is routed to the full table. Results go to results/arms.json, checkpointed per
variant; per-query cosines and overlaps to .work/perquery/.
"""
import collections
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import remex
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EG2, HF, Q, QUERY, ROOT, WORK, Index, compare, summary, test_queries  # noqa: E402

OUT = ROOT / "results" / "arms.json"
PQ = WORK / "perquery"
ROWS, D = 262144, 512


def q_row(W, bits):
    m = 2 ** (bits - 1) - 1
    s = np.abs(W).max(1, keepdims=True) / m
    s[s == 0] = 1
    return (np.clip(np.round(W / s), -m, m) * s).astype(np.float32), ROWS * (D * bits / 8 + 2)


def q_group(W, bits, g=32):
    G = W.reshape(-1, g)
    lo, hi = G.min(1, keepdims=True), G.max(1, keepdims=True)
    s = (hi - lo) / (2 ** bits - 1)
    s[s == 0] = 1
    return (np.round((G - lo) / s) * s + lo).reshape(W.shape).astype(np.float32), ROWS * (D * bits / 8 + D / g * 4)


def q_remex(W, bits, centered=False):
    mu = W.mean(0) if centered else None
    qz = remex.Quantizer(D, bits=bits, seed=0, mean=mu) if centered else remex.Quantizer(D, bits=bits, seed=0)
    out = np.empty_like(W)
    for s in range(0, ROWS, 16384):
        out[s:s + 16384] = qz.decode(qz.encode(W[s:s + 16384]))
    return out, ROWS * (D * bits / 8 + 2) + (D * 2 if centered else 0)


VARIANTS = {
    "int8_row": lambda W: q_row(W, 8),
    "int4_row": lambda W: q_row(W, 4),
    "int4_g32": lambda W: q_group(W, 4),
    "int3_g32": lambda W: q_group(W, 3),
    "int2_g32": lambda W: q_group(W, 2),
    "remex4": lambda W: q_remex(W, 4),
    "remex3": lambda W: q_remex(W, 3),
    "remex2": lambda W: q_remex(W, 2),
    "remex2_centered": lambda W: q_remex(W, 2, True),
}


def coverage(m):
    """Kept ids: tokens seen in corpus + training queries, added tokens, byte tokens, merge closure."""
    posts = [json.loads(l)["text"] for l in (WORK / "posts.jsonl").open()]
    docs = ["title: none | text: " + p for p in posts if p]
    qs = [QUERY + json.loads(l)["q"] for f in ["msmarco_train", "synth_train", "gpu_more"]
          for l in (Q / f"{f}.jsonl").open()]
    seen = set()
    for s in range(0, len(docs + qs), 4096):
        for ids in m.tok((docs + qs)[s:s + 4096], add_special_tokens=True)["input_ids"]:
            seen.update(ids)
    tj = json.loads((HF / "tokenizer.json").read_text())
    vocab = tj["model"]["vocab"]
    keep = seen | {a["id"] for a in tj["added_tokens"]} | {i for t, i in vocab.items() if t.startswith("<0x")}
    inv = {i: t for t, i in vocab.items()}
    built = {}
    for mg in tj["model"]["merges"]:
        a, b = mg if isinstance(mg, list) else mg.split(" ", 1)
        built.setdefault(a + b, (a, b))
    stack = [inv[i] for i in keep if i in inv]
    while stack:
        t = stack.pop()
        for part in built.get(t, ()):
            if vocab[part] not in keep:
                keep.add(vocab[part])
                stack.append(part)
    return np.array(sorted(keep)), {"posts": len(docs), "train_queries": len(qs), "seen": len(seen),
                                    "after_closure": len(keep), "tokenizer": tj}


def pruned_tokenizer(tj, keep):
    """eg2-query-student's rule: renumber kept tokens, drop merges whose parts or result were
    cut, byte fallback on. Used only to show what a fallback query becomes without the fallback."""
    from tokenizers import Tokenizer
    import copy
    tj = copy.deepcopy(tj)
    inv = {i: t for t, i in tj["model"]["vocab"].items()}
    new = {int(o): n for n, o in enumerate(keep)}
    tj["model"]["vocab"] = {inv[o]: new[o] for o in new if o in inv}
    kv = tj["model"]["vocab"]
    tj["model"]["merges"] = [mg for mg in tj["model"]["merges"]
                             if (lambda a, b: a in kv and b in kv and a + b in kv)(*(mg if isinstance(mg, list) else mg.split(" ", 1)))]
    tj["model"]["byte_fallback"] = True
    for a in tj["added_tokens"]:
        a["id"] = new[a["id"]]
    return Tokenizer.from_str(json.dumps(tj)), new


def save(res):
    OUT.write_text(json.dumps(res, indent=1))


def main():
    PQ.mkdir(parents=True, exist_ok=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    qs = test_queries()
    texts = [q["q"] for q in qs]
    kind = np.array(["real" if q["kind"] != "synth" else "synth" for q in qs])
    m, ix = EG2(), Index()
    W = m.full.numpy()
    ids = m.ids(texts)

    t = time.time()
    R = m.embed_ids(ids)
    top_r = ix.topk(R)
    res["reference"] = {"model": "google/embeddinggemma-2 bf16 checkpoint, fp32 compute", "n": len(texts),
                        "real": int((kind == "real").sum()), "synth": int((kind == "synth").sum()),
                        "table_shape": [ROWS, D], "table_bytes_bf16": ROWS * D * 2, "embed_s": time.time() - t}

    # Context: the corpus-side encoder (model_q4.onnx), whose query targets are stored in R2.
    pick = sorted(np.random.default_rng(0).choice(1498, 1000, replace=False))
    q4 = np.concatenate([np.load(Q / "real_test.eg2.npy"), np.load(Q / "synth_test.eg2.npy")[pick]])
    c, o = compare(q4, R, ix.topk(q4), top_r)
    res["baseline_q4_onnx"] = summary(c, o)
    save(res)

    def run(name, table, route_ids=None):
        m.set_table(table)
        E = m.embed_ids(route_ids or ids)
        c, o = compare(E, R, ix.topk(E), top_r)
        np.savez(PQ / f"{name}.npz", cos=c, ov=o)
        out = summary(c, o) | {k: summary(c[kind == k], o[kind == k]) for k in ["real", "synth"]}
        return out, c, o

    res.setdefault("arm1", {})
    for name, f in VARIANTS.items():
        if name in res["arm1"]:
            continue
        Wq, nbytes = f(W)
        out, _, _ = run(f"arm1_{name}", Wq)
        res["arm1"][name] = out | {"table_bytes": int(nbytes), "table_rmse_rel": float(
            np.sqrt(np.mean((Wq - W) ** 2)) / np.sqrt(np.mean(W ** 2)))}
        save(res)
        print(name, res["arm1"][name]["overlap@10"], res["arm1"][name]["cos_p01"], flush=True)
    m.set_table(W)

    keep, info = coverage(m)
    tj = info.pop("tokenizer")
    remap = np.full(ROWS, -1, np.int64)
    remap[keep] = np.arange(len(keep))
    covered = np.array([all(remap[i] >= 0 for i in x) for x in ids])
    pids = [[int(remap[i]) for i in x] if cov else x for x, cov in zip(ids, covered)]

    def arm2(name, Wfull):
        # Covered queries run on the pruned table; fallback queries on the full table.
        Ep = np.empty((len(ids), 768), np.float32)
        sel = np.flatnonzero(covered)
        m.set_table(Wfull[keep])
        Ep[sel] = m.embed_ids([pids[i] for i in sel])
        fb = np.flatnonzero(~covered)
        m.set_table(Wfull)
        if len(fb):
            Ep[fb] = m.embed_ids([ids[i] for i in fb])
        c, o = compare(Ep, R, ix.topk(Ep), top_r)
        np.savez(PQ / f"{name}.npz", cos=c, ov=o, covered=covered)
        return {"all": summary(c, o), "covered_route": summary(c[covered], o[covered]),
                "fallback_route": summary(c[~covered], o[~covered]) if (~covered).any() else None}

    tok_p, new = pruned_tokenizer(tj, keep)
    fb = np.flatnonzero(~covered)
    m.set_table(W[keep])
    En = m.embed_ids([tok_p.encode(QUERY + texts[i]).ids[:128] for i in fb]) if len(fb) else None
    nofb = None
    if En is not None:
        c, o = compare(En, R[fb], ix.topk(En), top_r[fb])
        nofb = summary(c, o)
    same = np.mean([tok_p.encode(QUERY + texts[i]).ids[:128] == pids[i] for i in np.flatnonzero(covered)])
    res["arm2"] = {"coverage": info | {"kept_rows": int(len(keep)),
                                       "table_bytes_bf16": int(len(keep) * D * 2),
                                       "pruned_tokenizer_gz_bytes": len(gzip.compress(tok_p.to_str().encode())),
                                       "full_tokenizer_gz_bytes": len(gzip.compress((HF / "tokenizer.json").read_bytes()))},
                   "fallback_share": float((~covered).mean()),
                   "fallback_share_real": float((~covered[kind == "real"]).mean()),
                   "fallback_share_synth": float((~covered[kind == "synth"]).mean()),
                   "pruned_tokenizer_matches_full_on_covered": float(same),
                   "fp32": arm2("arm2_fp32", W),
                   "fallback_queries_without_fallback": nofb}
    save(res)
    print("arm2", res["arm2"]["fallback_share"], res["arm2"]["fp32"]["all"]["overlap@10"], flush=True)

    res["combined"] = {}
    for name in ["int8_row", "int4_g32", "remex4", "remex3"]:
        Wq, nbytes = VARIANTS[name](W)
        r = arm2(f"combined_{name}", Wq)
        r["table_bytes"] = int(len(keep) * nbytes / ROWS)
        res["combined"][name] = r
        save(res)
        print("combined", name, r["all"]["overlap@10"], r["all"]["cos_p01"], flush=True)


if __name__ == "__main__":
    main()
