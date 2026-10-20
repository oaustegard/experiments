"""A4: EmbeddingGemma 2 itself with its 262,144-token vocabulary cut to the K tokens this
domain uses. It starts in EG2's space, so the question is only what the cut costs.

Usage: python3 -I run/a4_prune.py [--ks 8000 16000 32000 64000]

1. Token frequencies over the 61,686 post texts plus the training queries (never the tests).
2. Kept set = top-K by frequency + every added/special token + the 256 byte tokens, closed
   under BPE: each kept token's parts (from the merge that builds it) are kept too, so every
   kept token stays reachable.
3. Pruned tokenizer.json: vocab renumbered in original id order, merges whose result was
   cut are dropped, byte_fallback on, so a word whose token was cut splits into kept pieces
   or bytes. Pruned ONNX: the three embed_tokens initializers sliced to the kept rows. All
   other weights untouched.
4. For each K: top-10 overlap with full EG2 on both test sets, the share of test queries
   whose full-vocab tokenization survives intact, and the download (ONNX + gzip tokenizer).
Writes out/a4/k<K>/ and results/a4.json.
"""
import argparse
import collections
import copy
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_linear import kinds, score  # noqa: E402
from common import HF, QUERY, Index, Teacher  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "out" / "queries"
EMB = ["model_embed_tokens_weight_quant", "model_embed_tokens_weight_scales", "model_embed_tokens_weight_zp"]


def texts(name):
    return [json.loads(l)["q"] for l in (Q / f"{name}.jsonl").open()]


def frequencies(T):
    posts = [json.loads(l)["text"] for l in (ROOT / "out" / "posts.jsonl").open()]
    docs = ["title: none | text: " + T.clean(p) for p in posts if p]
    qs = [QUERY + T.clean(q) for q in texts("msmarco_train") + texts("synth_train")]
    c = collections.Counter()
    for enc in T.tok.encode_batch(docs + qs):
        c.update(enc.ids)
    return c


def kept_set(tj, freq, k):
    vocab = tj["model"]["vocab"]
    always = {a["id"] for a in tj["added_tokens"]} | {i for t, i in vocab.items() if t.startswith("<0x")}
    keep = always | {i for i, _ in freq.most_common(k)}
    inv = {i: t for t, i in vocab.items()}
    built = {}
    for m in tj["model"]["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        built.setdefault(a + b, (a, b))
    stack = [inv[i] for i in keep]
    while stack:
        t = stack.pop()
        for part in built.get(t, ()):
            if vocab[part] not in keep:
                keep.add(vocab[part])
                stack.append(part)
    return sorted(keep)


def prune_tokenizer(tj, keep):
    tj = copy.deepcopy(tj)
    old = tj["model"]["vocab"]
    new_id = {o: n for n, o in enumerate(keep)}
    inv = {i: t for t, i in old.items()}
    tj["model"]["vocab"] = {inv[o]: new_id[o] for o in keep}
    kv = tj["model"]["vocab"]
    ms = []
    for m in tj["model"]["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        if a in kv and b in kv and a + b in kv:
            ms.append(m)
    tj["model"]["merges"] = ms
    for a in tj["added_tokens"]:
        a["id"] = new_id[a["id"]]
    return tj, new_id


def prune_onnx(keep, dst):
    m = onnx.load(str(HF / "model_q4.onnx"))
    rows = np.array(keep)
    for init in m.graph.initializer:
        if init.name in EMB:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init)[rows], init.name))
    onnx.save(m, str(dst / "model_q4.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
              location="model_q4.onnx_data", size_threshold=1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=[8000, 16000, 32000, 64000])
    a = ap.parse_args()
    T = Teacher()
    tj = json.loads((HF / "tokenizer.json").read_text())
    freq = frequencies(T)
    ix = Index()
    tests = {t: (texts(t), np.load(Q / f"{t}.eg2.npy"), kinds(t)) for t in ["real_test", "synth_test"]}
    full_bytes = (HF / "model_q4.onnx").stat().st_size + (HF / "model_q4.onnx_data").stat().st_size
    res = {"full": {"onnx_bytes": full_bytes, "tokenizer_gz_bytes": len(gzip.compress((HF / "tokenizer.json").read_bytes())),
                    "vocab": len(tj["model"]["vocab"]), "distinct_tokens_seen": len(freq)}, "k": {}}
    for k in a.ks:
        dst = ROOT / "out" / "a4" / f"k{k}"
        dst.mkdir(parents=True, exist_ok=True)
        keep = kept_set(tj, freq, k)
        ptj, new_id = prune_tokenizer(tj, keep)
        (dst / "tokenizer.json").write_text(json.dumps(ptj, ensure_ascii=False))
        prune_onnx(keep, dst)
        P = Teacher.__new__(Teacher)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        P.sess = ort.InferenceSession(str(dst / "model_q4.onnx"), so, providers=["CPUExecutionProvider"])
        P.tok = Tokenizer.from_file(str(dst / "tokenizer.json"))
        P.specials = T.specials
        r = {"vocab": len(keep),
             "onnx_bytes": (dst / "model_q4.onnx").stat().st_size + (dst / "model_q4.onnx_data").stat().st_size,
             "tokenizer_gz_bytes": len(gzip.compress((dst / "tokenizer.json").read_bytes()))}
        for t, (qs, E, kd) in tests.items():
            intact = np.mean([all(i in new_id for i in T.tok.encode(QUERY + T.clean(q)).ids) for q in qs])
            r[t] = score(ix, P.embed(qs), E, kd) | {"intact_share": float(intact)}
        res["k"][k] = r
        print(k, json.dumps({kk: (v["overlap@10"] if isinstance(v, dict) else v) for kk, v in r.items()}), flush=True)
        (ROOT / "results" / "a4.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
