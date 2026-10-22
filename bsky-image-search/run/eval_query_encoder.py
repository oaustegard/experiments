"""Score a small query encoder (an ONNX in EG2's space) on text->image search over the EG2 image index.

Usage: python3 -I run/eval_query_encoder.py --model k4_q4w=DIR/model_q4w.onnx [--model k2_t3w=...] \
           [--name cuts] [--budget 70]
       (each model's tokenizer.json sits beside its .onnx)

Same queries and index as build_index.py's eval: the 5,248 images whose alt text is 25-300
characters, alt text + EG2's query prompt as the query, the image as the one relevant result,
over all image vectors at the given vision budget, on fp32 vectors and on the 2-bit codes.
The encoder must take input_ids + attention_mask (int64) and return a unit 768-d
`embedding` (eg2-layer-student run/export.py). Full EG2 q4 text is re-run on the same
queries so the comparison is paired. Writes results/eval_<name>.json and per-query ranks.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import remex
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import QUERY, assemble, rx2_scores, text_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def onnx_encoder(onnx_path, tok_path, max_tokens=128, batch=64):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    out_name = sess.get_outputs()[0].name
    tok = Tokenizer.from_file(str(tok_path))
    tok.no_padding()
    tj = json.loads(Path(tok_path).read_text())
    specials = sorted((t["content"] for t in tj["added_tokens"] if t.get("special")), key=len, reverse=True)

    def clean(text):  # as eg2-query-student common.Teacher.clean: literal special tokens become spaces
        for sp in specials:
            text = text.replace(sp, " ")
        return text

    def embed(texts):  # texts already carry the prompt; clean only the part after it
        enc = [tok.encode(QUERY + clean(t[len(QUERY):]) if t.startswith(QUERY) else clean(t)).ids[:max_tokens] for t in texts]
        out = np.empty((len(texts), 768), np.float32)
        order = np.argsort([len(e) for e in enc])
        for s in range(0, len(order), batch):
            rows = order[s:s + batch]
            S = max(len(enc[r]) for r in rows)
            ids = np.zeros((len(rows), S), np.int64)
            att = np.zeros((len(rows), S), np.int64)
            for b, r in enumerate(rows):
                ids[b, :len(enc[r])] = enc[r]
                att[b, :len(enc[r])] = 1
            e = sess.run([out_name], {"input_ids": ids, "attention_mask": att})[0]
            out[rows] = e / np.linalg.norm(e, axis=1, keepdims=True)
        return out
    return embed


def ranks(S, target):
    return (S > S[np.arange(len(target)), target][:, None]).sum(1)


def summary(rank):
    return {"hit@1": float((rank < 1).mean()), "hit@10": float((rank < 10).mean()),
            "mrr@10": float(np.where(rank < 10, 1 / (rank + 1), 0).mean())}


def paired(a, b, n=2000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=path/to/model.onnx (tokenizer.json beside it)")
    ap.add_argument("--name", default="cuts")
    ap.add_argument("--budget", type=int, default=70)
    a = ap.parse_args()
    X, idx, rows = assemble(a.budget)
    qz = remex.Quantizer(768, bits=2, seed=42, mean=X.mean(0).astype(np.float32))
    cv = qz.encode(X)
    scale, bias = qz._score_norms(cv).astype(np.float32), qz._row_bias(cv).astype(np.float32)
    cand = [k for k, i in enumerate(idx) if 25 <= len(rows[i]["alt"].strip()) <= 300]
    random.Random(0).shuffle(cand)
    texts = [QUERY + rows[idx[k]]["alt"].strip() for k in cand]
    target = np.array(cand)
    models = dict(m.split("=", 1) for m in a.model)
    Qs = {"eg2": text_encoder().embed([rows[idx[k]]["alt"].strip() for k in cand], prompt=QUERY)}
    for name, path in models.items():
        Qs[name] = onnx_encoder(path, Path(path).with_name("tokenizer.json"))(texts)
    res = {"n_images": len(X), "n_queries": len(cand), "budget": a.budget, "models": models,
           "query_cos_to_eg2": {name: {"min": float((Qs[name] * Qs["eg2"]).sum(1).min()),
                                       "median": float(np.median((Qs[name] * Qs["eg2"]).sum(1)))} for name in models}}
    R = {}
    for name, Q in Qs.items():
        for rep, S in [("fp32", Q @ X.T), ("rx2", rx2_scores(qz, cv, scale, bias, Q))]:
            R[(name, rep)] = ranks(S, target)
            res.setdefault(name, {})[rep] = summary(R[(name, rep)])
    for name in models:
        for rep in ("fp32", "rx2"):
            for m, f in [("hit@1", lambda r: (r < 1).astype(float)), ("hit@10", lambda r: (r < 10).astype(float))]:
                res[name][rep][m + "_minus_eg2"] = paired(f(R[(name, rep)]), f(R[("eg2", rep)]))
    print(json.dumps(res, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"eval_{a.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
