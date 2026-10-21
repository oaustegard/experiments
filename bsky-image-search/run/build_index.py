"""Turn the browser-computed EG2 image vectors into the page's 2-bit index, and score text->image search.

Usage: python3 -I run/build_index.py [--budget 70] [--eval 1000]

1. Assemble: out/emb/img_<a>_<b>.f32 + .idx.json (row numbers into out/images.jsonl) -> X [n,768].
2. Quantize: remex Quantizer(768, bits=2, seed=42, mean=X.mean) as the Bluesky text index did
   (embeddinggemma2-webgpu-tail/run/build_bsky_index.py): 192 B codes + (scale, bias) per image.
   Writes out/index/{codes.u8.bin, aux.f32.bin, rows.json, index.json}; rows.json carries only
   what the page shows (cid, did, rkey, handle, likes, thumb, aspect), never alt or post text.
3. Eval (--eval N > 0): N images whose alt text is 25-300 characters; the alt text, with EG2's
   query prompt, is the query and its own image the one relevant result. Query vectors from EG2
   q4 text ONNX on CPU (the same graph the page would run). hit@10 and MRR@10 over fp32 vectors
   and over the 2-bit codes. results/eval.json. Alt text describes the image, so this is an
   easy, caption-like query set: an upper bound on typed-query quality, not an estimate of it.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import remex
from remex.packing import pack

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
EXP = ROOT.parent
QUERY = "task: search result | query: "


def assemble(budget):
    rows = [json.loads(l) for l in (OUT / "images.jsonl").open()]
    vec, idx = [], []
    for f in sorted((OUT / "emb").glob(f"img_b{budget}_*.f32")):  # one vision token budget per index
        v = np.fromfile(f, np.float32).reshape(-1, 768)
        ids = json.loads(f.with_suffix(".idx.json").read_text())
        assert len(ids) == len(v), f
        vec.append(v)
        idx += ids
    X, idx = np.concatenate(vec), np.array(idx)
    order = np.argsort(idx)
    X, idx = X[order], idx[order]
    assert len(np.unique(idx)) == len(idx), "duplicate rows across upload files"
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, idx, rows


def rx2_scores(qz, cv, scale, bias, Q):
    dirs = qz.centroids.astype(np.float32)[cv.indices]
    return (qz._rotate_rows(Q).astype(np.float32) @ dirs.T) * scale + bias


def text_encoder():
    sys.path.insert(0, str(EXP / "eg2-query-student" / "run"))
    import common  # noqa: E402  (EG2 q4 text ONNX on CPU; reads common.HF)
    common.HF = OUT / "hf"
    if not (common.HF / "model_q4.onnx_data").exists():
        from huggingface_hub import hf_hub_download
        common.HF.mkdir(parents=True, exist_ok=True)
        for f in ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data", "tokenizer.json"]:
            p = hf_hub_download("onnx-community/embeddinggemma-2-ONNX", f, local_dir=str(OUT / "hf_dl"))
            (common.HF / Path(f).name).write_bytes(Path(p).read_bytes())
    return common.Teacher()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=70, help="vision soft tokens the vectors were made with")
    a = ap.parse_args()
    X, idx, rows = assemble(a.budget)
    print(f"{len(X)} image vectors (of {len(rows)} rows)", flush=True)
    qz = remex.Quantizer(768, bits=2, seed=42, mean=X.mean(0).astype(np.float32))
    cv = qz.encode(X)
    scale, bias = qz._score_norms(cv).astype(np.float32), qz._row_bias(cv).astype(np.float32)
    d = OUT / "index"
    d.mkdir(exist_ok=True)
    (d / "codes.u8.bin").write_bytes(np.ascontiguousarray(pack(cv.indices.astype(np.uint8), 2).reshape(len(X), 192)).tobytes())
    (d / "aux.f32.bin").write_bytes(np.stack([scale, bias], 1).astype("<f4").tobytes())
    keep = ["cid", "did", "rkey", "handle", "likes", "thumb", "aspect"]
    (d / "rows.json").write_text(json.dumps([{k: rows[i][k] for k in keep} for i in idx], separators=(",", ":")))
    (d / "index.json").write_text(json.dumps({"n": len(X), "d": 768, "bits": 2, "seed": 42, "rotation": "rht",
        "remex_version": remex.__version__, "model": "onnx-community/embeddinggemma-2-ONNX q4 (vision_encoder + model), transformers.js 4.3.1 WebGPU",
        "vision_soft_tokens": a.budget,
        "query_prompt": QUERY, "mu": [float(v) for v in X.mean(0)]}, separators=(",", ":")))
    print("index written:", {f.name: f.stat().st_size for f in d.iterdir()}, flush=True)
    if not a.eval:
        return
    cand = [k for k, i in enumerate(idx) if 25 <= len(rows[i]["alt"].strip()) <= 300]
    random.Random(0).shuffle(cand)
    cand = cand[: a.eval]
    T = text_encoder()
    Q = T.embed([rows[idx[k]]["alt"].strip() for k in cand], prompt=QUERY)
    target = np.array(cand)
    res = {"n_images": len(X), "n_queries": len(cand), "query": "alt text of the image, EG2 query prompt"}
    for rep, S in [("fp32", Q @ X.T), ("rx2", rx2_scores(qz, cv, scale, bias, Q))]:
        rank = (S > S[np.arange(len(target)), target][:, None]).sum(1)  # 0 = best
        res[rep] = {"hit@1": float((rank < 1).mean()), "hit@10": float((rank < 10).mean()),
                    "mrr@10": float(np.where(rank < 10, 1 / (rank + 1), 0).mean())}
        print(rep, res[rep], flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
