"""Screen off-the-shelf Ettin-based embedders on eg2-native-small's Bluesky task (torch, MPS, fp32).

Usage: .venv/bin/python -I run/screen.py [hf_id ...]     (default: CANDIDATES)

Same task, split and scoring as ../eg2-native-small/run/native.py: 61,686 AI-list posts (docs
truncated to 128 tokens), 1,498 held-out Haiku-synthetic queries, hit@10 = source post, on the
arm's own fp32 vectors and its own remex 2-bit codes, paired against EG2's committed codes.
This is a screen: fp32 torch, no ONNX/int8. Finalists go through run/compress.py.
Writes results/screen.json (merged per model).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT.parent
NS = EXP / "eg2-native-small"
Q = NS / "out" / "q"
POSTS = EXP / "eg2-query-student" / "out" / "posts.jsonl"
sys.path.insert(0, str(EXP / "eg2-query-student" / "run"))
from common import Index, boot_ci, overlap  # noqa: E402

import importlib.util  # load native.py's helpers by path (it has its own run/ dir)
_s = importlib.util.spec_from_file_location("native", NS / "run" / "native.py")
native = importlib.util.module_from_spec(_s)
_s.loader.exec_module(native)

CANDIDATES = [
    "MongoDB/mdbr-leaf-ir",  # calibration: native.py's int8 ONNX got 0.889
    "tasksource/ettin-17m-embed",
    "tomaarsen/ettin-encoder-17m-cmnrl-lr2e-05-bs128",
    "much1na/ettin-encoder-17m-v3-vendor-embeddistill-v2",
    "much1na/ettin-encoder-17m-gooaq-v2",
    "chungimungi/ettin-glint-kd",
    "capemox/ettin-encoder-17m-pretrain-50m",
    "tasksource/ettin-32m-embed",
    "tomaarsen/ettin-encoder-32m-nq",
    "much1na/ettin-32m-distil-gte-v2",
    "capemox/ettin-encoder-32m-pretrain-50m",
    "capemox/ettin-encoder-68m-contrastive-2.5m",
]


def perquery(arm, h):
    """SYM per-query hit@10 bools, synth_test file order (coord #20)."""
    (ROOT / "results" / f"{arm}_perquery.json").write_text(json.dumps(
        {"metric": "SYM hit@10, own remex 2-bit codes", "n": int(len(h)), "hits": [bool(x) for x in h]}))


def prompts(m):
    p = m.prompts or {}
    q = p.get("query", p.get("search_query", ""))
    d = p.get("document", p.get("passage", p.get("search_document", "")))
    return q, d


def main():
    ids = sys.argv[1:] or CANDIDATES
    posts = [json.loads(l) for l in POSTS.open()]
    texts = [p["text"] or "" for p in posts]
    synth = [json.loads(l) for l in (Q / "synth_test.jsonl").open()]
    real = [json.loads(l) for l in (Q / "real_test.jsonl").open()]
    src = np.array([r["post"] for r in synth])
    ix = Index()
    eg2_s = ix.topk(np.load(Q / "synth_test.eg2.npy"))
    eg2_r = ix.topk(np.load(Q / "real_test.eg2.npy"))
    hit_eg2 = (eg2_s == src[:, None]).any(1).astype(float)

    path = ROOT / "results" / "screen.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for hf in ids:
        m = SentenceTransformer(hf, device=dev, trust_remote_code=False)
        m.max_seq_length = native.MAX_TOKENS
        qp, dp = prompts(m)
        if hf == "MongoDB/mdbr-leaf-ir":
            qp, dp = native.LEAF_Q, ""
        nparam = sum(p.numel() for p in m.parameters())
        t0 = time.time()
        X = m.encode([dp + t for t in texts], batch_size=128, normalize_embeddings=True,
                     convert_to_numpy=True).astype(np.float32)
        tdoc = time.time() - t0
        Qs = m.encode([qp + r["q"] for r in synth], batch_size=256, normalize_embeddings=True).astype(np.float32)
        Qr = m.encode([qp + r["q"] for r in real], batch_size=256, normalize_embeddings=True).astype(np.float32)
        cache = ROOT / "out" / "screen" / hf.replace("/", "__")
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "docs.npy", X)
        np.save(cache / "synth.npy", Qs)
        np.save(cache / "real.npy", Qr)
        r = {"params_m": round(nparam / 1e6, 1), "dim": X.shape[1], "query_prompt": qp, "doc_prompt": dp,
             "doc_encode_s_mps": round(tdoc)}
        for rep in ("fp32", "rx2"):
            if rep == "fp32":
                ts, tr = native.topk(Qs @ X.T), native.topk(Qr @ X.T)
            else:
                (ts, _), (tr, _) = native.rx2_topk(X, Qs), native.rx2_topk(X, Qr)
            h = (ts == src[:, None]).any(1).astype(float)
            if rep == "rx2":
                perquery(f"screen_{hf.split('/')[-1]}", h)
            r[rep] = {"hit@10": boot_ci(h), "minus_eg2": native.paired(h, hit_eg2),
                      "hit@1": float((ts[:, 0] == src).mean()),
                      "real_overlap_with_eg2": boot_ci(overlap(tr, eg2_r))}
        res[hf] = r
        print(hf, json.dumps(r), flush=True)
        path.write_text(json.dumps(res, indent=1))
        del m
        if dev == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
