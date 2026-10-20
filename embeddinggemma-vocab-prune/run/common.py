"""Shared pieces: the fp32 EG2 reference, the committed 2-bit Bluesky index, the test queries.

The doc side is fixed: the 61,686 posts in ../embeddinggemma2-webgpu-tail/web/bsky, encoded
once by EG2 model_q4.onnx into remex 2-bit codes. Every arm here changes only the query
encoder's token embedding table (and, for arm 2, which rows of it exist).

Reference: google/embeddinggemma-2 (bf16 checkpoint) run in float32 through
sentence-transformers, query prompt "task: search result | query: ", mean pooling over all
tokens including the prompt, 768-d, L2-normalized: the same prompt, width and normalization
as the corpus-side query targets (eg2-query-student).
"""
import json
from pathlib import Path

import numpy as np
import remex
import torch
from remex.packing import unpack

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / ".work"
HF = WORK / "hf"
Q = WORK / "queries"
IDX = ROOT.parent / "embeddinggemma2-webgpu-tail" / "web" / "bsky"
QUERY = "task: search result | query: "
K = 10
torch.set_num_threads(4)


class Index:
    """The committed codes, scored as remex search_adc scores them (copied from
    eg2-query-student/run/common.py; the per-query constant q . mu is dropped)."""

    def __init__(self):
        meta = json.loads((IDX / "index.json").read_text())
        n, d = meta["n"], meta["d"]
        mu = np.array(json.loads(meta["mu"]) if isinstance(meta["mu"], str) else meta["mu"], np.float32)
        self.qz = remex.Quantizer(d, bits=2, seed=42, mean=mu)
        packed = np.frombuffer((IDX / "codes.u8.txt").read_bytes(), np.uint8).reshape(n, -1)
        idx = unpack(packed.reshape(-1), 2, n * d).reshape(n, d)
        aux = np.frombuffer((IDX / "aux.f32.txt").read_bytes(), "<f4").reshape(n, 2)
        self.scale, self.bias = aux[:, 0].copy(), aux[:, 1].copy()
        self.dirs = self.qz.centroids.astype(np.float32)[idx]

    def topk(self, Q, k=K, chunk=256):
        out = np.empty((len(Q), k), np.int64)
        for s in range(0, len(Q), chunk):
            Qr = self.qz._rotate_rows(np.asarray(Q[s:s + chunk], np.float32)).astype(np.float32)
            sc = (Qr @ self.dirs.T) * self.scale + self.bias
            part = np.argpartition(-sc, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sc, part, 1), axis=1)
            out[s:s + chunk] = np.take_along_axis(part, order, 1)
        return out


def test_queries(n_synth=1000, seed=0):
    """The 2,000 held-out queries: all 1,000 real (MS MARCO tech/AI, held out by topic) and
    1,000 of the 1,498 synthetic test queries (Haiku, from posts in the held-out pool)."""
    real = [json.loads(l) for l in (Q / "real_test.jsonl").open()]
    synth = [json.loads(l) for l in (Q / "synth_test.jsonl").open()]
    pick = sorted(np.random.default_rng(seed).choice(len(synth), n_synth, replace=False))
    return real + [synth[i] for i in pick]


class EG2:
    """fp32 EG2 whose token table can be swapped. Tokenization is done here so a caller can
    feed ids from a pruned tokenizer; embed(texts) uses the full tokenizer."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.st = SentenceTransformer(str(HF), device="cpu", model_kwargs={"dtype": torch.float32})
        self.st.eval()
        self.emb = next(m for n, m in self.st.named_modules() if n.endswith("embed_tokens"))
        self.full = self.emb.weight.data.clone()
        self.tok = self.st.tokenizer

    def set_table(self, W):
        self.emb.weight.data = torch.as_tensor(W, dtype=torch.float32)

    def ids(self, texts):
        return [self.tok(QUERY + t, truncation=True, max_length=128)["input_ids"] for t in texts]

    @torch.inference_mode()
    def embed_ids(self, ids, batch=64):
        order = np.argsort([len(x) for x in ids])
        out = np.empty((len(ids), 768), np.float32)
        for s in range(0, len(ids), batch):
            sel = order[s:s + batch]
            L = max(len(ids[i]) for i in sel)
            inp = torch.zeros((len(sel), L), dtype=torch.long)
            att = torch.zeros((len(sel), L), dtype=torch.long)
            for r, i in enumerate(sel):
                inp[r, :len(ids[i])] = torch.tensor(ids[i])
                att[r, :len(ids[i])] = 1
            v = self.st({"input_ids": inp, "attention_mask": att})["sentence_embedding"]
            out[sel] = v.float().numpy()
        return out

    def embed(self, texts):
        return self.embed_ids(self.ids(texts))


def compare(E, R, top_e, top_r):
    """Per-query cosine to the reference and top-10 overlap with the reference's top 10."""
    cos = np.sum(E * R, 1) / np.linalg.norm(E, axis=1) / np.linalg.norm(R, axis=1)
    ov = np.array([len(set(a) & set(b)) / K for a, b in zip(top_e, top_r)])
    return cos, ov


def summary(cos, ov, seed=0):
    rng = np.random.default_rng(seed)
    bs = [ov[rng.integers(0, len(ov), len(ov))].mean() for _ in range(2000)]
    return {"n": int(len(cos)), "overlap@10": float(ov.mean()),
            "overlap@10_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "cos_mean": float(cos.mean()), "cos_p01": float(np.percentile(cos, 1)), "cos_min": float(cos.min()),
            "pass": bool(ov.mean() >= 0.95 and np.percentile(cos, 1) >= 0.995)}
