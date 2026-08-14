#!/usr/bin/env python3
"""One-time SciFact corpus encode for the TTT-Embed x remex/remax experiment.

Produces `data/{Dm.npy,Q.npy,meta.json}` so the downstream numpy work (task
vector under an oracle teacher, remex 8/4/2/1-bit and remax k=8/4/2 sweeps)
never has to pay the encode again. See `RESULTS.md`.

Settings are pinned to the 2026-07-08 codec eval so its fidelity numbers carry
over unchanged:

  * encoder  jina-v5-nano `model.q4.onnx` @ release `v5-nano-8a7f00aa`
             (retrieval adapter merged, int4 MatMulNBits)
  * dim=256  Matryoshka truncation, max_length=384
  * doc text `title + ". " + text`
  * prefixes "Document: " / "Query: "
  * last-token pool (`mask.sum(-1) - 1`), then L2-normalize, fp32

The pooling/prefix/truncate/normalize order is copied from the mirror's own
torch-free reference loader (`scripts/embed_onnx.py`), which pins `model.onnx`
(fp32); the only deviation here is the q4 asset the issue specifies.

Batches are formed after sorting by token length, which cuts padding waste
roughly in half. Padding is masked out of both attention and the pool index, so
this is numerically inert -- `--parity-check` verifies that against unsorted
batching before the real run.

Run:  python3 encode.py            # full encode, resumable
      python3 encode.py --parity-check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

DATA = HERE / "data"
RAW = DATA / "raw"
CKPT = HERE / "checkpoints"

# --- pinned encoder -------------------------------------------------------
RELEASE_TAG = "v5-nano-8a7f00aa"
MIRROR = "oaustegard/jina-v5-nano-mirror"
Q4_URL = f"https://github.com/{MIRROR}/releases/download/{RELEASE_TAG}/model.q4.onnx"
Q4_SHA256 = "b8b18777a9b49bafb5d14f7db3e2687b7bc60485500c39cd9febdcf1d2552e15"
TOK_URL = (
    f"https://raw.githubusercontent.com/{MIRROR}/{RELEASE_TAG}/model/tokenizer.json"
)
MODEL_DIR = Path(
    os.environ.get("JINA_V5_NANO_CACHE", Path.home() / "models" / "jina-v5-nano-q4")
)

# --- pinned encode settings (must match the 2026-07-08 codec eval) --------
DIM = 256
MAX_LENGTH = 384
PAD_ID = 128001  # <|end_of_text|>, per upstream tokenizer_config.json
PROMPT_PREFIXES = {"query": "Query: ", "document": "Document: "}

# --- dataset --------------------------------------------------------------
HF = "https://huggingface.co/datasets"
SOURCES = {
    "corpus.parquet": f"{HF}/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "queries.parquet": f"{HF}/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
    "qrels_test.tsv": f"{HF}/BeIR/scifact-qrels/resolve/main/test.tsv",
}


def log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path, *, sha256: str | None = None, attempts: int = 8) -> Path:
    """Download with retry.

    HF load-balances its LFS redirect across `us.gcp.cdn.hf.co` and
    `us.aws.cdn.hf.co`; only the former is allowlisted from the claude.ai
    container, so a run there may need several attempts to land on GCP. Both
    resolve from CCotw, where this ran.
    """
    if dst.exists() and (sha256 is None or _sha256(dst) == sha256):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for i in range(1, attempts + 1):
        try:
            req = Request(url, headers={"User-Agent": "experiments/ttt-embed-quantized"})
            tmp = dst.with_suffix(dst.suffix + ".part")
            with urlopen(req, timeout=300) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if sha256 is not None:
                got = _sha256(tmp)
                if got != sha256:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"sha256 mismatch: got {got}, want {sha256}")
            tmp.rename(dst)
            return dst
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            last = exc
            log(f"  fetch {dst.name} attempt {i}/{attempts} failed: {exc}")
            time.sleep(min(2 ** i, 30))
    raise RuntimeError(f"could not fetch {url}: {last}")


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------
class Q4Encoder:
    def __init__(self, threads: int | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path = MODEL_DIR / "model.q4.onnx"
        tok_path = MODEL_DIR / "tokenizer.json"
        fetch(Q4_URL, onnx_path, sha256=Q4_SHA256)
        fetch(TOK_URL, tok_path)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        t0 = time.time()
        self.sess = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        log(f"session loaded in {time.time() - t0:.1f}s")
        self.tok = Tokenizer.from_file(str(tok_path))
        self.tok.enable_truncation(max_length=MAX_LENGTH)
        self.tok.enable_padding(pad_id=PAD_ID, pad_token="<|end_of_text|>")

    def token_lengths(self, texts: list[str], prompt: str) -> np.ndarray:
        pre = PROMPT_PREFIXES[prompt]
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=MAX_LENGTH)
        lens = np.array(
            [len(e.ids) for e in self.tok.encode_batch([pre + t for t in texts])],
            dtype=np.int32,
        )
        self.tok.enable_padding(pad_id=PAD_ID, pad_token="<|end_of_text|>")
        return lens

    def encode_batch(self, texts: list[str], prompt: str, dim: int = DIM) -> np.ndarray:
        """Encode one batch -> (n, dim) fp32, last-token pooled, L2-normalized."""
        pre = PROMPT_PREFIXES[prompt]
        enc = self.tok.encode_batch([pre + t for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        hidden = self.sess.run(
            ["last_hidden_state"], {"input_ids": ids, "attention_mask": mask}
        )[0]
        idx = mask.sum(axis=1) - 1
        pooled = hidden[np.arange(hidden.shape[0]), idx]  # (n, 768)
        if dim != pooled.shape[1]:
            pooled = pooled[:, :dim]  # Matryoshka truncate BEFORE normalizing
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def encode_all(
    enc: Q4Encoder,
    texts: list[str],
    prompt: str,
    *,
    batch_size: int,
    ckpt: Path | None = None,
    ckpt_every: int = 40,
) -> np.ndarray:
    """Length-sorted batched encode with resumable checkpoints."""
    n = len(texts)
    out = np.zeros((n, DIM), dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    if ckpt is not None and ckpt.exists():
        z = np.load(ckpt)
        out, done = z["out"], z["done"]
        log(f"resumed {prompt}: {int(done.sum())}/{n} rows already encoded")

    order = np.argsort(enc.token_lengths(texts, prompt), kind="stable")
    batches = [order[i : i + batch_size] for i in range(0, n, batch_size)]
    batches = [b for b in batches if not done[b].all()]
    log(f"{prompt}: {n} texts, {len(batches)} batches of {batch_size}")

    t0 = time.time()
    for bi, rows in enumerate(batches, 1):
        out[rows] = enc.encode_batch([texts[i] for i in rows], prompt)
        done[rows] = True
        if bi % 5 == 0 or bi == len(batches):
            seen = int(done.sum())
            rate = seen / max(time.time() - t0, 1e-9)
            eta = (n - seen) / max(rate, 1e-9)
            log(f"  {prompt} {seen}/{n} ({rate:.1f}/s, eta {eta / 60:.1f} min)")
        if ckpt is not None and (bi % ckpt_every == 0 or bi == len(batches)):
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt.with_suffix(".tmp.npz")
            np.savez(tmp, out=out, done=done)
            tmp.replace(ckpt)
    assert done.all(), f"{(~done).sum()} rows unencoded"
    return out


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_scifact() -> tuple[list[str], list[str], list[str], list[str], dict]:
    import pyarrow.parquet as pq

    for name, url in SOURCES.items():
        fetch(url, RAW / name)

    corpus = pq.read_table(RAW / "corpus.parquet").to_pylist()
    doc_ids = [str(r["_id"]) for r in corpus]
    doc_texts = [f"{r['title'] or ''}. {r['text'] or ''}" for r in corpus]

    with (RAW / "qrels_test.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    qrels: dict[str, dict[str, int]] = {}
    for r in rows:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])

    queries = pq.read_table(RAW / "queries.parquet").to_pylist()
    by_id = {str(r["_id"]): (r["text"] or "") for r in queries}
    missing = sorted(set(qrels) - set(by_id))
    if missing:
        raise RuntimeError(f"{len(missing)} qrels query ids absent from queries: {missing[:5]}")
    q_ids = sorted(qrels, key=int)  # test queries only, deterministic order
    q_texts = [by_id[q] for q in q_ids]
    return doc_ids, doc_texts, q_ids, q_texts, qrels


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------
def ndcg_at_k(sims: np.ndarray, doc_ids: list[str], q_ids: list[str], qrels: dict, k: int = 10) -> float:
    """Standard BEIR nDCG@k with binary gains (SciFact qrels are all score 1)."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    total = 0.0
    for qi, q in enumerate(q_ids):
        rel = qrels[q]
        top = np.argpartition(-sims[qi], k)[:k]
        top = top[np.argsort(-sims[qi][top])]
        gains = np.array([rel.get(doc_ids[i], 0) for i in top], dtype=np.float64)
        dcg = float((gains * discounts).sum())
        ideal = np.sort(np.array(list(rel.values()), dtype=np.float64))[::-1][:k]
        idcg = float((ideal * discounts[: len(ideal)]).sum())
        total += dcg / idcg if idcg > 0 else 0.0
    return total / len(q_ids)


def recall_at_k(sims: np.ndarray, doc_ids: list[str], q_ids: list[str], qrels: dict, k: int) -> float:
    total = 0.0
    for qi, q in enumerate(q_ids):
        rel = set(qrels[q])
        top = np.argpartition(-sims[qi], k)[:k]
        hit = sum(1 for i in top if doc_ids[i] in rel)
        total += hit / len(rel) if rel else 0.0
    return total / len(q_ids)


# --------------------------------------------------------------------------
def parity_check(enc: Q4Encoder, texts: list[str]) -> None:
    """Length-sorted batching must not change the vectors."""
    sample = texts[:48]
    a = np.vstack([enc.encode_batch(sample[i : i + 8], "document") for i in range(0, 48, 8)])
    order = np.argsort(enc.token_lengths(sample, "document"), kind="stable")
    b = np.zeros_like(a)
    for i in range(0, 48, 8):
        rows = order[i : i + 8]
        b[rows] = enc.encode_batch([sample[j] for j in rows], "document")
    cos = (a * b).sum(1)
    log(f"parity: min cos {cos.min():.6f}, mean {cos.mean():.6f}, max abs diff {np.abs(a - b).max():.2e}")
    # Also: batch size must not matter.
    c = np.vstack([enc.encode_batch(sample[i : i + 16], "document") for i in range(0, 48, 16)])
    log(f"batch-size parity (8 vs 16): min cos {(a * c).sum(1).min():.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--parity-check", action="store_true")
    args = ap.parse_args()

    doc_ids, doc_texts, q_ids, q_texts, qrels = load_scifact()
    log(f"scifact: {len(doc_ids)} docs, {len(q_ids)} test queries, "
        f"{sum(len(v) for v in qrels.values())} qrels")

    enc = Q4Encoder(threads=args.threads)
    if args.parity_check:
        parity_check(enc, doc_texts)
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Q = encode_all(enc, q_texts, "query", batch_size=args.batch_size,
                   ckpt=CKPT / "Q.npz")
    Dm = encode_all(enc, doc_texts, "document", batch_size=args.batch_size,
                    ckpt=CKPT / "Dm.npz")
    wall = time.time() - t0

    np.save(DATA / "Dm.npy", Dm)
    np.save(DATA / "Q.npy", Q)
    meta = {
        "doc_ids": doc_ids,
        "q_ids": q_ids,
        "qrels": qrels,
        "encoder": {
            "repo": MIRROR,
            "release": RELEASE_TAG,
            "asset": "model.q4.onnx",
            "sha256": Q4_SHA256,
            "dim": DIM,
            "max_length": MAX_LENGTH,
            "pooling": "last-token (mask.sum(-1) - 1)",
            "doc_text": 'title + ". " + text',
            "prefixes": PROMPT_PREFIXES,
            "normalize": "L2 after Matryoshka truncation, fp32",
        },
        "dataset": {"corpus": "BeIR/scifact", "qrels": "BeIR/scifact-qrels test.tsv"},
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=1))

    sims = Q @ Dm.T
    scores = {
        "ndcg@10": ndcg_at_k(sims, doc_ids, q_ids, qrels, 10),
        "recall@10": recall_at_k(sims, doc_ids, q_ids, qrels, 10),
        "recall@100": recall_at_k(sims, doc_ids, q_ids, qrels, 100),
    }
    log(f"Dm {Dm.shape} {Dm.dtype}  Q {Q.shape} {Q.dtype}  wall {wall / 60:.1f} min")
    for k, v in scores.items():
        log(f"fp32 {k} = {v:.4f}")
    (HERE / "sanity.json").write_text(
        json.dumps({**scores, "wall_clock_min": round(wall / 60, 2),
                    "n_docs": len(doc_ids), "n_queries": len(q_ids)}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
