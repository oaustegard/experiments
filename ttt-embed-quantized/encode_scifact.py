#!/usr/bin/env python3
"""Encode BEIR SciFact with jina-v5-nano q4 ONNX into the matrices the
TTT-Embed x remex/remax experiment consumes (issue #33).

One-time corpus encode. claude.ai measures this encoder at <2 docs/s on one
core and reaps detached jobs after ~100s, so 5,183 docs cannot be encoded
there; this script runs it once here and the `.npy` files are committed.

Settings are pinned to the 2026-07-08 codec eval so its fidelity numbers carry
over: dim=256 Matryoshka, max_length=384, doc text = `title + ". " + text`,
prefixes `Document: ` / `Query: `, last-token pool (`mask.sum(-1) - 1`), then
L2-normalize, fp32.

Two deliberate deviations from the issue's spec, both forced by this
container's egress policy and both verified rather than assumed:

* **Corpus source.** Every `*.hf.co` / `huggingface.co` host answers 403 at the
  egress proxy here, so `BeIR/scifact` is unreachable. The corpus is rebuilt
  from the *upstream* AllenAI SciFact release that BEIR itself derives from
  (`scifact.s3-us-west-2.amazonaws.com`, reachable). `--verify` asserts the
  BEIR-published shape this reproduces: 5,183 docs, 1,109 claims, and 300 test
  queries carrying 339 unique qrels pairs.
* **Encoder driver.** `scripts/embed_onnx.py` in the mirror repo hardcodes the
  847 MB fp32 `model.onnx` asset in `materialize()`, so it cannot load the q4
  asset this experiment is pinned to. Its pooling/prefix/normalize semantics
  are replicated verbatim below (see `_pool`), against the same tokenizer
  file and the same `pad_id=128001`.

Usage:

    python3 encode_scifact.py --fetch          # model + corpus into ~/.cache
    python3 encode_scifact.py                  # encode -> data/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.pipeline import retry  # noqa: E402

HERE = Path(__file__).resolve().parent

# --- pinned sources ---------------------------------------------------------
RELEASE_TAG = "v5-nano-8a7f00aa"
MIRROR = "https://github.com/oaustegard/jina-v5-nano-mirror"
RAW = "https://raw.githubusercontent.com/oaustegard/jina-v5-nano-mirror/main"
MODEL_URL = f"{MIRROR}/releases/download/{RELEASE_TAG}/model.q4.onnx"
MODEL_BYTES = 169_736_452
TOKENIZER_URL = f"{RAW}/model/tokenizer.json"
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

CACHE = Path(os.environ.get("TTT_EMBED_CACHE") or Path.home() / ".cache" / "ttt-embed-quantized")

# --- encode settings (must match the 2026-07-08 codec eval) -----------------
DIM = 256
MAX_LENGTH = 384
PAD_ID = 128001
PREFIXES = {"query": "Query: ", "document": "Document: "}

# --- BEIR-published shape this reconstruction must reproduce ----------------
N_DOCS, N_CLAIMS, N_TEST_QUERIES, N_TEST_QRELS = 5183, 1109, 300, 339


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------
def _download(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    def once() -> None:
        req = Request(url, headers={"User-Agent": "ttt-embed-quantized/1.0"})
        with urlopen(req) as resp, tmp.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)

    log(f"fetching {url}")
    retry(once, attempts=5)
    tmp.rename(dst)
    return dst


def fetch_all() -> None:
    model = _download(MODEL_URL, CACHE / "model.q4.onnx")
    got = model.stat().st_size
    if got != MODEL_BYTES:
        raise SystemExit(f"model.q4.onnx is {got} B, expected {MODEL_BYTES} B")
    _download(TOKENIZER_URL, CACHE / "tokenizer.json")

    tgz = _download(SCIFACT_URL, CACHE / "scifact-allenai.tar.gz")
    if not (CACHE / "data" / "corpus.jsonl").exists():
        with tarfile.open(tgz) as t:
            members = [m for m in t.getmembers() if m.name.startswith("data/") and m.isfile()]
            t.extractall(CACHE, members=members)
    log(f"cache ready: {CACHE}")


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------
def _jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_scifact(root: Path) -> tuple[list[str], list[str], list[str], list[str], dict]:
    """Rebuild BEIR SciFact's test split from the upstream AllenAI release.

    BEIR maps each abstract to one document (`text` = abstract sentences joined
    by a space) and each claim to one query, and takes the *dev* claims as its
    test split — the AllenAI test claims ship unlabelled. A claim's relevant
    documents are its `cited_doc_ids`; `evidence` is the strictly smaller
    SUPPORT/CONTRADICT subset (209 pairs) and reproduces neither BEIR's 339
    qrels nor its 300 judged queries, so `cited_doc_ids` is what BEIR used.
    """
    corpus = _jsonl(root / "corpus.jsonl")
    doc_ids = [str(d["doc_id"]) for d in corpus]
    doc_texts = [f'{d["title"]}. {" ".join(d["abstract"])}' for d in corpus]

    dev = _jsonl(root / "claims_dev.jsonl")
    q_ids = [str(c["id"]) for c in dev]
    q_texts = [c["claim"] for c in dev]

    known = set(doc_ids)
    qrels: dict[str, list[str]] = {}
    for c in dev:
        # dict-keyed dedup: one dev claim cites the same doc twice, and BEIR's
        # 339 (not 340) qrels means it counted the pair once.
        rel = list(dict.fromkeys(str(d) for d in c["cited_doc_ids"]))
        missing = [d for d in rel if d not in known]
        if missing:
            raise SystemExit(f"claim {c['id']} cites unknown docs {missing}")
        qrels[str(c["id"])] = rel

    n_claims = len(dev) + len(_jsonl(root / "claims_train.jsonl"))
    return doc_ids, doc_texts, q_ids, q_texts, {"qrels": qrels, "n_claims": n_claims}


def verify_shape(doc_ids, q_ids, meta) -> None:
    pairs = sum(len(v) for v in meta["qrels"].values())
    judged = sum(1 for v in meta["qrels"].values() if v)
    checks = [
        ("docs", len(doc_ids), N_DOCS),
        ("claims (train+dev)", meta["n_claims"], N_CLAIMS),
        ("test queries", len(q_ids), N_TEST_QUERIES),
        ("judged test queries", judged, N_TEST_QUERIES),
        ("test qrels pairs", pairs, N_TEST_QRELS),
    ]
    bad = [(n, g, w) for n, g, w in checks if g != w]
    for name, got, want in checks:
        log(f"  {'ok ' if got == want else 'BAD'} {name}: {got} (BEIR: {want})")
    if bad:
        raise SystemExit("reconstruction does not match BEIR's published shape")


# --------------------------------------------------------------------------
# encode
# --------------------------------------------------------------------------
class Encoder:
    """Torch-free q4 ONNX encoder — embed_onnx.py's semantics, q4 asset."""

    def __init__(self, model: Path, tokenizer: Path, threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        t0 = time.time()
        self.session = ort.InferenceSession(
            str(model), sess_options=so, providers=["CPUExecutionProvider"]
        )
        log(f"session loaded in {time.time() - t0:.1f}s")
        self.tok = Tokenizer.from_file(str(tokenizer))
        self.tok.enable_padding(pad_id=PAD_ID, pad_token="<|end_of_text|>")
        self.tok.enable_truncation(max_length=MAX_LENGTH)

    def _pool(self, texts: list[str], prompt: str) -> np.ndarray:
        enc = self.tok.encode_batch([f"{PREFIXES[prompt]}{t}" for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        hidden = self.session.run(
            ["last_hidden_state"], {"input_ids": ids, "attention_mask": mask}
        )[0]
        pooled = hidden[np.arange(hidden.shape[0]), mask.sum(axis=1) - 1][:, :DIM]
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.where(norms == 0, 1.0, norms)).astype(np.float32)

    def encode(self, texts: list[str], prompt: str, batch: int, ckpt: Path | None) -> np.ndarray:
        """Mini-batched encode, resumable from `ckpt`.

        Mini-batching is not a free choice: a one-shot batch makes the
        attention-mask Expand allocate tens of GB and OOM (`q4-official-vs-ours`
        hit this at 1,500 docs). Rows are unaffected — attention is masked and
        last-token pooling indexes true lengths, so padding width cannot move a
        pooled vector. `--check-batch-invariance` re-verifies that here.
        """
        done = np.load(ckpt) if ckpt and ckpt.exists() else np.zeros((0, DIM), np.float32)
        if len(done):
            log(f"  resuming {prompt} at {len(done)}/{len(texts)}")
        out = [done] if len(done) else []
        t0 = time.time()
        for i in range(len(done), len(texts), batch):
            out.append(self._pool(texts[i : i + batch], prompt))
            n = min(i + batch, len(texts))
            if ckpt and (n % (batch * 25) == 0 or n == len(texts)):
                # atomic: a reaped job must not leave a half-written .npy
                acc = np.vstack(out)
                tmp = ckpt.with_suffix(".npy.part")
                # write through a handle: np.save(path, …) would append a
                # second ".npy" and leave nothing at `tmp` to rename.
                with tmp.open("wb") as f:
                    np.save(f, acc)
                tmp.replace(ckpt)
                rate = (n - len(done)) / max(1e-6, time.time() - t0)
                log(f"  {prompt} {n}/{len(texts)}  {rate:.1f}/s")
        return np.vstack(out).astype(np.float32)


# --------------------------------------------------------------------------
# sanity check
# --------------------------------------------------------------------------
def ndcg_at_k(Q: np.ndarray, Dm: np.ndarray, doc_ids, q_ids, qrels, k: int = 10) -> float:
    """nDCG@10 with binary gains, BEIR's convention (ideal over min(|rel|, k))."""
    index = {d: i for i, d in enumerate(doc_ids)}
    scores = Q @ Dm.T
    top = np.argsort(-scores, axis=1)[:, :k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    total, n = 0.0, 0
    for row, qid in enumerate(q_ids):
        rel = {index[d] for d in qrels.get(qid, [])}
        if not rel:
            continue
        gains = np.array([1.0 if j in rel else 0.0 for j in top[row]])
        ideal = discount[: min(len(rel), k)].sum()
        total += float((gains * discount).sum() / ideal)
        n += 1
    return total / max(1, n)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download model + corpus, then exit")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    ap.add_argument("--check-batch-invariance", action="store_true",
                    help="re-verify that batch width does not move a pooled row")
    a = ap.parse_args()

    if a.fetch:
        fetch_all()
        return 0

    model, tokenizer = a.cache / "model.q4.onnx", a.cache / "tokenizer.json"
    corpus_root = a.cache / "data"
    if not (model.exists() and tokenizer.exists() and (corpus_root / "corpus.jsonl").exists()):
        raise SystemExit(f"cache incomplete at {a.cache} — run with --fetch first")

    doc_ids, doc_texts, q_ids, q_texts, meta = load_scifact(corpus_root)
    log("verifying reconstruction against BEIR's published shape:")
    verify_shape(doc_ids, q_ids, meta)

    a.out.mkdir(parents=True, exist_ok=True)
    enc = Encoder(model, tokenizer, threads=a.threads)

    if a.check_batch_invariance:
        probe = doc_texts[:32]
        wide = enc._pool(probe, "document")
        narrow = np.vstack([enc._pool(probe[i : i + 4], "document") for i in range(0, 32, 4)])
        log(f"batch invariance: max|Δ| = {np.abs(wide - narrow).max():.3e}")

    t0 = time.time()
    Dm = enc.encode(doc_texts, "document", a.batch, a.out / ".ckpt_docs.npy")
    Q = enc.encode(q_texts, "query", a.batch, a.out / ".ckpt_queries.npy")
    secs = time.time() - t0

    assert Dm.shape == (N_DOCS, DIM), Dm.shape
    assert Q.shape == (N_TEST_QUERIES, DIM), Q.shape

    np.save(a.out / "Dm.npy", Dm)
    np.save(a.out / "Q.npy", Q)
    (a.out / "meta.json").write_text(json.dumps({
        "doc_ids": doc_ids, "q_ids": q_ids, "qrels": meta["qrels"],
        "encoder": {
            "model": f"{MIRROR} @ {RELEASE_TAG} / model.q4.onnx",
            "dim": DIM, "max_length": MAX_LENGTH, "pool": "last-token", "norm": "l2", "dtype": "float32",
            "doc_text": 'title + ". " + text', "prefixes": PREFIXES,
        },
        "dataset": {
            "name": "BEIR SciFact (test split)",
            "rebuilt_from": SCIFACT_URL,
            "n_docs": len(doc_ids), "n_queries": len(q_ids),
            "n_qrels": sum(len(v) for v in meta["qrels"].values()),
        },
    }, indent=1))

    ndcg = ndcg_at_k(Q, Dm, doc_ids, q_ids, meta["qrels"])
    log(f"\nencode: {secs / 60:.1f} min for {N_DOCS + N_TEST_QUERIES} texts "
        f"({(N_DOCS + N_TEST_QUERIES) / secs:.1f} texts/s)")
    print(f"fp32 nDCG@10 (Q @ Dm.T) = {ndcg:.4f}")
    if not 0.60 <= ndcg <= 0.72:
        log("WARNING: outside the 0.60-0.72 band expected for SciFact — "
            "suspect pooling or prefixes")
    for f in (a.out / ".ckpt_docs.npy", a.out / ".ckpt_queries.npy"):
        f.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
