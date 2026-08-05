#!/usr/bin/env python3
"""Semantic lookup over this repo's markdown, for the "I never looked" failure.

CLAUDE.md tells you to grep METHODS.md before starting an experiment. That works
**if you already know the term**. Measured on the repo's own documented
rediscovery failures, grep finds the prior 5/5 when handed the right keyword
(`concurrency`, `ITQ`, `codebook`, `matryoshka`, `power analysis`) and 1/5 from
the words a person would actually type. This tool covers that gap — it is a
*complement* to the grep instruction, not a replacement for it.

    python3 repo-index/ask.py "about to fan out concurrent LLM calls"
    python3 repo-index/ask.py --build      # after adding or editing markdown

The index is ~0.12 MB (remex 2-bit over 384-d bekko-a8m vectors) and is committed.
It stores (path, line) pointers, not chunk text: the repo is the corpus.

First run downloads the encoder (~124 MB) to $BEKKO_HOME or ~/.cache/repo-index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("BEKKO_HOME", Path.home() / ".cache/repo-index"))
# Pinned mirror (MIT-licensed model, redistribution OK). Falls back to HF.
MODEL_SHA256 = "96d8cc6199e96357b21b2fb12f6d7ffd2d4abc7b182fe94b5468fbd6dc819af7"
MIRROR = os.environ.get(
    "REPO_INDEX_MIRROR",
    "https://github.com/oaustegard/experiments/releases/download/repo-index-model-v1")
HF = "https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m/resolve/main"
FILES = {"onnx/model.onnx": "model.onnx", "tokenizer.json": "tokenizer.json",
         "config.json": "config.json"}
SKIP = {".git", "node_modules", "__pycache__", ".venv"}
MIN_CHARS, MAX_CHARS = 200, 2000
BITS, DIM = 2, 384


def ensure_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        p = MODEL_DIR / local
        if not p.exists():
            for base, path in ((MIRROR, local), (HF, remote)):
                try:
                    print(f"fetching {local} from {base.split('/')[2]} ...", file=sys.stderr)
                    urllib.request.urlretrieve(f"{base}/{path}", p)
                    break
                except Exception as e:  # noqa: BLE001 - mirror may not exist yet
                    print(f"  {type(e).__name__}, trying next source", file=sys.stderr)
            else:
                raise SystemExit(f"could not fetch {local}")
    got = _sha256(MODEL_DIR / "model.onnx")
    if got != MODEL_SHA256:
        raise SystemExit(
            f"encoder sha256 mismatch\n  expected {MODEL_SHA256}\n  got      {got}\n"
            "The index was built with a specific encoder; a different one silently "
            "changes the embedding space. Delete the cache and refetch.")
    return MODEL_DIR


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def provenance(qz) -> dict:
    """Everything that defines the embedding space this index lives in.

    remex's own rule: *the rotation is part of the encoding*. It is regenerated
    from the seed at query time, and numpy's QR can drift across BLAS builds — so
    a CI-built index queried on a different machine could land in a different
    space with no error, just quietly worse recall. Fingerprint it so that is
    detected instead.
    """
    import numpy as _np
    import onnxruntime
    from remex.rotation import haar_rotation
    R = haar_rotation(DIM, seed=0)
    return {
        "encoder_sha256": MODEL_SHA256, "dim": DIM, "bits": BITS, "seed": 0,
        "rotation": qz.rotation,
        "rotation_fingerprint": hashlib.sha256(
            _np.ascontiguousarray(R).tobytes()).hexdigest()[:32],
        "onnxruntime": onnxruntime.__version__, "numpy": _np.__version__,
        # informational only -- a git checkout reports 0.0.0+unknown while CI
        # installs 0.6.0. The enforced invariant is the fingerprint, not this.
        "remex": getattr(__import__("remex"), "__version__", "unknown"),
    }


class Encoder:
    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = ensure_model()
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=512)
        self.tok.enable_padding(pad_id=0, pad_token="<pad>")
        self.sess = ort.InferenceSession(str(d / "model.onnx"),
                                         providers=["CPUExecutionProvider"])

    def __call__(self, texts: list[str], batch: int = 16) -> np.ndarray:
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), DIM), dtype=np.float32)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            e = self.tok.encode_batch([texts[i] for i in idx])
            ids = np.array([x.ids for x in e], dtype=np.int64)
            am = np.array([x.attention_mask for x in e], dtype=np.int64)
            h = self.sess.run(None, {"input_ids": ids, "attention_mask": am})[0]
            m = am.astype(np.float32)[..., None]
            out[idx] = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.clip(n, 1e-9, None)).astype(np.float32)


def chunk_repo() -> list[dict]:
    out = []
    for p in sorted(REPO.rglob("*.md")):
        if any(d in p.parts for d in SKIP):
            continue
        rel = str(p.relative_to(REPO))
        lines = p.read_text(errors="ignore").split("\n")
        cur, start, head = [], 1, ""
        def flush():
            body = "\n".join(cur).strip()
            if len(body) >= MIN_CHARS:
                for i in range(0, len(body), MAX_CHARS):
                    out.append({"f": rel, "s": start,
                                "t": f"# {rel}\n{head}\n{body[i:i + MAX_CHARS]}"})
        for i, ln in enumerate(lines, 1):
            if re.match(r"^#{1,4}\s", ln) and cur:
                flush(); cur, start, head = [], i, ln.strip()
            cur.append(ln)
        flush()
    return out


def build() -> None:
    import remex
    chunks = chunk_repo()
    print(f"chunking {len({c['f'] for c in chunks})} markdown files "
          f"-> {len(chunks)} chunks", file=sys.stderr)
    vecs = Encoder()(([c["t"] for c in chunks]))
    qz = remex.Quantizer(d=DIM, bits=BITS, seed=0)
    codes = qz.encode(vecs)
    # pack to the real 2 bits/dim; raw `indices` is uint8, i.e. 4x larger
    np.save(HERE / "index.npy", remex.pack(codes.indices.ravel(), BITS))
    (HERE / "shape.json").write_text(json.dumps(list(codes.indices.shape)))
    json.dump([{"f": c["f"], "s": c["s"]} for c in chunks],
              open(HERE / "pointers.json", "w"), separators=(",", ":"))
    prov = provenance(qz) | {"n_chunks": len(chunks)}
    json.dump(prov, open(HERE / "manifest.json", "w"), indent=1, sort_keys=True)
    size = (HERE / "index.npy").stat().st_size + (HERE / "pointers.json").stat().st_size
    print(f"wrote index: {size / 2**20:.2f} MB", file=sys.stderr)


def query(q: str, k: int) -> None:
    import remex
    ptr = json.load(open(HERE / "pointers.json"))
    shape = json.loads((HERE / "shape.json").read_text())
    idx = remex.unpack(np.load(HERE / "index.npy"), BITS,
                       shape[0] * shape[1]).reshape(shape).astype(np.uint8)
    qz = remex.Quantizer(d=DIM, bits=BITS, seed=0)
    want = json.load(open(HERE / "manifest.json"))
    have = provenance(qz)
    if have["rotation_fingerprint"] != want["rotation_fingerprint"]:
        print(f"WARNING: rotation fingerprint differs from the one this index was "
              f"built with ({want['rotation_fingerprint'][:12]} vs "
              f"{have['rotation_fingerprint'][:12]}).\n"
              f"  built with numpy {want['numpy']} / ort {want['onnxruntime']}; "
              f"you have numpy {have['numpy']} / ort {have['onnxruntime']}.\n"
              f"  Results will be silently degraded. Re-run --build.",
              file=sys.stderr)
    cv = remex.CompressedVectors(indices=idx, norms=np.ones(len(ptr), dtype=np.float32),
                                 d=DIM, bits=BITS, rotation=qz.rotation)
    xhat = qz.decode(cv)
    xhat /= np.clip(np.linalg.norm(xhat, axis=1, keepdims=True), 1e-9, None)
    qv = Encoder()([q])[0]
    seen, shown = set(), 0
    for i in np.argsort(-(xhat @ qv)):
        key = (ptr[i]["f"], ptr[i]["s"] // 40)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {ptr[i]['f']}:{ptr[i]['s']}")
        shown += 1
        if shown >= k:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("-k", type=int, default=8)
    a = ap.parse_args()
    if a.build:
        build()
    elif a.query:
        query(" ".join(a.query), a.k)
    else:
        ap.print_help()
