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
import fnmatch
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
# Directories of machine-generated model output — an experiment's *data*, not its
# findings. 173 files, all `run_NN.md`-shaped near-duplicates, and they were 20%
# of the corpus. Measured: excluding them took agreement-with-grep on
# keyword-bearing queries from 8/10 to 10/10 (both misses were identifier lookups
# answered with LLM chatter) while the 5 rediscovery cases stayed 5/5.
GENERATED = {"outputs", "prompts"}
MIN_CHARS, MAX_CHARS = 200, 2000
BITS, DIM, SEED = 2, 384, 0
# Pinned explicitly, never left to remex's default. remex documents that
# default as deliberately changeable, and decoding under the wrong rotation is
# "50%-different, not slightly off ... total and silent". Costs nothing to pin.
ROTATION = "haar"


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
                    print(f"  {type(e).__name__} — expected if the release mirror "
                          f"has not been published; falling back", file=sys.stderr)
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


def quantizer():
    """The codec, with the stored rotation bound to it.

    remex's own rule is that *the rotation is part of the encoding*. This used to
    regenerate R from the seed at query time and merely fingerprint it. That is
    the weaker form of the rule: it turns a wrong rotation into a warning and a
    rebuild rather than a non-event, and it leaves the index depending on three
    upstream things staying still — remex's `rotation` default, remex's
    construction of it, and numpy's `default_rng` stream (which NEP 19
    explicitly does not guarantee across feature releases).

    Storing the matrix costs 576 KB once and removes all three. It is the same
    reasoning remex applies to its own containers, which record the rotation
    rather than assuming it.
    """
    import remex
    qz = remex.Quantizer(d=DIM, bits=BITS, seed=SEED, rotation=ROTATION)
    p = HERE / "rotation.npy"
    if p.exists():
        R = np.load(p)
        if R.shape != (DIM, DIM) or R.dtype != np.float32:
            raise SystemExit(f"rotation.npy is {R.shape}/{R.dtype}, expected "
                             f"({DIM}, {DIM})/float32")
        qz.R = R
    return qz


def provenance(qz) -> dict:
    """Everything that defines the embedding space this index lives in."""
    import numpy as _np
    import onnxruntime
    return {
        "encoder_sha256": MODEL_SHA256, "dim": DIM, "bits": BITS, "seed": SEED,
        "rotation": qz.rotation,
        # of the stored matrix, so this checks the artifact rather than a
        # regeneration of it
        "rotation_sha256": hashlib.sha256(
            _np.ascontiguousarray(qz.R).tobytes()).hexdigest(),
        # analytic from (d, bits) -- no RNG, no BLAS. Not stored, only
        # fingerprinted: a change upstream is caught, not silently absorbed.
        "codebook_sha256": hashlib.sha256(
            _np.ascontiguousarray(qz.boundaries).tobytes()
            + _np.ascontiguousarray(qz.centroids).tobytes()).hexdigest()[:32],
        "onnxruntime": onnxruntime.__version__, "numpy": _np.__version__,
        # informational only -- a git checkout reports 0.0.0+unknown while CI
        # installs 0.6.0. The enforced invariants are the hashes, not this.
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
        if any(d in p.parts for d in SKIP | GENERATED):
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
    # build from the seed, then persist R alongside the codes it produced
    qz = remex.Quantizer(d=DIM, bits=BITS, seed=SEED, rotation=ROTATION)
    np.save(HERE / "rotation.npy", np.ascontiguousarray(qz.R))
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


def excluded(path: str, patterns: list[str]) -> bool:
    """Match a pointer path against --exclude patterns.

    A pattern with no glob metacharacter is treated as a substring, so
    `--exclude ms13-campaign` does the obvious thing; anything with `*?[` is
    matched as a glob against the whole relative path (`--exclude '*/vendor/*'`).
    """
    for pat in patterns:
        g = pat if any(c in pat for c in "*?[") else f"*{pat}*"
        if fnmatch.fnmatch(path, g):
            return True
    return False


def query(q: str, k: int, exclude: list[str] | None = None) -> None:
    import remex
    ptr = json.load(open(HERE / "pointers.json"))
    shape = json.loads((HERE / "shape.json").read_text())
    idx = remex.unpack(np.load(HERE / "index.npy"), BITS,
                       shape[0] * shape[1]).reshape(shape).astype(np.uint8)
    qz = quantizer()
    want = json.load(open(HERE / "manifest.json"))
    have = provenance(qz)
    for key, what in (("rotation_sha256", "rotation.npy"),
                      ("codebook_sha256", "remex Lloyd-Max codebook")):
        if want.get(key) and have[key] != want[key]:
            raise SystemExit(
                f"{what} does not match the one this index was built with.\n"
                f"  expected {want[key][:16]}  got {have[key][:16]}\n"
                f"  Codes decoded under the wrong transform are ~50% different, "
                f"not slightly off. Refusing to return results; re-run --build.")
    cv = remex.CompressedVectors(indices=idx, norms=np.ones(len(ptr), dtype=np.float32),
                                 d=DIM, bits=BITS, rotation=qz.rotation)
    xhat = qz.decode(cv)
    xhat /= np.clip(np.linalg.norm(xhat, axis=1, keepdims=True), 1e-9, None)
    qv = Encoder()([q])[0]
    seen, shown = set(), 0
    for i in np.argsort(-(xhat @ qv)):
        key = (ptr[i]["f"], ptr[i]["s"] // 40)
        if key in seen or (exclude and excluded(ptr[i]["f"], exclude)):
            continue
        seen.add(key)
        print(f"  {ptr[i]['f']}:{ptr[i]['s']}")
        shown += 1
        if shown >= k:
            break


def verify() -> None:
    """Does regenerating R from the seed still reproduce the stored matrix?

    Purely diagnostic — the query path uses the stored one either way, so a
    divergence here is not a failure, it is the pin doing its job. Kept out of
    the query path because the Householder QR is O(d^3).
    """
    from remex.rotation import ROTATION_CODES  # noqa: F401  (fail loudly if gone)
    import remex
    stored = np.load(HERE / "rotation.npy")
    fresh = remex.Quantizer(d=DIM, bits=BITS, seed=SEED, rotation=ROTATION).R
    same = np.array_equal(stored, fresh)
    print(f"rotation: stored vs regenerated from seed -> "
          f"{'identical' if same else 'DIVERGED'}")
    if not same:
        print(f"  max abs delta {np.abs(stored - fresh).max():.3e} — the stored "
              f"matrix is authoritative; nothing is broken.")
    print(f"remex default rotation is {remex.Quantizer(d=8, bits=2).rotation!r}; "
          f"this index pins {ROTATION!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="check the stored rotation against seed regeneration")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--exclude", action="append", metavar="PAT", default=[],
                    help="drop results whose path matches; substring, or a glob "
                         "if it contains *?[. Repeatable.")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.verify:
        verify()
    elif a.query:
        query(" ".join(a.query), a.k, a.exclude)
    else:
        ap.print_help()
