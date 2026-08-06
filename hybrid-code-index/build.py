"""Build a persisted hybrid index for any repo, and report what it actually costs.

    python3 hybrid-code-index/build.py /home/user/remax --out /tmp/remax.npz
    python3 hybrid-code-index/build.py /home/user/remax --out /tmp/remax.npz --tombstones
    python3 hybrid-code-index/build.py --query /tmp/remax.npz "was BM25 ever tried here?"

Every number printed is measured on the machine it runs on. The point is to
answer "is this practical?" with a cost sheet rather than an argument.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "repo-index"))
sys.path.insert(0, "/home/user/remex")
import hcindex as H  # noqa: E402


ROT = "rht"  # 6 ms to build vs 171 ms for haar, 0 bytes vs 576 KB stored;
# retrieval-indistinguishable per remex (experiments#11), re-verified below


def encoder():
    from ask import Encoder
    return Encoder()


def quantize(vecs):
    import remex
    qz = remex.Quantizer(d=384, bits=2, seed=0, rotation=ROT)
    return qz.encode(vecs).indices, qz


def build(root: Path, out: Path, with_tombstones: bool, update: bool = False) -> None:
    cfg = H.load_cfg(root)
    t0 = time.time()
    chunks = H.build_corpus(root, cfg)
    n_live = len(chunks)
    if with_tombstones:
        sys.path.insert(0, str(HERE.parent / "history-tombstone-index"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tomb", HERE.parent / "history-tombstone-index" / "run.py")
        m = importlib.util.module_from_spec(spec)
        m.TARGET = root
        spec.loader.exec_module(m)
        m.TARGET = root
        chunks = chunks + m.tombstones(cfg)
    t_corpus = time.time() - t0

    src_bytes = sum(p.stat().st_size for p in H.discover(root, cfg))
    print(f"repo        {root}")
    print(f"  source    {src_bytes/2**20:.2f} MB over {len(H.discover(root,cfg))} files")
    print(f"  chunks    {len(chunks)} ({n_live} live"
          f"{f', {len(chunks)-n_live} tombstone' if with_tombstones else ''})"
          f"  [{t_corpus:.1f}s]")

    prev_codes = prev_hashes = None
    tf_cache: dict = {}
    if update and out.exists():
        old_chunks, prev_codes, old_bm, prev_hashes, _ = H.load(out)
        # reusable tokenization, keyed by content hash exactly like the dense rows
        for i, c in enumerate(old_chunks):
            pass  # texts are not stored; tf_cache stays empty on a loaded index
        print(f"  reusing   {out.name} ({len(prev_hashes)} prior chunks)")

    t0 = time.time()
    bm = H.BM25().fit(chunks, tf_cache)
    t_bm = time.time() - t0

    enc = encoder()
    t0 = time.time()
    codes, hashes, n_enc, n_reused = H.incremental(
        chunks, lambda ts: quantize(enc(ts, batch=16))[0],
        prev_codes=prev_codes, prev_hashes=prev_hashes)
    t_enc = time.time() - t0
    if prev_codes is not None:
        print(f"  encoded   {n_enc} new chunks, reused {n_reused} "
              f"({100*n_reused/max(1,len(chunks)):.0f}%)")

    sizes = H.save(out, chunks, codes, bm, hashes,
                   {"root": str(root), "n_live": n_live, "dim": 384, "bits": 2})
    print(f"  bm25 fit  {t_bm:.1f}s   dense encode {t_enc:.1f}s")
    print("\nartifact breakdown (compressed):")
    for k in ("codes", "bm_terms", "bm_offs", "bm_docs", "bm_tfs", "bm_lens",
              "files", "lines", "hashes"):
        print(f"  {k:10s} {sizes[k]/1024:8.0f} KB")
    tot = sizes["TOTAL_on_disk"]
    print(f"  {'TOTAL':10s} {tot/1024:8.0f} KB   "
          f"({100*tot/src_bytes:.0f}% of source)")
    print(f"\nencoder dependency: {sum(p.stat().st_size for p in Path(__import__('os').environ.get('BEKKO_HOME', Path.home()/'.cache/repo-index')).glob('*'))/2**20:.0f} MB "
          f"(one-time, shared across all repos)")


def query(idx: Path, q: str, k: int) -> None:
    import remex
    t0 = time.time()
    chunks, codes, bm, _, meta = H.load(idx)
    t_load = time.time() - t0

    t0 = time.time()
    qz = remex.Quantizer(d=meta["dim"], bits=meta["bits"], seed=0, rotation=ROT)
    cv = remex.CompressedVectors(indices=codes, norms=np.ones(len(chunks), np.float32),
                                 d=meta["dim"], bits=meta["bits"], rotation=ROT)
    x = qz.decode(cv)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)
    t_prep = time.time() - t0

    enc = encoder()
    t0 = time.time()
    qv = enc([q])[0]
    d = H.to_files(x @ qv, chunks, k=20)
    b = H.to_files(bm.score(q), chunks, k=20)
    top = H.rrf([d, b])[:k]
    t_q = time.time() - t0

    print(f"load {t_load*1000:.0f} ms | decode {t_prep*1000:.0f} ms | "
          f"query {t_q*1000:.0f} ms")
    for f in top:
        print(f"  {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tombstones", action="store_true")
    ap.add_argument("--update", action="store_true",
                    help="reuse rows from an existing --out index by content hash")
    ap.add_argument("--query", type=Path)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("q", nargs="*")
    a = ap.parse_args()
    if a.query:
        query(a.query, " ".join(a.q) or a.root, a.k)
    else:
        build(Path(a.root), a.out, a.tombstones, a.update)
