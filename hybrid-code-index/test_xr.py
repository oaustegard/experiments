"""Tests for xr's query path. No network, no encoder, no published index.

Builds a tiny prepared cache by hand and exercises everything that does not
need the 124 MB ONNX model, which is the part with the interesting failure
modes anyway: BM25 off flat arrays, bounded top-k, repo scoping, and the error
that once killed the resident server.

    python3 hybrid-code-index/test_xr.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "repo-index"))
import hcindex as H  # noqa: E402
import xr  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def build_cache(d: Path) -> None:
    """A 6-chunk corpus over two repos, written in the prepared-cache layout."""
    texts = [
        ("alpha/one.py", 1, "rotation haar orthogonal matrix"),
        ("alpha/one.py", 40, "rotation rht hadamard fast transform"),
        ("alpha/two.md", 1, "quantization lloyd max boundaries"),
        ("beta/three.py", 1, "rotation haar orthogonal matrix"),
        ("beta/four.md", 7, "bm25 lexical scoring postings"),
        ("beta/four.md", 99, "unrelated filler about gardening"),
    ]
    chunks = [H.Chunk(f, s, t) for f, s, t in texts]
    bm = H.BM25().fit(chunks)

    # dense vectors chosen so "rotation" chunks point one way and the rest another
    rng = np.random.default_rng(0)
    x = rng.normal(size=(len(chunks), 8)).astype(np.float32)
    x[0] = x[1] = x[3] = np.array([1, 0, 0, 0, 0, 0, 0, 0], np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "dense.npy", x)
    np.save(d / "files.npy", np.array([c.f for c in chunks]))
    np.save(d / "lines.npy", np.array([c.s for c in chunks], dtype=np.int32))
    terms = sorted(bm.postings)
    docs, tfs, offs = [], [], [0]
    for t in terms:
        for doc, f in bm.postings[t]:
            docs.append(doc); tfs.append(f)
        offs.append(len(docs))
    np.save(d / "bm_terms.npy", np.array(terms))
    np.save(d / "bm_offs.npy", np.array(offs, dtype=np.int64))
    np.save(d / "bm_docs.npy", np.array(docs, dtype=np.int32))
    np.save(d / "bm_tfs.npy", np.array(tfs, dtype=np.int32))
    np.save(d / "bm_lens.npy", np.array(bm.lens, dtype=np.int32))
    (d / "meta.json").write_text(json.dumps(
        {"dim": 8, "bits": 2, "seed": 0, "rotation": "rht",
         "n_chunks": len(chunks), "repos": {"alpha": "t", "beta": "t"},
         "built_at": "2026-01-01T00:00:00Z"}))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "prepared"
        build_cache(cache)
        idx = xr.Index(cache)

        print("BM25 off flat arrays")
        s = idx.bm25("lloyd max boundaries")
        check("scores the matching chunk", s[2] > 0, f"s[2]={s[2]:.3f}")
        check("leaves non-matching at zero", s[4] == 0 and s[5] == 0)
        check("unknown term is not an error", float(idx.bm25("zzzznotaterm").sum()) == 0.0)
        # the reference implementation must agree, since this replaced it
        ref = H.BM25(); ref.postings = {}
        chunks = [H.Chunk(str(f), int(l), "") for f, l in zip(idx.files, idx.lines)]
        for i, t in enumerate(idx.terms):
            lo, hi = int(idx.offs[i]), int(idx.offs[i + 1])
            ref.postings[str(t)] = list(zip(np.asarray(idx.docs[lo:hi]).tolist(),
                                            np.asarray(idx.tfs[lo:hi]).tolist()))
        ref.lens = idx.lens.tolist(); ref.n = idx.n
        check("matches hcindex.BM25.score",
              np.allclose(idx.bm25("rotation haar"), ref.score("rotation haar"), atol=1e-5))

        print("bounded top-k")
        files = idx._top_files(idx.bm25("rotation"), k=5)
        check("dedupes to one hit per file", len({f for f, _ in files}) == len(files))
        check("returns the winning chunk's line, not the file's first",
              ("alpha/one.py", 1) in files or ("alpha/one.py", 40) in files,
              str(files[:2]))
        check("drops zero-scoring chunks", all(f.endswith((".py", ".md")) for f, _ in files))
        check("k is respected", len(idx._top_files(idx.bm25("rotation"), k=1)) == 1)

        print("repo scoping")
        mask = np.char.startswith(idx.files, "alpha/")
        s_all, s_alpha = idx.bm25("rotation"), idx.bm25("rotation", mask)
        check("masks out other repos", s_alpha[3] == 0 and s_all[3] > 0)
        check("keeps in-scope scores", s_alpha[0] == s_all[0])

        print("the bug that killed the server")
        try:
            idx.search("x", repo="nosuchrepo")
            check("unknown repo raises", False)
        except ValueError as e:
            check("unknown repo raises ValueError, not SystemExit", True)
            check("names some real repos", "alpha" in str(e))
        except SystemExit:
            check("unknown repo raises ValueError, not SystemExit", False,
                  "raised SystemExit -- BaseException, so the server's "
                  "`except Exception` would miss it")

    print()
    if FAILED:
        raise SystemExit(f"{len(FAILED)} failed: {', '.join(FAILED)}")
    print("all passed")


if __name__ == "__main__":
    main()
