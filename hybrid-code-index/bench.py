"""Head-to-head: dense vs stored-BM25 vs ripgrep, alone and RRF-fused.

Three query classes, because the arms are expected to differ by class and a
single pooled number would hide that:

  rediscovery : "I'm about to do X, has this been done?" — no keyword available.
                Answer key: the repo's own documented rediscovery failures.
  keyword     : an identifier or term you already know. Answer key: what grep
                over the same file types returns.
  duplication : "does something like this already exist?", queried with a file's
                own text. Answer key: METHODS.md's duplication map, written
                before any of this existed, for an unrelated purpose.

Also measured as a variant, not assumed: whether indexing this repo's 13.6 MB of
generated .json results data helps, hurts, or is inert. That is the one place a
build-time exclusion might earn its place here, and the user's framing was that
those should be rare and repo-specific rather than baked into the tool.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "repo-index"))
sys.path.insert(0, "/home/user/remex")
import hcindex as H  # noqa: E402
from ask import Encoder  # noqa: E402

REDISCOVERY = [
    ("About to fan out many concurrent LLM calls through a gateway for an "
     "extraction pipeline — any throughput limits I should know about?",
     ["METHODS.md", "phase-a-bridges", "te-bridges"]),
    ("Planning to learn a rotation matrix on the training vectors to improve "
     "binary quantization recall.", ["METHODS.md", "recall-per-byte", "remax"]),
    ("I want to compare a quantizer's bytes per vector against an uncompressed "
     "baseline — what should I count?", ["METHODS.md", "remex-vs-higgs"]),
    ("Does truncating embedding dimensions beat quantizing them at a fixed "
     "storage budget?", ["METHODS.md", "bekko-embedding-bench"]),
    ("How should I pick a sample size before spending compute on a benchmark?",
     ["METHODS.md", "bekko-embedding-bench"]),
]

KEYWORD = [
    ("what does ascii_fold do", "ascii_fold"),
    ("why is the grid cache keyed on a version constant", "GRID_VERSION"),
    ("Lloyd-Max codebook", "Lloyd-Max"),
    ("where do spoke checkouts resolve from", "EXPERIMENTS_SPOKES_ROOT"),
    ("how do I run the _lib tests", "_lib/tests"),
    ("SPECTER2 embeddings", "SPECTER2"),
    ("what k does reciprocal rank fusion use", "RRF"),
    ("nDCG scores", "nDCG"),
    ("retention window of the EarthCam API", "EarthCam"),
    ("bit packing for sub-byte storage", "bit-pack"),
]

DUP_GROUPS = [
    ["muninn-embedder-bakeoff/bench.py", "lfm25-embedder-remax_kb/bench_muninn.py",
     "jina-int8-remax_kb/bench.py"],
    ["phase-a-bridges/scripts/common.py", "te-bridges/scripts/te_common.py"],
    ["lexical-kb/skill_template/search.py", "kb-packer-web/vendor/search.py"],
    ["muninn-rm3/bench.py", "lexical-kb-phase0/sweep.py"],
]


_RG_CACHE: dict = {}
_GOLD_CACHE: dict = {}


def keyword_gold(term: str, cfg) -> set[str]:
    import subprocess
    if term in _GOLD_CACHE:
        return _GOLD_CACHE[term]
    globs = []
    for e in cfg["extensions"]:
        globs += ["--glob", f"*{e}"]
    r = subprocess.run(["rg", "-l", "-i", *globs, "--", term, "."],
                       cwd=REPO, capture_output=True, text=True)
    _GOLD_CACHE[term] = {f.lstrip("./") for f in r.stdout.split()}
    return _GOLD_CACHE[term]


def main() -> None:
    cfg = H.load_cfg(REPO)
    # the harness embeds its own queries verbatim; leaving it in the corpus
    # makes it retrieve itself (measured: 4 of 9 NL queries, first run of
    # code-index-duplication)
    cfg["exclude"] = list(cfg["exclude"]) + ["hybrid-code-index/*"]

    t0 = time.time()
    chunks = H.build_corpus(REPO, cfg)
    files = sorted({c.f for c in chunks})
    print(f"corpus: {len(chunks)} chunks / {len(files)} files "
          f"({time.time()-t0:.1f}s)", flush=True)
    is_json = np.array([c.f.endswith(".json") for c in chunks])
    print(f"  of which .json: {is_json.sum()} chunks ({100*is_json.mean():.0f}%)",
          flush=True)

    t0 = time.time()
    bm = H.BM25().fit(chunks)
    print(f"bm25 fit {time.time()-t0:.1f}s, ~{bm.nbytes()/2**20:.2f} MB stored, "
          f"{len(bm.postings)} terms", flush=True)

    t0 = time.time()
    enc = Encoder()
    mat = enc([c.text for c in chunks], batch=16)
    print(f"dense encode {time.time()-t0:.0f}s", flush=True)

    results = {}
    for corpus_name, mask in (("all", np.zeros(len(chunks), bool)),
                              ("no-json", is_json)):
        qcache: dict = {}

        def dense_rank(q, exclude=None, k=20):
            if q not in qcache:
                qcache[q] = enc([q])[0]
            s = mat @ qcache[q]
            return H.to_files(np.where(mask, -np.inf, s), chunks, exclude, k)

        bcache: dict = {}

        def bm25_rank(q, exclude=None, k=20):
            if q not in bcache:
                bcache[q] = bm.score(q)
            s = bcache[q]
            return H.to_files(np.where(mask, -np.inf, s), chunks, exclude, k)

        def rg_rank(q, exclude=None, k=20):
            if q not in _RG_CACHE:
                _RG_CACHE[q] = H.rg_files(q, REPO, cfg)
            out = _RG_CACHE[q]
            if corpus_name == "no-json":
                out = [f for f in out if not f.endswith(".json")]
            if exclude:
                out = [f for f in out if not any(H.match(f, p) for p in exclude)]
            return out[:k]

        arms = {
            "dense": lambda q, e=None: dense_rank(q, e),
            "bm25": lambda q, e=None: bm25_rank(q, e),
            "rg": lambda q, e=None: rg_rank(q, e),
            "rrf(dense,bm25)": lambda q, e=None: H.rrf([dense_rank(q, e), bm25_rank(q, e)]),
            "rrf(dense,rg)": lambda q, e=None: H.rrf([dense_rank(q, e), rg_rank(q, e)]),
            "rrf(all3)": lambda q, e=None: H.rrf([dense_rank(q, e), bm25_rank(q, e),
                                                  rg_rank(q, e)]),
        }

        print(f"\n=== corpus: {corpus_name} ===")
        hdr = f"{'arm':17s} {'redisc':>8s} {'keyword':>8s} {'dup':>8s} {'TOTAL':>8s}"
        print(hdr); print("-" * len(hdr))
        for name, fn in arms.items():
            a = sum(any(any(g.lower() in f.lower() for g in gold) for f in fn(q)[:5])
                    for q, gold in REDISCOVERY)
            b = 0
            for q, term in KEYWORD:
                gold = keyword_gold(term, cfg)
                b += bool(set(fn(q)[:5]) & gold)
            c = 0
            for grp in DUP_GROUPS:
                for qf in grp:
                    if qf not in files:
                        continue
                    body = "\n".join((REPO / qf).read_text(errors="ignore")
                                     .split("\n")[:cfg["win"]])
                    drop = [f"{Path(qf).parent}/*"]
                    sib = [s for s in grp if s != qf]
                    c += any(s in fn(body, drop)[:5] for s in sib)
            tot = a + b + c
            print(f"{name:17s} {a:>6d}/5 {b:>6d}/10 {c:>6d}/9 {tot:>6d}/24")
            results[f"{corpus_name}|{name}"] = {"rediscovery": a, "keyword": b,
                                                "duplication": c, "total": tot}

    json.dump(results, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
