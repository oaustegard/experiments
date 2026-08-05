"""Build a retrieval sidecar for the experiments repo itself, and test it on the
repo's own documented rediscovery failures.

This repo's CLAUDE.md states the failure mode plainly: *"The failure mode is
'I never looked', not 'I looked in the right folder and missed it'."* It has
happened at least three times:

  1. `te-bridges` repeated `phase-a-bridges`' Cloudflare-gateway concurrency
     lesson from the adjacent directory, losing 18-20% of its extractions.
  2. `recall-per-byte` re-derived an ITQ overfitting result `remax#46` had
     already written up.
  3. (this session) `bekko-embedding-bench` reproduced
     `remex-vs-higgs-ablation`'s shared-codebook accounting trap, which is
     recorded in METHODS.md.

That is the task this sidecar should be judged on, and it is *not* the sklearn
task: there, the query already contained the identifiers, so grep had something
to match and dense only tied it. Here the querier by construction does **not**
know the keyword — that is what "I never looked" means. So this is the case where
semantic retrieval has a claim grep cannot make.

Chunks are markdown sections. Following the code-sidecar finding, the artifact
stores `(path, line)` pointers rather than chunk text: the repo is the corpus.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("REMEX_ROOT", "/home/user/remex"))
from bekko import BekkoEncoder  # noqa: E402

import remex  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
SKIP = {".git", "node_modules", "__pycache__", ".venv"}
MIN_CHARS, MAX_CHARS = 200, 2000


def md_chunks(path: Path, rel: str) -> list[dict]:
    """Split markdown on headings, then hard-wrap overlong sections."""
    text = path.read_text(errors="ignore")
    lines = text.split("\n")
    out, cur, start, head = [], [], 1, ""
    def flush(end: int) -> None:
        body = "\n".join(cur).strip()
        if len(body) >= MIN_CHARS:
            for i in range(0, len(body), MAX_CHARS):
                out.append({"file": rel, "start": start,
                            "text": f"# {rel}\n{head}\n{body[i:i+MAX_CHARS]}"})
    for i, ln in enumerate(lines, 1):
        if re.match(r"^#{1,4}\s", ln) and cur:
            flush(i); cur, start, head = [], i, ln.strip()
        cur.append(ln)
    flush(len(lines))
    return out


def build_corpus() -> list[dict]:
    chunks = []
    for p in sorted(REPO.rglob("*.md")):
        if any(d in p.parts for d in SKIP):
            continue
        chunks += md_chunks(p, str(p.relative_to(REPO)))
    return chunks


def main() -> None:
    chunks = build_corpus()
    print(f"experiments repo: {len(chunks)} markdown chunks from "
          f"{len({c['file'] for c in chunks})} files", flush=True)

    enc = BekkoEncoder("a8m", threads=4)
    import time
    t0 = time.time()
    vecs = enc.encode([c["text"] for c in chunks], batch_size=16)
    build_s = time.time() - t0
    print(f"encoded in {build_s:.1f}s ({len(chunks)/build_s:.0f} chunks/s)", flush=True)

    # remex 2-bit — this experiment's own measured sweet spot
    qz = remex.Quantizer(d=384, bits=2, seed=0)
    cv = qz.encode(vecs)
    xhat = qz.decode(cv)
    xhat /= np.clip(np.linalg.norm(xhat, axis=1, keepdims=True), 1e-9, None)

    ptr = json.dumps([{"f": c["file"], "s": c["start"]} for c in chunks],
                     separators=(",", ":")).encode()
    vec_b = len(chunks) * 384 * 2 // 8
    print(f"\nsidecar: vectors {vec_b/2**20:.2f} MB (remex 2-bit) + "
          f"pointers {len(zlib.compress(ptr,9))/2**20:.2f} MB gzip "
          f"= {(vec_b+len(zlib.compress(ptr,9)))/2**20:.2f} MB", flush=True)

    np.save(HERE / "repo_kb_vecs.npy", xhat.astype(np.float32))
    json.dump([{"f": c["file"], "s": c["start"]} for c in chunks],
              open(HERE / "repo_kb_ptr.json", "w"))

    # ── the real test: the repo's own rediscovery failures ──────────────────
    CASES = [
        ("About to fan out many concurrent LLM calls through a gateway for an "
         "extraction pipeline — any throughput limits I should know about?",
         ["METHODS.md", "phase-a-bridges", "te-bridges"]),
        ("Planning to learn a rotation matrix on the training vectors to improve "
         "binary quantization recall.",
         ["METHODS.md", "recall-per-byte", "remax"]),
        ("I want to compare a quantizer's bytes per vector against an "
         "uncompressed baseline — what should I count?",
         ["METHODS.md", "remex-vs-higgs"]),
        ("Does truncating embedding dimensions beat quantizing them at a fixed "
         "storage budget?", ["METHODS.md", "bekko-embedding-bench"]),
        ("How should I pick a sample size before spending compute on a benchmark?",
         ["METHODS.md", "bekko-embedding-bench"]),
    ]
    print("\n=== the task this repo actually has: 'I never looked' ===")
    print("(gold = any chunk whose path contains a documented prior)\n")
    dense_hits = grep_hits = 0
    for q, gold_subs in CASES:
        qv = enc.encode([q], sort_by_length=False)[0]
        order = np.argsort(-(xhat @ qv))[:5]
        top = [chunks[i]["file"] for i in order]
        hit = any(any(g.lower() in f.lower() for g in gold_subs) for f in top)
        dense_hits += hit
        # grep arm: the querier does not know the keyword, so use content words
        words = [w for w in re.findall(r"[a-z]{5,}", q.lower())][:6]
        found = set()
        for w in words:
            r = subprocess.run(["rg", "-l", "-i", "--glob", "*.md", "--", w, "."],
                               cwd=REPO, capture_output=True, text=True)
            found.update(r.stdout.split())
        ghit = any(any(g.lower() in f.lower() for g in gold_subs) for f in list(found)[:5])
        grep_hits += ghit
        print(f"  {'HIT ' if hit else 'MISS'} dense | {'HIT ' if ghit else 'MISS'} grep(top5)"
              f"  {q[:62]}...")
        print(f"        dense top-3: {top[:3]}")
    print(f"\n  dense {dense_hits}/{len(CASES)}   grep {grep_hits}/{len(CASES)}")


if __name__ == "__main__":
    main()
