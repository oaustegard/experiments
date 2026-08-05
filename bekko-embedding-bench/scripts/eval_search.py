"""mini-CTXBench: file-discovery recall for grep vs dense vs fused.

Metric follows the 2026-07-04/05 replication (memories ea313ee8, 554bf3dd):
gold = the fix-PR's diff file set, score = recall@5 / recall@10 over *files*,
plus raw-ingestible token accounting (Oskar's currency) — the tokens an agent
would actually have to read to see the ranked result.

Arms
----
rg      naive ripgrep over identifiers extracted from the issue, files ranked
        by number of matching identifiers then by hit count. This is the arm
        that has beaten every semantic tier across two prior runs.
dense   bekko cosine over chunks; a file scores as its best chunk.
rrf     reciprocal-rank fusion of the two, k=60.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bekko import BekkoEncoder, matryoshka  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("BEKKO_BENCH_REPO", "/home/user/sklearn-bench"))
RRF_K = 60

CAMEL = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b")
SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
BACKTICK = re.compile(r"`([^`\n]{2,60})`")
DOTTED = re.compile(r"\b(?:sklearn|np|numpy)\.([A-Za-z_][A-Za-z0-9_.]*)")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")

# Tokens that match half the tree and localize nothing.
STOP = {
    "sklearn", "python", "numpy", "scipy", "array", "arrays", "value", "values",
    "error", "errors", "result", "results", "expected", "actual", "versions",
    "steps", "code", "reproduce", "describe", "https", "github", "com", "self",
    "import", "return", "none", "true", "false", "shape", "dtype", "data",
    "test", "tests", "bug", "issue", "example", "examples", "default", "using",
    "used", "with", "from", "this", "that", "when", "which", "should", "would",
    "machine", "executable", "platform", "system", "install", "version",
}


def extract_identifiers(text: str, cap: int = 12) -> list[str]:
    """Identifiers a developer would paste into ripgrep, most specific first."""
    scored: dict[str, int] = defaultdict(int)
    for m in BACKTICK.findall(text):
        for w in WORD.findall(m):
            scored[w] += 4
    for w in CAMEL.findall(text):
        scored[w] += 3
    for w in DOTTED.findall(text):
        scored[w.split(".")[-1]] += 3
    for w in SNAKE.findall(text):
        scored[w] += 2
    out = [
        w
        for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])
        if w.lower() not in STOP and len(w) > 3
    ]
    return out[:cap]


# ── arms ────────────────────────────────────────────────────────────────────
def arm_rg(idents: list[str]) -> tuple[list[str], float, int]:
    """Rank files by (#distinct identifiers matched, #hits). Returns ranking,
    wall seconds, and raw ingestible characters of the tool output."""
    t0 = time.time()
    per_file_ids: dict[str, set] = defaultdict(set)
    per_file_hits: dict[str, int] = defaultdict(int)
    raw_chars = 0
    for ident in idents:
        try:
            p = subprocess.run(
                ["rg", "-n", "--no-heading", "-w", "-g", "*.py", "-g", "*.pyx",
                 "-g", "*.pxd", "-g", "*.tp", "--", ident, "sklearn"],
                cwd=REPO, capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            continue
        out = p.stdout
        raw_chars += len(out)
        for line in out.split("\n"):
            if not line.strip():
                continue
            f = line.split(":", 1)[0]
            per_file_ids[f].add(ident)
            per_file_hits[f] += 1
    ranked = sorted(
        per_file_hits,
        key=lambda f: (-len(per_file_ids[f]), -per_file_hits[f], f),
    )
    return ranked, time.time() - t0, raw_chars


def arm_dense(
    qvec: np.ndarray, mat: np.ndarray, chunks: list[dict], topn: int = 200
) -> tuple[list[str], list[int]]:
    """Best-chunk-per-file ranking. Returns file ranking and the chunk indices
    backing the top files (for honest token accounting)."""
    sims = mat @ qvec
    order = np.argsort(-sims)[:topn]
    best: dict[str, float] = {}
    backing: dict[str, int] = {}
    for i in order:
        f = chunks[i]["file"]
        if f not in best:
            best[f] = float(sims[i])
            backing[f] = int(i)
    ranked = sorted(best, key=lambda f: -best[f])
    return ranked, [backing[f] for f in ranked]


def rrf(*rankings: list[str]) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for r in rankings:
        for i, f in enumerate(r):
            score[f] += 1.0 / (RRF_K + i + 1)
    return sorted(score, key=lambda f: -score[f])


def recall_at(ranked: list[str], gold: list[str], k: int) -> float:
    g = set(gold)
    return len(g & set(ranked[:k])) / len(g) if g else 0.0


def approx_tokens(chars: int) -> int:
    """~4 chars/token, the accounting used in the prior two runs."""
    return round(chars / 4)
