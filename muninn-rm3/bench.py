#!/usr/bin/env python3
"""Pure RM3 (model-free, agent-free) on the muninn corpus — the platform-less
site-search floor. BM25 on the raw query vs BM25 + RM3 pseudo-relevance feedback,
no agent expansion (site search has no Claude in the loop).

Reconstructs the 73-post corpus from muninn.kb, builds a lexical index at two
granularities (whole-doc + 500-char chunks), and scores R@5/@10 (distinct-post)
vs the phase-0 gold.

Baselines (same gold/corpus): lexical + AGENT expansion 1.00/1.00 (skill path,
NOT available to site search); Jina-q4->remax d=512/k=4 0.833/1.00; Jina-fp32
float 0.90/1.00.
"""
from __future__ import annotations

import json, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path("/home/user/claude-workspace")
HERE = Path(__file__).resolve().parent
KB = ROOT / ".spokes/muninn.austegard.com/knowledge/muninn.kb"
BUILD = ROOT / "experiments/lexical-kb/build_lexkb.py"
SEARCH = ROOT / "experiments/lexical-kb/skill_template/search.py"
CORPUS = HERE / "corpus"
sys.path.insert(0, str(ROOT / "experiments/lexical-kb-phase0"))
from sweep import QUERIES, stem  # noqa: E402

KS = (5, 10)


def reconstruct():
    if CORPUS.exists() and any(CORPUS.rglob("*.txt")):
        return
    z = zipfile.ZipFile(KB)
    chunks = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    from collections import OrderedDict
    posts = OrderedDict()
    for c in chunks:
        posts.setdefault(c["meta"]["source_path"], []).append(c["text"])
    for sp, texts in posts.items():
        dest = CORPUS / sp.replace(".html", ".txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n\n".join(texts), encoding="utf-8")
    print(f"reconstructed {len(posts)} posts -> {CORPUS}", flush=True)


def build(target_chars, out):
    subprocess.run([sys.executable, str(BUILD), str(CORPUS), "--out", str(out),
                    "--name", "muninn", "--ext", "txt", "--target-chars", str(target_chars)],
                   check=True, capture_output=True, text=True)


def ranked_posts(bundle, query, use_rm3, k_chunks=40):
    cmd = [sys.executable, str(SEARCH), "--index", str(bundle), "--query", query, "--k", str(k_chunks)]
    if use_rm3:
        cmd.append("--rm3")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    hits = json.loads(out)["hits"]
    seen, posts = set(), []
    for h in hits:
        sp = stem(h["meta"]["source_path"])
        if sp not in seen:
            seen.add(sp); posts.append(sp)
    return posts


def score(posts, gold):
    pos = {p: i + 1 for i, p in enumerate(posts)}
    return {k: sum(1 for g in gold if pos.get(g, 999) <= k) / len(gold) for k in KS}


def run(label, bundle, use_rm3):
    a5 = a10 = 0.0
    per = []
    for q in QUERIES:
        r = score(ranked_posts(bundle, q["query"], use_rm3), set(q["gold"]))
        a5 += r[5]; a10 += r[10]; per.append((q["label"], r[5]))
    n = len(QUERIES)
    print(f"{label:<26} R@5={a5/n:.3f} R@10={a10/n:.3f}   " +
          " ".join(f"{lbl.split()[0]}:{v:.2f}" for lbl, v in per), flush=True)
    return a5 / n, a10 / n


def main():
    reconstruct()
    rows = []
    for tc, gran in [(0, "whole-doc"), (500, "500-char")]:
        bundle = HERE / f"kb_{gran}"
        build(tc, bundle)
        print(f"\n--- granularity: {gran} ---", flush=True)
        rows.append((f"BM25 raw ({gran})", *run(f"BM25 raw ({gran})", bundle, False)))
        rows.append((f"BM25+RM3 ({gran})", *run(f"BM25+RM3 ({gran})", bundle, True)))
    print("\n" + "=" * 60)
    print(f"{'method':<26}{'R@5':>8}{'R@10':>8}")
    for lbl, r5, r10 in rows:
        print(f"{lbl:<26}{r5:>8.3f}{r10:>8.3f}")
    print(f"{'lexical+AGENT-expand*':<26}{1.000:>8.3f}{1.000:>8.3f}  (skill path; NOT for site search)")
    print(f"{'Jina-q4->remax d512/k4*':<26}{0.833:>8.3f}{1.000:>8.3f}  (needs inference platform)")
    print("* baselines, same gold/corpus. n=5 — directional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
