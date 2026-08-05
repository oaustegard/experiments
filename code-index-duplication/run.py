"""Would a semantic index over this repo's .py files catch its documented duplication?

`METHODS.md` records a duplication map: several groups of files that are
near-identical reimplementations of each other, found by hand after the fact.
That map is an answer key nobody wrote for this purpose, which makes it a fair
test — the groups were recorded before this experiment existed.

Two query modes, because they correspond to different moments:

  code : you are about to write a file and have a draft. Query with the draft,
         ask whether something like it already exists. This is the realistic
         duplication-prevention moment.
  nl   : you describe what you want in words first.

Scored as hit@5: does a *known sibling* of the query file appear in the top 5
files, excluding the query file itself?

Two corpus variants, because the header is a live confound: the .md chunker
prepends `# <relpath>`, and `bench.py` appearing in two paths could drive a
match on filename alone rather than on content. bekko-embedding-bench already
measured path-only retrieval at r@5 0.304 on a related task — real but not
dominant — so it must be controlled here, not assumed away.

Flat line-window chunking: AST vs flat was noise in bekko-embedding-bench
(+0.063, p=0.424), so the simpler one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "repo-index"))
sys.path.insert(0, "/home/user/remex")
import ask  # noqa: E402

# This experiment's own directory is excluded: run.py contains every NL query
# verbatim, so leaving it in makes the harness retrieve itself. It ranked in the
# top-5 for 4 of 9 NL queries on the first run before this was noticed.
SKIP = {".git", "__pycache__", ".venv", "node_modules", "code-index-duplication"}
WIN, STRIDE = 60, 45

# ── answer key: METHODS.md "Duplication map", recorded before this experiment ──
GROUPS = [
    ["muninn-embedder-bakeoff/bench.py",
     "lfm25-embedder-remax_kb/bench_muninn.py",
     "jina-int8-remax_kb/bench.py"],
    ["phase-a-bridges/scripts/common.py", "te-bridges/scripts/te_common.py"],
    ["lexical-kb/skill_template/search.py", "kb-packer-web/vendor/search.py"],
    ["muninn-rm3/bench.py", "lexical-kb-phase0/sweep.py"],
]
# natural-language description of each group, written from METHODS.md's own words
NL = {
    "muninn-embedder-bakeoff/bench.py":
        "score R@5 and R@10 over the muninn corpus, collapsing chunk hits to "
        "distinct posts against a fixed topical gold set",
    "lfm25-embedder-remax_kb/bench_muninn.py":
        "benchmark an embedder on the muninn blog corpus and report recall at 5 and 10",
    "jina-int8-remax_kb/bench.py":
        "evaluate retrieval recall for a quantized embedder over muninn posts",
    "phase-a-bridges/scripts/common.py":
        "shared helpers for a bridge-paper extraction pipeline: retry with "
        "backoff, chunking, atomic checkpoint save and load",
    "te-bridges/scripts/te_common.py":
        "common utilities for the cross-domain extraction runs, including "
        "jittered retry and unicode normalization",
    "lexical-kb/skill_template/search.py":
        "BM25 search over a packed knowledge base file with stemming",
    "kb-packer-web/vendor/search.py":
        "query a packed .kb knowledge base using lexical scoring",
    "muninn-rm3/bench.py":
        "RM3 pseudo-relevance feedback benchmark reusing a shared query and gold set",
    "lexical-kb-phase0/sweep.py":
        "sweep chunk granularity for BM25 retrieval and report recall",
}


def chunk_py() -> list[dict]:
    out = []
    for p in sorted(REPO.rglob("*.py")):
        if any(d in p.parts for d in SKIP):
            continue
        rel = str(p.relative_to(REPO))
        lines = p.read_text(errors="ignore").split("\n")
        for s in range(0, max(1, len(lines)), STRIDE):
            body = "\n".join(lines[s:s + WIN]).strip()
            if len(body) < 120:
                continue
            out.append({"f": rel, "s": s + 1, "body": body})
            if s + WIN >= len(lines):
                break
    return out


def rank_files(scores, chunks, exclude: str, k=5) -> list[str]:
    """Excludes the query file's whole directory, matching `ask.py --file`.

    Excluding only the file itself scores higher (9/9) but does not describe the
    shipped tool: same-directory neighbours fill every slot in real use, which is
    why --file drops the directory. Reported numbers should describe what the
    tool does, not the most flattering configuration of the harness.
    """
    d = str(Path(exclude).parent)
    best: dict[str, float] = {}
    for i in np.argsort(-scores):
        f = chunks[i]["f"]
        if f == exclude or str(Path(f).parent) == d or f in best:
            continue
        best[f] = float(scores[i])
        if len(best) >= k:
            break
    return list(best)


def grep_arm(query_file: str, k=5) -> list[str]:
    """Ideal-keyword grep: the most distinctive def name in the query file."""
    txt = (REPO / query_file).read_text(errors="ignore")
    names = [ln.split("def ")[1].split("(")[0] for ln in txt.split("\n")
             if ln.strip().startswith("def ")]
    hits: list[str] = []
    for n in sorted(names, key=len, reverse=True)[:3]:
        r = subprocess.run(["rg", "-l", "--glob", "*.py", "--", f"def {n}", "."],
                           cwd=REPO, capture_output=True, text=True)
        for f in r.stdout.split():
            f = f.lstrip("./")
            if f != query_file and f not in hits:
                hits.append(f)
    return hits[:k]


def main() -> None:
    chunks = chunk_py()
    files = {c["f"] for c in chunks}
    print(f"corpus: {len(chunks)} chunks from {len(files)} .py files", flush=True)
    missing = [f for g in GROUPS for f in g if f not in files]
    if missing:
        print(f"WARNING answer-key files absent from corpus: {missing}", flush=True)

    enc = ask.Encoder()
    rows = []
    for variant in ("with-path-header", "content-only"):
        texts = [(f"# {c['f']}\n{c['body']}" if variant == "with-path-header"
                  else c["body"]) for c in chunks]
        mat = enc(texts, batch=16)
        for mode in ("code", "nl"):
            hits = tot = 0
            detail = []
            for g in GROUPS:
                for q in g:
                    if q not in files:
                        continue
                    siblings = [s for s in g if s != q]
                    if mode == "code":
                        qtext = "\n".join(
                            (REPO / q).read_text(errors="ignore").split("\n")[:WIN])
                    else:
                        qtext = NL[q]
                    qv = enc([qtext])[0]
                    top = rank_files(mat @ qv, chunks, exclude=q)
                    ok = any(s in top for s in siblings)
                    hits += ok; tot += 1
                    detail.append({"variant": variant, "mode": mode, "query": q,
                                   "hit": ok, "top": top, "siblings": siblings})
            print(f"  {variant:18s} {mode:5s}  hit@5 {hits}/{tot}", flush=True)
            rows.append({"variant": variant, "mode": mode, "hits": hits,
                         "n": tot, "detail": detail})

    # grep baseline, corpus-independent
    gh = gt = 0
    gdetail = []
    for g in GROUPS:
        for q in g:
            if q not in files:
                continue
            sib = [s for s in g if s != q]
            top = grep_arm(q)
            ok = any(s in top for s in sib)
            gh += ok; gt += 1
            gdetail.append({"query": q, "hit": ok, "top": top})
    print(f"  {'grep (ideal def name)':18s} {'--':5s}  hit@5 {gh}/{gt}")
    rows.append({"variant": "grep", "mode": "ideal-keyword", "hits": gh,
                 "n": gt, "detail": gdetail})

    json.dump(rows, open(HERE / "results.json", "w"), indent=1)
    print("\nper-query detail in results.json")


if __name__ == "__main__":
    main()
