"""Are PR descriptions a searchable decision log worth indexing?

The tombstone experiment showed the working tree cannot hold *mechanism* that
was deleted. This asks the adjacent question about *rationale*: code says what,
commit messages say what changed, and PR descriptions say **why**, including
what was considered and rejected. None of that is in the tree.

Compared against two things that already exist, so the number is marginal value
rather than raw capability:

  tree       remax's working tree, including a CLAUDE.md that deliberately
             documents decisions -- e.g. a table of "anti-goals overridden by
             shipped work" naming three reversals and why each was accepted.
  tombstones deleted files, from history-tombstone-index.

Queries are "why" questions written from CLAUDE.md's *claims* rather than from
PR bodies, so the query text is not copied from the corpus under test.

**Threat to external validity, stated before the result:** every remax PR is
authored through Claude sessions, median body 2,727 chars, none empty. That is
far richer than a typical repo, where PR bodies are frequently "fixes #12" or
blank. A positive result here says PR bodies are worth indexing *when they are
written like this*, not that they are in general.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "hybrid-code-index"))
sys.path.insert(0, str(REPO / "repo-index"))
sys.path.insert(0, "/home/user/remex")
import hcindex as H  # noqa: E402
from ask import Encoder  # noqa: E402

TARGET = Path("/home/user/remax")
OWNER, NAME = "oaustegard", "remax"


def fetch_prs(cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text())
    tok = os.environ["GH_TOKEN"]
    out = []
    for page in (1, 2, 3):
        req = urllib.request.Request(
            f"https://api.github.com/repos/{OWNER}/{NAME}/pulls"
            f"?state=all&per_page=100&page={page}",
            headers={"Authorization": f"Bearer {tok}",
                     "Accept": "application/vnd.github+json"})
        batch = json.load(urllib.request.urlopen(req))
        if not batch:
            break
        out += batch
    prs = [{"n": p["number"], "t": p["title"], "b": p.get("body") or "",
            "merged": bool(p.get("merged_at")),
            "at": (p.get("merged_at") or p["created_at"])[:10]} for p in out]
    cache.write_text(json.dumps(prs))
    return prs


def pr_chunks(prs: list[dict], cfg: dict) -> list[H.Chunk]:
    """One chunk per ~2000 chars of PR body, headed with number/title/date.

    Unlike code, a PR body has no line semantics worth preserving, so `s` is the
    character offset -- enough to point a reader at the right part of the thread.
    """
    out = []
    for p in prs:
        if not p["merged"]:
            continue
        head = f"PR #{p['n']} [{p['at']}]: {p['t']}"
        body = p["b"].strip()
        if len(body) < cfg["min_chars"]:
            out.append(H.Chunk(f"[PR #{p['n']}] {p['t'][:60]}", 0, head))
            continue
        for i in range(0, len(body), cfg["max_chars"]):
            out.append(H.Chunk(f"[PR #{p['n']}] {p['t'][:60]}", i,
                               f"{head}\n{body[i:i + cfg['max_chars']]}"))
    return out


# "why" questions, written from CLAUDE.md's claims rather than from PR bodies.
# gold is a substring matched against the retrieved chunk's file label, so a
# `[PR #66]` label matches "PR #66" and a tree path matches its filename.
RATIONALE = [
    ("Why does this library compile a C popcount kernel when its anti-goals say "
     "no C or C++ bindings?", ["_native", "CLAUDE.md"]),
    ("Why is the number of RHT rounds floored at two rather than one?",
     ["ROTATION_LSH", "rotation.py", "PR #60"]),
    ("Why does the stored corpus format record which rotation encoded it?",
     ["corpus.py", "rotation.json", "PR #62"]),
    ("Why was the benchmark harness removed from the installable package?",
     ["PR #65", "CROSSOVER"]),
    ("Why did the query path get faster without any neighbour changing?",
     ["PR #66", "QUERY_PATH_SPEED"]),
    ("Why was assignment to rotations_ restored with a write-through setter "
     "instead of left broken?", ["PR #61"]),
    ("Why keep a learned-rotation writeup when the approach was rejected?",
     ["LEARNED_ROTATION", "PR #57", "PR #58"]),
    ("Why does this project refuse to add a GPU or torch dependency?",
     ["CLAUDE.md", "README"]),
]


def main() -> None:
    cfg = H.load_cfg(TARGET)
    prs = fetch_prs(HERE / "prs.json")
    tree = H.build_corpus(TARGET, cfg)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "t", REPO / "history-tombstone-index" / "run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tombs = m.tombstones(cfg)
    pr = pr_chunks(prs, cfg)

    print(f"tree       {len(tree):4d} chunks")
    print(f"tombstones {len(tombs):4d} chunks")
    print(f"PR bodies  {len(pr):4d} chunks from {sum(1 for p in prs if p['merged'])} "
          f"merged PRs (+{100*len(pr)/len(tree):.0f}% over tree)\n", flush=True)

    enc = Encoder()
    corpora = {
        "tree": tree,
        "tree+tombstones": tree + tombs,
        "tree+PRs": tree + pr,
        "tree+tombstones+PRs": tree + tombs + pr,
    }
    rows = {}
    print(f"{'corpus':24s} {'rationale hit@5':>16s}")
    print("-" * 41)
    for name, chunks in corpora.items():
        mat = enc([c.text for c in chunks])
        bm = H.BM25().fit(chunks)
        hits, detail = 0, []
        for q, gold in RATIONALE:
            d = H.to_files(mat @ enc([q])[0], chunks, k=20)
            b = H.to_files(bm.score(q), chunks, k=20)
            top = H.rrf([d, b])[:5]
            ok = any(any(g.lower() in f.lower() for g in gold) for f in top)
            hits += ok
            detail.append({"q": q, "hit": ok, "top": top})
        print(f"{name:24s} {hits:>14d}/8")
        rows[name] = {"hits": hits, "n": len(RATIONALE), "detail": detail}

    json.dump(rows, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
