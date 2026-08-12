"""Does repo size crowd small repos out of account-wide xr results, and if so
what fixes it?

Five arms, all scored off ONE encode + ONE dense scan + ONE BM25 pass per query,
so the arms differ only in how candidates are pooled and fused.

  A baseline      current xr: top-25 files per arm, RRF, top-k
  B cap           baseline, then at most `CAP` files per repo in the final list
  C size-prior    RRF contribution scaled by 1/log(1+chunks_in_repo)
  D diverse-pool  at most `CAP` files per repo while BUILDING each arm's 25
  O oracle        baseline restricted to the ground-truth repo (`-r`)

O is the ceiling: the gap between A and O is what scoping buys, i.e. the size of
the crowding problem. If A ~= O there is nothing to fix.

Metric: recall@k -- did any file the PR actually changed land in the top k.
"""
import collections
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/claude/cw/scripts")

import numpy as np
import xr

K = 10
CAP = 2
POOL = 25

idx = xr.Index(max(Path(xr.CACHE).glob("prepared-*")))
chunks = json.loads(Path("repo_chunks.json").read_text())
queries = json.loads(Path("queries.json").read_text())
repo_of = np.array([str(f).split("/")[0] for f in idx.files])
W = {r: 1.0 / np.log(1.0 + n) for r, n in chunks.items()}


def top_files(scores, k, cap=None):
    """Best chunk per file over a bounded pool, optionally capped per repo."""
    m = min(3000, len(scores))
    cand = np.argpartition(-scores, m - 1)[:m]
    cand = cand[np.argsort(-scores[cand])]
    out, seen, per = [], set(), collections.Counter()
    for i in cand:
        s = scores[i]
        if not np.isfinite(s) or s <= 0:
            continue
        f = str(idx.files[i])
        if f in seen:
            continue
        r = repo_of[i]
        if cap and per[r] >= cap:
            continue
        seen.add(f); per[r] += 1
        out.append(f)
        if len(out) >= k:
            break
    return out


def fuse(lists, weights=None, k=60):
    s = collections.defaultdict(float)
    for lst in lists:
        for rank, f in enumerate(lst, 1):
            w = 1.0 if weights is None else weights.get(f.split("/")[0], 1.0)
            s[f] += w / (k + rank)
    return sorted(s, key=lambda f: -s[f])


def capped(files, cap):
    out, per = [], collections.Counter()
    for f in files:
        r = f.split("/")[0]
        if per[r] >= cap:
            continue
        per[r] += 1; out.append(f)
    return out


STRIP = os.environ.get("STRIP") == "1"


def strip_identifiers(text, truth):
    """Remove every token the ground-truth PATHS give away.

    PR text names the files it changed, so the localization task above is
    identifier-rich -- and identifier-rich retrieval is the regime where plain
    lexical matching already wins. The crowding anecdote came from a CONCEPTUAL
    query ("fan out concurrent LLM calls through a Cloudflare gateway"), which
    has no such handles. Deleting the path tokens turns each query into its
    identifier-poor twin while keeping the same ground truth.
    """
    bad = set()
    for t in truth:
        for part in re.split(r"[/._\-]", t):
            if len(part) > 2:
                bad.add(part.lower())
                bad.update(w.lower() for w in
                           re.findall(r"[a-z]+|[A-Z][a-z]*", part) if len(w) > 2)
    keep = [w for w in re.split(r"(\W+)", text) if w.lower() not in bad]
    return "".join(keep)


enc = xr.Encoder()
rows = []
for n, q in enumerate(queries, 1):
    qtext = strip_identifiers(q["q"], q["truth"]) if STRIP else q["q"]
    qv = np.asarray(enc([qtext])[0], dtype=np.float32)
    i, sc = idx.qz.search_adc(idx.cv, qv, k=idx.cv.n)
    dense = np.empty(idx.n, dtype=np.float32); dense[i] = sc
    lex = idx.bm25(qtext)
    truth = set(q["truth"])

    d, b = top_files(dense, POOL), top_files(lex, POOL)
    A = fuse([d, b])[:K]
    B = capped(fuse([d, b]), CAP)[:K]
    C = fuse([d, b], weights=W)[:K]
    dc, bc = top_files(dense, POOL, CAP), top_files(lex, POOL, CAP)
    D = fuse([dc, bc])[:K]

    mask = repo_of == q["repo"]
    do = top_files(np.where(mask, dense, -np.inf), POOL)
    bo = top_files(np.where(mask, lex, 0.0), POOL)
    O = fuse([do, bo])[:K]

    def first(v, truth=truth):
        """1-based rank of the first ground-truth file, or None."""
        for j, f in enumerate(v, 1):
            if f in truth:
                return j
        return None
    rows.append({"repo": q["repo"], "pr": q["pr"],
                 **{k_: first(v) for k_, v in
                    (("A", A), ("B", B), ("C", C), ("D", D), ("O", O))},
                 "A_repos": len({f.split('/')[0] for f in A}),
                 "A_top_repo": collections.Counter(
                     f.split('/')[0] for f in A).most_common(1)[0][0]})
    if n % 25 == 0:
        print(f"  {n}/{len(queries)}", flush=True)

Path("results_strip.json" if STRIP else "results.json").write_text(
    json.dumps(rows, indent=1))

ARMS = ["A", "B", "C", "D", "O"]
def rate(rs, a, k):
    hit = sum(1 for r in rs if r[a] is not None and r[a] <= k)
    return 100.0 * hit / max(len(rs), 1)

# Terciles over the repos that actually have queries -- ranking all 65 puts
# every queried repo in the top half and leaves the "small" bucket empty.
qrepos = sorted({r["repo"] for r in rows}, key=lambda r: -chunks[r])
t3 = len(qrepos) // 3
tier = {r: ("large" if i < t3 else "small" if i >= 2 * t3 else "mid")
        for i, r in enumerate(qrepos)}

for k in (1, 3, 5, 10):
    print(f"\nrecall@{k} over {len(rows)} PR-localization queries (cap={CAP})")
    print("      " + "  ".join(f"{a:>6}" for a in ARMS))
    print("all   " + "  ".join(f"{rate(rows,a,k):6.1f}" for a in ARMS))
    for t in ("large", "mid", "small"):
        rs = [r for r in rows if tier.get(r["repo"]) == t]
        print(f"{t:6s}" + "  ".join(f"{rate(rs,a,k):6.1f}" for a in ARMS)
              + f"   n={len(rs)}")
print("\nchunk range by tier: " + ", ".join(
    f"{t}={min(chunks[r] for r in qrepos if tier[r]==t)}-"
    f"{max(chunks[r] for r in qrepos if tier[r]==t)}"
    for t in ("large", "mid", "small")))

print("\nwhere the baseline's own repo is not the one dominating its top-10:")
miss = [r for r in rows if r["A_top_repo"] != r["repo"]]
print(f"  {len(miss)}/{len(rows)} queries; "
      f"mean distinct repos in top-10 = "
      f"{np.mean([r['A_repos'] for r in rows]):.1f}")
c = collections.Counter(r["A_top_repo"] for r in miss)
print("  repos doing the crowding:", dict(c.most_common(5)))
Path("/tmp/eval2.done" if STRIP else "/tmp/eval.done").write_text("ok")
