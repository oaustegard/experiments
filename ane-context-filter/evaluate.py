"""Teacher agreement: how much of what Jev would keep does each cheap arm keep, per budget.

For each labelled delegation in the chosen split, every arm scores the context
chunks, and the shipped `select` packs them into 5K, 10K and 20K-token budgets.
User messages are force-kept by `select` for every arm alike, so they are left
out of the comparison. Two numbers per arm and budget:

  chunk_recall  share of the non-user chunks Jev's selection keeps that the arm also keeps
  mass_recall   Jev-score mass of the arm's non-user chunks over that of Jev's own selection

Arms whose scores are not probabilities are mapped to rank percentiles, so the
filter's score floor (0.15) cuts the same bottom share for every arm.

Usage: EVAL_DATA=<private dir> python3 evaluate.py [--split test|val|train|all] [--arms recency,bm25,...]
Writes results/agreement_<split>.json (numbers only).
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

sys.path.insert(0, str(experiment("subagent-context-filter")))
import jevfilter  # noqa: E402

DATA = Path(os.environ.get("EVAL_DATA") or HERE / "data")
BUDGETS = (5_000, 10_000, 20_000)
DELEG = ("Agent", "Task")


def split_of(session):
    """80/10/10 by session, stable across runs."""
    h = int(hashlib.sha256(session.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def context(chunks, deleg_chunk):
    return [c for c in chunks if c["id"] < deleg_chunk and not (
        c["kind"] == "tool" and (c.get("tool") in DELEG
                                 or c.get("tool", "").endswith(("create_session", "send_message"))))]


def chunk_text(c):
    v = jevfilter.view(c, 1.0)
    return " ".join(str(x) for k, x in v.items() if k != "kind")


TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+")


def toks(s):
    return [t.lower() for t in TOKEN.findall(s)]


def bm25(task, ctx, k1=1.2, b=0.75):
    docs = [toks(chunk_text(c)) for c in ctx]
    n = len(docs)
    avg = sum(map(len, docs)) / max(1, n)
    df = Counter(t for d in docs for t in set(d))
    q = set(toks(task))
    out = {}
    for c, d in zip(ctx, docs):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        out[c["id"]] = s
    return out


def recency(task, ctx):
    return {c["id"]: i for i, c in enumerate(ctx)}


ARMS = {"recency": recency, "bm25": bm25}


def percentiles(scores):
    order = sorted(scores, key=lambda k: scores[k])
    n = max(1, len(order) - 1)
    return {k: i / n for i, k in enumerate(order)}


def nonuser(ctx, keep):
    users = {c["id"] for c in ctx if c["kind"] == "user"}
    return {k for k in keep if k not in users}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    labels = {}
    for line in open(DATA / "labels.jsonl"):
        r = json.loads(line)
        if "error" not in r:
            labels[r["key"]] = {int(k): v for k, v in r["scores"].items()}
    rows = [json.loads(l) for l in open(DATA / "delegations.jsonl")]
    rows = [r for r in rows if f"{r['session']}:{r['deleg_chunk']}" in labels
            and (args.split == "all" or split_of(r["session"]) == args.split)]
    cache = {}
    res = {a: {b: {"chunk": [], "mass": []} for b in BUDGETS} for a in args.arms.split(",")}
    jev_sizes = {b: [] for b in BUDGETS}
    for r in rows:
        if r["session"] not in cache:
            cache[r["session"]] = json.loads((DATA / "chunks" / f"{r['session']}.json").read_text())
        ctx = context(cache[r["session"]], r["deleg_chunk"])
        jev = labels[f"{r['session']}:{r['deleg_chunk']}"]
        for b in BUDGETS:
            J = nonuser(ctx, jevfilter.select(ctx, jev, b))
            jev_sizes[b].append(len(J))
            if not J:
                continue
            jmass = sum(jev.get(k, 0) for k in J)
            for arm in res:
                s = percentiles(ARMS[arm](r["task"], ctx))
                A = nonuser(ctx, jevfilter.select(ctx, s, b))
                res[arm][b]["chunk"].append(len(A & J) / len(J))
                res[arm][b]["mass"].append(sum(jev.get(k, 0) for k in A) / jmass if jmass else 0)
    out = {"split": args.split, "delegations": len(rows),
           "sessions": len({r["session"] for r in rows}),
           "jev_nonuser_chunks_kept_median": {str(b): statistics.median(v) if v else None
                                              for b, v in jev_sizes.items()},
           "arms": {a: {str(b): {"chunk_recall_mean": round(statistics.mean(d["chunk"]), 3) if d["chunk"] else None,
                                 "mass_recall_mean": round(statistics.mean(d["mass"]), 3) if d["mass"] else None,
                                 "n": len(d["chunk"])}
                        for b, d in per.items()} for a, per in res.items()}}
    (HERE / "results" / f"agreement_{args.split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
