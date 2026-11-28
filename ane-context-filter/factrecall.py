"""Fact recall: does a subagent given the task plus filtered context produce the facts it needs?

The decisive test for the students. On the held-out test sessions, a Sonnet labeller
(subagent-context-filter/label.py, unchanged) writes 4 tasks per session with 2-4 required facts
each, as regexes, plus the chunk ids that hold them. For each task and each arm, the session's
chunks are scored, packed into the budget with the shipped `select`, and a fresh tool-less Sonnet
executor writes the deliverable from that context alone (subagent-context-filter/run_eval.py's
executor). Facts are scored by regex; misses can then go to that experiment's adjudicator.

Arms, at 5K and 10K tokens:
  recency       newest chunks first
  jev           Jev, flat, 4x view (the teacher; the shipped filter's scorer)
  student17     Ettin-17m, pointwise BCE (round 1)
  student17lw   Ettin-17m, listwise (round 2)
Non-Jev scores are rank-mapped (ties shared), as in evaluate.py.

Usage (EVAL_DATA = the private data dir; transcripts never enter this repo):
  EVAL_DATA=<dir> python3 factrecall.py prep           # chunks + rendered text for the test sessions
  EVAL_DATA=<dir>/fr EVAL_MODEL='sonnet[1m]' python3 ../subagent-context-filter/label.py
  EVAL_DATA=<dir> python3 factrecall.py scores         # every arm's chunk scores per task (Jev calls here)
  EVAL_DATA=<dir> python3 factrecall.py run [--workers 6] [--limit N]
  EVAL_DATA=<dir> python3 factrecall.py summary        # writes results/factrecall.json (counts only)
Use the TypeSafe transport for Jev (unset CF_API_TOKEN, CF_ACCOUNT_ID, CF_GATEWAY_ID).
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

# The skill's jevfilter has the max_tokens_exceeded fix (delegating-with-context 0.3.1). Import it
# first so every later `import jevfilter` (run_eval, train) gets this module from the cache.
SKILL = Path(os.environ.get("JEVFILTER_DIR", "/home/user/claude-skills/delegating-with-context/scripts"))
sys.path.insert(0, str(SKILL))
import jevfilter as jf  # noqa: E402

assert "max_tokens_exceeded" in Path(jf.__file__).read_text(), f"unpatched jevfilter at {jf.__file__}"
sys.path.insert(1, str(experiment("subagent-context-filter")))
import run_eval  # noqa: E402

DATA = Path(os.environ.get("EVAL_DATA") or HERE / "data")
FR = DATA / "fr"
BUDGETS = (5_000, 10_000)
STUDENTS = {"student17": "student17", "student17lw": "student17lw"}
MIN_SESSION_TOKENS = 12_000   # below this every arm passes the whole session at 10K


def split_of(session):
    h = int(hashlib.sha256(session.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def percentiles(scores):
    vals = sorted(set(scores.values()))
    n = max(1, len(vals) - 1)
    rank = {v: i / n for i, v in enumerate(vals)}
    return {k: rank[v] for k, v in scores.items()}


def test_sessions():
    labelled = {json.loads(l)["key"].split(":")[0] for l in open(DATA / "labels.jsonl")}
    return sorted(s for s in labelled if split_of(s) == "test")


def prep():
    (FR / "chunks").mkdir(parents=True, exist_ok=True)
    (FR / "rendered").mkdir(exist_ok=True)
    kept = []
    for s in test_sessions():
        chunks = json.loads((DATA / "chunks" / f"{s}.json").read_text())
        text = "\n\n".join(jf.render(c, max_chars=10**9) for c in chunks)
        if jf.tokens(text) < MIN_SESSION_TOKENS:
            continue
        (FR / "chunks" / f"{s}.json").write_text(json.dumps(chunks))
        (FR / "rendered" / f"{s}.txt").write_text(text)
        kept.append((s, len(chunks), jf.tokens(text)))
    for k in kept:
        print(*k)
    print(f"{len(kept)} sessions prepared in {FR}")


def tasks():
    return [t for f in sorted(glob.glob(str(FR / "tasks" / "*.json"))) for t in json.load(open(f))]


def chunks_of(task):
    return json.loads((FR / "chunks" / f"{task['task_id'].rsplit('-', 1)[0]}.json").read_text())


def student_scorer(name):
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE))
    import train
    tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-17m")
    model = train.Scorer("jhu-clsp/ettin-encoder-17m")
    model.load_state_dict(torch.load(DATA / "models" / name / "final.pt", map_location="cpu"))
    model.eval()
    enc = train.Encoder(tok)

    @torch.no_grad()
    def score(task, chunks):
        n = len(chunks)
        items = [(c["id"], enc(task, c, n - i)) for i, c in enumerate(chunks)]
        items.sort(key=lambda x: len(x[1]))
        out = {}
        for s in range(0, len(items), 32):
            b = items[s: s + 32]
            ids, att = train.collate([{"ids": x[1]} for x in b], tok.pad_token_id)
            for (cid, _), p in zip(b, torch.sigmoid(model(ids, att)).tolist()):
                out[cid] = p
        return out
    return score


def scores_cmd():
    path = FR / "arm_scores.jsonl"
    done = set()
    if path.exists():
        done = {(d["task_id"], d["arm"]) for d in map(json.loads, open(path))}
    scorers = {n: student_scorer(d) for n, d in STUDENTS.items()}
    with open(path, "a") as f:
        for t in tasks():
            chunks = chunks_of(t)
            for arm in ("recency", "jev", *scorers):
                if (t["task_id"], arm) in done:
                    continue
                if arm == "recency":
                    s = {c["id"]: i for i, c in enumerate(chunks)}
                elif arm == "jev":
                    s, st = jf.score(t["task"], chunks, 4.0)
                else:
                    s = scorers[arm](t["task"], chunks)
                f.write(json.dumps({"task_id": t["task_id"], "arm": arm,
                                    "scores": {str(k): v for k, v in s.items()}}) + "\n")
                f.flush()
            print(t["task_id"], "scored", flush=True)


def load_scores():
    out = {}
    for d in map(json.loads, open(FR / "arm_scores.jsonl")):
        out[(d["task_id"], d["arm"])] = {int(k): v for k, v in d["scores"].items()}
    return out


def context_for(task, chunks, scores, budget):
    keep = jf.select(chunks, scores, budget)
    kept = [c for c in chunks if c["id"] in keep]
    body = "\n\n".join(jf.render(c, keep[c["id"]]) for c in kept)
    header = (f"Context from the parent session, selected for this task: {len(kept)} of {len(chunks)} "
              f"transcript chunks (~{jf.tokens(body)} tokens; ids in brackets). Chunks not shown were judged "
              f"irrelevant; if something you need is missing, say so in your reply.")
    return header + "\n\n" + body, [c["id"] for c in kept]


def run_one(task, arm, budget, scores):
    out = FR / "results" / f"{task['task_id']}.{arm}.{budget}.json"
    if out.exists():
        return json.loads(out.read_text())
    chunks = chunks_of(task)
    raw = scores[(task["task_id"], arm)]
    s = raw if arm == "jev" else percentiles(raw)
    ctx, kept = context_for(task, chunks, s, budget)
    r = {"task_id": task["task_id"], "arm": arm, "budget": budget, "context_tokens": jf.tokens(ctx),
         "must_recall": [i in kept for i in task["must_chunks"]]}
    try:
        ex = run_eval.claude(run_eval.EXEC_PROMPT.format(task=task["task"], context=ctx))
        r["executor"] = {k: v for k, v in ex.items() if k != "result"}
        r["deliverable"] = ex["result"]
        r["facts"] = run_eval.score_facts(ex["result"], task["facts"]) if ex["result"] else None
    except Exception as e:  # recorded, never scored as a pass or a fail
        r["error"] = f"{type(e).__name__}: {e}"[:400]
    out.write_text(json.dumps(r))
    return r


def run_cmd(workers, limit, arms, budgets):
    (FR / "results").mkdir(exist_ok=True)
    os.makedirs(run_eval.EMPTY, exist_ok=True)
    scores = load_scores()
    ts = tasks()[: limit or None]
    jobs = [(t, arm, b) for t in ts for b in budgets for arm in arms]
    print(f"{len(ts)} tasks, {len(jobs)} executor runs", flush=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_one, *j, scores) for j in jobs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            f = r.get("facts")
            print(f"{i}/{len(jobs)} {r['task_id']} {r['arm']} {r['budget']} "
                  f"{'ERR' if f is None else f'{sum(f)}/{len(f)}'}", flush=True)


def summary_cmd():
    rows = [json.loads(p.read_text()) for p in (FR / "results").glob("*.json")]
    ts = tasks()
    out = {"sessions": len({t["task_id"].rsplit("-", 1)[0] for t in ts}), "tasks": len(ts),
           "facts": sum(len(t["facts"]) for t in ts), "arms": {}}
    for arm in ("recency", "jev", *STUDENTS):
        for b in BUDGETS:
            rs = [r for r in rows if r["arm"] == arm and r["budget"] == b]
            ok = [r for r in rs if r.get("facts") is not None]
            mr = [x for r in rs for x in r["must_recall"]]
            out["arms"][f"{arm}@{b}"] = {
                "runs": len(rs), "errors": len(rs) - len(ok),
                "facts_found": sum(sum(r["facts"]) for r in ok),
                "facts_total": sum(len(r["facts"]) for r in ok),
                "evidence_chunk_recall": round(sum(mr) / len(mr), 3) if mr else None,
                "cost_usd": round(sum((r.get("executor") or {}).get("cost_usd") or 0 for r in rs), 2)}
    (HERE / "results" / "factrecall.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "scores", "run", "summary"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="recency,jev," + ",".join(STUDENTS))
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    a = ap.parse_args()
    {"prep": prep, "scores": scores_cmd, "summary": summary_cmd}.get(
        a.cmd, lambda: run_cmd(a.workers, a.limit, a.arms.split(","), [int(x) for x in a.budgets.split(",")]))()


if __name__ == "__main__":
    main()
