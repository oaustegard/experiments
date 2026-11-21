"""Score run files with monad-weight-edit's scorer, plus the reasoning-length numbers this experiment needs.

    python3 score.py runs/*__dev.jsonl --json results/gen_dev.json

Per task and overall: acc, loop, open (think never closed), mean_tok (all answers), think_closed (mean
think tokens over answers that closed their think block), tok_per_correct (all generated tokens divided by
correct answers: the cost of one right answer, the number a speedup has to move). IFEval needs lm_eval.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "monad-weight-edit"))
from summarize import score  # noqa: E402


def summarize(path, tasks=None):
    recs = [json.loads(l) for l in open(path)]
    by = {}
    for r in recs:
        if tasks and r["task"] not in tasks:
            continue
        s = score(r)
        s["n_think"] = r.get("n_think")
        s["closed"] = r["closed"]
        by.setdefault(r["task"], []).append(s)

    def agg(xs):
        n = len(xs)
        cl = [x for x in xs if x["closed"] and x["n_think"] is not None]
        corr = sum(x["correct"] for x in xs)
        toks = sum(x["n_tok"] for x in xs)
        return {"n": n, "correct": corr, "acc": round(corr / n, 4), "loop": round(sum(x["loop"] for x in xs) / n, 4),
                "open": round(sum(x["open"] for x in xs) / n, 4), "mean_tok": round(toks / n, 1),
                "think_closed": round(sum(x["n_think"] for x in cl) / len(cl), 1) if cl else None,
                "tok_per_correct": round(toks / corr, 1) if corr else None}
    res = {t: agg(xs) for t, xs in by.items()}
    res["all"] = agg([x for xs in by.values() for x in xs])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    tasks = set(a.tasks.split(",")) if a.tasks else None
    out = {Path(p).stem: summarize(p, tasks) for p in a.runs}
    for k, v in out.items():
        r = v["all"]
        print(f"{k:28s} acc={r['acc']:.3f} loop={r['loop']:.3f} open={r['open']:.3f} tok={r['mean_tok']} "
              f"think={r['think_closed']} tok/correct={r['tok_per_correct']}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
