"""Pool the two hierarchical-filter runs into paired per-task comparisons.

Usage: python3 pool.py <eval_dir> [<eval_dir> ...]   (each holds results/ and adjudication.json)
Writes results/summary_pooled.json (counts only).
"""
import glob
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["jev4", "hier4", "hier8", "full"]


def load(d: str) -> dict:
    up = {tuple(k) for k in json.load(open(os.path.join(d, "adjudication.json")))["upgrades"]}
    res = {}
    for f in glob.glob(os.path.join(d, "results", "*.json")):
        tid, arm = os.path.basename(f)[:-5].rsplit(".", 1)
        res.setdefault(tid, {})[arm] = json.load(open(f))
    out = {}
    for t, arms in res.items():
        out[t] = {}
        for a in ARMS:
            hits = [x or (t, a, i) in up for i, x in enumerate(arms[a]["facts"])]
            out[t][a] = {"recall": sum(hits) / len(hits), "jev": arms[a].get("filter", {}).get("jev_input_tokens")}
    return out


def main(dirs: list[str]) -> None:
    r = {}
    for d in dirs:
        r.update(load(d))
    rnd = random.Random(0)
    summary = {"tasks": len(r), "pairs": {}}
    for a, b in [("hier8", "jev4"), ("hier4", "jev4"), ("hier8", "full"), ("jev4", "full")]:
        d = [r[t][a]["recall"] - r[t][b]["recall"] for t in r]
        means = sorted(st.mean(rnd.choice(d) for _ in d) for _ in range(5000))
        summary["pairs"][f"{a}_minus_{b}"] = {
            "mean_pct": round(100 * st.mean(d), 1), "ci_lo_pct": round(100 * means[125], 1),
            "ci_hi_pct": round(100 * means[4875], 1), "worse": sum(x < 0 for x in d), "better": sum(x > 0 for x in d)}
    for a in ("hier4", "hier8"):
        ratio = [r[t][a]["jev"] / r[t]["jev4"]["jev"] for t in r]
        summary[f"{a}_jev_token_saving_pct"] = round(100 * (1 - st.median(ratio)), 1)
    json.dump(summary, open(os.path.join(HERE, "results", "summary_pooled.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
