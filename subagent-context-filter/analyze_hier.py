"""Aggregate the hierarchical-filter run (arms none, jev4, hier4, hier8, full); prints markdown.

Usage: EVAL_DATA=<dir> python3 analyze_hier.py   (reads <dir>/results, <dir>/adjudication.json)
Writes results/summary_hier.json (counts only, no transcript text).
"""
import glob
import json
import os
import random
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("EVAL_DATA") or os.path.join(HERE, "data")
ARMS = ["none", "jev4", "hier4", "hier8", "full"]
FILTERS = ["jev4", "hier4", "hier8"]

res = defaultdict(dict)
for f in glob.glob(os.path.join(DATA, "results", "*.json")):
    tid, arm = os.path.basename(f)[:-5].rsplit(".", 1)
    res[tid][arm] = json.load(open(f))
tids = sorted(t for t in res if all(a in res[t] and "facts" in res[t][a] for a in ARMS))
ADJ = os.path.join(DATA, "adjudication.json")
adj = json.load(open(ADJ)) if os.path.exists(ADJ) else None
UP = {tuple(k) for k in (adj or {}).get("upgrades", [])}


def facts(t, a, regex_only=False):
    f = res[t][a]["facts"]
    return f if regex_only else [x or (t, a, i) in UP for i, x in enumerate(f)]


def recall(t, a):
    return sum(facts(t, a)) / len(facts(t, a))


def boot_ci(vals, n=5000, seed=0):
    rnd = random.Random(seed)
    means = sorted(st.mean(rnd.choice(vals) for _ in vals) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def jev_tokens(f):
    return f["jev_input_tokens"]


def windows(f):
    return f.get("windows", 0) + f.get("windows_l1", 0) + f.get("windows_l2", 0)


skipped = sorted(set(res) - set(tids))
print(f"{len(tids)} tasks with every arm scored ({len(skipped)} incomplete: {skipped}); "
      f"{sum(len(facts(t, 'none')) for t in tids)} facts\n")
if adj:
    print(f"Adjudicator: TPR {adj['tpr']:.0%} on {adj['n_pos']} regex hits, TNR {adj['tnr']:.0%} on "
          f"{adj['n_neg']} wrong-task deliverables; {len(UP)} of {adj['misses']} regex misses upgraded.\n")
    print("Regex-only: " + ", ".join(
        f"{a} {sum(x for t in tids for x in facts(t, a, True))}" for a in ARMS) + "\n")
print("| arm | facts found | tasks fully correct | 95% CI (tasks) | context passed, est. tokens (median) |")
print("|---|---|---|---|---|")
for a in ARMS:
    f = [x for t in tids for x in facts(t, a)]
    passed = [float(all(facts(t, a))) for t in tids]
    lo, hi = boot_ci(passed)
    print(f"| {a} | {sum(f)}/{len(f)} ({sum(f)/len(f):.0%}) | {int(sum(passed))}/{len(tids)} | {lo:.0%}-{hi:.0%} "
          f"| {int(st.median(res[t][a]['context_tokens'] for t in tids)):,} |")

print("\nPaired per-task fact recall, bootstrap 95% CI:\n")
for a, b in [("jev4", "full"), ("hier4", "full"), ("hier8", "full"), ("hier4", "jev4"), ("hier8", "jev4")]:
    d = [recall(t, a) - recall(t, b) for t in tids]
    lo, hi = boot_ci(d)
    print(f"- {a} - {b}: {st.mean(d):+.1%} ({lo:+.1%} to {hi:+.1%}); worse on {sum(x < 0 for x in d)}, "
          f"better on {sum(x > 0 for x in d)}, tied on {sum(x == 0 for x in d)}")

print("\n| filter | Jev calls (median) | Jev input tokens (median) | Jev $ (median) | seconds (median) "
      "| chunks scored at level 2 (median) | must-have chunks kept |")
print("|---|---|---|---|---|---|---|")
for a in FILTERS:
    fl = [res[t][a]["filter"] for t in tids]
    mr = [x for f in fl for x in f["must_recall"]]
    cand = st.median(f.get("candidates", len(f.get("kept_ids", []))) for f in fl) if a != "jev4" else "all"
    print(f"| {a} | {st.median(f['calls'] for f in fl)} | {int(st.median(jev_tokens(f) for f in fl)):,} "
          f"| ${st.median(jev_tokens(f) for f in fl) * 0.042 / 1e6:.4f} | {st.median(f['seconds'] for f in fl):.1f} "
          f"| {cand} | {sum(mr)}/{len(mr)} ({sum(mr)/len(mr):.0%}) |")

# Where hier lost a must-have chunk: pruned at level 1, or scored and out-ranked at level 2?
print("\nMust-have chunks hier dropped, by cause:\n")
tasks = {t["task_id"]: t for f in glob.glob(os.path.join(DATA, "tasks", "*.json")) for t in json.load(open(f))}
chunk_seg = {}
for s in {t.rsplit("-", 1)[0] for t in tids}:
    import sys
    sys.path.insert(0, HERE)
    import hierfilter as hf
    segs = hf.segments(json.load(open(os.path.join(DATA, "chunks", f"{s}.json"))))
    chunk_seg[s] = {c["id"]: i for i, sg in enumerate(segs) for c in sg}
for a in ("hier4", "hier8"):
    pruned = ranked_out = 0
    for t in tids:
        f = res[t][a]["filter"]; s = t.rsplit("-", 1)[0]
        for cid, kept in zip(tasks[t]["must_chunks"], f["must_recall"]):
            if not kept:
                if chunk_seg[s][cid] in f["segments_kept"]:
                    ranked_out += 1
                else:
                    pruned += 1
    print(f"- {a}: {pruned} pruned with their segment at level 1, {ranked_out} scored but out-ranked at level 2")

print("\n| task | " + " | ".join(ARMS) + " |")
print("|---|" + "---|" * len(ARMS))
for t in tids:
    print(f"| {t} | " + " | ".join(f"{sum(facts(t, a))}/{len(facts(t, a))}" for a in ARMS) + " |")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump({
    "tasks": len(tids), "facts": sum(len(facts(t, "none")) for t in tids),
    "adjudicator": ({k: adj[k] for k in ("tpr", "tnr", "n_pos", "n_neg", "misses")} | {"upgrades": len(UP)}) if adj else None,
    "arms": {a: {"facts_found": sum(x for t in tids for x in facts(t, a)),
                 "facts_found_regex_only": sum(x for t in tids for x in facts(t, a, True)),
                 "tasks_fully_correct": sum(all(facts(t, a)) for t in tids),
                 "context_tokens_median": st.median(res[t][a]["context_tokens"] for t in tids)} for a in ARMS},
    "filter": {a: {"calls_median": st.median(res[t][a]["filter"]["calls"] for t in tids),
                   "jev_tokens_median": st.median(jev_tokens(res[t][a]["filter"]) for t in tids),
                   "seconds_median": st.median(res[t][a]["filter"]["seconds"] for t in tids),
                   "must_kept": sum(x for t in tids for x in res[t][a]["filter"]["must_recall"]),
                   "must_total": sum(len(res[t][a]["filter"]["must_recall"]) for t in tids)} for a in FILTERS},
    "per_task": {t: {a: [sum(facts(t, a)), len(facts(t, a))] for a in ARMS} for t in tids},
}, open(os.path.join(HERE, "results", "summary_hier.json"), "w"), indent=1)
