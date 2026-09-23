"""Aggregate data/results into the RESULTS.md tables (prints markdown)."""
import glob
import json
import os
import random
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "data", "results")
ARMS = ["none", "jev1", "jev4", "full", "brief"]
# Opus 5.5 list prices, $/MTok: the orchestrator a real run would have.
OPUS_IN, OPUS_OUT, OPUS_WRITE_1H, OPUS_READ = 4.0, 20.0, 8.0, 0.20

res = defaultdict(dict)
for f in glob.glob(os.path.join(R, "*.json")):
    tid, arm = os.path.basename(f)[:-5].rsplit(".", 1)
    res[tid][arm] = json.load(open(f))
tids = sorted(t for t in res if all(a in res[t] and "facts" in res[t][a] for a in ARMS))
ADJ = os.path.join(HERE, "data", "adjudication.json")
adj = json.load(open(ADJ)) if os.path.exists(ADJ) else None
UP = {tuple(k) for k in (adj or {}).get("upgrades", [])}


def facts(t, a, regex_only=False):
    f = res[t][a]["facts"]
    return f if regex_only else [x or (t, a, i) in UP for i, x in enumerate(f)]


def boot_ci(vals, n=5000, seed=0):
    rnd = random.Random(seed)
    means = sorted(st.mean(rnd.choice(vals) for _ in vals) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


print(f"{len(tids)} tasks with every arm scored; {sum(len(facts(t, 'none')) for t in tids)} facts\n")
if adj:
    print(f"Adjudicator (Sonnet 5, blind to arm): TPR {adj['tpr']:.0%} on {adj['n_pos']} regex hits, "
          f"TNR {adj['tnr']:.0%} on {adj['n_neg']} wrong-task deliverables; {len(UP)} of {adj['misses']} regex misses "
          f"upgraded to found.\n")
    print("Regex-only fact recall: " + ", ".join(
        f"{a} {sum(x for t in tids for x in facts(t, a, True))}/{sum(len(facts(t, a)) for t in tids)}" for a in ARMS) + "\n")
print("| arm | facts found | tasks fully correct | 95% CI (task) | passed context, est. tokens, median | executor s, median |")
print("|---|---|---|---|---|---|")
for a in ARMS:
    f = [x for t in tids for x in facts(t, a)]
    passed = [float(all(facts(t, a))) for t in tids]
    lo, hi = boot_ci(passed)
    ctx = [res[t][a]["context_tokens"] for t in tids]
    wall = [res[t][a]["executor"]["wall_s"] for t in tids]
    print(f"| {a} | {sum(f)}/{len(f)} ({sum(f)/len(f):.0%}) | {int(sum(passed))}/{len(tids)} | {lo:.0%}-{hi:.0%} "
          f"| {int(st.median(ctx)):,} | {st.median(wall):.1f} |")

print("\nPaired differences in fact recall per task (arm minus full), bootstrap 95% CI:\n")
for a in ["jev1", "jev4", "brief"]:
    d = [sum(facts(t, a)) / len(facts(t, a)) - sum(facts(t, "full")) / len(facts(t, "full")) for t in tids]
    lo, hi = boot_ci(d)
    print(f"- {a}: mean {st.mean(d):+.1%} (95% CI {lo:+.1%} to {hi:+.1%}); worse on {sum(x < 0 for x in d)}, "
          f"better on {sum(x > 0 for x in d)}, tied on {sum(x == 0 for x in d)}")

print("\nFilter cost and must-have chunk recall:\n")
print("| arm | windows, median | Jev input tokens, median | Jev $, median | filter s, median | must-have chunks kept |")
print("|---|---|---|---|---|---|")
for a in ["jev1", "jev4"]:
    fl = [res[t][a]["filter"] for t in tids]
    mr = [x for f in fl for x in f["must_recall"]]
    print(f"| {a} | {st.median(f['windows'] for f in fl)} | {int(st.median(f['jev_input_tokens'] for f in fl)):,} "
          f"| ${st.median(f['jev_input_tokens'] for f in fl) * 0.042 / 1e6:.4f} | {st.median(f['seconds'] for f in fl):.1f} "
          f"| {sum(mr)}/{len(mr)} ({sum(mr)/len(mr):.0%}) |")

w = [res[t]["brief"]["writer"] for t in tids]
print(f"\nBrief writer (same model, reading the full transcript): output tokens median "
      f"{int(st.median(x['output_tokens'] for x in w)):,}, wall median {st.median(x['wall_s'] for x in w):.1f} s, "
      f"cost median ${st.median(x['cost_usd'] for x in w):.3f}.")

print("\nOrchestrator-side handoff cost at Opus 5.5 prices, per delegation (medians):\n")
brief_out = st.median(x["output_tokens"] for x in w)
full_ctx = st.median(res[t]["full"]["context_tokens"] for t in tids)
for a in ["jev1", "jev4"]:
    ctx = st.median(res[t][a]["context_tokens"] for t in tids)
    jev = st.median(res[t][a]["filter"]["jev_input_tokens"] for t in tids) * 0.042 / 1e6
    print(f"- {a}: Jev ${jev:.4f} + first-turn cache write of ~{int(ctx):,} tokens ${ctx * OPUS_WRITE_1H / 1e6:.3f}; "
          f"each later subagent turn reads it for ${ctx * OPUS_READ / 1e6:.4f}")
print(f"- brief: {int(brief_out):,} output tokens ${brief_out * OPUS_OUT / 1e6:.3f} and "
      f"~{brief_out / 60:.0f} s of orchestrator generation at ~60 tok/s, before the subagent starts")
print(f"- fork: ~{int(full_ctx):,} inherited tokens read at ${full_ctx * OPUS_READ / 1e6:.4f} per subagent turn")

print("\nPer task (facts found per arm):\n")
print("| task | " + " | ".join(ARMS) + " |")
print("|---|" + "---|" * len(ARMS))
for t in tids:
    print(f"| {t} | " + " | ".join(f"{sum(facts(t, a))}/{len(facts(t, a))}" for a in ARMS) + " |")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
summary = {
    "tasks": len(tids), "facts": sum(len(facts(t, "none")) for t in tids),
    "adjudicator": {k: adj[k] for k in ("tpr", "tnr", "n_pos", "n_neg", "misses")} | {"upgrades": len(UP)} if adj else None,
    "arms": {a: {"facts_found": sum(x for t in tids for x in facts(t, a)),
                 "facts_found_regex_only": sum(x for t in tids for x in facts(t, a, True)),
                 "tasks_fully_correct": sum(all(facts(t, a)) for t in tids),
                 "context_tokens_median": st.median(res[t][a]["context_tokens"] for t in tids)} for a in ARMS},
    "filter": {a: {"windows_median": st.median(res[t][a]["filter"]["windows"] for t in tids),
                   "jev_tokens_median": st.median(res[t][a]["filter"]["jev_input_tokens"] for t in tids),
                   "seconds_median": st.median(res[t][a]["filter"]["seconds"] for t in tids),
                   "must_kept": sum(x for t in tids for x in res[t][a]["filter"]["must_recall"]),
                   "must_total": sum(len(res[t][a]["filter"]["must_recall"]) for t in tids)} for a in ("jev1", "jev4")},
    "brief_writer": {"output_tokens_median": st.median(x["output_tokens"] for x in w),
                     "wall_s_median": st.median(x["wall_s"] for x in w)},
    "per_task": {t: {a: [sum(facts(t, a)), len(facts(t, a))] for a in ARMS} for t in tids},
}
json.dump(summary, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)
