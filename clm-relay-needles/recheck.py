"""Check the RESULTS.md table and headline figures against results/summary.json (seconds, no API calls)."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
summ = json.loads((HERE / "results/summary.json").read_text())
text = (HERE / "RESULTS.md").read_text()
bad = 0
rows = {(t["variant"], t["n_chunks"], t["cond"]): t for t in summ["table"]}
seen = 0
for line in text.splitlines():
    m = re.match(r"\| (labeled|unlabeled) \| (\d+) \| (\w+) \| ([\d.]+) \| (\d+) \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \|", line)
    if not m:
        continue
    seen += 1
    t = rows[(m[1], int(m[2]), m[3])]
    want = (f"{t['recall_mean']:.3f}", t["corrupted"], t["noise_kept"], t["final_tokens_mean"],
            float(t["cost_usd"]), float(t["jev_usd"]), t["truncations"])
    got = (m[4], int(m[5]), int(m[6]), int(m[7]), float(m[8]), float(m[9]), int(m[10]))
    if want != got:
        bad += 1
        print("MISMATCH", m[1], m[2], m[3], "table", got, "summary", want)
if seen != len(rows):
    bad += 1
    print(f"table has {seen} rows, summary has {len(rows)}")
eps = summ["episodes"]
jev = [e for e in eps if e["cond"] == "jev"]
checks = {
    "835 needles": sum(e["needles"] for e in jev) == 835 and all(e["recall"] == 1 for e in jev),
    "total $22.51 subagent": round(sum(e["cost_usd"] for e in eps), 2) == 22.51,
    "state96 one corrupted": next(e for e in eps if (e["cond"], e["variant"], e["n"]) == ("state", "unlabeled", 96))["corrupted"] == 1,
    "both96 recall 0.755": round(next(e for e in eps if (e["cond"], e["variant"], e["n"]) == ("both", "unlabeled", 96))["recall"], 3) == 0.755,
}
for k, ok in checks.items():
    if not ok:
        bad += 1
        print("FAIL", k)
# Phase 2 table
tally = {(r["n"], r["cond"]): r for r in json.loads((HERE / "results_tally/summary.json").read_text())}
seen2 = 0
for line in text.split("# Phase 2")[1].splitlines():
    m = re.match(r"\| (\d+) \| (\d+) \| ([\w+]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.—]+) \| ([\d.—]+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
    if not m:
        continue
    seen2 += 1
    r = tally[(int(m[1]), m[3])]
    f = lambda x: "—" if x is None else f"{x:.3f}"
    want = (r["events"], f"{r['holder_acc']:.3f}", f"{r['count_exact']:.3f}", float(r["count_mae"]), f(r["replay_holder"]),
            f(r["replay_count"]), r["carried_tokens"], float(r["relay_usd"]), float(r["answer_usd"]), float(r["jev_usd"]))
    got = (int(m[2]), m[4], m[5], float(m[6]), m[7], m[8], int(m[9]), float(m[10]), float(m[11]), float(m[12]))
    if want != got:
        bad += 1
        print("MISMATCH phase2", m[1], m[3], "table", got, "summary", want)
if seen2 != len(tally):
    bad += 1
    print(f"phase 2 table has {seen2} rows, summary has {len(tally)}")
if round(sum(r["relay_usd"] + r["answer_usd"] + r["jev_usd"] for r in tally.values()), 2) != 15.21:
    bad += 1
    print("FAIL phase 2 spend $15.21")
print("recheck:", "OK" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)
