"""Collect every number RESULTS.md cites into results/summary.json.

Run after `python3 ../monad-weight-edit/collect.py <arm> --prefix mlp-gen-<arm>-` for the tested arms.
F is the loop-edited fp32 model from monad-weight-edit (A8p__gpu_test.jsonl).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MWE = HERE.parent / "monad-weight-edit"
sys.path.insert(0, str(MWE))
from stats import mcnemar  # noqa: E402
from summarize import score, summarize  # noqa: E402

R = HERE / "results"
s = {"premises": json.loads((R / "premises.json").read_text())}
s["screen_kl"] = {k: round(v["mean_kl"], 4) for k, v in json.loads((R / "screen_kl.json").read_text()).items()}
pairs = json.loads((R / "pairs_A8p.json").read_text())
s["pairs"] = {k: {"pairs": v["pairs"], "tf_kl": round(v["tf_kl"], 4)} for k, v in pairs["sets"].items()}


def bench(job):
    o = json.loads((MWE / "bridge/results" / job / "result.json").read_text())
    out = {"device": json.loads((MWE / "bridge/results" / job / "result.json").read_text()).get("device")}
    for b in o["outputs"]["bench"]:
        out[f"B{b['B']}"] = round(sorted(x["decodeMedMs"] for x in b["runs"])[1], 2)
    return out


s["bench"] = {"F": bench("mwe-bench-A8p-1")}
for arm in ("A8p_q8", "A8p_q4", "A8p_P8_q4", "A8p_P16_q4", "A8p_P24_q4"):
    s["bench"][arm] = bench(f"mlp-bench-{arm}")
s["sizes_mb"] = {}
for arm in ("A8p", "A8p_q8", "A8p_q4", "A8p_P8_q4", "A8p_P16_q4", "A8p_P24_q4"):
    m = json.loads((MWE / "export" / arm / "meta.json").read_text())
    s["sizes_mb"][arm] = {"data": round(m["data_bytes"] / 1e6, 1), "shipped": round(m["packed_bytes"] / 1e6, 1)}

F = MWE / "runs/A8p__gpu_test.jsonl"
s["test"] = {"F": summarize(F)}
s["stats_vs_F"] = {}
a = {json.loads(l)["uid"]: json.loads(l) for l in open(F)}
for arm in ("A8p_q8", "A8p_q4", "A8p_P16_q4", "A8p_P8"):
    p = MWE / f"runs/{arm}__gpu_test.jsonl"
    if not p.exists():
        continue
    s["test"][arm] = summarize(p)
    b = {json.loads(l)["uid"]: json.loads(l) for l in open(p)}
    st = {}
    for t in ("mmlu", "gsm8k", "ifeval", "all"):
        us = [u for u in a if u in b and (t == "all" or a[u]["task"] == t)]
        sa, sb = [score(a[u]) for u in us], [score(b[u]) for u in us]
        row = {"n": len(us)}
        for k in ("correct", "loop"):
            up = sum((not x[k]) and y[k] for x, y in zip(sa, sb))
            down = sum(x[k] and not y[k] for x, y in zip(sa, sb))
            row[k] = {"F": sum(x[k] for x in sa), "arm": sum(y[k] for y in sb), "only_arm": up, "only_F": down,
                      "p": mcnemar(up, down)}
        st[t] = row
    s["stats_vs_F"][arm] = st
    s["test"][arm]["all"]["acc"] = round(st["all"]["correct"]["arm"] / st["all"]["n"], 4)
s["test"]["F"]["all"]["acc"] = round(sum(score(r)["correct"] for r in a.values()) / len(a), 4)
(R / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps({k: s[k] for k in ("screen_kl", "bench", "sizes_mb")}, indent=1))
for arm, v in s["test"].items():
    print(arm, {t: (v[t].get("acc"), v[t]["loop"]) for t in v})
