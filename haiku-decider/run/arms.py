"""Run decide.py arms over the eval sets and summarise them.

    python3 run/arms.py run [--decide PATH/decide.py] [--tag v020] [--sections 150]
    python3 run/arms.py summary [--tag v020]         # -> results/summary_<tag>.json

Two runs per set, three samples each:
  old    k=3, the CLI's default effort, no angle     (the 0.1.1 default)
  stack  k=3 at low effort, one sample per angle: plain, literal, sceptic
Single-sample arms are read off those runs' per-sample distributions, so they cost no further calls: sample 0
of `stack` is one plain low-effort call, samples 1 and 2 are the literal and sceptic calls, and sample 0 of
`old` is one plain call at default effort. All metrics here are on the uncalibrated distributions.
"""
import argparse
import gzip
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SETS = ("bundled", "papers", "sections")
RUNS = {"old": ["-k", "3", "--effort", "default", "--angles", "plain"], "stack": ["--stack", "--effort", "low"]}


def items_file(name, decide, n_sections):
    if name == "bundled":
        return Path(decide).resolve().parents[1] / "assets" / "eval.jsonl"
    out = HERE / "out" / f"{name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with gzip.open(HERE / "data" / f"{name}.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    if name == "sections" and n_sections:
        rows = random.Random(0).sample(rows, n_sections)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def run(a):
    env = {**os.environ, "DECIDER_WORKERS": str(a.workers), "DECIDER_TIMEOUT": "120"}
    for name in SETS:
        src = items_file(name, a.decide, a.sections)
        for arm, flags in RUNS.items():
            out = HERE / "results" / f"{a.tag}__{name}__{arm}.jsonl"
            if out.exists():
                continue
            r = subprocess.run([sys.executable, a.decide, "eval", str(src), "--out", str(out) + ".part", "--transport", "cli", *flags],
                               capture_output=True, text=True, env=env, check=False)
            if r.returncode == 0:
                os.replace(str(out) + ".part", out)
            print(f"RUN_DONE {name} {arm} rc={r.returncode} {r.stderr[-200:]}", flush=True)
    print("done -> results", flush=True)


def metrics(rows):
    """rows: [(probs, label)] -> accuracy, ECE over the top probability in 10 bins, AUROC of top probability for right vs wrong."""
    if not rows:
        return None
    conf, right = [], []
    for p, y in rows:
        top = max(p, key=p.get)
        conf.append(p[top])
        right.append(top == y)
    n = len(rows)
    ece = 0.0
    for b in range(10):
        idx = [i for i, c in enumerate(conf) if b / 10 < c <= (b + 1) / 10 or (b == 0 and c == 0)]
        if idx:
            ece += len(idx) / n * abs(sum(right[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    pos = [c for c, r in zip(conf, right) if r]
    neg = [c for c, r in zip(conf, right) if not r]
    au = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg)) if pos and neg else None
    return {"n": n, "accuracy": round(sum(right) / n, 3), "ece": round(ece, 3), "auroc": None if au is None else round(au, 3)}


def summary(a):
    out = {}
    for name in SETS:
        res = {}
        for arm in RUNS:
            f = HERE / "results" / f"{a.tag}__{name}__{arm}.jsonl"
            if not f.exists():
                continue
            rows = [r for r in map(json.loads, f.read_text().splitlines()) if "raw" in r]
            full = [r for r in rows if len(r["per_sample"]) == 3]
            if arm == "old":
                res["old: 3 samples, default effort"] = metrics([(r["raw"], r["label"]) for r in rows])
                res["1 sample, default effort, plain"] = metrics([(r["per_sample"][0], r["label"]) for r in full])
            else:
                res["stack: plain + literal + sceptic, low effort"] = metrics([(r["raw"], r["label"]) for r in rows])
                for i, ang in enumerate(("plain", "literal", "sceptic")):
                    res[f"1 sample, low effort, {ang}"] = metrics([(r["per_sample"][i], r["label"]) for r in full])
                uni = [r for r in rows if r["agreement"] == 1]
                spl = [r for r in rows if r["agreement"] < 1]
                res["stack, unanimous"] = metrics([(r["raw"], r["label"]) for r in uni])
                res["stack, split"] = metrics([(r["raw"], r["label"]) for r in spl])
            lat = sorted({(r["id"], r["usage"]["latency_ms"], r["usage"]["cost_usd"]) for r in rows})
            if lat:
                res[f"{arm} request"] = {"latency_ms_median": sorted(x[1] for x in lat)[len(lat) // 2],
                                         "cost_usd_mean": round(sum(x[2] for x in lat) / len(lat), 5), "requests": len(lat)}
            by_type = {}
            for t in ("predicate", "choice", "score"):
                sub = [r for r in rows if r["type"] == t]
                if sub:
                    by_type[t] = metrics([(r["raw"], r["label"]) for r in sub])
            res[f"{arm} by type"] = by_type
        out[name] = res
    (HERE / "results" / f"summary_{a.tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, res in out.items():
        print(f"== {name}")
        for k, v in res.items():
            if v and "accuracy" in v:
                print(f"  {k:46s} n {v['n']:5d}  acc {v['accuracy']:.3f}  ece {v['ece']:.3f}  auroc {v['auroc']}")
            elif v and "latency_ms_median" in v:
                print(f"  {k:46s} {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--decide", default=str(HERE.parent.parent / "claude-skills" / "deciding-with-confidence" / "scripts" / "decide.py"))
    ap.add_argument("--tag", default="v020")
    ap.add_argument("--sections", type=int, default=150)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summary(a)
