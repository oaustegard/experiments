"""Typed Decisions (LocalLLaMA/typed-decisions) test split through deciding-with-confidence, zero-shot.

    python3 run/td.py run [--workers 6]     # -> results/td/test.jsonl, one line per case, resumable
    python3 run/td.py score                 # -> results/td/summary.json

Each of the 400 cases is one System One request (state + five questions) answered by decide.py's own
`--stack` path at low effort through the `cli` transport. The response keeps each answer's calibrated
`probabilities`, the uncalibrated pool and every sample, so four arms come from one run:
  stack       the shipped stack: three samples pooled, pooled temperatures applied
  stack raw   the same pool before calibration
  one         the stack's first sample (plain angle, options in listed order), single-sample temperatures
  one raw     that sample before calibration
The calibration was fitted on life-science and support text, never on this benchmark.
Scoring copies the dataset's own score_with_jobs.py: argmax agreement with the soft gold's argmax,
KL(gold || prediction), Brier over options, ECE over the top probability in 10 bins.
"""
import argparse
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

TD = experiment("browser-decider") / "data" / "td"
DECIDE = HERE.parent.parent / "claude-skills" / "deciding-with-confidence" / "scripts"
OUT = HERE / "results" / "td"
WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")
# from the dataset card, 2026-11-11: general models scored zero-shot on the same 2,000 decisions
CARD = {"TypeSafe Jev 1.13.0": (0.727, 1.442, 0.148, 0.144), "openjev (27B)": (0.728, 0.506, 0.191, 0.101),
        "clef (27B)": (0.728, 0.191, 0.103, 0.025), "meraGPT Decider 1": (0.768, 0.096, 0.052, 0.180),
        "Quyet-1.0-Large (31B)": (0.791, 0.281, 0.114, 0.093), "Teacher self-agreement": (0.735, None, None, None),
        "Prior (ignores the input)": (0.470, 0.347, 0.189, 0.088)}


def cases():
    for wf in WORKFLOWS:
        df = pd.read_parquet(TD / wf / "test.parquet")
        for _, r in df.iterrows():
            yield {"id": r["id"], "workflow": wf, "state": json.loads(r["state"]),
                   "questions": json.loads(r["questions"]), "gold": json.loads(r["gold"])}


def run(a):
    sys.path.insert(0, str(DECIDE))
    import decide
    decide.TIMEOUT = 180
    transport = decide.pick_transport("cli")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "test.jsonl"
    done = {json.loads(x)["id"] for x in path.read_text().splitlines()} if path.exists() else set()
    todo = [c for c in cases() if c["id"] not in done]
    lock = threading.Lock()
    failed = [0]

    def one(c):
        errors: list = []
        res = decide.decide({"state": c["state"], "questions": c["questions"]}, k=3, transport=transport,
                            errors=errors, angles=decide.STACK)
        if res is None:
            failed[0] += 1
            print(f"FAILED {c['id']}: {errors[:1]}", flush=True)
            return
        ans = {q: {"type": v.get("type"), "probabilities": v.get("probabilities"),
                   "raw": v.get("diagnostics", {}).get("raw"), "per_sample": v.get("diagnostics", {}).get("per_sample"),
                   "cues": v.get("diagnostics", {}).get("cues")} for q, v in res["answers"].items()}
        with lock, path.open("a") as f:
            f.write(json.dumps({"id": c["id"], "workflow": c["workflow"], "answers": ans, "usage": res["usage"]}) + "\n")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, todo))
    print(f"done: {len(todo) - failed[0]} answered, {failed[0]} failed", flush=True)


def score_one(probs, gold):
    """The dataset's scorer: (argmax agrees, KL(gold||p), Brier, top probability)."""
    opts = list(gold["probabilities"].keys())
    g = [float(gold["probabilities"][o]) for o in opts]
    p = [float((probs or {}).get(o, 0.0)) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    return p.index(max(p)) == g.index(max(g)), kl, sum((pi - gi) ** 2 for pi, gi in zip(p, g)), max(p)


def summarise(rows):
    n = len(rows)
    bins = [[0, 0, 0.0] for _ in range(10)]
    for ok, _, _, conf in rows:
        b = bins[min(int(conf * 10), 9)]
        b[0] += 1
        b[1] += ok
        b[2] += conf
    ece = sum(b[0] / n * abs(b[1] / b[0] - b[2] / b[0]) for b in bins if b[0])
    return {"n": n, "accuracy": sum(r[0] for r in rows) / n, "kl_from_gold": sum(r[1] for r in rows) / n,
            "brier": sum(r[2] for r in rows) / n, "ece": ece}


def score(a):
    sys.path.insert(0, str(DECIDE))
    import decide
    cal = decide.load_calibration()
    gold = {c["id"]: c for c in cases()}
    got = [json.loads(x) for x in (OUT / "test.jsonl").read_text().splitlines()]
    missing = [i for i in gold if i not in {g["id"] for g in got}]

    def single(ans):
        p = ans["per_sample"][0]
        t = {"noul": "predicate"}.get(ans["type"], ans["type"])
        return decide.temper(decide.floored(p), cal.get(t, 1.0))

    arms = {"stack": lambda ans: ans["probabilities"], "stack raw": lambda ans: ans["raw"],
            "one": single, "one raw": lambda ans: ans["per_sample"][0]}
    out = {"cases": len(got), "missing": missing, "arms": {}, "by_workflow": {}, "by_type": {}, "card": CARD}
    for arm, f in arms.items():
        rows, by_wf, by_t = [], {}, {}
        for g in got:
            for qid, gq in gold[g["id"]]["gold"].items():
                r = score_one(f(g["answers"][qid]), gq)
                rows.append(r)
                by_wf.setdefault(g["workflow"], []).append(r)
                by_t.setdefault(gq["type"], []).append(r)
        out["arms"][arm] = summarise(rows)
        out["by_workflow"][arm] = {k: summarise(v) for k, v in by_wf.items()}
        out["by_type"][arm] = {k: summarise(v) for k, v in by_t.items()}
    # agreement: unanimous stack answers against split ones
    uni, spl = [], []
    for g in got:
        for qid, gq in gold[g["id"]]["gold"].items():
            ans = g["answers"][qid]
            tops = [max(s, key=s.get) for s in ans["per_sample"]]
            r = score_one(ans["probabilities"], gq)
            (uni if len(set(tops)) == 1 else spl).append(r)
    out["stack_unanimous"], out["stack_split"] = summarise(uni), summarise(spl)
    lat = sorted(g["usage"]["latency_ms"] for g in got)
    out["usage"] = {"cost_usd_per_case_stack": sum(g["usage"]["cost_usd"] for g in got) / len(got),
                    "latency_ms_p50_stack": lat[len(lat) // 2]}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"cases {len(got)}, missing {len(missing)}")
    for arm, m in out["arms"].items():
        print(f"  {arm:10s} acc {m['accuracy']:.4f}  KL {m['kl_from_gold']:.3f}  Brier {m['brier']:.3f}  ECE {m['ece']:.3f}")
    for name, (acc, kl, br, ece) in CARD.items():
        print(f"  {name:28s} acc {acc:.3f}  KL {kl}  Brier {br}  ECE {ece}")
    print("  unanimous", round(out["stack_unanimous"]["accuracy"], 3), out["stack_unanimous"]["n"],
          "split", round(out["stack_split"]["accuracy"], 3), out["stack_split"]["n"])
    for arm in ("one", "stack"):
        print(f"  {arm} by workflow", {k: round(v["accuracy"], 3) for k, v in out["by_workflow"][arm].items()})
        print(f"  {arm} by type", {k: round(v["accuracy"], 3) for k, v in out["by_type"][arm].items()})
    print(" ", out["usage"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)
