"""Aggregate every arm's results into results/summary.json (means and SDs over seeds).

    python3 -I run/summarize.py
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORKFLOWS  # noqa: E402

RES = ROOT / "results"
METRICS = ("accuracy", "kl_from_gold", "brier", "ece")


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(st.mean(vals), 4), "sd": round(st.stdev(vals), 4) if len(vals) > 1 else None,
            "n": len(vals), "values": [round(v, 4) for v in vals]}


def main():
    out = {"spec": {}, "lowo": {}}
    for init in ("e17c", "stock", "stock-e10", "monad"):
        runs = [r for r in (load(f"spec-{init}-s{s}") for s in range(3)) if r]
        if runs:
            out["spec"][init] = {m: agg([r["test"][m] for r in runs]) for m in METRICS}
            out["spec"][init]["by_type"] = {t: agg([r["by"][f"type={t}"]["accuracy"] for r in runs])
                                            for t in ("choice", "noul", "score")}
            out["spec"][init]["by_workflow"] = {w: agg([r["by"][f"workflow={w}"]["accuracy"] for r in runs]) for w in WORKFLOWS}
            out["spec"][init]["best_epoch"] = [r["best_epoch"] for r in runs]
            out["spec"][init]["wall_min"] = agg([r["wall_sec"] / 60 for r in runs])
        per = {}
        for w in WORKFLOWS:
            rs = [r for r in (load(f"lowo-{init}-{w}-s{s}") for s in range(3)) if r]
            if rs:
                per[w] = {m: agg([r["test"][m] for r in rs]) for m in METRICS}
        if per:
            complete = len(per) == len(WORKFLOWS)
            out["lowo"][init] = {"per_workflow": per, "complete": complete,
                                 "mean_accuracy": round(st.mean(v["accuracy"]["mean"] for v in per.values()), 4)}
    for k in ("packed_spec-stock-s0-t4-w8L63", "ood_spec-stock-s0", "bridge_decider-final-1", "page_build"):
        r = load(k)
        if r:
            out[k] = r if k != "bridge_decider-final-1" else {x: r[x] for x in r if x not in ("sample", "describe")}
    (RES / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"spec": {k: v["accuracy"] for k, v in out["spec"].items()},
                      "lowo": {k: (v["mean_accuracy"], v["complete"]) for k, v in out["lowo"].items()}}, indent=1))


if __name__ == "__main__":
    main()
