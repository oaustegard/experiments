"""A6: score the Ettin label curve (browser-decider/run/train_enc.py --n-cases, Colab group `curve`)
against EG2's A4 on the same case draws.

    python3 -I run/a6_ettin.py RESULTS_DIR     # the unzipped bd_out_curve.zip results/ directory

Writes results/a6.json: per run, the curve per n, and the paired EG2-minus-Ettin difference per
(workflow, n, draw) with a bootstrap interval over the 12 pairs.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]


def main(src):
    src = Path(src)
    eg2 = {(r["held_out"], r["n"], r["draw"]): r["accuracy"]
           for r in json.loads((ROOT / "results" / "a4.json").read_text())["runs"] if r["start"] == "a0"}
    runs, pairs = [], {}
    for w in WORKFLOWS:
        for n in (64, 256):
            for d in (0, 1, 2):
                f = src / f"curve-{w}-n{n}-d{d}.json"
                r = json.loads(f.read_text())
                acc = r["test"]["accuracy"]
                runs.append({"workflow": w, "n": n, "draw": d, "accuracy": acc, "best_epoch": r["best_epoch"],
                             "kl_from_gold": r["test"]["kl_from_gold"], "wall_sec": r.get("wall_sec")})
                pairs.setdefault(n, []).append(eg2[(w, n, d)] - acc)
    rng = np.random.default_rng(0)
    out = {"runs": runs, "curve": {}, "eg2_minus_ettin": {}}
    for n in (64, 256):
        accs = [r["accuracy"] for r in runs if r["n"] == n]
        out["curve"][f"ettin_n{n}"] = float(np.mean(accs))
        out["curve"][f"eg2_n{n}"] = float(np.mean([eg2[(w, n, d)] for w in WORKFLOWS for d in (0, 1, 2)]))
        diff = np.array(pairs[n])
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        out["eg2_minus_ettin"][f"n{n}"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                                              float(np.percentile(boots, 97.5))]}
        out["curve"][f"ettin_n{n}_by_workflow"] = {w: float(np.mean([r["accuracy"] for r in runs
                                                                    if r["n"] == n and r["workflow"] == w]))
                                                   for w in WORKFLOWS}
    (ROOT / "results" / "a6.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
