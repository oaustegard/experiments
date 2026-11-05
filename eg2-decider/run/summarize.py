"""Collect the headline numbers RESULTS.md cites into results/summary.json.

    python3 -I run/summarize.py
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"


EMBED_S = {"full": 442, "cut4": 86}  # cls prompt, each run alone on the CPU


def r3(x):
    return round(float(x), 4)


def load(p):
    p = RES / p
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out = {"reference": {"prior": 0.470, "jev_zero_shot": 0.727, "ettin_decider_specialist": 0.740,
                         "ettin_decider_lowo": 0.353, "ettin_decider_ood": 0.279, "jev_ood": 0.894,
                         "mrm8488_frozen": 0.677, "mrm8488_end_to_end": 0.764}}
    for enc, sub in (("full", ""), ("cut4", "cut4/")):
        a0, a1 = load(f"{sub}a0.json"), load(f"{sub}a1.json")
        if not a0:
            continue
        e = {"a0": {"choice": a0["choice"], "td_test_accuracy": r3(a0["td_test"]["accuracy"]),
                    "td_test_kl": r3(a0["td_test"]["kl_from_gold"]), "td_by_type": a0["td_test"]["by_type"],
                    "ood_accuracy": r3(a0["ood"]["accuracy"]), "ood_by_subset": a0["ood"]["by_subset"]}}
        if a1:
            runs = a1["runs"]
            e["a1"] = {"td_test_accuracy": r3(a1["mean_td_test_accuracy"]), "td_test_sd": r3(a1["sd_td_test_accuracy"]),
                       "td_test_kl": r3(np.mean([r["td_test"]["kl_from_gold"] for r in runs])),
                       "td_test_brier": r3(np.mean([r["td_test"]["brier"] for r in runs])),
                       "td_test_ece": r3(np.mean([r["td_test"]["ece"] for r in runs])),
                       "td_by_type": {t: r3(np.mean([r["td_test"]["by_type"][t] for r in runs]))
                                      for t in runs[0]["td_test"]["by_type"]},
                       "td_by_workflow": {w: r3(np.mean([r["td_test"]["by_subset"][w] for r in runs]))
                                          for w in runs[0]["td_test"]["by_subset"]},
                       "ood_accuracy": r3(a1["mean_ood_accuracy"]),
                       "ood_by_subset": {k: r3(np.mean([r["ood"]["by_subset"][k] for r in runs]))
                                         for k in runs[0]["ood"]["by_subset"]}}
        out[enc] = e
    a2, a3, a4 = load("a2.json"), load("a3.json"), load("a4.json")
    if a2:
        out["full"]["a2"] = {"mean_accuracy": r3(a2["mean_accuracy"]),
                             "per_workflow": {k: r3(v) for k, v in a2["per_workflow"].items()}}
    if a3:
        out["full"]["a3"] = {"td_test_accuracy": r3(a3["mean_td_test_accuracy"]), "td_test_sd": r3(a3["sd_td_test_accuracy"]),
                             "ood_accuracy": r3(a3["mean_ood_accuracy"]),
                             "ood_by_subset": {k: r3(v) for k, v in a3["mean_ood_by_subset"].items()},
                             "n_train_rows": a3["n_train_rows"], "n_sources": a3["n_sources"]}
    if a4:
        out["full"]["a4"] = {k: r3(v) for k, v in a4["curve"].items()}
    for name in ("timing", "a3_diag"):
        d = load(f"{name}.json")
        if d:
            out["full"][name] = {k: (r3(v) if isinstance(v, float) else v) for k, v in d.items()}
    # state embedding wall-clock, 1,672 unique states per prompt on 4 vCPU (out/*embed*.log)
    out["embed_seconds_per_prompt"] = {"full": EMBED_S["full"], "cut4": EMBED_S["cut4"]}
    (RES / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
