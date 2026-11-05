"""Compare the WGSL training run (a GPU Bridge result) with the seeded CPU run of the same recipe.

    python3 -I run/compare.py RESULT_DIR    # the `gpu_bridge.py fetch` output dir; writes results/trajectory.json

Per-step loss: relative difference |gpu - cpu| / cpu, mean over epoch 1 and over the whole run (P2: under
1% over epoch 1). Per epoch: val / test / reply_thread metrics side by side. Final epoch: per-example
probability differences and how often the two models' answers (p > 0.5) agree.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def main():
    rd = Path(sys.argv[1])
    res = json.loads((rd / "result.json").read_text()) if (rd / "result.json").exists() else None
    if res is None:
        res = json.loads(next(rd.glob("*.json")).read_text())
    out = res.get("outputs", res)
    cpu = json.loads((HERE / "results" / "cpu_traj.json").read_text())
    gl = np.load(rd / "losses.npy").astype(np.float64)
    cl = np.array(cpu["steps"], np.float64)
    n = min(len(gl), len(cl))
    per_ep = len(cl) // len(cpu["epochs"])
    rel = np.abs(gl[:n] - cl[:n]) / np.abs(cl[:n])
    r = {"device": {k: res.get(k) for k in ("device", "adapter", "tabName", "ua", "runMs")},
         "steps": {"gpu": len(gl), "cpu": len(cl), "per_epoch": per_ep},
         "loss_rel_diff": {"epoch1_mean": float(rel[:per_ep].mean()), "all_mean": float(rel.mean()), "all_max": float(rel.max()),
                           "by_epoch_mean": [float(rel[i * per_ep:(i + 1) * per_ep].mean()) for i in range(n // per_ep)],
                           "first10": rel[:10].tolist()},
         "epochs": []}
    for ge, ce in zip(out["epochs"], cpu["epochs"]):
        r["epochs"].append({"epoch": ge["epoch"], "gpu_train_s": ge["trainMs"] / 1000, "cpu_train_s_cumulative": ce["train_s"],
                            **{k: {"gpu": {m: ge[k][m] for m in ("accuracy", "recall", "false_pos_rate", "mean_abs_err")},
                                   "cpu": {m: ce[k][m] for m in ("accuracy", "recall", "false_pos_rate", "mean_abs_err")}}
                               for k in ("val", "test", "reply_thread")}})
    agree = {}
    for k in ("val", "test", "reply_thread"):
        gp = np.load(rd / f"probs_{k}.npy").astype(np.float64)
        cp = np.fromfile(HERE / "fixtures" / f"cpu_probs_{k}.bin", np.float32).astype(np.float64)
        d = np.abs(gp - cp)
        agree[k] = {"n": len(cp), "max_abs_prob_diff": float(d.max()), "mean_abs_prob_diff": float(d.mean()),
                    "answer_agreement": float(((gp > 0.5) == (cp > 0.5)).mean())}
    r["final_probs"] = agree
    r["timing"] = {"gpu_train_wall_s": out["trainWallMs"] / 1000, "gpu_init_ms": out["initMs"], "gpu_sync_step_ms": out["syncStepMs"],
                   "gpu_sync_step_batch": out["syncStepBatch"], "cpu_train_wall_s": cpu["train_wall_s"],
                   "cpu_note": "the CPU run shared its 4 vCPUs with a SwiftShader job for about its first 300 steps"}
    (HERE / "results" / "trajectory.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "epochs"}, indent=1))
    for e in r["epochs"]:
        print(e["epoch"], {k: (round(e[k]["gpu"]["accuracy"], 4), round(e[k]["cpu"]["accuracy"], 4)) for k in ("val", "test", "reply_thread")})


if __name__ == "__main__":
    main()
