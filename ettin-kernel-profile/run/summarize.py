"""Derive the headline numbers of a profile run into results/summary.json.

    python3 -I run/summarize.py [results/ettin-prof-1/result.json]

Percentages are of the full graph's GPU time (SpikeEngine, timestamp queries) or wall time (Trainer).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def pct(x, tot):
    return round(100 * x / tot, 1)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/ettin-prof-1/result.json"
    r = json.loads(src.read_text())
    o = r["outputs"]
    s = {"device": r["device"], "adapter": r["adapter"], "ua": r["ua"], "runMs": r["runMs"], "sha": o["sha"],
         "workloads": o["workloads"],
         "null_us": {k: round(v, 2) for k, v in o["null"].items()}}
    S = {"dispatches": o["S"]["steps"], "byGroup": o["S"]["byGroup"]}
    for w in ("one", "b16"):
        W = o["S"][w]
        g = W["e2e"]["gpu"]
        S[w] = {
            "gpu_ms": round(g, 2), "wall_ms": round(W["e2e"]["wall"], 2), "score_wall_ms": round(W["score_wall_ms"], 1),
            "outside_gpu_ms": round(W["score_wall_ms"] - g, 1),
            "iso_sum_ms": round(W["isoSum_ms"], 2),
            "iso_pct": {k: pct(v, W["isoSum_ms"]) for k, v in W["isoByGroup_ms"].items()},
            "drop_saves_pct": {k: round(100 * (1 - v["gpu"] / g), 1) for k, v in W["drop"].items()},
            "dispatch_floor_pct": pct(o["S"]["steps"] * o["null"]["rwBinding_us"] / 1000, g),
        }
        glob = [x["us"] for x in W["iso"] if x["group"] == "attn"]
        S[w]["attn_global_us"] = round(max(glob), 1)
        S[w]["attn_local_us"] = round(min(glob), 1)
        lora = [x for x in W["iso"] if x["group"] == "lora"]
        S[w]["lora_A_us"] = round(sum(x["us"] for x in lora if x["grid"][0] == 1))     # X A^T: N = 32 (rank 8 padded)
        S[w]["lora_B_us"] = round(sum(x["us"] for x in lora if x["grid"][0] != 1))     # Y += T B^T: K = 32, full-width Y
        S[w]["lora_B_pct"] = round(100 * S[w]["lora_B_us"] / 1000 / S[w]["gpu_ms"], 1)
        S[w]["elementwise_drop_pct"] = round(sum(S[w]["drop_saves_pct"][k] for k in ("norm", "rope", "glu")), 1)
        S[w]["base_plus_attn_pct"] = round(S[w]["iso_pct"]["base_mm"] + S[w]["iso_pct"]["attn"], 1)
    s["S"] = S
    m = o["merge"]
    s["merge"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "questions"} for k, v in m.items()}
    T = {"batch": o["T"]["batch"]}
    for ph in ("forward", "step"):
        P = o["T"][ph]
        iso = P["isoSum_ms"]
        roles = {}
        for k, v in P["isoByRole_ms"].items():
            role = k.split(":", 1)[1]
            roles[role] = roles.get(role, 0) + v
        T[ph] = {"dispatches": P["dispatches"], "wall_ms": round(P["e2e_wall_ms"], 1), "iso_sum_ms": round(iso, 1),
                 "iso_pct": {k: pct(v, iso) for k, v in sorted(roles.items(), key=lambda x: -x[1])},
                 "iso_pct_dir": {k: pct(v, iso) for k, v in P["isoByRole_ms"].items()},
                 "drop_saves_pct": {k: round(100 * (1 - v / P["e2e_wall_ms"]), 1) for k, v in P["drop"].items()}}
        T[ph]["lora_pct"] = round(T[ph]["iso_pct"].get("lora_mm", 0) + T[ph]["iso_pct"].get("lora_grad", 0), 1)
    s["T"] = T
    out = HERE / "results/summary.json"
    out.write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({"one": S["one"], "step": T["step"]["iso_pct"]}, indent=1))


if __name__ == "__main__":
    main()
