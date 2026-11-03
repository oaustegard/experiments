"""Collect the numbers RESULTS.md cites into results/summary.json."""
import json
from pathlib import Path
from summarize import summarize

H = Path(__file__).resolve().parent
R = H / "results"
j = lambda p: json.loads((R / p).read_text())
s = {}
s["test_gpu"] = {n: summarize(H / f"runs/{n}__gpu_test.jsonl") for n in ("base", "A8p")}
s["dev_gpu"] = {n: summarize(H / f"runs/{n}__gpu_dev.jsonl") for n in ("base", "L33", "L33rand", "N5", "A31p", "A3p", "A8p")}
if (H / "runs/base__test.jsonl").exists():
    s["test_cpu_base"] = summarize(H / "runs/base__test.jsonl")
s["dev_cpu_base"] = summarize(H / "runs/base__dev.jsonl")
s["stats_test_gpu"] = j("stats_test_gpu.json")
ld = j("loopdir.json")
s["loopdir"] = {"auroc_loop": {str(r["layer"]): round(r["auroc_loop"], 3) for r in ld["layers"]},
                "auroc_pre": {str(r["layer"]): round(r["auroc_pre"], 3) for r in ld["layers"]},
                "norm_gap_L33": round(ld["layers"][33]["norm_gap"], 1), "resid_norm_L33": round(ld["layers"][33]["resid_norm"], 1),
                "counts": ld["counts"], "n_loops": len(ld["loops"])}
w, wp = j("writers.json"), j("writers_perp.json")
s["writers"] = {"mlp22": round(w["mlp"][22], 1), "mlp29": round(w["mlp"][29], 1), "att31": round(w["att"][31], 1),
                "perp_att31": round(wp["att"][31], 1), "perp_att32": round(wp["att"][32], 1), "perp_att34": round(wp["att"][34], 1),
                "top_neurons": w["top_neurons"][:5]}
L = j("layers.json")
s["layers"] = {"single_kl_min": round(min(L["single_kl"]), 3), "greedy_kl": {str(g["k"]): round(g["kl"], 3) for g in L["greedy"]},
               "oneshot_kl": {k: round(v, 3) for k, v in L["oneshot"].items()}}
bench = {}
for jid in ("mwe-bench-base-1", "mwe-bench-base-2", "mwe-bench-A8p-1"):
    o = json.loads((H / f"bridge/results/{jid}/result.json").read_text())
    bench[jid] = {"device": o["device"], **{f"B{b['B']}_decode_ms": round(sorted(x["decodeMedMs"] for x in b["runs"])[1], 1) for b in o["outputs"]["bench"]}}
s["bench"] = bench
s["gpu_timing"] = {n: {k: round(v, 1) for k, v in j(f"gpu_timing_{n}.json").items() if k in ("tok_per_s", "step_med_ms_median", "tokens")} for n in ("base", "A8p")}
s["parity"] = {"mac_max_abs_logit_diff": 5.53e-05, "mac_greedy_agree": "120/120", "swiftshader_max_abs_logit_diff": 2.68,
               "wasm_max_abs_logit_diff": 3.62e-05, "A8p_onnx_vs_torch": 5.15e-05, "torch_vs_ort_cpu": 6.68e-05}
s["sink"] = {"token0_proj_on_u": -11790, "token0_resid_norm": 15285, "median_resid_norm": 228, "cos_u_sink": -0.771}
(R / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps({k: s[k] for k in ("bench", "gpu_timing")}, indent=1))
for n, v in s["dev_gpu"].items():
    print(n, {k: (v[k].get("acc"), v[k]["loop"]) for k in v if k != "all"}, v["all"]["loop"])
