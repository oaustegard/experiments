"""Collect the headline numbers RESULTS.md cites into results/summary.json."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
j = lambda f: json.load(open(R / f))  # noqa: E731
par = j("parity_tess.json")["summary"]
beam = j("beam_check.json")["models"]
pack = j("pack_sweep.json")["by"]
base = {r["model"]: r for r in j("tess_baseline.json")["overall"]}
b1 = j("bridge_tess-gpu-1_mac.json")["outputs"]["models"]
b2 = j("bridge_tess-gpu-2_mac.json")["outputs"]["models"]
k = j("kaggle_t4_server_bench_1.json")["results"]
fl = j("flops.json")
torch_check = j("torch_check.json")
s = {
    "parity": {
        "lines_per_model": par["best_eng"]["lines"],
        "best_max_dp": max(par[f"best_{l}"]["max_dp"] for l in ("eng", "fra", "grc")),
        "fast_max_dp": max(par[f"fast_{l}"]["max_dp"] for l in ("eng", "fra", "grc")),
        "argmax_all_steps_share_min": min(v["argmax_agree_all_steps"] for v in par.values()),
        "greedy_eq_beam_min": min(v["greedy_eq_beam"] for v in par.values()),
        "greedy_eq_beam_max": max(v["greedy_eq_beam"] for v in par.values()),
    },
    "beam": {"lines": sum(v["lines"] for v in beam.values()), "identical": sum(v["identical"] for v in beam.values())},
    "baseline_binary": {"best_cer": base["best"]["mean_cer"], "fast_cer": base["fast"]["mean_cer"],
                        "best_s_per_line_4proc": base["best"]["sec_per_line"]},
    "pack": {k2.replace("/", "_"): v["cer"] for k2, v in pack.items()},
    "pack_same_text": {k2.replace("/", "_"): v["same_text_as_f32"] for k2, v in pack.items()},
    "mac": {
        "lines": b2["best"]["batches"]["greedy_343"]["lines"],
        "best_gpu_ms_r1": b1["best"]["batches"]["v2_343"]["gpu_ms"],
        "best_gpu_ms": b2["best"]["batches"]["greedy_343"]["gpu"],
        "fast_gpu_ms": b2["fast"]["batches"]["greedy_343"]["gpu"],
        "best_parity_beam": b2["best"]["parity"]["beam"],
        "fast_parity_beam": b2["fast"]["parity"]["beam"],
        "best_beam_decode_ms_r2": b2["best"]["batches"]["beam_343"]["decode"],
        "best_stages_ms": b2["best"]["stages"],
    },
    "t4": {f"{m}_{mode}_b4096": round(k[m][f"{mode}_b4096"]["lines_per_s"], 1)
           for m in ("best", "fast") for mode in ("exact", "cudnn")},
    "t4_parity": {m: {"max_dp": k[m]["exact_vs_ref_max_dp"], "argmax_lines": k[m]["exact_vs_ref_argmax_lines_equal"]}
                  for m in ("best", "fast")},
    "flops": {"best_eng_x3_share": fl["[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]"]["share"]["x3"],
              "best_eng_seq_share": fl["[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]"]["sequential_share"],
              "fast_eng_vs_best": fl["[1,36,0,1Ct3,3,16Mp3,3Lfys48Lfx96Lrx96Lfx192O1c1]"]["vs_eng_best"]},
    "torch_cpu": torch_check.get("throughput", {}).get("lines_per_s"),
}
(R / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s, indent=1))
