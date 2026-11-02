"""Collect the numbers RESULTS.md cites into results/headline.json (rounded as cited)."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
s = json.loads((R / "summary.json").read_text())
br5 = json.loads((R / "bridge_i2w-render-5.json").read_text())
lay = json.loads((R / "bridge_i2w-layout-6_raw.json").read_text())
svg = json.loads((R / "svg_arm.json").read_text())
ims = list(s["images"])
h = {}
h["gap_vs_best_db"] = {k: round(-v["median_delta_db_vs_best"], 2) for k, v in s["by_n"].items()}
h["gap_vs_jpeg_down_db"] = {k: round(-v["jpeg_down_delta"], 2) for k, v in s["by_n"].items()}
h["gap_vs_avif_down_db"] = {k: round(-v["avif_down_delta"], 2) for k, v in s["by_n"].items()}
h["median_psnr"] = {k: round(v["median_psnr"], 2) for k, v in s["by_n"].items()}
h["median_bytes"] = {k: int(round(v["median_bytes"])) for k, v in s["by_n"].items()}
h["prefix_cost_db"] = {k: round(-v["median_prefix_minus_plain_db"], 2) for k, v in s["prefix_cost"].items()}
h["untrained_prefix_gap_db"] = {k: round(v["median_trained_minus_untrained_db"], 2) for k, v in s["untrained_prefix"].items()}
res = s["residual"]
h["residual"] = {"median_gain_db": round(st.median(r["gain_db"] for r in res), 2),
                 "median_extra_bytes": int(st.median(r["extra_bytes"] for r in res)),
                 "median_splat_gain_same_bytes_db": round(st.median(r["splat_db_per_extra_bytes_at_same_cost"] for r in res), 2)}
h["bytes"] = {"per_splat_gz": round(s["bytes"]["per_splat_gz_median"], 1), "overhead_gz": int(s["bytes"]["overhead_gz_median"]),
              "raw_bits_per_splat": s["bytes"]["raw_bits_per_splat"]}
h["columnar_smaller_pct"] = round(100 * (1 - s["encodings"]["columnar"]["median_gz_ratio_vs_packed_decimal"]), 1)
g = [r["psnr_gpu_vs_torch"] for im in ims for r in s["images"][im]["gpu_check"]]
h["swiftshader_vs_torch_db_min"] = round(min(g), 1)
h["swiftshader_max_abs"] = max(r["max_abs"] for im in ims for r in s["images"][im]["gpu_check"])
sh = {x["id"]: x for x in br5["shaders"]}
full = sh["kodim23_prefix1024_n1024"]
h["iphone"] = {"device": br5["device"], "max_abs_vs_torch": max(x["max_abs_vs_torch"] for x in br5["shaders"]),
               "frac_px_differ_max_pct": round(100 * max(x["frac_px_differ"] for x in br5["shaders"]), 2),
               "psnr_1024": round(full["psnr_gpu"], 2),
               "ms_1024": {k: v["median"] for k, v in full["ms"].items()},
               "ms_16": {k: v["median"] for k, v in sh["kodim23_prefix1024_n16"]["ms"].items()}}
logs = {l.split(" ", 1)[0]: json.loads(l.split(" ", 1)[1]) for l in lay["logs"]}
h["layout_ms_768"] = {k.replace("kodim23_prefix1024_n1024_", ""): v["768x512"]["median"] for k, v in logs.items()}
h["layout_slowdown"] = {k: round(v / h["layout_ms_768"]["u32"], 2) for k, v in h["layout_ms_768"].items()}
h["layout_error_4096"] = lay.get("error")
h["svg_arm"] = {k: {"gz": v["wgsl_gz"], "psnr": round(v["psnr_shader"], 2) if "psnr_shader" in v else None, "error": v.get("error")} for k, v in svg.items()}
h["kodim23"] = {f"n{r['n']}": {"bytes": r["gz_packed"], "psnr": round(r["psnr"], 2)} for r in s["images"]["kodim23"]["prefix1024"]["points"]}
fit_s = [s["images"][im]["prefix1024"]["seconds"] for im in ims if im != "kodim23"]
h["fit_minutes_median"] = round(st.median(fit_s) / 60, 1)
gaps = [r["psnr"] - r["avif_psnr_at_bytes"] for im in ims for r in s["images"][im]["vs_baselines"] if r["n"] == 1024]
h["avif_gap_1024_db"] = {"min": round(-max(gaps), 1), "max": round(-min(gaps), 1)}
h["splat_cap_one_array"] = ((2047 - 1) * 32 - 12 * 6) // 61
(R / "headline.json").write_text(json.dumps(h, indent=1))
print(json.dumps(h, indent=1))
