"""Pool the deep-zoom bridge results into results/deepzoom.json.

    python3 deepzoom_analyze.py
Reads results/bridge_dz-<image>-arms-1.json (six DIV2K images, four tile arms) and bridge_dz-<image>-n<k>-1.json
(splats per tile), plus results/deepzoom_private.json if present (numbers only, for a photo that is not in the repo).
PSNR per level is pooled over that level's tiles (sum of squared errors over all core pixels).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
IDS = ["0801", "0802", "0803", "0804", "0805", "0806"]
ARMS = ["residual", "prefix", "indep", "indep2"]


def load(p):
    d = json.loads(p.read_text())
    return d, d["outputs"]


def med(xs):
    return round(st.median(xs), 2)


def main():
    out = {"images": {}, "median": {}, "tile_seconds": {}, "sweep": {}, "draw_ms": [], "seams": {}}
    secs = []
    for im in IDS:
        d, o = load(R / f"bridge_dz-{im}-arms-1.json")
        e = {"device": d["device"], "source": [o["Ws"], o["Hs"]], "base_size": [o["W0"], o["H0"]], "base_psnr": round(o["base"]["psnr"], 2),
             "base_seconds": round(o["base"]["seconds"], 2), "levels": [[lv["W"], lv["H"]] for lv in o["levels"]], "detail_bytes": o.get("detailBytes")}
        for arm in ARMS:
            a = o["arms"][arm]
            e[arm] = {f"L{lv['L']}": {k: v for k, v in lv.items() if k != "L"} for lv in a["levels"]}
            if arm == "residual":
                secs += [t["seconds"] for t in a["tiles"]]
                e["tile_total_ms_median"] = med([t["totalMs"] for t in a["tiles"]])
        res = e["residual"]
        for L in ["L1", "L2"]:
            r = res[L]
            e[f"gain_{L}"] = round(r["detail"] - r["base"], 2)
            e[f"vs_bicubic_{L}"] = round(r["detail"] - r["bicubic"], 2)
            e[f"vs_indep_{L}"] = round(r["detail"] - e["indep"][L]["detail"], 2)
            e[f"vs_indep2_{L}"] = round(r["detail"] - e["indep2"][L]["detail"], 2)
            e[f"prefix_cost_{L}"] = round(r["detail"] - e["prefix"][L]["detail"], 2)
            e[f"drawn_minus_trained_{L}"] = round(r["drawn"] - r["detail"], 3)
            # splats on screen for a view at this level: base + every level up to it (residual) or the level alone
            e[f"splats_drawn_{L}"] = {"residual": 1024 + sum(res[f"L{k}"]["splats"] for k in range(1, int(L[1]) + 1)),
                                      "indep": e["indep"][L]["splats"], "indep2": e["indep2"][L]["splats"]}
        sm = o.get("seams") or []
        if sm:
            s = sm[0]
            e["seams_L2"] = {"bins_px": s["bins"], "detail_db": s["detailDb"], "base_db": s["baseDb"], "pixels": s["n"],
                             "edge_minus_far_detail": round(s["detailDb"][0] - s["detailDb"][-1], 2),
                             "edge_minus_far_base": round(s["baseDb"][0] - s["baseDb"][-1], 2)}
        out["images"][im] = e
    I = out["images"].values()
    for k in ["gain_L1", "gain_L2", "vs_bicubic_L1", "vs_bicubic_L2", "vs_indep_L1", "vs_indep_L2", "vs_indep2_L1", "vs_indep2_L2",
              "prefix_cost_L1", "prefix_cost_L2"]:
        out["median"][k] = med([e[k] for e in I])
    for L in ["L1", "L2"]:
        for f in ["base", "below", "detail", "bicubic"]:
            out["median"][f"{f}_{L}"] = med([e["residual"][L][f] for e in I])
        out["median"][f"indep_{L}"] = med([e["indep"][L]["detail"] for e in I])
        out["median"][f"indep2_{L}"] = med([e["indep2"][L]["detail"] for e in I])
    out["median"]["seam_edge_minus_far_detail"] = med([e["seams_L2"]["edge_minus_far_detail"] for e in I])
    out["median"]["seam_edge_minus_far_base"] = med([e["seams_L2"]["edge_minus_far_base"] for e in I])
    out["median"]["bicubic_lead_L2"] = round(-out["median"]["vs_bicubic_L2"], 2)
    out["median"]["seam_drop_detail"] = round(-out["median"]["seam_edge_minus_far_detail"], 2)
    out["median"]["seam_drop_base"] = round(-out["median"]["seam_edge_minus_far_base"], 2)
    out["median"]["drawn_minus_trained_max_abs"] = max(abs(e[f"drawn_minus_trained_{L}"]) for e in I for L in ["L1", "L2"])
    out["median"]["base_psnr"] = med([e["base_psnr"] for e in I])
    out["tile_seconds"] = {"device": "Mac (M5 Max)", "tiles": len(secs), "median": round(st.median(secs), 3), "min": round(min(secs), 3), "max": round(max(secs), 3)}
    # frame times from the first run (draw rows), M5
    d, o = load(R / "bridge_dz-0801-res-1.json") if (R / "bridge_dz-0801-res-1.json").exists() else (None, None)
    if o:
        out["draw_ms"] = o["draws"]
    # splats per tile
    for p in sorted(R.glob("bridge_dz-*-n*-1.json")):
        d, o = load(p)
        im, n = p.stem.split("-")[1], int(p.stem.split("-")[2][1:])
        a = o["arms"]["residual"]
        out["sweep"].setdefault(im, {})[n] = {f"L{lv['L']}": {"detail": lv["detail"], "splats": lv["splats"], "seconds": lv["seconds"]} for lv in a["levels"]}
        out["sweep"][im][n]["tile_seconds_median"] = round(st.median(t["seconds"] for t in a["tiles"]), 3)
    for im in out["sweep"]:
        e = out["images"][im]
        out["sweep"][im][256] = {L: {"detail": e["residual"][L]["detail"], "splats": e["residual"][L]["splats"], "seconds": e["residual"][L]["seconds"]} for L in ["L1", "L2"]}
    for im in out["sweep"]:
        e = out["images"][im]
        for n in (512, 1024):
            if n in out["sweep"][im]:
                for L in ["L1", "L2"]:
                    sw = out["sweep"][im][n][L]
                    sw["vs_bicubic"] = round(sw["detail"] - e["residual"][L]["bicubic"], 2)
                    sw["vs_256"] = round(sw["detail"] - e["residual"][L]["detail"], 2)
                    if n == 512:
                        sw["vs_indep_same_n"] = round(sw["detail"] - e["indep2"][L]["detail"], 2)
        priv = R / "deepzoom_private.json"
    if priv.exists():
        out["private_photo"] = json.loads(priv.read_text())
    (R / "deepzoom.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["median"], indent=1))
    print(json.dumps(out["tile_seconds"]))


if __name__ == "__main__":
    main()
