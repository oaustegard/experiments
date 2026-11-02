"""Turn the sweep's runs into rate-distortion numbers, comparisons and a plot.

    python3 analyze.py [--gpu]   -> results/summary.json, results/rd.png
--gpu also renders every full shader in headless Chromium and checks it against the torch decode.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codec  # noqa: E402
from run_sweep import IMAGES, RUNS  # noqa: E402

KS = [16, 32, 64, 128, 256, 512, 1024]
BASE_ARMS = ["jpeg", "jpeg_down", "webp", "webp_down", "avif", "avif_down"]


def frontier(points):
    """Upper-left Pareto frontier of (bytes, psnr), sorted by bytes."""
    pts = sorted(points)
    out, best = [], -1e9
    for b, p in pts:
        if p > best:
            out.append((b, p))
            best = p
    return out


def psnr_at(front, b):
    """PSNR the arm reaches at b bytes (log-bytes interpolation on its frontier); None outside its range."""
    if not front or b < front[0][0]:
        return None
    if b >= front[-1][0]:
        return front[-1][1]
    for (b0, p0), (b1, p1) in zip(front, front[1:]):
        if b0 <= b <= b1:
            t = (math.log(b) - math.log(b0)) / (math.log(b1) - math.log(b0))
            return p0 + t * (p1 - p0)


def bytes_for(front, p):
    """Bytes the arm needs to reach PSNR p; None if it never does or starts above it."""
    if not front or p > front[-1][1]:
        return None
    if p <= front[0][1]:
        return None
    for (b0, p0), (b1, p1) in zip(front, front[1:]):
        if p0 <= p <= p1:
            t = (p - p0) / (p1 - p0)
            return math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))


def run_points(run, residual=None, ks=KS, encodings=("packed", "columnar", "hex")):
    tgt = run["target"].numpy()
    rows = []
    for k in ks:
        if k > run["P"].shape[0]:
            continue
        img = codec.render_image(run["P"], run["bg"], run["H"], run["W"], run["A"], run["layout"], n=k,
                                 residual=residual if k == run["P"].shape[0] else None)
        row = {"n": k, "psnr": codec.psnr8(img.numpy(), tgt)}
        for enc in encodings:
            src = codec.emit_wgsl(run, n=k, encoding="packed" if enc == "hex" else enc, hexlit=enc == "hex",
                                  residual=residual if k == run["P"].shape[0] else None)
            r = codec.rate(src)
            row[f"gz_{enc}"] = r["gz"]
            row[f"min_{enc}"] = r["min"]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    base = json.loads((HERE / "results" / "baselines.json").read_text())
    svg_arm = json.loads((HERE / "results" / "svg_arm.json").read_text()) if (HERE / "results" / "svg_arm.json").exists() else {}
    cache_path = HERE / "results" / "points_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    summary = {"images": {}}
    prev = json.loads((HERE / "results" / "summary.json").read_text()) if (HERE / "results" / "summary.json").exists() else {"images": {}}
    for im in IMAGES:
        img_path = HERE / "data" / "kodak" / f"{im}.png"
        S = {}
        for arm in ["prefix1024", "plain1024", "plain64", "plain256", "res16"]:
            f = RUNS / f"{im}_{arm}.json"
            if not f.exists():
                continue
            key = f"{im}_{arm}:{f.stat().st_mtime_ns}"
            if key not in cache:
                run, res = codec.load_run(f, img_path)
                ks = KS if arm in ("prefix1024", "plain1024") else [run["P"].shape[0]]
                encs = ("packed", "columnar", "hex") if arm == "prefix1024" else ("packed",)
                cache[key] = {"points": run_points(run, res, ks, encs), "seconds": run["seconds"]}
                cache_path.write_text(json.dumps(cache))
                print(im, arm, "done", flush=True)
            S[arm] = cache[key]
        if "prefix1024" not in S:
            continue
        # empty-shader overhead (decoder only, background, zero splats)
        run, _ = codec.load_run(RUNS / f"{im}_prefix1024.json")
        S["overhead_gz"] = codec.rate(codec.emit_wgsl(run, n=0))["gz"]
        fronts = {arm: frontier([(r["bytes"], r["psnr"]) for r in base[im][arm]]) for arm in BASE_ARMS}
        cmp_rows = []
        for r in S["prefix1024"]["points"]:
            b, p = r["gz_packed"], r["psnr"]
            row = {"n": r["n"], "bytes": b, "psnr": p}
            for arm in BASE_ARMS:
                pa = psnr_at(fronts[arm], b)
                row[f"{arm}_psnr_at_bytes"] = pa
                bb = bytes_for(fronts[arm], p)
                row[f"{arm}_bytes_for_psnr"] = bb
            cands = [row[f"{arm}_psnr_at_bytes"] for arm in BASE_ARMS if row[f"{arm}_psnr_at_bytes"] is not None]
            row["best_baseline_psnr_at_bytes"] = max(cands) if cands else None
            row["best_baseline_arm"] = max(((row[f"{arm}_psnr_at_bytes"], arm) for arm in BASE_ARMS
                                           if row[f"{arm}_psnr_at_bytes"] is not None), default=(None, None))[1]
            row["delta_db_vs_best"] = (p - row["best_baseline_psnr_at_bytes"]) if cands else None
            bcands = [row[f"{arm}_bytes_for_psnr"] for arm in BASE_ARMS if row[f"{arm}_bytes_for_psnr"] is not None]
            row["best_baseline_bytes_for_psnr"] = min(bcands) if bcands else None
            cmp_rows.append(row)
        S["vs_baselines"] = cmp_rows
        S["svg_arm"] = {k: v for k, v in svg_arm.items() if k.startswith(im + "_")}
        if a.gpu:
            sys.path.insert(0, str(HERE))
            import verify_gpu
            run, res = codec.load_run(RUNS / f"{im}_prefix1024.json", img_path)
            rows, _ = verify_gpu.check(run, None, ns=(64, None), encodings=("packed",))
            S["gpu_check"] = rows
            if "res16" in S:
                run2, res2 = codec.load_run(RUNS / f"{im}_res16.json", img_path)
                rows2, _ = verify_gpu.check(run2, res2, ns=(None,), encodings=("packed",))
                S["gpu_check_res16"] = rows2
            print(im, "gpu", [(r["n"], r.get("psnr_gpu_vs_torch"), r.get("max_abs")) for r in S["gpu_check"]], flush=True)
        if not a.gpu:  # keep GPU checks from an earlier --gpu run
            for k in ("gpu_check", "gpu_check_res16"):
                if k in prev["images"].get(im, {}):
                    S[k] = prev["images"][im][k]
        summary["images"][im] = S
    # aggregates
    agg = {}
    ims = [im for im in IMAGES if im in summary["images"]]
    for i, k in enumerate(KS):
        rows = [summary["images"][im]["vs_baselines"][i] for im in ims]
        d = [r["delta_db_vs_best"] for r in rows if r["delta_db_vs_best"] is not None]
        agg[str(k)] = {
            "median_bytes": float(np.median([r["bytes"] for r in rows])),
            "median_psnr": float(np.median([r["psnr"] for r in rows])),
            "median_delta_db_vs_best": float(np.median(d)) if d else None,
            "n_images": len(rows),
            "avif_down_delta": float(np.median([r["psnr"] - r["avif_down_psnr_at_bytes"] for r in rows if r["avif_down_psnr_at_bytes"] is not None])) if any(r["avif_down_psnr_at_bytes"] is not None for r in rows) else None,
            "jpeg_down_delta": float(np.median([r["psnr"] - r["jpeg_down_psnr_at_bytes"] for r in rows if r["jpeg_down_psnr_at_bytes"] is not None])) if any(r["jpeg_down_psnr_at_bytes"] is not None for r in rows) else None,
            "webp_down_delta": float(np.median([r["psnr"] - r["webp_down_psnr_at_bytes"] for r in rows if r["webp_down_psnr_at_bytes"] is not None])) if any(r["webp_down_psnr_at_bytes"] is not None for r in rows) else None,
            "avif_delta": float(np.median([r["psnr"] - r["avif_psnr_at_bytes"] for r in rows if r["avif_psnr_at_bytes"] is not None])) if any(r["avif_psnr_at_bytes"] is not None for r in rows) else None,
        }
    summary["by_n"] = agg
    # prefix cost: dedicated plain runs vs the prefix run's same-length prefix
    pc = {}
    for k, arm in [(64, "plain64"), (256, "plain256"), (1024, "plain1024")]:
        diffs = []
        for im in ims:
            S = summary["images"][im]
            if arm in S:
                plain = [r for r in S[arm]["points"] if r["n"] == k][0]["psnr"]
                pre = [r for r in S["prefix1024"]["points"] if r["n"] == k][0]["psnr"]
                diffs.append(pre - plain)
        if diffs:
            pc[str(k)] = {"median_prefix_minus_plain_db": float(np.median(diffs)), "values": diffs}
    summary["prefix_cost"] = pc
    # untrained prefixes of plain1024 vs trained prefixes
    up = {}
    for k in KS[:-1]:
        diffs = []
        for im in ims:
            S = summary["images"][im]
            if "plain1024" in S:
                a_ = [r for r in S["plain1024"]["points"] if r["n"] == k][0]["psnr"]
                b_ = [r for r in S["prefix1024"]["points"] if r["n"] == k][0]["psnr"]
                diffs.append(b_ - a_)
        if diffs:
            up[str(k)] = {"median_trained_minus_untrained_db": float(np.median(diffs)), "values": diffs}
    summary["untrained_prefix"] = up
    # residual
    rr = []
    for im in ims:
        S = summary["images"][im]
        if "res16" in S:
            full = S["prefix1024"]["points"][-1]
            res = S["res16"]["points"][-1]
            half = S["prefix1024"]["points"][-2]
            slope = (full["psnr"] - half["psnr"]) / (full["gz_packed"] - half["gz_packed"])  # dB per byte, 512->1024
            rr.append({"image": im, "gain_db": res["psnr"] - full["psnr"], "extra_bytes": res["gz_packed"] - full["gz_packed"],
                       "splat_db_per_extra_bytes_at_same_cost": slope * (res["gz_packed"] - full["gz_packed"])})
    summary["residual"] = rr
    # encodings
    enc = {}
    for e in ["columnar", "hex"]:
        ratios = [r[f"gz_{e}"] / r["gz_packed"] for im in ims for r in summary["images"][im]["prefix1024"]["points"]]
        enc[e] = {"median_gz_ratio_vs_packed_decimal": float(np.median(ratios)), "min": float(min(ratios)), "max": float(max(ratios))}
    summary["encodings"] = enc
    ovh = [summary["images"][im]["overhead_gz"] for im in ims]
    per = [(summary["images"][im]["prefix1024"]["points"][-1]["gz_packed"] - summary["images"][im]["overhead_gz"]) / 1024 for im in ims]
    summary["bytes"] = {"overhead_gz_median": float(np.median(ovh)), "per_splat_gz_median": float(np.median(per)),
                        "raw_bits_per_splat": codec.Layout().bits}
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(summary, base, ims)
    print(json.dumps({k: summary[k] for k in ["by_n", "prefix_cost", "untrained_prefix", "residual", "encodings", "bytes"]}, indent=1))


def plot(summary, base, ims):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 3
    rows = math.ceil(len(ims) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
    colors = {"jpeg_down": "#c0504d", "webp_down": "#9bbb59", "avif": "#4f81bd", "avif_down": "#8064a2"}
    for ax, im in zip(axes.flat, ims):
        S = summary["images"][im]
        pts = S["prefix1024"]["points"]
        ax.plot([r["gz_packed"] for r in pts], [r["psnr"] for r in pts], "-o", color="#111", ms=4, lw=2,
                label="shader (prefixes of one run)", zorder=5)
        for arm in ["plain64", "plain256"]:
            if arm in S:
                r = S[arm]["points"][-1]
                ax.plot(r["gz_packed"], r["psnr"], "x", color="#e36c09", ms=8, mew=2, label="shader, dedicated plain run" if arm == "plain64" else None, zorder=6)
        if "plain1024" in S:
            p2 = S["plain1024"]["points"]
            ax.plot([r["gz_packed"] for r in p2], [r["psnr"] for r in p2], ":", color="#e36c09", lw=1.5, label="prefixes of a plain run")
        if "res16" in S:
            r = S["res16"]["points"][-1]
            ax.plot(r["gz_packed"], r["psnr"], "s", color="#17becf", ms=6, label="+ SIREN residual", zorder=6)
        for arm, c in colors.items():
            f = frontier([(r["bytes"], r["psnr"]) for r in base[im][arm]])
            f = [x for x in f if x[0] <= 40000]
            ax.plot([x[0] for x in f], [x[1] for x in f], "-", color=c, lw=1.3, label=arm.replace("_down", " (downscaled)"))
        sv = S.get("svg_arm") or {}
        if sv:
            xs = sorted(sv.values(), key=lambda v: v["wgsl_gz"])
            ax.plot([v["wgsl_gz"] for v in xs], [v.get("psnr_shader", float("nan")) for v in xs], "--^", color="#7f7f7f", ms=5, label="image-to-svg → forge shader")
        ax.set_xscale("log")
        ax.set_xlim(300, 40000)
        ax.set_title(im)
        ax.set_xlabel("bytes (gzip)")
        ax.set_ylabel("PSNR dB")
        ax.grid(alpha=0.3, which="both")
    for ax in list(axes.flat)[len(ims):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "results" / "rd.png", dpi=110)


if __name__ == "__main__":
    main()
