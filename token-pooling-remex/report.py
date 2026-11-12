#!/usr/bin/env python3
"""Render the result tables into RESULTS.md (between the tables markers) and
write summary.json with every number the prose cites."""
from __future__ import annotations

import json, re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORPORA = ("scifact", "nfcorpus")
MODELS = ("aware", "baseline")
# iso-byte comparisons within a model: (a, b), a - b. Arms with a haar s1 twin use the
# per-query mean of both seeds.
ISO = [("pf2 2-bit", "none 1-bit"), ("pf4 4-bit", "none 1-bit"), ("pf4 4-bit", "pf2 2-bit"),
       ("pf2 4-bit", "none 2-bit")]
# pooling cost per model, and the aware-minus-baseline difference of that cost
POOL_COST = ["pf2 fp32", "pf4 fp32", "mt32 fp32", "mt16 fp32", "pf2 2-bit", "pf4 4-bit"]


def boot(delta, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(n, len(delta)))].mean(1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(t):
    m, lo, hi = t
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def arm(pq, name):
    """Per-query nDCG@10; mean over seeds when a haar s1 twin exists."""
    twin = f"{name} haar s1"
    return (pq[name] + pq[twin]) / 2 if twin in pq else pq[name]


def main():
    out, summary = [], {}
    loaded = {}
    for c in CORPORA:
        for m in MODELS:
            j = HERE / f"results_{m}_{c}.json"
            if j.exists():
                loaded[m, c] = (json.load(open(j)), dict(np.load(HERE / f"results_{m}_{c}_perquery.npz")))
    for c in CORPORA:
        if not all((m, c) in loaded for m in MODELS):
            continue
        (ja, pa), (jb, pb) = loaded["aware", c], loaded["baseline", c]
        rb = {r["arm"]: r for r in jb["rows"]}
        out.append(f"### {c}: every arm (nDCG@10; Δ = arm − the same model's unpooled fp32; aware − baseline = paired per-query difference of absolute nDCG@10; all CIs 95% paired bootstrap)\n")
        out.append("| arm | tok/doc | B/doc | aware | aware Δ | baseline | baseline Δ | aware − baseline |")
        out.append("|---|---:|---:|---:|---|---:|---|---|")
        s = summary.setdefault(c, {"rows": {}, "iso": {}, "pool_cost": {}})
        for r in ja["rows"]:
            b = rb[r["arm"]]
            ab = boot(pa[r["arm"]] - pb[r["arm"]])
            isref = r["arm"] == "none fp32"
            da_s = "ref" if isref else fmt((r["delta"], *r["ci"])); db_s = "ref" if isref else fmt((b["delta"], *b["ci"]))
            out.append(f"| {r['arm']} | {r['tokens_per_doc']:.1f} | {r['bytes_per_doc']:,} | {r['ndcg10']:.4f} | "
                       f"{da_s} | {b['ndcg10']:.4f} | {db_s} | {fmt(ab)} |")
            s["rows"][r["arm"]] = dict(bytes=r["bytes_per_doc"], aware=r["ndcg10"], baseline=b["ndcg10"],
                                       aware_delta=r["delta"], baseline_delta=b["delta"], aware_minus_baseline=ab)
        out.append("")
        out.append(f"### {c}: same bytes, split differently between token count and bits (ΔnDCG@10 a − b, 95% CI; an arm with a `haar s1` twin is the per-query mean of its default-rotation and haar-seed-1 runs)\n")
        out.append("| a | b | B/doc a / b | aware Δ | baseline Δ |")
        out.append("|---|---|---|---|---|")
        ra = {r["arm"]: r for r in ja["rows"]}
        for a, b in ISO:
            ta, tb = boot(arm(pa, a) - arm(pa, b)), boot(arm(pb, a) - arm(pb, b))
            out.append(f"| {a} | {b} | {ra[a]['bytes_per_doc']:,} / {ra[b]['bytes_per_doc']:,} | {fmt(ta)} | {fmt(tb)} |")
            s["iso"][f"{a} vs {b}"] = dict(aware=ta, baseline=tb)
        out.append("")
        out.append(f"### {c}: does pooling-aware training shrink what pooling loses? (ΔnDCG@10 pooled − unpooled at the same codec, negative = pooling lost; same rotation averaging as above, so the 2/4-bit rows differ from the single-run Δs in the first table)\n")
        out.append("| pooled arm | unpooled ref | aware pooled − unpooled | baseline pooled − unpooled | aware − baseline |")
        out.append("|---|---|---|---|---|")
        for p in POOL_COST:
            ref = "none " + p.split(" ", 1)[1]
            da, db = arm(pa, p) - arm(pa, ref), arm(pb, p) - arm(pb, ref)
            out.append(f"| {p} | {ref} | {fmt(boot(da))} | {fmt(boot(db))} | {fmt(boot(da - db))} |")
            s["pool_cost"][p] = dict(aware=boot(da), baseline=boot(db), diff=boot(da - db))
        out.append("")
        g = "; ".join(f"{k} {ja['geometry'][k]['norm_of_mean']} / {jb['geometry'][k]['norm_of_mean']}" for k in ja["geometry"])
        nq = len(next(iter(pa.values())))
        out.append(f"{c}: {ja['n_docs']:,} docs, {ja['n_tokens']:,} tokens (same tokenizer and length cap for both models), {nq} queries. "
                   f"Norm of the corpus mean, aware / baseline, by pooling (remex's centering predictor): {g}. "
                   f"Bench time aware {ja['seconds']} s, baseline {jb['seconds']} s on 4 vCPU.\n")
    # codec costs on unpooled tokens, two-rotation means where available
    for c in CORPORA:
        for m in MODELS:
            if (m, c) not in loaded or c not in summary:
                continue
            pq = loaded[m, c][1]
            mean = lambda n: float(arm(pq, n).mean())
            summary[c].setdefault("drops", {})[m] = dict(
                two_to_one_bit=mean("none 1-bit") - mean("none 2-bit"),
                fp32_to_two_bit=mean("none 2-bit") - mean("none fp32"),
                fp32_to_one_bit=mean("none 1-bit") - mean("none fp32"),
                pf4_4bit_vs_fp32=mean("pf4 4-bit") - mean("none fp32"))
    json.dump(summary, open(HERE / "summary.json", "w"), indent=1)
    res = HERE / "RESULTS.md"
    txt = res.read_text() if res.exists() else "<!-- tables:start -->\n<!-- tables:end -->\n"
    txt = re.sub(r"<!-- tables:start -->.*<!-- tables:end -->",
                 "<!-- tables:start -->\n" + "\n".join(out) + "\n<!-- tables:end -->", txt, flags=re.S)
    res.write_text(txt)
    print("\n".join(out))


if __name__ == "__main__":
    main()
