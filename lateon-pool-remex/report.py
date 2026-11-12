#!/usr/bin/env python3
"""Render result tables into RESULTS.md (between the tables markers), write summary.json
with every number the prose cites, and draw frontier.png (nDCG@10 against bytes/doc)."""
from __future__ import annotations

import json, re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORPORA = ("scifact", "nfcorpus")
ISO = [("st pf4 4-bit", "none 1-bit"), ("st pf2 2-bit", "none 1-bit"), ("st pf4 4-bit", "st pf2 2-bit"),
       ("st pf2 4-bit", "none 2-bit")]
C_GRID = (0.5, 1.0, 2.0, 4.0)
BITS = (4, 3, 2, 1)


def boot(delta, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(n, len(delta)))].mean(1)
    return [round(float(delta.mean()), 4), round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def load(corpus):
    f = HERE / f"results_{corpus}.jsonl"
    if not f.exists():
        return None
    rows = {}
    for line in open(f):
        r = json.loads(line)
        rows[r["arm"]] = r
    return rows


def pq(rows, name, avg_seed=True):
    """Per-query nDCG@10; mean over both rotations when a haar s1 twin exists."""
    x = np.array(rows[name]["perquery"])
    twin = f"{name} haar s1"
    if avg_seed and twin in rows:
        x = (x + np.array(rows[twin]["perquery"])) / 2
    return x


def main():
    out, summary = [], {}
    for corpus in CORPORA:
        rows = load(corpus)
        if not rows or "none fp32" not in rows:
            continue
        S = summary[corpus] = {}
        ref = pq(rows, "none fp32")
        S["rows"] = {}
        out.append(f"### {corpus}: every arm (nDCG@10; Δ = arm − unpooled fp32, 95% paired bootstrap; "
                   f"B/doc counts 2 B of side scalar per coded vector, codes-only in brackets)\n")
        out.append("| arm | tok/doc | B/doc (codes only) | nDCG@10 | Δ vs none fp32 | norm of mean |")
        out.append("|---|---:|---:|---:|---|---:|")
        for name, r in rows.items():
            d = boot(np.array(r["perquery"]) - ref)
            S["rows"][name] = dict(ndcg=r["ndcg10"], tok=r["tokens_per_doc"], bytes=r["bytes_per_doc"],
                                   bytes_codes=r["bytes_per_doc_codes_only"], delta=d)
            nm = "" if r.get("norm_of_mean") is None else f"{r['norm_of_mean']:.3f}"
            out.append(f"| {name} | {r['tokens_per_doc']:.1f} | {r['bytes_per_doc']:,.0f} "
                       f"({r['bytes_per_doc_codes_only']:,.0f}) | {r['ndcg10']:.4f} | {fmt(d)} | {nm} |")
        out.append("")

        # unpooled codec drops
        S["drops"] = {}
        for a, b, key in (("none 2-bit", "none 1-bit", "two_to_one_bit"), ("none fp32", "none 2-bit", "fp32_to_two_bit"),
                          ("none fp32", "none 4-bit", "fp32_to_four_bit"), ("none fp32", "st pf4 4-bit", "st_pf4_4bit_vs_fp32")):
            if a in rows and b in rows:
                S["drops"][key] = boot(pq(rows, b) - pq(rows, a))

        # side-info diagnostics
        S["side"] = {}
        side_lines = []
        for b in (2, 1):
            base = f"none {b}-bit"
            for v in ("exact-side", "no-side"):
                nm = f"{base} {v}"
                if nm in rows and base in rows:
                    d = boot(np.array(rows[nm]["perquery"]) - np.array(rows[base]["perquery"]))
                    S["side"][f"{b}-bit {v}"] = d
                    side_lines.append(f"| {b}-bit | {v} | {rows[nm]['ndcg10']:.4f} | {rows[base]['ndcg10']:.4f} | {fmt(d)} |")
        if side_lines:
            out.append(f"### {corpus}: per-vector side information (unpooled; Δ = variant − fp16 μ·x, same rotation)\n")
            out.append("| bits | variant | nDCG@10 | fp16 side | Δ |")
            out.append("|---|---|---:|---:|---|")
            out += side_lines + [""]

        # iso-byte (library pooling)
        S["iso"] = {}
        out.append(f"### {corpus}: library pooling at about the same bytes (Δ a − b; seed twins averaged)\n")
        out.append("| a | b | B/doc a / b | Δ |")
        out.append("|---|---|---|---|")
        for a, b in ISO:
            if a in rows and b in rows:
                d = boot(pq(rows, a) - pq(rows, b))
                S["iso"][f"{a} vs {b}"] = d
                out.append(f"| {a} | {b} | {rows[a]['bytes_per_doc']:,.0f} / {rows[b]['bytes_per_doc']:,.0f} | {fmt(d)} |")
        if "wu pf4 fp32" in rows and "st pf4 fp32" in rows:
            for s in ("fp32", "4-bit"):
                d = boot(pq(rows, f"wu pf4 {s}") - pq(rows, f"st pf4 {s}"))
                S["iso"][f"wu pf4 {s} vs st pf4 {s}"] = d
                out.append(f"| wu pf4 {s} | st pf4 {s} | {rows[f'wu pf4 {s}']['bytes_per_doc']:,.0f} / "
                           f"{rows[f'st pf4 {s}']['bytes_per_doc']:,.0f} | {fmt(d)} |")
        out.append("")

        # integration
        S["adapt"] = {}
        out.append(f"### {corpus}: integrated pooling (Ward cut at E_pool ≤ c·D_b) against the unpooled codec at "
                   f"the same bits and against uniform Ward at the same tokens and bits (seed twins averaged)\n")
        out.append("| bits | c | τ | tokens kept | B/doc | nDCG@10 | adapt − none at b | adapt − match | match nDCG@10 |")
        out.append("|---|---|---:|---:|---:|---:|---|---|---:|")
        n_none = rows["none fp32"]["tokens_per_doc"]
        for b in BITS:
            for c in C_GRID:
                a, m, nb = f"adapt c{c:g}@{b} {b}-bit", f"match c{c:g}@{b} {b}-bit", f"none {b}-bit"
                if a not in rows or m not in rows:
                    continue
                kept = rows[a]["tokens_per_doc"] / n_none
                d_none = boot(pq(rows, a) - pq(rows, nb))
                d_match = boot(pq(rows, a) - pq(rows, m))
                S["adapt"][f"c{c:g}@{b}"] = dict(kept=round(kept, 4), tok=rows[a]["tokens_per_doc"],
                                                bytes=rows[a]["bytes_per_doc"], ndcg=rows[a]["ndcg10"],
                                                vs_none=d_none, vs_match=d_match, match_ndcg=rows[m]["ndcg10"],
                                                tau=rows[a]["tau"])
                fp = f"adapt c{c:g}@{b} fp32"
                if fp in rows:
                    S["adapt"][f"c{c:g}@{b}"]["pool_only_vs_none_fp32"] = boot(pq(rows, fp) - ref)
                out.append(f"| {b} | {c:g} | {rows[a]['tau']:.4f} | {kept:.1%} | {rows[a]['bytes_per_doc']:,.0f} | "
                           f"{rows[a]['ndcg10']:.4f} | {fmt(d_none)} | {fmt(d_match)} | {rows[m]['ndcg10']:.4f} |")
        out.append("")
        Db = json.load(open(HERE / ".work" / corpus / "distortion.json")) if (HERE / ".work" / corpus / "distortion.json").exists() else None
        if Db:
            S["D_b"] = {k: round(v, 5) for k, v in Db.items()}
            out.append(f"{corpus}: D_b (mean squared error per unit token, centered remex with the fp16 side scalar): " +
                       ", ".join(f"{k}-bit {v:.4f}" for k, v in sorted(Db.items(), key=lambda kv: -int(kv[0]))) + ".\n")

    json.dump(summary, open(HERE / "summary.json", "w"), indent=1)
    text = "\n".join(out)
    res = HERE / "RESULTS.md"
    if res.exists():
        body = res.read_text()
        new = re.sub(r"<!-- tables:start -->.*<!-- tables:end -->",
                     lambda _: f"<!-- tables:start -->\n{text}\n<!-- tables:end -->", body, flags=re.S)
        res.write_text(new)
    else:
        print(text)
    frontier(summary)


def frontier(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = [c for c in CORPORA if c in summary]
    if not cs:
        return
    fig, axes = plt.subplots(1, len(cs), figsize=(6.5 * len(cs), 4.6), squeeze=False)
    style = {"none": ("#444444", "o"), "st": ("#1f77b4", "s"), "wu": ("#17becf", "D"),
             "adapt": ("#d62728", "^"), "match": ("#ff7f0e", "v")}
    for ax, corpus in zip(axes[0], cs):
        rows = load(corpus)
        for fam, (col, mk) in style.items():
            pts = [(r["bytes_per_doc"], r["ndcg10"]) for r in rows.values()
                   if r.get("family") == fam and not r.get("seed_arm") and r["bits"] is not None]
            if pts:
                x, y = zip(*pts)
                ax.scatter(x, y, c=col, marker=mk, s=28, label=fam, alpha=0.85)
        ax.axhline(rows["none fp32"]["ndcg10"], color="#888", lw=0.8, ls="--", label="unpooled fp32")
        ax.set_xscale("log"); ax.set_xlabel("bytes per document (codes + 2 B side scalar per vector)")
        ax.set_ylabel("nDCG@10"); ax.set_title(f"LateOn, {corpus}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(HERE / "frontier.png", dpi=130)


if __name__ == "__main__":
    main()
