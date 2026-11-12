#!/usr/bin/env python3
"""Render result tables into RESULTS.md (between the tables markers), write summary.json
with every number the prose cites, and draw frontier.png (nDCG@10 against bytes/doc).

An arm's score is the per-query mean over every rotation it was run with: the default
(rht, seed 42) plus any `<arm> haar sK` twins. Paired CIs (5,000-resample bootstrap over
queries) are taken on those seed-averaged per-query scores; seed counts are listed."""
from __future__ import annotations

import json, re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORPORA = ("scifact", "nfcorpus")
C_GRID = (0.5, 1.0, 2.0, 4.0)
BITS = (4, 3, 2, 1)

# a - b at about the same bytes (or a at fewer bytes than b)
ISO = [("st pf4 4-bit", "none 1-bit"), ("st pf2 2-bit", "none 1-bit"), ("st pf4 4-bit", "st pf2 2-bit"),
       ("st pf2 4-bit", "none 2-bit"),
       ("adapt c1@2 2-bit", "none 1-bit"), ("adapt c1@2 2-bit", "st pf4 4-bit"),
       ("adapt c4@3 3-bit", "st pf4 4-bit"), ("adapt c4@3 3-bit", "st pf2 2-bit"),
       ("cadapt c1@2 2-bit", "st pf4 4-bit"), ("cadapt c4@3 3-bit", "st pf4 4-bit"),
       # against the cheapest 1-bit codec: no side scalar, codes only
       ("st pf4 4-bit", "none 1-bit no-side"), ("st pf2 2-bit", "none 1-bit no-side"),
       ("st pf4 4-bit no-side", "none 1-bit no-side")]
# adaptive allocation against uniform allocation at the same tokens and bits
CONTRASTS = [("adapt c1@2 2-bit", "match c1@2 2-bit"), ("adapt c1@3 3-bit", "match c1@3 3-bit"),
             ("adapt c2@3 3-bit", "match c2@3 3-bit"), ("adapt c4@3 3-bit", "match c4@3 3-bit"),
             ("cadapt c1@2 2-bit", "cmatch c1@2 2-bit"), ("cadapt c4@3 3-bit", "cmatch c4@3 3-bit"),
             ("cadapt c2@2 2-bit", "cmatch c2@2 2-bit"), ("cadapt c0.5@2 2-bit", "cmatch c0.5@2 2-bit"),
             ("cadapt c1@3 3-bit", "cmatch c1@3 3-bit"), ("cadapt c2@3 3-bit", "cmatch c2@3 3-bit"),
             # which tree: cosine-distance Ward (library) against true Ward, same cut rule
             ("cadapt c1@2 2-bit", "adapt c1@2 2-bit"), ("cadapt c4@3 3-bit", "adapt c4@3 3-bit"),
             ("cadapt c4@3 fp32", "adapt c4@3 fp32"), ("cadapt c1@2 fp32", "adapt c1@2 fp32"),
             ("cu pf4 fp32", "st pf4 fp32"), ("wu pf4 fp32", "st pf4 fp32"),
             # pooling alone, fp32
             ("adapt c4@3 fp32", "none fp32"), ("adapt c1@2 fp32", "none fp32"),
             ("cadapt c4@3 fp32", "none fp32"), ("cadapt c1@2 fp32", "none fp32")]
SIDE = [("none 1-bit no-side", "none 1-bit"), ("none 2-bit no-side", "none 2-bit"),
        ("none 1-bit exact-side", "none 1-bit"), ("none 2-bit exact-side", "none 2-bit"),
        ("st pf4 4-bit no-side", "st pf4 4-bit"), ("st pf2 2-bit no-side", "st pf2 2-bit"),
        ("adapt c1@2 2-bit no-side", "adapt c1@2 2-bit")]


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
        if "no-side" in r["arm"]:  # round 1 wrote these with the 2 B counted; they carry no side scalar
            r["bytes_per_doc"] = r["bytes_per_doc_codes_only"]
        rows[r["arm"]] = r
    return rows


def runs(rows, name):
    """Per-query arrays for every rotation of this arm (default first)."""
    out = [np.array(rows[name]["perquery"])] if name in rows else []
    out += [np.array(rows[k]["perquery"]) for k in sorted(rows) if k.startswith(name + " haar s")
            and k[len(name) + 7:].isdigit()]
    return out


def pq(rows, name):
    r = runs(rows, name)
    return np.mean(r, axis=0)


def seed_note(rows, name):
    r = runs(rows, name)
    ms = [float(x.mean()) for x in r]
    return len(r), (round(min(ms), 4), round(max(ms), 4))


def shared(rows, a, b):
    """Per-query a - b averaged over the rotations both arms ran with (fp32 arms have none,
    so they pair with every rotation of the other arm)."""
    keys = [""] + [f" haar s{k}" for k in (1, 2, 3)]
    fa, fb = rows[a]["bits"] is None, rows[b]["bits"] is None
    diffs = []
    for k in keys:
        ka, kb = (a if fa else a + k), (b if fb else b + k)
        if ka in rows and kb in rows and not (fa and fb and k):
            diffs.append(np.array(rows[ka]["perquery"]) - np.array(rows[kb]["perquery"]))
    return np.mean(diffs, axis=0), len(diffs)


def per_seed_diffs(rows, a, b):
    """Differences of the means for seeds both arms share (default + haar sK)."""
    out = []
    if a in rows and b in rows:
        out.append(round(rows[a]["ndcg10"] - rows[b]["ndcg10"], 4))
    for k in (1, 2, 3):
        ka, kb = f"{a} haar s{k}", f"{b} haar s{k}"
        if ka in rows and kb in rows:
            out.append(round(rows[ka]["ndcg10"] - rows[kb]["ndcg10"], 4))
    return out


def main():
    out, summary = [], {}
    for corpus in CORPORA:
        rows = load(corpus)
        if not rows or "none fp32" not in rows:
            continue
        S = summary[corpus] = {}
        ref = pq(rows, "none fp32")
        base = [k for k in rows if " haar s" not in k]

        # ---------------------------------------------------------------- every arm
        S["rows"] = {}
        out.append(f"### {corpus}: every arm (nDCG@10 averaged over the rotations run, n = count, range = "
                   f"lowest to highest single run; Δ = arm − unpooled fp32, 95% paired bootstrap over queries; "
                   f"B/doc counts 2 B of side scalar per coded vector except `no-side` arms)\n")
        out.append("| arm | tok/doc | B/doc | nDCG@10 | n | range | Δ vs none fp32 |")
        out.append("|---|---:|---:|---:|---:|---|---|")
        for name in base:
            r = rows[name]
            x = pq(rows, name)
            n, rng = seed_note(rows, name)
            d = boot(x - ref)
            S["rows"][name] = dict(ndcg=round(float(x.mean()), 4), n=n, range=list(rng), tok=r["tokens_per_doc"],
                                   bytes=r["bytes_per_doc"], kept=round(r["tokens_per_doc"] / rows["none fp32"]["tokens_per_doc"], 4),
                                   delta=d)
            rng_s = "" if n == 1 else f"{rng[0]:.4f}–{rng[1]:.4f}"
            out.append(f"| {name} | {r['tokens_per_doc']:.1f} | {r['bytes_per_doc']:,.0f} | {x.mean():.4f} | {n} | "
                       f"{rng_s} | {fmt(d)} |")
        out.append("")

        # ---------------------------------------------------------------- unpooled codec drops
        S["drops"] = {}
        for a, b, key in (("none 2-bit", "none 1-bit", "two_to_one_bit"), ("none fp32", "none 2-bit", "fp32_to_two_bit"),
                          ("none fp32", "none 4-bit", "fp32_to_four_bit"), ("none fp32", "st pf4 4-bit", "st_pf4_4bit_vs_fp32")):
            if a in rows and b in rows:
                S["drops"][key] = boot(pq(rows, b) - pq(rows, a))

        # ---------------------------------------------------------------- same bytes
        def table(title, pairs, key):
            S[key] = {}
            lines = []
            for a, b in pairs:
                if a not in rows or b not in rows:
                    continue
                dd, n_sh = shared(rows, a, b)
                d = boot(dd)
                ps = per_seed_diffs(rows, a, b)
                S[key][f"{a} vs {b}"] = dict(delta=d, per_seed=ps, n_shared=n_sh)
                pos = sum(v > 0 for v in ps)
                seeds_s = f"{pos} of {len(ps)}" if ps else "fp32 pair"
                lines.append(f"| {a} | {b} | {rows[a]['bytes_per_doc']:,.0f} / {rows[b]['bytes_per_doc']:,.0f} | "
                             f"{n_sh} | {fmt(d)} | {seeds_s} |")
            if lines:
                out.append(title + "\n")
                out.append("| a | b | B/doc a / b | shared rotations | Δ a − b (mean over shared rotations) | rotations with a > b |")
                out.append("|---|---|---|---|---|---|")
                out.extend(lines + [""])

        table(f"### {corpus}: about the same or fewer bytes (paired over the rotations both arms ran)", ISO, "iso")
        table(f"### {corpus}: adaptive against uniform allocation, tree against tree, pooling alone", CONTRASTS, "contrasts")
        table(f"### {corpus}: per-vector side information (`no-side` drops the 2 B scalar and decodes with its "
              f"corpus average; `exact-side` uses remex's float32 fields)", SIDE, "side")

        # ---------------------------------------------------------------- integration sweep
        S["adapt"] = {}
        out.append(f"### {corpus}: integrated cut E_pool ≤ c·D_b on each tree, against the unpooled codec at the "
                   f"same bits and uniform pooling at the same tokens and bits (rotation-averaged where run)\n")
        out.append("| tree | bits | c | tokens kept | B/doc | nDCG@10 | − none at b | − uniform at same tokens | uniform nDCG@10 |")
        out.append("|---|---|---|---:|---:|---:|---|---|---:|")
        n_none = rows["none fp32"]["tokens_per_doc"]
        for fam, mfam, tree in (("adapt", "match", "true Ward"), ("cadapt", "cmatch", "cosine Ward")):
            for b in BITS:
                for c in C_GRID:
                    a, m, nb = f"{fam} c{c:g}@{b} {b}-bit", f"{mfam} c{c:g}@{b} {b}-bit", f"none {b}-bit"
                    if a not in rows or m not in rows:
                        continue
                    kept = rows[a]["tokens_per_doc"] / n_none
                    matched = abs(rows[m]["tokens_per_doc"] / rows[a]["tokens_per_doc"] - 1) < 0.02
                    d_none = boot(shared(rows, a, nb)[0])
                    d_match = boot(shared(rows, a, m)[0])
                    k = f"{fam} c{c:g}@{b}"
                    S["adapt"][k] = dict(kept=round(kept, 4), tok=rows[a]["tokens_per_doc"], bytes=rows[a]["bytes_per_doc"],
                                         ndcg=round(float(pq(rows, a).mean()), 4), vs_none=d_none, vs_match=d_match,
                                         match_ndcg=round(float(pq(rows, m).mean()), 4), matched=matched)
                    fp = f"{fam} c{c:g}@{b} fp32"
                    if fp in rows:
                        S["adapt"][k]["pool_only_vs_none_fp32"] = boot(pq(rows, fp) - ref)
                    mt = "" if matched else f" (uniform floor {rows[m]['tokens_per_doc']:.1f} tok/doc, not matched)"
                    out.append(f"| {tree} | {b} | {c:g} | {kept:.1%} | {rows[a]['bytes_per_doc']:,.0f} | "
                               f"{pq(rows, a).mean():.4f} | {fmt(d_none)} | {fmt(d_match)}{mt} | {pq(rows, m).mean():.4f} |")
        out.append("")
        dpath = HERE / ".work" / corpus / "distortion.json"
        if dpath.exists():
            Db = json.load(open(dpath))
            S["D_b"] = {k: round(v, 5) for k, v in Db.items()}
            out.append(f"{corpus}: D_b (mean squared error per unit token, centered remex with the fp16 side scalar): " +
                       ", ".join(f"{k}-bit {v:.4f}" for k, v in sorted(Db.items(), key=lambda kv: -int(kv[0]))) +
                       f". Norm of the corpus mean {rows['none 1-bit']['norm_of_mean']:.3f}. "
                       f"{len(ref)} queries, {rows['none fp32']['tokens_per_doc']:.1f} tokens/doc unpooled.\n")

    json.dump(summary, open(HERE / "summary.json", "w"), indent=1)
    text = "\n".join(out)
    res = HERE / "RESULTS.md"
    if res.exists() and "<!-- tables:start -->" in res.read_text():
        body = res.read_text()
        res.write_text(re.sub(r"<!-- tables:start -->.*<!-- tables:end -->",
                              lambda _: f"<!-- tables:start -->\n{text}\n<!-- tables:end -->", body, flags=re.S))
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
    fig, axes = plt.subplots(1, len(cs), figsize=(6.5 * len(cs), 4.8), squeeze=False)
    style = {"none": ("#444444", "o", "unpooled"), "st": ("#1f77b4", "s", "library pooling (pf2, pf4, 32 cap)"),
             "adapt": ("#d62728", "^", "integrated, true Ward"), "cadapt": ("#9467bd", "P", "integrated, cosine Ward"),
             "match": ("#ff7f0e", "v", "uniform Ward, same tokens")}
    for ax, corpus in zip(axes[0], cs):
        rows = load(corpus)
        for fam, (col, mk, lab) in style.items():
            pts = [(r["bytes_per_doc"], float(pq(rows, k).mean())) for k, r in rows.items()
                   if r.get("family") == fam and " haar s" not in k and r["bits"] is not None and r.get("side", "fp16") == "fp16"]
            pts = [p for p in pts if p[0] >= 300]
            if pts:
                x, y = zip(*pts)
                ax.scatter(x, y, c=col, marker=mk, s=30, label=lab, alpha=0.85)
            if fam in ("none", "st"):  # label the bit width of the reference points
                for k, r in rows.items():
                    if r.get("family") == fam and " haar s" not in k and r["bits"] and r["bytes_per_doc"] >= 300:
                        ax.annotate(f"{r['bits']}b", (r["bytes_per_doc"], float(pq(rows, k).mean())),
                                    textcoords="offset points", xytext=(4, -9), fontsize=6.5, color=col)
        ax.axhline(rows["none fp32"]["ndcg10"], color="#888", lw=0.8, ls="--", label="unpooled fp32")
        ax.set_xscale("log"); ax.set_xlabel("bytes per document (codes + 2 B per vector)")
        ax.set_ylabel("nDCG@10 (rotation-averaged)"); ax.set_title(f"LateOn, {corpus}")
        lo = rows["none fp32"]["ndcg10"]
        ax.set_ylim(lo - 0.12, lo + 0.03)
        ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(HERE / "frontier.png", dpi=130)


if __name__ == "__main__":
    main()
