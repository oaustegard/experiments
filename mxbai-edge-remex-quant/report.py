#!/usr/bin/env python3
"""Render the results tables into RESULTS.md between the tables markers."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPORA = ["scifact", "nfcorpus"]


def fmt_ci(r):
    return "ref" if "delta" not in r else f"{r['delta']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"


def main():
    res = {c: json.load(open(HERE / f"results_{c}.json")) for c in CORPORA}
    ext = {c: json.load(open(HERE / f"results_extra_{c}.json")) for c in CORPORA}
    out = []
    by = {c: {r["arm"]: r for r in res[c]["rows"]} for c in CORPORA}
    for c in CORPORA:
        for r in ext[c]["rows"]:
            by[c][r["arm"]] = {**r, "bytes_per_doc": None}
    arms = list(by["scifact"])
    out.append("### Every arm, both corpora (nDCG@10; Δ vs mxbai fp32 with 95% paired-bootstrap CI; bytes per document as stored)\n")
    out.append("| arm | SciFact B/doc | SciFact nDCG@10 | SciFact Δ | NFCorpus B/doc | NFCorpus nDCG@10 | NFCorpus Δ |")
    out.append("|---|---:|---:|---|---:|---:|---|")
    for a in arms:
        cells = []
        for c in CORPORA:
            r = by[c].get(a)
            b = "" if r is None or r.get("bytes_per_doc") is None else f"{r['bytes_per_doc']:,.0f}"
            if r is not None and "bytes_per_doc_with_table" in r:
                b += f" ({r['bytes_per_doc_with_table']:,.0f})"
            cells += [b, "" if r is None else f"{r['ndcg10']:.4f}", "" if r is None else fmt_ci(r)]
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    out.append("\nplaid bytes in parentheses include the fp16 centroid table (K × 64 × 2 B; K = 16,384 SciFact, 8,192 NFCorpus) amortised over the corpus. "
               "Follow-up arms (extra.py) have the same token count as their bench.py counterparts.\n")
    out.append("### Head-to-head (paired bootstrap, per-query ΔnDCG@10, a − b)\n")
    out.append("| a | b | SciFact | NFCorpus |")
    out.append("|---|---|---|---|")
    for p_s, p_n in zip(ext["scifact"]["paired"], ext["nfcorpus"]["paired"]):
        f = lambda p: f"{p['delta']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] ({p['wins']}/{p['losses']})"
        out.append(f"| {p_s['a']} | {p_s['b']} | {f(p_s)} | {f(p_n)} |")
    out.append("\n### Token geometry\n")
    out.append("| | SciFact | NFCorpus |")
    out.append("|---|---:|---:|")
    for k in ext["scifact"]["geometry"]:
        out.append(f"| {k} | {ext['scifact']['geometry'][k]} | {ext['nfcorpus']['geometry'][k]} |")
    for c in CORPORA:
        m = res[c]["meta"]
        out.append(f"\n{c}: {m['n_docs']:,} docs, {m['n_tokens']:,} tokens ({m['tokens_per_doc']}/doc, "
                   f"{m['pooled_tokens_per_doc']}/doc after pool2), {m['n_queries']} queries, bench {m['seconds']:.0f} s.")
    md = (HERE / "RESULTS.md").read_text()
    a, b = md.index("<!-- tables:start -->"), md.index("<!-- tables:end -->")
    (HERE / "RESULTS.md").write_text(md[: a + len("<!-- tables:start -->")] + "\n" + "\n".join(out) + "\n" + md[b:])
    print("RESULTS.md tables rendered")


if __name__ == "__main__":
    main()
