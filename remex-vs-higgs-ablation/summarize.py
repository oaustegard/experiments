#!/usr/bin/env python3
"""Turn results.json into the markdown tables RESULTS.md quotes.

Generated rather than hand-typed on purpose: every number in the writeup
should be traceable to a rerun of this script, and a table transcribed by
hand is a table that silently drifts from the data behind it.

    python3 summarize.py > tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plot import AXIS_FLIP, BITS, HIGGS, REMEX, marginals

HERE = Path(__file__).resolve().parent
RES = HERE / "results.json"

METRIC_LABEL = {"cosine": "cosine", "ip": "inner product"}


def headline(res):
    print("## Headline: remex vs HIGGS-like, matched actual bytes\n")
    print("recall@10 vs fp32 exact search, mean over 5 rotation seeds "
          "(worst seed in parentheses).\n")
    for key in [k for k in res if "|" in k]:
        name, metric = key.split("|")
        block = res[key]
        print(f"**{name} — {METRIC_LABEL[metric]}**\n")
        print("| bits | B/vec remex | B/vec HIGGS-like | remex R@10 | "
              "HIGGS-like R@10 | delta | remex rho | HIGGS-like rho |")
        print("|---|---|---|---|---|---|---|---|")
        for b in BITS:
            cell = block.get(str(b))
            if not cell or REMEX not in cell or HIGGS not in cell:
                continue
            r, h = cell[REMEX], cell[HIGGS]
            d = h["recall@10"]["mean"] - r["recall@10"]["mean"]
            print(f"| {b} | {r['bytes']['total']:.0f} | {h['bytes']['total']:.0f} "
                  f"| {r['recall@10']['mean']:.3f} ({r['recall@10']['min']:.3f}) "
                  f"| {h['recall@10']['mean']:.3f} ({h['recall@10']['min']:.3f}) "
                  f"| {d:+.3f} | {r['spearman']['mean']:.4f} "
                  f"| {h['spearman']['mean']:.4f} |")
        print()


def axis_tables(res):
    print("## Marginal effect of each axis\n")
    print("Mean change in recall@10 from flipping one axis with the other two "
          "held fixed, averaged over the 4 cells that differ only in that "
          "axis, +/- 1 sd across those cells. Positive = the HIGGS-lineage "
          "setting wins.\n")
    for key in [k for k in res if "|" in k]:
        name, metric = key.split("|")
        m = marginals(res, key)
        print(f"**{name} — {METRIC_LABEL[metric]}**\n")
        print("| bits | " + " | ".join(AXIS_FLIP) + " |")
        print("|---|" + "---|" * len(AXIS_FLIP))
        for b in BITS:
            row = []
            for ax in AXIS_FLIP:
                v = m[ax].get(b)
                row.append(f"{v[0]:+.4f} ± {v[1]:.4f}" if v else "—")
            if any(c != "—" for c in row):
                print(f"| {b} | " + " | ".join(row) + " |")
        print()


def axis_summary(res):
    """One number per axis per metric, pooled over corpora and bit widths."""
    print("## Axis summary, pooled over corpora and bit widths\n")
    print("| metric | axis | mean delta R@10 | sd | max |delta| | n cells |")
    print("|---|---|---|---|---|---|")
    for metric in ("cosine", "ip"):
        for ax in AXIS_FLIP:
            vals = []
            for key in [k for k in res if k.endswith(f"|{metric}")]:
                m = marginals(res, key)
                vals += [v[0] for v in m[ax].values()]
            if not vals:
                continue
            a = np.array(vals)
            print(f"| {METRIC_LABEL[metric]} | {ax} | {a.mean():+.4f} | "
                  f"{a.std():.4f} | {np.abs(a).max():.4f} | {a.size} |")
    print()


def controls(res):
    print("## Controls\n")
    print("| corpus / metric | bits | fp32 | naive uniform (no rotation) | "
          "LM+QJL (`prod`) | remex |")
    print("|---|---|---|---|---|---|")
    for key in [k for k in res if "|" in k]:
        name, metric = key.split("|")
        block = res[key]
        for b in BITS:
            cell = block.get(str(b))
            if not cell:
                continue
            u = cell.get("control:uniform-norot")
            q = cell.get("control:lm+qjl")
            r = cell.get(REMEX)
            qs = f"{q['recall@10']['mean']:.3f}" if q else "n/a"
            print(f"| {name} / {METRIC_LABEL[metric]} | {b} | 1.000 "
                  f"| {u['recall@10']['mean']:.3f} | {qs} "
                  f"| {r['recall@10']['mean']:.3f} |")
    print()


def budget(res):
    print("## Actual bytes per vector, itemised\n")
    print("Payload is identical across codebooks by construction; the arms "
          "differ only in the side channel. The shared column is the "
          "index-level cost (rotation + codebook) amortized over the whole "
          "index, not per vector.\n")
    print("The **cosine-opt** column is the honest total for a cosine-only "
          "index: documents are unit-norm there, so remex's stored fp32 norm "
          "is a constant 1.0 and a real deployment would drop it. The "
          "block-scale arm has no such saving — its scales stay live because "
          "they carry per-block variance, not just the global norm. Every "
          "recall number in this writeup is measured with the norm *stored* "
          "(the conservative choice against remex); this column is what "
          "remex's byte cost would fall to if it were dropped.\n")
    print("| corpus | arm | bits | payload B | side B | total B | cosine-opt B "
          "| shared (KiB) | grid dim m |")
    print("|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for key in [k for k in res if k.endswith("|cosine")]:
        name = key.split("|")[0]
        block = res[key]
        for b in BITS:
            cell = block.get(str(b))
            if not cell:
                continue
            for arm in (REMEX, HIGGS):
                r = cell[arm]
                k = (name, arm, b)
                if k in seen:
                    continue
                seen.add(k)
                opt = (r["bytes"]["payload"] if "exactnorm" in arm
                       else r["bytes"]["total"])
                print(f"| {name} | {arm} | {b} | {r['bytes']['payload']:.0f} "
                      f"| {r['bytes']['side']:.0f} | {r['bytes']['total']:.0f} "
                      f"| {opt:.0f} | {r['shared_bytes'] / 1024:.0f} "
                      f"| {r['codebook_m']} |")
    print()


#: Corpus sizes actually swept, for the shared-byte amortization table.
CORPUS_N = {"arxiv768": 750, "glove100": 20_000, "nfcorpus1024": 3_633}


def shared_amortization(res):
    """What the index-level bytes cost per vector AT THE SIZES ACTUALLY RUN.

    The headline tables exclude the rotation and the codebook because they are
    shared across the index. That is the convention both lineages use and it is
    right in the limit — but it is not free here, and it is not symmetric: a
    Haar rotation is d*d fp32 (2.4 MB at d=768) while a vector codebook is up
    to 65536*m fp32 (2.1 MB at m=8). At 750 or 20,000 documents those are not
    rounding errors, and for the vector arm at 4 bits the codebook costs about
    as much per vector as the entire payload.

    So this table states the crossover instead of hiding it: `N for <5%` is the
    index size at which shared bytes fall below 5% of the per-vector total.
    """
    print("## Shared bytes: what amortization actually costs at these corpus sizes\n")
    print("Shared = rotation + codebook, divided by the number of documents in "
          "that corpus. `true B/vec` is payload + side + shared. This is the "
          "column the headline tables leave out.\n")
    print("| corpus | N | bits | arm | B/vec (headline) | shared B/vec | "
          "true B/vec | N for <5% |")
    print("|---|---|---|---|---|---|---|---|")
    for key in [k for k in res if k.endswith("|cosine")]:
        name = key.split("|")[0]
        n = CORPUS_N.get(name)
        if not n:
            continue
        for b in BITS:
            cell = res[key].get(str(b))
            if not cell:
                continue
            for arm in (REMEX, HIGGS):
                r = cell[arm]
                head = r["bytes"]["total"]
                sh = r["shared_bytes"] / n
                need = int(np.ceil(r["shared_bytes"] / (0.05 * head)))
                print(f"| {name} | {n:,} | {b} | {arm} | {head:.0f} | {sh:.1f} "
                      f"| {head + sh:.1f} | {need:,} |")
    print()


def timing(res):
    t = res.get("_timing")
    if not t:
        return
    print("## Axis A wall-clock — rotation apply, 4096 vectors\n")
    print("| d | Haar (dense) | RHT | speedup | Haar build | RHT build |")
    print("|---|---|---|---|---|---|")
    for d, row in sorted(t.items(), key=lambda kv: int(kv[0])):
        print(f"| {d} | {row['haar'] * 1e3:.1f} ms | {row['rht'] * 1e3:.1f} ms "
              f"| {row['speedup']:.1f}x | {row['haar_build_s'] * 1e3:.0f} ms "
              f"| {row['rht_build_s'] * 1e3:.2f} ms |")
    print()


def main():
    res = json.loads(RES.read_text())
    headline(res)
    axis_tables(res)
    axis_summary(res)
    controls(res)
    budget(res)
    shared_amortization(res)
    timing(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
