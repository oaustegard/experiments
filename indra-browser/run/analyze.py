"""Arms against a base run on OmniParseBench: headline and family scores, the paired
difference with a 95% bootstrap interval (documents resampled, since a document's tests pass
or fail together), and how often the arm's page output equals the base's.

    python3 run/analyze.py /tmp/indra-eval/out fp32b ARM [ARM ...] [--json results/arms.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILY = {"present": "text", "order": "text", "repeat": "text", "page_near_empty": "text",
          "table_cell": "tables", "layout_kind": "layout"}


def load(run: Path):
    tests, pages = {}, {}
    for f in (run / "pages").glob("*.json"):
        r = json.loads(f.read_text())
        pages[r["page_id"]] = r
        for ln in r["lines"]:
            tests[ln["test_id"]] = (ln["document"], FAMILY[ln["test_type"]], ln["result"] == "pass")
    return tests, pages


def headline(ids, tests, docs_w=None):
    fam = defaultdict(lambda: [0.0, 0.0])
    for t in ids:
        d, f, ok = tests[t]
        w = 1.0 if docs_w is None else docs_w[d]
        fam[f][0] += w * ok
        fam[f][1] += w
    sc = {f: a / b for f, (a, b) in fam.items() if b}
    return float(np.mean(list(sc.values()))), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("base")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    bt, bp = load(a.out / a.base)
    ids = sorted(bt)
    docs = sorted({bt[t][0] for t in ids})
    rng = np.random.default_rng(0)
    draws = [rng.choice(len(docs), len(docs)) for _ in range(a.boot)]
    weights = []
    for d in draws:
        c = np.bincount(d, minlength=len(docs))
        weights.append({docs[i]: float(c[i]) for i in range(len(docs))})
    base_h, base_f = headline(ids, bt)
    base_boot = [headline(ids, bt, w)[0] for w in weights]
    res = {"base": a.base, "base_headline": base_h, "base_families": base_f, "arms": {}}
    print(f"{a.base}: headline {100 * base_h:.2f} " + " ".join(f"{k} {100 * v:.2f}" for k, v in sorted(base_f.items())))
    for arm in a.arms:
        at, ap_ = load(a.out / arm)
        # an arm run on some pages only (--pages with --base): the rest are the base's
        for p, r in bp.items():
            if p not in ap_:
                ap_[p] = r
                for ln in r["lines"]:
                    at.setdefault(ln["test_id"], bt[ln["test_id"]])
        if set(at) != set(bt):
            print(f"{arm}: {len(set(bt) - set(at))} tests missing, skipped")
            continue
        h, f = headline(ids, at)
        boot = np.array([headline(ids, at, w)[0] - b for w, b in zip(weights, base_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        same = sum(ap_[p]["raw"].get("markdown") == bp[p]["raw"].get("markdown") for p in bp if "raw" in ap_.get(p, {}) and "raw" in bp[p])
        flips = sum(at[t][2] != bt[t][2] for t in ids)
        res["arms"][arm] = {"headline": h, "families": f, "delta": h - base_h, "ci95": [float(lo), float(hi)],
                            "pages_same_markdown": same, "pages": len(bp), "tests_flipped": flips}
        print(f"{arm}: headline {100 * h:.2f} ({100 * (h - base_h):+.2f}, 95% CI {100 * lo:+.2f} to {100 * hi:+.2f}) "
              + " ".join(f"{k} {100 * v:.2f}" for k, v in sorted(f.items()))
              + f"; same markdown on {same}/{len(bp)} pages; {flips} of {len(ids)} tests flipped")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
