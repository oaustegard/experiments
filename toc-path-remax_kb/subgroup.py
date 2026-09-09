"""EXPLORATORY, not pre-registered.

The mechanism B1 is supposed to exploit is "the heading names something the
chunk body does not". If the heading path adds no vocabulary the body already
has, B1 cannot help there by construction. So the honest place to look for an
effect, if one exists anywhere, is the subset of queries whose gold chunk gains
real terms from its heading path.

A null in that subset makes the overall null stronger. A positive there would be
actionable — apply the prefix only where it adds terms — but it is a post-hoc
split and would need its own pre-registered run to count as a finding.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
N_BOOT = 10_000


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def paired(a: np.ndarray, b: np.ndarray, rng):
    d = b - a
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    gold = {g["id"]: g for g in
            (json.loads(l) for l in (DATA / "gold_chunks.jsonl").read_text().splitlines())}
    qrecs = [json.loads(l) for l in (DATA / "queries.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(20260909)

    new_counts = []
    for r in qrecs:
        g = gold[r["gold_id"]]
        new_counts.append(len(toks(g["heading_path"]) - toks(g["text"])))
    new_counts = np.array(new_counts)

    groups = {
        "heading adds 0 new tokens": new_counts == 0,
        "heading adds 1-2 new tokens": (new_counts >= 1) & (new_counts <= 2),
        "heading adds >=3 new tokens": new_counts >= 3,
        "heading adds >=6 new tokens": new_counts >= 6,
    }

    out = {}
    print("EXPLORATORY subgroups — fused R@1, arm A -> arm B1\n")
    for mode in ("fused", "dense", "bm25"):
        ra = np.array([1.0 if (r is not None and r <= 1) else 0.0
                       for r in res["ranks"]["A"][mode]])
        rb = np.array([1.0 if (r is not None and r <= 1) else 0.0
                       for r in res["ranks"]["B1"][mode]])
        print(f"  {mode}")
        for name, mask in groups.items():
            if mask.sum() < 20:
                continue
            m, lo, hi = paired(ra[mask], rb[mask], rng)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"    {name:<28} n={int(mask.sum()):4d}  "
                  f"A {ra[mask].mean():.3f} -> B1 {rb[mask].mean():.3f}   "
                  f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")
            out[f"{mode}/{name}"] = {"n": int(mask.sum()), "a": float(ra[mask].mean()),
                                     "b1": float(rb[mask].mean()), "diff": float(m),
                                     "lo": float(lo), "hi": float(hi)}
        print()

    (HERE / "subgroups.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
