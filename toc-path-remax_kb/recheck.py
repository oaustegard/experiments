"""Assert RESULTS.md's numbers against the artifacts. Sub-5-minute fixture.

Guards the prose against drift from the data between full rebuilds. Reads the
committed JSON only — no embedding, no .kb, no network.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
TOL = 5e-4
fails: list[str] = []


def check(label: str, got: float, want: float, tol: float = TOL) -> None:
    if abs(got - want) > tol:
        fails.append(f"{label}: artifact {got:.4f} vs prose {want:.4f}")


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    ana = json.loads((HERE / "analysis.json").read_text())
    sub = json.loads((HERE / "subgroups.json").read_text())
    prose = (HERE / "RESULTS.md").read_text()

    # Headline
    check("n_queries", res["n_queries"], 799, tol=0)
    check("A fused R@1", res["arms"]["A"]["fused"]["R@1"], 0.592, tol=1e-3)
    check("B1 fused R@1", res["arms"]["B1"]["fused"]["R@1"], 0.586, tol=1e-3)

    d = ana["contrasts"]["fused/A->B1/R@1"]
    check("A->B1 fused diff", d["diff"], -0.0063)
    check("A->B1 fused lo", d["lo"], -0.0325)
    check("A->B1 fused hi", d["hi"], +0.0188)

    c = ana["control_contrasts"]["fused/A->B1/R@1"]
    check("control diff", c["diff"], +0.2450)
    check("control lo", c["lo"], +0.1950)
    check("control hi", c["hi"], +0.2950)

    # Every contrast RESULTS.md claims excludes zero, and no others.
    EXCLUDING_ZERO = {
        "dense/A->B1/MRR": (-0.0200, -0.0378, -0.0019),
        "dense/B1->B2/R@1": (+0.0350, +0.0038, +0.0676),
        "dense/B1->B2/MRR": (+0.0234, +0.0031, +0.0444),
        "bm25/A->B1/R@10": (+0.0113, +0.0013, +0.0225),
        "bm25/A->B2/R@3": (-0.0375, -0.0626, -0.0138),
        "bm25/A->B2/R@5": (-0.0288, -0.0513, -0.0075),
        "bm25/B1->B2/R@3": (-0.0388, -0.0613, -0.0163),
        "bm25/B1->B2/R@5": (-0.0275, -0.0463, -0.0088),
        "bm25/B1->B2/R@10": (-0.0175, -0.0325, -0.0025),
        "bm25/B1->B2/MRR": (-0.0191, -0.0369, -0.0015),
        "fused/A->B2/R@3": (-0.0300, -0.0551, -0.0063),
        "fused/B1->B2/R@5": (-0.0188, -0.0363, -0.0013),
    }
    excluding = {k for k, v in ana["contrasts"].items() if v["lo"] > 0 or v["hi"] < 0}
    if excluding != set(EXCLUDING_ZERO):
        fails.append(
            f"significant set changed; gained {sorted(excluding - set(EXCLUDING_ZERO))}, "
            f"lost {sorted(set(EXCLUDING_ZERO) - excluding)}")
    for key, (diff, lo, hi) in EXCLUDING_ZERO.items():
        v = ana["contrasts"][key]
        check(f"{key} diff", v["diff"], diff)
        check(f"{key} lo", v["lo"], lo)
        check(f"{key} hi", v["hi"], hi)

    # The headline null: no fused B1 contrast may become significant.
    for k in ("R@1", "R@3", "R@5", "R@10", "MRR"):
        v = ana["contrasts"][f"fused/A->B1/{k}"]
        if v["lo"] > 0 or v["hi"] < 0:
            fails.append(f"fused/A->B1/{k} now excludes zero; RESULTS.md reports a null")

    # Full absolute table
    for arm, dense, bm25, fused in (
        ("A", 0.504, 0.554, 0.592), ("B0", 0.488, 0.552, 0.593),
        ("B1", 0.481, 0.561, 0.586), ("B2", 0.516, 0.546, 0.598),
    ):
        a = res["arms"][arm]
        check(f"{arm} dense R@1", a["dense"]["R@1"], dense, tol=1e-3)
        check(f"{arm} bm25 R@1", a["bm25"]["R@1"], bm25, tol=1e-3)
        check(f"{arm} fused R@1", a["fused"]["R@1"], fused, tol=1e-3)

    # Saturation stop was not tripped
    if res["arms"]["A"]["fused"]["R@1"] > 0.90:
        fails.append("arm A fused R@1 > 0.90; PLAN.md says stop, RESULTS.md reports arms")

    # Monotone dense subgroup claim
    order = ["heading adds 0 new tokens", "heading adds 1-2 new tokens",
             "heading adds >=3 new tokens", "heading adds >=6 new tokens"]
    got = [sub[f"dense/{k}"]["diff"] for k in order]
    for label, want in zip(order, (-0.0078, -0.0353, -0.0294, -0.0667)):
        check(f"dense subgroup {label}", sub[f"dense/{label}"]["diff"], want)
    if not (got[3] < got[0]):
        fails.append("dense subgroup no longer deepens with foreign-token count")

    # B2 structural claims
    b2 = [json.loads(l) for l in
          (HERE / "data" / "chunks_B2.jsonl").read_text().splitlines()]
    a_n = len([1 for _ in (HERE / "data" / "chunks_A.jsonl").read_text().splitlines()])
    if len(b2) != 2675 or a_n != 1871:
        fails.append(f"chunk counts changed: A={a_n} B2={len(b2)}")
    check("B2 gold mean", res["b2_gold_mean"], 1.69, tol=5e-3)

    # Numbers quoted in prose must exist in the artifacts
    for lit in ("0.592", "0.586", "+0.2450", "0.0113", "0.0200", "0.0300"):
        if lit not in prose:
            fails.append(f"RESULTS.md no longer quotes {lit}")

    if fails:
        print("RECHECK FAILED")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print(f"recheck ok — {len(ana['contrasts'])} contrasts, "
          f"{res['n_queries']} queries, prose matches artifacts")


if __name__ == "__main__":
    main()
