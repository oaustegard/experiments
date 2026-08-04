#!/usr/bin/env python3
"""Check RESULTS.md prose against the committed result artifacts.

Sub-5-minute fixture, no network, no models: it re-derives every headline
number from results_*.json and asserts the claims the writeup actually makes.
Run after editing either the prose or the results.

    python3 recheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAIL: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"{'ok  ' if cond else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(label)


def mean(key: str, rows: list[dict]) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def main() -> int:
    # ── Part A ──────────────────────────────────────────────────────────────
    A = json.load(open(HERE / "results_parta.json"))
    inst = json.load(open(HERE / "instances.json"))
    check("instance set is committed and n=6", len(inst) == 6, f"n={len(inst)}")

    ast8 = [r for r in A if r["mode"] == "ast" and r["variant"] == "a8m"]
    check("grep baseline r@5 == 0.667", abs(mean("rg_r5", ast8) - 2 / 3) < 5e-3,
          f"{mean('rg_r5', ast8):.3f}")
    check("dense beats grep at r@5 (ast/a8m)", mean("dense_r5", ast8) > mean("rg_r5", ast8),
          f"{mean('dense_r5', ast8):.3f} > {mean('rg_r5', ast8):.3f}")
    check("RRF is best arm at r@5", mean("rrf_r5", ast8) >= mean("dense_r5", ast8),
          f"{mean('rrf_r5', ast8):.3f}")

    # decision gate
    poor = [r for r in ast8 if r["issue"] == 22186]
    rich = [r for r in ast8 if r["issue"] != 22186]
    check("identifier-poor instance yields 0 identifiers", poor[0]["n_idents"] == 0)
    check("GATE part 1: grep scores 0 on poor stratum", mean("rg_r5", poor) == 0.0)
    check("GATE part 1: bekko clears grep on poor stratum",
          mean("dense_r5", poor) > mean("rg_r5", poor), f"{mean('dense_r5', poor):.3f} > 0")
    check("GATE part 2: no regression on rich stratum r@5",
          mean("dense_r5", rich) >= mean("rg_r5", rich) - 1e-9,
          f"{mean('dense_r5', rich):.3f} >= {mean('rg_r5', rich):.3f}")
    check("poor stratum is n=1 (thin-slice caveat is real)", len(poor) == 1)

    # axes both small
    flat8 = [r for r in A if r["mode"] == "flat" and r["variant"] == "a8m"]
    ast25 = [r for r in A if r["mode"] == "ast" and r["variant"] == "a25m"]
    if flat8 and ast25:
        d_chunk = abs(mean("dense_r5", flat8) - mean("dense_r5", ast8))
        d_enc = abs(mean("dense_r5", ast25) - mean("dense_r5", ast8))
        gap = mean("dense_r5", ast8) - mean("rg_r5", ast8)
        check("chunking effect smaller than dense-vs-grep gap", d_chunk < gap,
              f"{d_chunk:.3f} < {gap:.3f}")
        check("encoder effect smaller than dense-vs-grep gap", d_enc < gap,
              f"{d_enc:.3f} < {gap:.3f}")

    # ── Part B ──────────────────────────────────────────────────────────────
    B = json.load(open(HERE / "results_partb.json"))
    for f in B["fidelity"]:
        check(f"int8-embtable fidelity holds ({f['model']}/{f['dist']})",
              f["per_doc_cosine_vs_own_fp32"] > 0.9994,
              f"{f['per_doc_cosine_vs_own_fp32']:.5f}")
    check("second distribution was actually measured",
          {f["dist"] for f in B["fidelity"]} == {"blog", "code"})

    J = json.load(open(HERE / "results_jina.json"))
    jr = {(r["dist"], r["dim"]): r["r@10"] for r in J["retrieval"]}
    br = {(r["dist"], r["dim"]): r["r@10"]
          for r in B["retrieval"] if r["model"] == "bekko-a25m"}
    contested = [(d, k) for (d, k) in jr if (d, k) in br]
    jina_wins = sum(1 for key in contested if jr[key] > br[key])
    check("jina wins the large majority of iso-byte cells vs bekko-a25m",
          jina_wins >= len(contested) - 1, f"{jina_wins}/{len(contested)}")
    check("jina q4 and bekko-a8m are comparable size (size argument is moot)",
          abs(J["model_mb"] - 124.1) < 15, f"jina {J['model_mb']} MB vs bekko-a8m 124.1 MB")

    # ── composition ─────────────────────────────────────────────────────────
    C = json.load(open(HERE / "results_compose.json"))
    cells = 0
    for v in ("a8m", "a25m"):
        for dim in (384, 256, 128, 64):
            g = lambda p: [r["r@10"] for r in C if r["variant"] == v and r["dim"] == dim
                           and r["codec"] == "remex" and r["param"] == p][0]
            if g(2) > g(1):
                cells += 1
    check("2-bit beats 1-bit in ALL 8 cells (bekko is Jina-side)", cells == 8, f"{cells}/8")

    def best_at(v, b):
        s = [r for r in C if r["variant"] == v and r["bytes"] == b]
        return max(r["r@10"] for r in s) if s else None

    q96 = [r["r@10"] for r in C if r["variant"] == "a25m" and r["codec"] == "remex"
           and r["dim"] == 384 and r["param"] == 2][0]
    f1536 = [r["r@10"] for r in C if r["variant"] == "a25m" and r["codec"] == "fp32"
             and r["dim"] == 384][0]
    check("quantization at 96 B >= full fp32 at 1536 B (16x smaller, not worse)",
          q96 >= f1536, f"{q96:.3f} >= {f1536:.3f}")

    f512 = [r["r@10"] for r in C if r["variant"] == "a25m" and r["codec"] == "fp32"
            and r["dim"] == 128][0]
    q48 = [r["r@10"] for r in C if r["variant"] == "a25m" and r["codec"] == "remex"
           and r["dim"] == 384 and r["param"] == 1][0]
    check("remex 48 B matches fp32-truncated 512 B", q48 >= f512, f"{q48:.3f} >= {f512:.3f}")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
