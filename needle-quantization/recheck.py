#!/usr/bin/env python3
"""Check every number quoted in RESULTS.md against the artifacts.

    python3 recheck.py        # exits non-zero on any disagreement
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from analyze import load, mcnemar, routable, size_mb  # noqa: E402
from run_quant import ARMS  # noqa: E402

FAILURES: list[str] = []


def eq(label: str, got, want, tol: float = 5e-4) -> None:
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        FAILURES.append(f"{label}: RESULTS.md says {want}, artifacts say {got}")
    print(f"  {'ok ' if ok else 'FAIL'} {label:52} {got}")


def acc(arm: str) -> float:
    return load(arm)["summary"]["tool_acc_routable"]


def main() -> int:
    print("the ladder")
    table = {
        "all3": (17.77, 0.6481), "prot3": (18.46, 0.6481), "engram-tern": (13.74, 0.6481),
        "all2": (12.36, 0.6296), "mhc2": (13.41, 0.6296), "shipped": (13.74, 0.6111),
        "emb2": (12.69, 0.6111), "all4": (23.17, 0.5926),
        "prot-tern": (13.74, 0.2222), "all-tern": (12.36, 0.0370),
    }
    for arm, (mb, routable_acc) in table.items():
        eq(f"{arm} MB", size_mb(arm), mb, tol=6e-3)
        eq(f"{arm} routable", acc(arm), routable_acc, tol=6e-4)
    eq("ten arms", len(ARMS), 10)

    print("the control")
    eq("shipped == the sibling controls", acc("shipped"), 0.6111)
    from _lib.paths import experiment
    for sib, fname in [("needle-bsky", "results_tuned-min.json"),
                       ("needle-tool-naming", "results_canon_flat.json"),
                       ("needle-layer-pruning", "results_control.json")]:
        d = json.loads((experiment(sib) / fname).read_text())
        eq(f"  {sib}", d["summary"]["tool_acc_routable"], acc("shipped"))

    print("bit width does not matter above 2")
    usable = [a for a in ARMS if a not in ("prot-tern", "all-tern")]
    eq("non-ternary-bulk arms", len(usable), 8)
    lo, hi = min(acc(a) for a in usable), max(acc(a) for a in usable)
    eq("span low", lo, 0.5926, tol=6e-4)
    eq("span high", hi, 0.6481, tol=6e-4)
    eq("span in queries of 54", round((hi - lo) * 54), 3)
    sig = sum(1 for a in usable
              if mcnemar(routable(load("shipped")), routable(load(a)))[2] < 0.05)
    eq("significant against shipped", sig, 0)
    eq("bytes span low", min(size_mb(a) for a in usable), 12.36, tol=6e-3)
    eq("bytes span high", max(size_mb(a) for a in usable), 23.17, tol=6e-3)
    eq("byte range factor", round(23.17 / 12.36, 1), 1.9, tol=0.06)

    print("all4 buys nothing")
    eq("all4 vs shipped bytes", round(100 * (size_mb("all4") / size_mb("shipped") - 1), 1), 68.6, tol=0.06)
    eq("all4 delta", round(acc("all4") - acc("shipped"), 3), -0.018, tol=1e-3)
    eq("all4 delta is one query", round((acc("shipped") - acc("all4")) * 54), 1)

    print("the 4-bit protection is not load-bearing")
    eq("all2 bytes", round(100 * (size_mb("all2") / size_mb("shipped") - 1), 1), -10.0, tol=0.06)
    eq("all2 routable", acc("all2"), 0.6296, tol=6e-4)
    eq("all2 p", round(mcnemar(routable(load("shipped")), routable(load("all2")))[2], 2), 1.00, tol=6e-3)
    eq("emb2 lands on shipped", acc("emb2"), acc("shipped"))
    eq("emb2 bytes", round(100 * (size_mb("emb2") / size_mb("shipped") - 1), 1), -7.6, tol=0.06)
    eq("mhc2 bytes", round(100 * (size_mb("mhc2") / size_mb("shipped") - 1), 1), -2.4, tol=0.06)
    eq("emb2 is one query below mhc2", round((acc("mhc2") - acc("emb2")) * 54), 1)

    print("ternary: same bytes, by construction")
    eq("prot-tern byte-identical to shipped", size_mb("prot-tern"), size_mb("shipped"))
    eq("all-tern byte-identical to all2", size_mb("all-tern"), size_mb("all2"))
    eq("prot-tern routable", acc("prot-tern"), 0.2222, tol=6e-4)
    eq("prot-tern loss in points", round((acc("shipped") - acc("prot-tern")) * 100), 39)
    eq("all-tern loss in points", round((acc("shipped") - acc("all-tern")) * 100), 57)
    eq("all-tern refuses everything", load("all-tern")["summary"]["refusal_acc"], 1.0)
    for arm in ("prot-tern", "all-tern"):
        eq(f"{arm} p < 0.0001",
           mcnemar(routable(load("shipped")), routable(load(arm)))[2] < 1e-4, True)
    from needle.model.export import _packed_row_bytes
    from needle.model.quantize import TERNARY_BITS
    from needle.model.export import TERNARY_RECORD_BITS
    eq("_packed_row_bytes(ternary) == _packed_row_bytes(2 bits)",
       _packed_row_bytes(512, TERNARY_RECORD_BITS, 128), _packed_row_bytes(512, 2, 128))
    eq("TERNARY_BITS is log2(3), not the storage width", round(TERNARY_BITS, 2), 1.58, tol=6e-3)

    print("the Engram tables ternarize for free")
    eq("engram-tern bytes unchanged", size_mb("engram-tern"), size_mb("shipped"))
    eq("engram-tern routable", acc("engram-tern"), 0.6481, tol=6e-4)
    eq("engram-tern cost under 0.05", acc("shipped") - acc("engram-tern") < 0.05, True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} disagreement(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULTS.md agrees with the artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
