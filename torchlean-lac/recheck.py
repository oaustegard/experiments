#!/usr/bin/env python3
"""Check RESULTS.md against the Lean sources without a Lean toolchain.

Runs in under a second. Set TORCHLEAN=<path to a TorchLean checkout> to also
re-check the claims that quote TorchLean's own source.
"""
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = (HERE / "RESULTS.md").read_text()
CORE = (HERE / "LAC" / "Core.lean").read_text()
CHECK = (HERE / "LAC" / "Check.lean").read_text()

failures = []


def expect(ok, msg):
    if not ok:
        failures.append(msg)


# 1. Every theorem named in the RESULTS table is declared in Check.lean.
TABLE = ["score_eq", "score_self", "score_lt_self", "score_le_self", "score_gap"]
for name in TABLE:
    expect(f"`{name}`" in RESULTS, f"RESULTS.md does not name {name}")
    expect(
        re.search(rf"^theorem {re.escape(name)}\b", CHECK, re.M) is not None,
        f"Check.lean does not declare theorem {name}",
    )

# 2. No escape hatches in either source.
for label, src in (("Core.lean", CORE), ("Check.lean", CHECK)):
    for hatch in (r"\bsorry\b", r"\baxiom\b", r"\bnative_decide\b"):
        expect(
            re.search(hatch, src) is None,
            f"{label} contains {hatch}",
        )

# 3. Core.lean carries the memory encoding RESULTS.md describes.
for decl in ("paraKey", "paraQuery", "score", "Mem", "rowScores"):
    expect(
        re.search(rf"^(def|abbrev|theorem) {re.escape(decl)}\b", CORE, re.M) is not None,
        f"Core.lean does not declare {decl}",
    )
expect("Spec.Tensor ℝ [2]" in CORE, "Core.lean key/query is not Tensor ℝ [2]")
expect("Spec.Tensor ℝ [C, 3]" in CORE, "Core.lean Mem is not Tensor ℝ [C, 3]")

# 4. The score definition in both files is the parabolic dot product.
for label, src in (("Core.lean", CORE), ("Check.lean", CHECK)):
    expect(
        "2 * j * i - j * j" in src,
        f"{label} score is not 2*j*i - j*j",
    )

# 5. Optional: re-check the quotes taken from TorchLean itself.
tl = os.environ.get("TORCHLEAN")
if tl:
    tl = Path(tl)
    scalar = (tl / "NN/Spec/Core/Scalar.lean").read_text()
    expect(
        "abbrev SpecScalar := ℝ" in scalar,
        "SpecScalar is no longer ℝ; the DynamicalSystem claim needs revisiting",
    )
    graph = (tl / "NN/IR/Graph.lean").read_text()
    for absent in ("argmax", "gather", "oneHot"):
        expect(
            re.search(rf"^\s*\| {absent}\b", graph, re.M) is None,
            f"NN/IR/Graph.lean now has an {absent} op; the lowering claim is stale",
        )
    reductions = (tl / "NN/Spec/Core/TensorReductionShape/Reductions.lean").read_text()
    expect("def argmax" in reductions, "Spec.Tensor.argmax moved")
    hits = 0
    for p in (tl / "NN").rglob("*.lean"):
        hits += len(re.findall(r"^\s*(?:private )?(?:theorem|lemma) .*argmax", p.read_text(), re.M))
    expect(hits == 3, f"argmax lemma count is {hits}, not the 3 RESULTS.md reports")
else:
    print("TORCHLEAN unset — skipped the checks that quote TorchLean source")

if failures:
    print(f"FAIL ({len(failures)})")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("OK")
