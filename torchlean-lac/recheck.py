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
CAPACITY = (HERE / "LAC" / "Capacity.lean").read_text()

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

# 1b. Every capacity theorem named in RESULTS.md is declared in Capacity.lean, and the
#     file states the score over TorchLean's FP32 operations, not a private rounding model.
CAPACITY_TABLE = [
    "r32_dyadic", "r32_int", "r32_odd_sq", "r32_neg_odd_sq", "r32_two_odd_sq",
    "score32_exact", "read_exact_below_ceiling", "read_ties_above_ceiling",
    "capacity_ceiling", "generic_even_of_ge", "not_both_representable_above_ceiling",
    "kernel_tie_4097", "kernel_distinct_4096", "kernel_tie_11585", "kernel_separates_11587",
]
for name in CAPACITY_TABLE:
    expect(f"`{name}`" in RESULTS, f"RESULTS.md does not name {name}")
    expect(
        re.search(rf"^theorem {re.escape(name)}\b", CAPACITY, re.M) is not None,
        f"Capacity.lean does not declare theorem {name}",
    )
expect("NF.ofReal" in CAPACITY and "FP32" in CAPACITY, "Capacity.lean does not use FP32/NF.ofReal")
expect(
    "((key j).1 * query i + (key j).2).val" in CAPACITY,
    "score32 is no longer FP32 multiply-then-add",
)
expect("by\n  decide" in CAPACITY, "kernel instances are not closed by plain `decide`")

# 1c. The numeric boundary claims in RESULTS.md, from the same script that produced them.
sys.path.insert(0, str(HERE))
import capacity_numerics as cn  # noqa: E402

expect(not cn.model_b_fails(4096), "fp32 pipeline fails at 4096")
expect(cn.model_b_fails(4097), "fp32 pipeline succeeds at 4097")
expect(cn.model_b_fails(5793), "fp32 pipeline succeeds at 5793")
expect(not cn.model_b_fails(5794), "fp32 pipeline fails at 5794")
expect(cn.model_a_ties(11585), "exact-then-round does not tie at 11585")
expect(not cn.model_a_ties(11587), "exact-then-round ties at 11587")

# 2. No escape hatches in any source.
for label, src in (("Core.lean", CORE), ("Check.lean", CHECK), ("Capacity.lean", CAPACITY)):
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
    fp32core = (tl / "NN/Floats/FP32/Core.lean").read_text()
    expect("def fexp32 : ℤ → ℤ := FLTExp (-149) 24" in fp32core, "fexp32 changed")
    expect("def rnd32 : ℝ → ℤ := neuralNearestEven" in fp32core, "rnd32 is no longer nearest-even")
    bridge = (tl / "NN/Floats/IEEEExec/Bridge/FP32/RoundDyadic.lean").read_text()
    expect(
        "theorem toReal_roundDyadicToIEEE32_eq_fp32Round" in bridge,
        "the bit-level/FP32 bridge theorem moved",
    )
else:
    print("TORCHLEAN unset — skipped the checks that quote TorchLean source")

if failures:
    print(f"FAIL ({len(failures)})")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("OK")
