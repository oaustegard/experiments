#!/usr/bin/env python3
"""Confirm the checks added after mutation testing actually go red.

`mutate.py` reported 55 survivors over `grids.py` and `quantizers.py`.  Some
were equivalent mutants, some were tuning constants, and some were real holes.
`gate.py` grew checks for the real ones — but a check written *because* a
mutant survived has not been shown to catch it until it has been run against
that exact mutant and observed failing.  This does that, one mutation at a
time, restoring the file after each.

It is the permanent form of the transient experiment: `mutate.py` samples
every third token and takes ~45 minutes, while this pins the specific
mutations the gate is claimed to have closed and runs in a few minutes, so it
can be re-run whenever the gate changes.

Run:  python3 verify_kills.py     (~5 min; exit 1 if any mutant survives)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: (file, 1-based line, before, after, what the mutation breaks, which check
#: is claimed to catch it).  Every entry is a survivor from mutate.log.
CASES = [
    ("grids.py", 39, "1", "2",
     "drops b=1 from the Max (1960) anchor table by collapsing a dict key",
     "the Max (1960) anchor table still covers b=1..5"),
    ("grids.py", 130, "-", "+",
     "uniform_levels spans [+clip, +clip] — a degenerate grid",
     "uniform floor control"),
    ("grids.py", 166, "*", "/",
     "ScalarCodebook decision boundaries scaled by 0.5 instead of averaged",
     "scalar codec idempotence / encoder attains the codebook's distortion"),
    ("grids.py", 166, "+", "-",
     "ScalarCodebook boundaries from level DIFFERENCES, not midpoints",
     "scalar codec idempotence / encoder distortion"),
    ("grids.py", 194, "1", "2",
     "VectorCodebook returns the SECOND nearest codepoint",
     "vector encoder attains the codebook's distortion"),
    ("quantizers.py", 177, "min", "max",
     "d=100 collapses to ONE block, so per-block scale becomes a global scale",
     "blockscale arm has >1 block"),
    ("quantizers.py", 206, "*", "/",
     "payload byte count becomes d/bits/8",
     "payload == d*bits/8"),
    ("quantizers.py", 208, "+", "-",
     "total bytes reported as payload MINUS side channels",
     "total == payload + side"),
    ("quantizers.py", 232, "==", "!=",
     "the two axis-B norm modes are swapped",
     "arm round-trip error == codebook MSE"),
]

#: Survivors checked by hand and confirmed to have no observable effect at any
#: configuration the sweep uses.  Listed so the reasoning is on the record
#: rather than implied by their absence — an unexplained survivor and an
#: equivalent mutant look identical in a report.
EQUIVALENT = [
    ("grids.py", 126, "k > 1 -> k >= 1",
     ("k = 2**bits >= 2 for every rate in BITS; the branch differs only at "
      "bits=0, which the sweep never runs")),
    ("grids.py", 128, "m < best -> m <= best",
     ("keeps the last tie rather than the first in uniform_levels' clip "
      "search; measured, all 111 candidate clips give distinct float64 MSEs "
      "at both b=2 and b=4, so there is no tie to break")),
    ("grids.py", 152, "a * pdf(a) -> a / pdf(a)",
     ("enters _scalar_mse_gaussian only through Sum(fb - fa), which cancels "
      "term-by-term for any level set symmetric about 0")),
    ("grids.py", 154, "- (fb - fa) -> + (fb - fa)",
     "same cancellation: the sum telescopes to f(inf) - f(-inf) = 0"),
    ("grids.py", 232, "dist**2 -> dist**3",
     ("_lloyd's in-loop MSE is a convergence probe; selection and reporting "
      "both use held_out_mse, so the value is discarded")),
    ("quantizers.py", 73, "reshape(-1, B) -> reshape(-2, B)",
     "numpy treats any negative dimension as 'infer'"),
    ("quantizers.py", 120, "Y * sign -> Y / sign",
     "sign is +-1, so multiply and divide are the same map"),
]


def mutate_line(text: str, line: int, before: str, after: str) -> str:
    lines = text.splitlines(keepends=True)
    src = lines[line - 1]
    if before not in src:
        raise SystemExit(f"line {line} does not contain {before!r}: {src!r}")
    lines[line - 1] = src.replace(before, after, 1)
    return "".join(lines)


def main() -> int:
    cmd = [sys.executable, str(HERE / "gate.py"), "--fast"]
    print("baseline: gate must pass on unmutated code")
    if subprocess.call(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL) != 0:
        print("BASELINE RED — fix that first; kill counts against an already "
              "red gate mean nothing.")
        return 2

    survived = []
    for i, (fname, line, before, after, breaks, catcher) in enumerate(CASES, 1):
        path = HERE / fname
        original = path.read_text()
        try:
            path.write_text(mutate_line(original, line, before, after))
            rc = subprocess.call(cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        finally:
            path.write_text(original)     # restore even on Ctrl-C
        ok = rc != 0
        if not ok:
            survived.append((fname, line, before, after, catcher))
        print(f"  [{i}/{len(CASES)}] {fname}:{line} {before} -> {after:<5} "
              f"{'KILLED' if ok else 'SURVIVED'}  ({breaks})")

    print("\n" + "=" * 70)
    if survived:
        print(f"{len(survived)}/{len(CASES)} mutants SURVIVED — the check "
              f"named for each was not shown to catch it:")
        for fname, line, before, after, catcher in survived:
            print(f"  {fname}:{line} {before} -> {after}   claimed catcher: "
                  f"{catcher}")
        return 1
    print(f"all {len(CASES)} mutants killed — every check added after "
          f"mutation testing was observed going red on the mutation that "
          f"motivated it")
    print(f"\n{len(EQUIVALENT)} further survivors are equivalent mutants, "
          f"verified by inspection rather than by a check:")
    for fname, line, what, why in EQUIVALENT:
        print(f"  {fname}:{line}  {what}\n      {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
