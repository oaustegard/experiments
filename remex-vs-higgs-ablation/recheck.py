#!/usr/bin/env python3
"""Sub-5-minute re-verification. Run this on any touch of this directory.

Trust decays. The expensive artifacts here — corpora, trained grids, the sweep
— took hours and are gitignored or large, so nobody is going to rebuild them to
check a one-line edit. This is the cheap standing check that makes *not*
rebuilding them safe, and it targets the failure modes that actually occurred
in this experiment rather than the ones that sound plausible.

Four phases, cheapest first:

  1. ARTIFACT INTEGRITY  results.json has the shape the writeup assumes
  2. PROSE vs ARTIFACT   the headline numbers in RESULTS.md are RECOMPUTED
                         from results.json by a code path that shares nothing
                         with summarize.py, and must match what the prose says
  3. SELF-CONSISTENCY    gate.log's stated totals match its own check lines,
                         and RESULTS.md quotes those totals correctly
  4. LIVE CHECKS         the fast gate passes; a sample of the pinned mutants
                         still die

Phase 2 and phase 3 are the ones worth having. Two of run 2's logged errors
(ERRORS.md run 2 #7 and #8) were prose that disagreed with the artifacts it
described, and no amount of re-running the science would have found them — the
numbers were right and the sentences about them were wrong.

What this does NOT do: it does not re-derive the science. Phase 2 recomputes
the pooled axis deltas from stored per-cell recalls, so it catches prose drift
and arithmetic drift, not a wrong recall in results.json. Rebuilding that is
`build_corpora.py` + `run_ablation.py`, which is hours. See ERRORS.md.

Run:  python3 recheck.py     (~90 s; non-zero exit on any failure)
"""
from __future__ import annotations

import itertools
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROT = ("haar", "rht")
NRM = ("exactnorm", "blockscale")
CB = ("scalar", "vector")
AXES = {"A": (0, ROT), "B": (1, NRM), "C": (2, CB)}
CORPORA = ("arxiv768", "glove100", "nfcorpus1024")
BITS = ("1", "2", "3", "4", "6", "8")

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{label}: {detail}")
    return ok


# --------------------------------------------------------------------------
# 1. artifact integrity


def phase1(res) -> None:
    print("\n1. ARTIFACT INTEGRITY")
    blocks = [k for k in res if not k.startswith("_")]
    check(len(blocks) == 6, "results.json has 6 corpus x metric blocks",
          f"{sorted(blocks)}")
    check("_timing" in res, "axis-A timing block present")
    for b in blocks:
        want = set(BITS) | {"fp32"}
        check(set(res[b]) == want, f"{b} covers every bit width",
              f"{sorted(res[b])}")


# --------------------------------------------------------------------------
# 2. prose vs artifact — the phase that earns its keep


def pooled_axis_means(res) -> dict[tuple[str, str], float]:
    """Recompute the pooled marginal effect of each axis, from scratch.

    Deliberately does NOT import plot.marginals or summarize.py: a check that
    calls the code it is checking is a changelog. This walks results.json and
    averages, by hand, the delta from flipping one axis with the other two
    held fixed.
    """
    out: dict[tuple[str, str], float] = {}
    for metric in ("cosine", "ip"):
        for ax, (pos, vals) in AXES.items():
            deltas = []
            for corpus in CORPORA:
                block = res[f"{corpus}|{metric}"]
                for bits in BITS:
                    cells = block[bits]
                    held = [x for i, x in enumerate((ROT, NRM, CB)) if i != pos]
                    for combo in itertools.product(*held):
                        lab: list[str | None] = [None, None, None]
                        k = 0
                        for i in range(3):
                            if i != pos:
                                lab[i] = combo[k]
                                k += 1
                        lo, hi = list(lab), list(lab)
                        lo[pos], hi[pos] = vals[0], vals[1]
                        a, b = "+".join(lo), "+".join(hi)  # type: ignore[arg-type]
                        if a in cells and b in cells:
                            deltas.append(cells[b]["recall@10"]["mean"]
                                          - cells[a]["recall@10"]["mean"])
            out[(metric, ax)] = st.mean(deltas)
    return out


def parse_results_axis_table(text: str) -> dict[tuple[str, str], float]:
    """Pull the six pooled axis numbers out of the RESULTS.md prose table."""
    got: dict[tuple[str, str], float] = {}
    # rows look like: | **A** rotation: ... | +0.0005 ± 0.0016 | -0.0003 ± 0.0024 |
    row = re.compile(
        r"\|\s*\*\*([ABC])\*\*[^|]*\|\s*\*?\*?([+−-][\d.]+)[^|]*\|"
        r"\s*\*?\*?([+−-][\d.]+)[^|]*\|")
    for m in row.finditer(text):
        ax = m.group(1)
        for metric, raw in (("cosine", m.group(2)), ("ip", m.group(3))):
            got[(metric, ax)] = float(raw.replace("−", "-"))
    return got


def phase2(res, results_md: str) -> None:
    print("\n2. PROSE vs ARTIFACT (recomputed by a disjoint path)")
    recomputed = pooled_axis_means(res)
    stated = parse_results_axis_table(results_md)
    check(len(stated) == 6, "RESULTS.md axis table parsed",
          f"found {len(stated)} of 6 entries")
    for key in sorted(recomputed):
        if key not in stated:
            check(False, f"{key[0]} axis {key[1]} stated in RESULTS.md", "missing")
            continue
        # RESULTS.md quotes 4 decimal places, so agreement to 5e-5 is exact
        # agreement at the precision the prose claims.
        ok = abs(recomputed[key] - stated[key]) < 5e-5
        check(ok, f"{key[0]} axis {key[1]}",
              f"results.json -> {recomputed[key]:+.4f}, "
              f"RESULTS.md says {stated[key]:+.4f}")


# --------------------------------------------------------------------------
# 3. self-consistency of the gate log and the prose that quotes it


def phase3(gate_log: str, results_md: str) -> None:
    print("\n3. SELF-CONSISTENCY")
    n_checks = len(re.findall(r"^\s+\[(?:PASS|FAIL)\]", gate_log, re.MULTILINE))
    n_kb = len(re.findall(r"\(known-bad\)", gate_log))
    n_lim = len(re.findall(r"^\s+\[cannot catch\]", gate_log, re.MULTILINE))
    verdict = re.search(
        r"PASSED — (\d+) checks, (\d+) known-bad rejected, (\d+) coverage",
        gate_log)
    check(verdict is not None, "gate.log ends in a PASSED verdict")
    if not verdict:
        return
    sc, sk, sl = (int(x) for x in verdict.groups())
    check(sc == n_checks, "gate.log check count matches its own lines",
          f"stated {sc}, counted {n_checks}")
    check(sk == n_kb, "gate.log known-bad count matches its own lines",
          f"stated {sk}, counted {n_kb}")
    check(sl == n_lim, "gate.log coverage-limit count matches its own lines",
          f"stated {sl}, counted {n_lim}")

    # RESULTS.md quotes those totals; a stale quote is exactly ERRORS.md
    # run 2 #7, which shipped in the first draft of this rerun.
    flat = " ".join(results_md.split())
    quoted = re.search(
        r"PASSED — (\d+) checks, (\d+) known-bad rejected, (\d+) coverage", flat)
    check(quoted is not None, "RESULTS.md quotes the gate verdict")
    if quoted:
        qc, qk, ql = (int(x) for x in quoted.groups())
        check((qc, qk, ql) == (sc, sk, sl),
              "RESULTS.md's quoted gate totals match gate.log",
              f"prose {qc}/{qk}/{ql} vs log {sc}/{sk}/{sl}")

    m = re.search(r"`\[cannot catch\]` — (\d+) entries", flat)
    check(m is not None and int(m.group(1)) == n_lim,
          "RESULTS.md's coverage-limit count matches gate.log",
          f"prose {m.group(1) if m else '?'} vs log {n_lim}")


# --------------------------------------------------------------------------
# 4. live checks


def phase4(quick: bool) -> None:
    print("\n4. LIVE CHECKS")
    rc = subprocess.call([sys.executable, str(HERE / "gate.py"), "--fast"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    check(rc == 0, "gate.py --fast passes", f"exit {rc}")

    sys.path.insert(0, str(HERE))
    from verify_kills import CASES, mutate_line
    # A sample, not all nine: this is the standing check, and `verify_kills.py`
    # is the exhaustive one.  Picked to span both files and both kinds of hole
    # (an unrun encoder, a byte-accounting tautology).
    sample = [c for c in CASES
              if (c[0], c[1]) in {("grids.py", 166), ("grids.py", 194),
                                  ("quantizers.py", 206)}]
    if quick:
        sample = sample[:1]
    for fname, line, before, after, breaks, _ in sample:
        path = HERE / fname
        original = path.read_text()
        try:
            path.write_text(mutate_line(original, line, before, after))
            rc = subprocess.call([sys.executable, str(HERE / "gate.py"), "--fast"],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        finally:
            path.write_text(original)
        check(rc != 0, f"mutant {fname}:{line} {before}->{after} still dies",
              breaks)


def main() -> int:
    quick = "--quick" in sys.argv
    print("=" * 70)
    print("RECHECK — remex-vs-higgs-ablation" + (" [quick]" if quick else ""))
    print("=" * 70)
    res = json.loads((HERE / "results.json").read_text())
    results_md = (HERE / "RESULTS.md").read_text()
    gate_log = (HERE / "gate.log").read_text()

    phase1(res)
    phase2(res, results_md)
    phase3(gate_log, results_md)
    phase4(quick)

    print("\n" + "-" * 70)
    if FAILURES:
        print(f"RECHECK FAILED — {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  * {f}")
        return 1
    print("RECHECK PASSED — artifacts, prose and gate agree.")
    print("Not covered: the recall numbers themselves. Phase 2 recomputes the "
          "pooled\ndeltas FROM results.json, so a wrong recall stored there is "
          "invisible here.\nThat needs build_corpora.py + run_ablation.py "
          "(hours). See ERRORS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
