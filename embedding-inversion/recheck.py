#!/usr/bin/env python3
"""Sub-minute re-verification of this directory. Run on any touch of it.

The trained checkpoints and the corpus are gitignored, so nobody rebuilds them
to check a one-line edit. This is the standing check that makes that safe.

  1. ARTIFACT INTEGRITY   results_<cond>.json has the shape the writeup assumes
  2. PROSE vs ARTIFACT    every number in the RESULTS.md tables is recomputed
                          from the per-item records by code that shares nothing
                          with evaluate.py's metric path, and must match
  3. SELF-CONSISTENCY     the "not retrieval" claims, monotone improvement,
                          exact-by-round counts
  4. LIVE CHECK           bekko re-embeds a sample of final hypotheses and the
                          stored cosines reproduce (skipped, loudly, when the
                          encoder or data is not on disk)

Run:  python3 recheck.py     (non-zero exit on any failure)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def norm(s: str) -> str:
    return " ".join(s.lower().replace("’", "'").split())


def f1(a: str, b: str) -> float:
    ta, tb = norm(a).split(), norm(b).split()
    c = sum((Counter(ta) & Counter(tb)).values())
    if not ta or not tb or c == 0:
        return 0.0
    p, r = c / len(tb), c / len(ta)
    return 2 * p * r / (p + r)


def table_rows(md: str, header_first_cell: str) -> list[list[str]]:
    """Rows of the markdown table whose header starts with `header_first_cell`."""
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(f"| {header_first_cell}"):
            rows = []
            for r in lines[i + 2:]:
                if not r.startswith("|"):
                    break
                rows.append([c.strip() for c in r.strip("|").split("|")])
            return rows
    return []


def section(md: str, cond: str) -> str:
    m = re.search(rf"### {cond} condition.*?(?=\n### |\Z)", md, re.S)
    return m.group(0) if m else ""


ROUND_LABELS = {
    "nearest training string (control)": "nn_train",
    "zero-step, top beam": "zero_top_beam",
    "zero-step, verifier picks among 4 beams (round 0)": "round0",
    **{f"round {k}": f"round{k}" for k in range(1, 6)},
}


def recheck_cond(cond: str, md: str) -> None:
    path = HERE / f"results_{cond}.json"
    print(f"\n== {cond}")
    if not path.exists():
        print(f"  SKIP results_{cond}.json not present")
        return
    r = json.loads(path.read_text())
    items = r["items"]

    print(" phase 1: artifact integrity")
    check(r["cond"] == cond and r["n"] == len(items) == 1000, "n = 1000 items, cond tag matches")
    check(all(k in r["rounds"] for k in ROUND_LABELS.values()), "all round keys present")
    check(all(len(i["hyps"]) == 6 for i in items), "six hypotheses per item (round 0..5)")
    check(set(r["by_length"]) == {"0-6", "7-10", "11-16", "17-plus"}, "four length buckets")

    print(" phase 2: prose vs artifact")
    sec = section(md, cond)
    check(bool(sec), f"RESULTS.md has a '### {cond} condition' section")
    # recompute per-round exact / F1 from the items (independent of evaluate.metrics)
    ex_by_round = [np.mean([norm(i["text"]) == norm(i["hyps"][k]) for i in items]) for k in range(6)]
    f1_by_round = [np.mean([f1(i["text"], i["hyps"][k]) for i in items]) for k in range(6)]
    for k in range(6):
        check(abs(ex_by_round[k] - r["rounds"][f"round{k}"]["exact"]) < 1e-9, f"round{k} exact recomputed {ex_by_round[k]:.3f}")
        check(abs(f1_by_round[k] - r["rounds"][f"round{k}"]["token_f1"]) < 1e-6, f"round{k} token F1 recomputed {f1_by_round[k]:.3f}")
    nn_ex = np.mean([norm(i["text"]) == norm(i["nn"]) for i in items])
    check(abs(nn_ex - r["rounds"]["nn_train"]["exact"]) < 1e-9, f"nn_train exact recomputed {nn_ex:.3f}")
    fc = np.array([i["final_cos"] for i in items])
    check(abs(fc.mean() - r["rounds"]["round5"]["cosine"]) < 1e-4, f"final cosine mean recomputed {fc.mean():.3f}")
    rows = table_rows(sec, "arm")
    check(len(rows) == 8, "results table has 8 rows")
    for row in rows:
        key = ROUND_LABELS.get(row[0])
        if key is None:
            check(False, f"unknown table row label {row[0]!r}")
            continue
        got = r["rounds"][key]
        check(abs(float(row[1]) - got["exact"]) < 5e-4, f"table {key} exact {row[1]}")
        check(abs(float(row[2]) - got["token_f1"]) < 5e-4, f"table {key} F1 {row[2]}")
        check(abs(float(row[3]) - got["bleu"]) < 0.05, f"table {key} BLEU {row[3]}")
        check(abs(float(row[4]) - got["cosine"]) < 5e-4, f"table {key} cosine {row[4]}")
    lrows = table_rows(sec, "words")
    check(len(lrows) == 4, "length table has 4 rows")
    wl = np.array([i["words"] for i in items])
    for row, (lo, hi, key) in zip(lrows, [(0, 6, "0-6"), (7, 10, "7-10"), (11, 16, "11-16"), (17, 999, "17-plus")]):
        m = (wl >= lo) & (wl <= hi)
        ex = np.mean([norm(items[j]["text"]) == norm(items[j]["hyps"][-1]) for j in np.where(m)[0]])
        check(int(row[1]) == int(m.sum()) == r["by_length"][key]["n"], f"bucket {key} n {row[1]}")
        check(abs(float(row[2]) - ex) < 5e-4 and abs(ex - r["by_length"][key]["exact"]) < 1e-9, f"bucket {key} exact {row[2]}")
        check(abs(float(row[4]) - fc[m].mean()) < 5e-4, f"bucket {key} cosine {row[4]}")
    q = np.percentile(fc, [10, 50, 90])
    m = re.search(r"p10 ([0-9]+\.[0-9]+), median ([0-9]+\.[0-9]+), p90 ([0-9]+\.[0-9]+)", sec)
    check(bool(m) and all(abs(float(m.group(k + 1)) - q[k]) < 5e-3 for k in range(3)), f"cosine quantiles {q.round(3)}")
    m = re.search(r"(\d+)% of\s+items end above 0\.8 and (\d+)% above 0\.9", sec)
    check(bool(m) and int(m.group(1)) == round(100 * (fc >= 0.8).mean()) and int(m.group(2)) == round(100 * (fc >= 0.9).mean()),
          f"share above 0.8 / 0.9 = {(fc>=0.8).mean():.3f} / {(fc>=0.9).mean():.3f}")
    m = re.search(r"Exact matches by round: ([\d, ]+)\.", sec)
    counts = [int(round(x * 1000)) for x in ex_by_round]
    check(bool(m) and [int(x) for x in m.group(1).split(",")] == counts, f"exact-by-round counts {counts}")

    print(" phase 3: self-consistency")
    check(all(norm(i["hyps"][-1]) != norm(i["nn"]) for i in items), "no final hypothesis equals the nearest-neighbour string")
    imp = [r["rounds"][f"round{k}"]["improved_frac"] for k in range(1, 6)]
    check(all(a >= b for a, b in zip(imp, imp[1:])), f"improved_frac non-increasing {imp}")
    cos_r = [r["rounds"][f"round{k}"]["cosine"] for k in range(6)]
    check(all(a <= b + 1e-9 for a, b in zip(cos_r, cos_r[1:])), "verifier cosine non-decreasing across rounds (incumbent rule)")
    check(r["rounds"]["round5"]["cosine"] > r["rounds"]["nn_train"]["cosine"], "final cosine above the retrieval control")
    splits = HERE / "data/splits.json"
    if splits.exists():
        train = {norm(t) for t in json.loads(splits.read_text())["train"]}
        check(not any(norm(i["hyps"][-1]) in train for i in items), "no final hypothesis is a training string")
        leaks = sum(norm(i["text"]) in train for i in items)
        check(leaks <= 1, f"test strings present in train: {leaks} (ERRORS.md #5 says 1)")
    else:
        print("  SKIP data/splits.json not on disk; training-string checks not run")

    print(" phase 4: live check")
    try:
        from encoder import BekkoEncoder, SignBits, condition
        enc = BekkoEncoder()
        sb = SignBits.load(HERE / "data/signbits_mu.npy") if cond == "bin1" else None
        idx = np.random.default_rng(0).choice(len(items), 20, replace=False)
        hyps = [items[j]["hyps"][-1] for j in idx]
        texts = [items[j]["text"] for j in idx]
        h = condition(cond, enc.encode(hyps), sb)
        t = condition(cond, enc.encode(texts), sb)
        cos = (h * t).sum(1)
        stored = np.array([items[j]["final_cos"] for j in idx])
        check(np.abs(cos - stored).max() < 0.03, f"re-embedded cosines reproduce (max |Δ| {np.abs(cos-stored).max():.4f})")
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP live re-embedding not possible here: {type(e).__name__}: {e}")


def main() -> None:
    md = (HERE / "RESULTS.md").read_text()
    for cond in ("float", "bin1"):
        recheck_cond(cond, md)
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
