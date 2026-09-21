"""Sub-2-minute integrity check of the evidence chain, meant to be re-run cheaply
after any of fixture.json, results/folds.json, results/selected.json or
RESULTS.md changes.

(a) fixture.json: corpus_sha256 recomputes from its own entries; n_labels and
    label_counts agree with a recount over the memories' own tags; every
    memory's labels are a subset of its tags and of the global label set; no
    memory carries a private tag (common.PRIVATE_TAGS).
(b) results/folds.json: every id is a fixture id; fold_of has one entry per id;
    the number of distinct folds equals n_splits; fold sizes are within one of
    n / n_splits of each other (StratifiedKFold's own balance guarantee).
(c) RESULTS.md, if it exists: every number in results/headline.json appears in
    it, formatted to 3 decimals. Skipped silently (not a failure) if RESULTS.md
    does not exist yet.
(d) results/selected.json: each selected arm's micro-F1 recomputes to 1e-9 from
    its cached results/proba_{arm}_C{c}.npy file and a label matrix rebuilt
    from fixture.json + results/folds.json.

Non-zero exit on any failure.

Usage: python3 recheck.py
"""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import MIN_LABEL_COUNT, PRIVATE_TAGS, load_fixture, text_hash
from probe import SEED, micro_f1

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

failures = []


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def check_fixture():
    fx = load_fixture()
    entries = fx["memories"]

    recomputed = text_hash("\n".join(f"{e['id']} {e['sha256']}" for e in entries))
    check("fixture corpus_sha256 recomputes from its own entries",
          recomputed == fx["corpus_sha256"],
          f"stored={fx['corpus_sha256'][:12]} recomputed={recomputed[:12]}")

    counts = collections.Counter()
    for e in entries:
        counts.update(e["tags"])
    labels = fx["labels"]
    check("n_labels == len(labels)", fx["n_labels"] == len(labels))
    check("label_counts keys == labels", set(fx["label_counts"].keys()) == set(labels))
    mismatches = [t for t in labels if counts.get(t, 0) != fx["label_counts"].get(t)]
    check("label_counts values match a recount from memories' own tags", not mismatches,
          f"{len(mismatches)} mismatched: {mismatches[:5]}")
    under = [t for t in labels if counts.get(t, 0) < MIN_LABEL_COUNT]
    check(f"every label has >= {MIN_LABEL_COUNT} uses", not under, f"{under[:5]}")

    label_set = set(labels)
    bad = [e["id"] for e in entries
           if not (set(e["labels"]) <= set(e["tags"]) and set(e["labels"]) <= label_set)]
    check("every memory's labels are a subset of its tags and of the global label set",
          not bad, f"{len(bad)} bad, e.g. {bad[:5]}")

    hits = [e["id"] for e in entries if set(e["tags"]) & PRIVATE_TAGS]
    check("no memory carries a private tag", not hits, f"{len(hits)} hit, e.g. {hits[:5]}")


def check_folds():
    p = RESULTS / "folds.json"
    if not p.exists():
        check("results/folds.json exists", False)
        return
    folds = json.loads(p.read_text())
    fixture_ids = {m["id"] for m in load_fixture()["memories"]}
    ids = folds["ids"]

    check("folds.json ids are a subset of fixture ids", set(ids) <= fixture_ids,
          f"{len(set(ids) - fixture_ids)} ids not in fixture")
    check("folds.json ids has no duplicates", len(ids) == len(set(ids)))

    fold_of = folds["fold_of"]
    check("len(fold_of) == folds['n'] == len(ids)",
          len(fold_of) == folds["n"] == len(ids),
          f"len(fold_of)={len(fold_of)} n={folds['n']} len(ids)={len(ids)}")

    n_splits = folds["n_splits"]
    sizes = collections.Counter(fold_of)
    check("number of distinct folds == n_splits", len(sizes) == n_splits,
          f"got folds {sorted(sizes.keys())}, n_splits={n_splits}")

    lo, hi = folds["n"] // n_splits, -(-folds["n"] // n_splits)  # floor, ceil
    bad_sizes = {k: v for k, v in sizes.items() if not (lo <= v <= hi)}
    check(f"every fold size is within one of n/n_splits ({lo}-{hi})", not bad_sizes,
          f"{bad_sizes}")


def check_results_md():
    md_path = HERE / "RESULTS.md"
    hl_path = RESULTS / "headline.json"
    if not md_path.exists():
        print("[SKIP] RESULTS.md does not exist yet — skipping headline-number check")
        return
    if not hl_path.exists():
        check("results/headline.json exists (needed to check RESULTS.md against it)", False)
        return
    headline = json.loads(hl_path.read_text())
    text = md_path.read_text()
    missing = []
    for name, v in headline.items():
        if not isinstance(v, (int, float)):
            continue
        s = f"{v:.3f}"
        if s not in text:
            missing.append(f"{name}={s}")
    check("every results/headline.json number appears in RESULTS.md to 3 decimals",
          not missing, f"{len(missing)} missing: {missing[:8]}")


def check_selected_recompute():
    p = RESULTS / "selected.json"
    if not p.exists():
        check("results/selected.json exists", False)
        return
    selected = json.loads(p.read_text())
    sm = selected.get("selected_metrics", {})
    if not sm:
        check("selected.json has selected_metrics", False)
        return

    fixture = load_fixture()
    id_to_labels = {m["id"]: m["labels"] for m in fixture["memories"]}
    labels = fixture["labels"]
    label_idx = {lab: j for j, lab in enumerate(labels)}
    folds = json.loads((RESULTS / "folds.json").read_text())
    ids = folds["ids"]
    n, L = len(ids), len(labels)
    Y = np.zeros((n, L), dtype=np.int8)
    for i, mid in enumerate(ids):
        for lab in id_to_labels[mid]:
            Y[i, label_idx[lab]] = 1
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    Y_shuf = Y[perm]

    for arm, m in sm.items():
        c = m["C"]
        proba_path = RESULTS / f"proba_{arm}_C{c:g}.npy"
        if not proba_path.exists():
            check(f"cached proba exists for {arm} @ C{c:g}", False)
            continue
        P = np.load(proba_path)
        y_use = Y_shuf if arm == "shuffled" else Y
        recomputed = micro_f1(y_use, P)
        ok = abs(recomputed - m["micro_f1"]) < 1e-9
        check(f"selected.json micro-F1 for {arm} recomputes to 1e-9 from cached proba", ok,
              f"stored={m['micro_f1']!r} recomputed={recomputed!r} diff={abs(recomputed - m['micro_f1']):.2e}")


def main():
    t0 = time.time()
    check_fixture()
    check_folds()
    check_results_md()
    check_selected_recompute()
    el = time.time() - t0
    print(f"\n{len(failures)} failure(s) in {el:.1f}s")
    if failures:
        for f in failures:
            print(f" - {f}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
