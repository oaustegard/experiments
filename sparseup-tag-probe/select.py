"""Per-arm oracle C selection over the swept regularisation grid.

TF-IDF and gte-small rows are L2-normalised (row norm 1.0); SPARSEUP rows are
not (mean row L2 norm ~32, max cell value ~3.9). A single C is therefore a
very different effective prior per arm, and the C=1 numbers alone are not a
fair arm comparison. This reads every results/probe_C*.json to discover which
C values were swept, then for EACH ARM INDEPENDENTLY picks the C in that same
swept grid with the highest pooled micro-F1 (an oracle per arm, not one C
chosen for all of them — every arm still only gets to pick from the grid every
other arm was also swept on).

Every number in results/selected.json is recomputed from the cached
results/proba_{arm}_C{c}.npy matrices and a label matrix rebuilt from
fixture.json + results/folds.json (never copied from probe_C*.json), so it is
reproducible byte-for-byte from the cache alone — recheck.py depends on this.

Also writes results/headline.json: the flat set of metric numbers RESULTS.md
is expected to quote, for recheck.py's grep check.

Usage: python3 select.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

from common import load_fixture
from probe import SEED, bootstrap_ci_diff, macro_f1, micro_f1, p_at_5, per_label_f1

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

ARMS = ("sparseup", "sparseup_binary", "tfidf_word", "tfidf_char",
        "tfidf_word+char", "gte_small", "prior", "shuffled")


def ctag(c):
    return f"C{c:g}"


def discover_Cs():
    Cs = []
    for p in sorted(RESULTS.glob("probe_C*.json")):
        m = re.match(r"probe_C([0-9.]+)\.json$", p.name)
        if m:
            Cs.append(float(m.group(1)))
    return sorted(set(Cs))


def load_Y():
    """Exactly as probe.py builds it: fixture memories' labels, in the row
    order results/folds.json pinned (== probe.py's `labelled` order)."""
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
    return Y, labels, label_idx


def load_proba(arm, c):
    p = RESULTS / f"proba_{arm}_{ctag(c)}.npy"
    return np.load(p) if p.exists() else None


def main():
    Cs = discover_Cs()
    if not Cs:
        print("No results/probe_C*.json found; run probe.py --C <value> at least once first.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Swept C grid: {Cs}", file=sys.stderr)

    Y, labels, label_idx = load_Y()
    n = Y.shape[0]
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    Y_shuf = Y[perm]

    # ---- "same C for all" view: recomputed arm x metric grid, at every swept C
    grid = {}
    missing = []
    for c in Cs:
        grid[ctag(c)] = {}
        for arm in ARMS:
            P = load_proba(arm, c)
            if P is None:
                missing.append([arm, c])
                continue
            y_use = Y_shuf if arm == "shuffled" else Y
            grid[ctag(c)][arm] = {
                "micro_f1": micro_f1(y_use, P),
                "macro_f1": macro_f1(y_use, P),
                "p_at_5": p_at_5(y_use, P),
            }
    if missing:
        print(f"Missing cached proba (arm, C) -- skipped in the grid and in selection: {missing}",
              file=sys.stderr)

    # ---- per-arm oracle: best C by pooled micro-F1, same grid offered to every arm
    best_c = {}
    for arm in ARMS:
        candidates = [(c, grid[ctag(c)][arm]["micro_f1"]) for c in Cs if arm in grid[ctag(c)]]
        if candidates:
            best_c[arm] = max(candidates, key=lambda t: t[1])[0]

    # ---- full recompute (incl. per-label F1) at each arm's own selected C
    selected_proba = {arm: load_proba(arm, c) for arm, c in best_c.items()}
    selected_metrics = {}
    for arm, P in selected_proba.items():
        y_use = Y_shuf if arm == "shuffled" else Y
        selected_metrics[arm] = {
            "C": best_c[arm],
            "micro_f1": micro_f1(y_use, P),
            "macro_f1": macro_f1(y_use, P),
            "p_at_5": p_at_5(y_use, P),
            "per_label_f1": per_label_f1(y_use, P).tolist(),
        }

    # ---- binarization gap: sparseup at its best C minus sparseup_binary at its best C
    gap = None
    if "sparseup" in selected_metrics and "sparseup_binary" in selected_metrics:
        gap = {m: selected_metrics["sparseup"][m] - selected_metrics["sparseup_binary"][m]
               for m in ("micro_f1", "macro_f1", "p_at_5")}

    # ---- literal-mention split (same grouping probe.py uses), from results/arm1.json
    literal_split = None
    arm1_path = RESULTS / "arm1.json"
    if arm1_path.exists() and {"sparseup", "tfidf_word+char"} <= set(selected_metrics):
        arm1 = json.loads(arm1_path.read_text())
        lit_rates = {lab: e.get("literal_mention_rate") for lab, e in arm1.get("per_label", {}).items()
                     if e.get("literal_mention_rate") is not None}
        group_hi = [label_idx[l] for l in labels if lit_rates.get(l, 0.0) >= 0.5]
        group_lo = [label_idx[l] for l in labels if lit_rates.get(l, 0.0) < 0.5]

        def mean_f1(arm, idxs):
            f1s = np.array(selected_metrics[arm]["per_label_f1"])
            return float(np.nanmean(f1s[idxs])) if idxs else None

        literal_split = {
            "n_labels_high_literal_rate_ge_0.5": len(group_hi),
            "n_labels_low_literal_rate_lt_0.5": len(group_lo),
            "mean_f1": {
                "sparseup": {"high_literal": mean_f1("sparseup", group_hi),
                             "low_literal": mean_f1("sparseup", group_lo)},
                "tfidf_word+char": {"high_literal": mean_f1("tfidf_word+char", group_hi),
                                     "low_literal": mean_f1("tfidf_word+char", group_lo)},
            },
        }

    # ---- paired bootstrap CIs (each arm at its own selected C)
    boot = None
    needed = {"sparseup", "tfidf_word+char", "sparseup_binary"}
    if needed <= set(selected_proba):
        boot = {
            "sparseup_minus_tfidf_word+char": {
                "micro_f1": bootstrap_ci_diff(Y, selected_proba["sparseup"], selected_proba["tfidf_word+char"], micro_f1),
                "p_at_5": bootstrap_ci_diff(Y, selected_proba["sparseup"], selected_proba["tfidf_word+char"], p_at_5),
            },
            "sparseup_minus_sparseup_binary": {
                "micro_f1": bootstrap_ci_diff(Y, selected_proba["sparseup"], selected_proba["sparseup_binary"], micro_f1),
                "p_at_5": bootstrap_ci_diff(Y, selected_proba["sparseup"], selected_proba["sparseup_binary"], p_at_5),
            },
        }
    else:
        print(f"Skipping bootstrap CIs -- missing selected proba for {needed - set(selected_proba)}",
              file=sys.stderr)

    out = {
        "n": n,
        "n_labels": len(labels),
        "C_grid": Cs,
        "selection_method": ("per-arm oracle: for each arm independently, the C in the swept grid "
                              "with the highest pooled micro-F1 -- NOT a single C chosen for all arms. "
                              "Every arm chooses from the same grid."),
        "selected_C": best_c,
        "selected_metrics": selected_metrics,
        "same_C_grid": grid,
        "binarization_gap_sparseup_minus_binary": gap,
        "literal_mention_split": literal_split,
        "bootstrap_ci_diff": boot,
        "missing_proba": missing,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selected.json").write_text(json.dumps(out, indent=1))

    # ---- headline.json: flat metric numbers for recheck.py to grep RESULTS.md for
    headline = {}
    for arm in ("sparseup", "sparseup_binary", "tfidf_word", "tfidf_char",
                "tfidf_word+char", "gte_small", "prior", "shuffled"):
        if arm in selected_metrics:
            m = selected_metrics[arm]
            headline[f"{arm}_micro_f1"] = round(m["micro_f1"], 3)
            headline[f"{arm}_macro_f1"] = round(m["macro_f1"], 3)
            headline[f"{arm}_p_at_5"] = round(m["p_at_5"], 3)
    if gap:
        for k, v in gap.items():
            headline[f"binarization_gap_{k}"] = round(v, 3)
    if boot:
        for cmp_name, metrics in boot.items():
            for metric_name, v in metrics.items():
                headline[f"{cmp_name}_{metric_name}_diff"] = round(v["mean_diff"], 3)
                headline[f"{cmp_name}_{metric_name}_ci_lo"] = round(v["ci_lo"], 3)
                headline[f"{cmp_name}_{metric_name}_ci_hi"] = round(v["ci_hi"], 3)
    if literal_split:
        for arm, groups in literal_split["mean_f1"].items():
            for g, v in groups.items():
                if v is not None:
                    headline[f"literal_split_{arm}_{g}"] = round(v, 3)
    arm1_path = RESULTS / "arm1.json"
    if arm1_path.exists():
        arm1 = json.loads(arm1_path.read_text())
        for hd in ("exact", "all_parts"):
            for k, v in arm1["micro_recall"].get(hd, {}).items():
                if v is not None:
                    headline[f"arm1_{hd}_recall_k{k}"] = round(v, 3)
    (RESULTS / "headline.json").write_text(json.dumps(headline, indent=1))

    print(f"\nSwept C grid: {Cs}")
    print("selection: per-arm oracle over that grid (highest pooled micro-F1), not one C for all\n")
    print(f"{'arm':<18}{'best C':>10}{'micro-F1':>10}{'macro-F1':>10}{'P@5':>8}")
    for arm in ARMS:
        if arm in selected_metrics:
            m = selected_metrics[arm]
            print(f"{arm:<18}{m['C']:>10g}{m['micro_f1']:>10.4f}{m['macro_f1']:>10.4f}{m['p_at_5']:>8.4f}")
        else:
            print(f"{arm:<18}{'--':>10}{'n/a':>10}{'n/a':>10}{'n/a':>8}")
    if gap:
        print(f"\nbinarization gap (sparseup@C{best_c.get('sparseup','?'):g} - "
              f"sparseup_binary@C{best_c.get('sparseup_binary','?'):g}): "
              f"micro-F1 {gap['micro_f1']:+.4f}, macro-F1 {gap['macro_f1']:+.4f}, P@5 {gap['p_at_5']:+.4f}")
    if boot:
        b1 = boot["sparseup_minus_tfidf_word+char"]["micro_f1"]
        b2 = boot["sparseup_minus_sparseup_binary"]["micro_f1"]
        print(f"sparseup - tfidf_word+char micro-F1: {b1['mean_diff']:+.4f} "
              f"[{b1['ci_lo']:+.4f}, {b1['ci_hi']:+.4f}]")
        print(f"sparseup - sparseup_binary micro-F1: {b2['mean_diff']:+.4f} "
              f"[{b2['ci_lo']:+.4f}, {b2['ci_hi']:+.4f}]")
    print(f"\nWrote {RESULTS / 'selected.json'} and {RESULTS / 'headline.json'}")


if __name__ == "__main__":
    main()
