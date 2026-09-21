"""Per-arm oracle C selection over the swept regularisation grid, selecting on
micro-AP (threshold-free) rather than a fixed 0.5 threshold.

Why: only 0.3-0.6% of probability cells exceed 0.5 for the sparse/TF-IDF arms
(most one-vs-rest classifiers here are calibrated toward low positive rates),
so F1@0.5 rewards whichever arm's scores happen to cluster near 1.0 rather
than whichever arm ranks documents against labels correctly. AP does not
depend on where that mass sits. The old micro-F1@0.5 oracle C is still
reported (as a secondary column) so the write-up can show both and how much
they disagree.

TF-IDF and gte-small rows are also L2-normalised while SPARSEUP rows are not
(mean row L2 norm ~32), which is why every arm gets its OWN swept grid and its
own oracle C rather than one C for all of them — and, this round, an arm may
have a partial grid (a new arm or a new C can lag the others), so C values are
discovered per arm from whatever results/proba_{arm}_C*.npy files exist on
disk, not assumed to be the same set for every arm.

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
from sklearn.metrics import average_precision_score

from common import load_fixture
from probe import SEED, bootstrap_ci_diff, macro_f1, micro_f1, p_at_5, per_label_ap

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Display order; an arm with no cached proba anywhere is simply skipped.
ARMS = ("sparseup", "sparseup_binary", "tfidf_word", "tfidf_char",
        "tfidf_word+char", "tfidf_word+char_trunc512", "gte_small", "prior", "shuffled")

THRESHOLDS = [round(0.005 * i, 3) for i in range(1, 191)]  # 0.005 .. 0.950 step 0.005

# The four bootstrap comparisons this round asks for (each side at its own selected C).
BOOT_COMPARISONS = (
    ("sparseup", "tfidf_word+char"),
    ("sparseup", "tfidf_word+char_trunc512"),
    ("sparseup", "sparseup_binary"),
    ("sparseup", "gte_small"),
)


def micro_ap(Y, P):
    return float(average_precision_score(Y, P, average="micro"))


def macro_ap(Y, P):
    return float(average_precision_score(Y, P, average="macro"))


def best_threshold_micro_f1(Y, P):
    """Single global threshold maximising pooled micro-F1; also returns macro-F1
    AT THAT SAME threshold (not independently optimised)."""
    best_t, best_f1 = THRESHOLDS[0], -1.0
    for t in THRESHOLDS:
        f1 = micro_f1(Y, P, thresh=t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1, macro_f1(Y, P, thresh=best_t)


def discover_Cs_per_arm():
    """{arm: sorted [C, ...]} from whatever results/proba_{arm}_C{c}.npy exist on
    disk -- the ground truth for what each arm actually has cached, independent
    of which arms a given probe_C*.json's 'arms' dict happens to list (an arm
    added after some C values were already swept won't be in those older files
    even once its proba for a LATER C exists)."""
    out = {}
    for p in RESULTS.glob("proba_*_C*.npy"):
        m = re.match(r"proba_(.+)_C([0-9.]+)\.npy$", p.name)
        if not m:
            continue
        arm, c = m.group(1), float(m.group(2))
        out.setdefault(arm, set()).add(c)
    return {arm: sorted(cs) for arm, cs in out.items()}


def ctag(c):
    return f"C{c:g}"


def load_proba(arm, c):
    p = RESULTS / f"proba_{arm}_{ctag(c)}.npy"
    return np.load(p) if p.exists() else None


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


def main():
    avail = discover_Cs_per_arm()
    if not avail:
        print("No results/proba_*_C*.npy files found; run probe.py --C <value> at least once first.",
              file=sys.stderr)
        sys.exit(1)
    for arm in ARMS:
        print(f"  {arm}: C in {avail.get(arm, [])}", file=sys.stderr)

    Y, labels, label_idx = load_Y()
    n = Y.shape[0]
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    Y_shuf = Y[perm]

    def y_for(arm):
        return Y_shuf if arm == "shuffled" else Y

    # ---- "same C for all" view: recomputed arm x metric grid, at every C an arm has
    grid = {}
    for arm in ARMS:
        for c in avail.get(arm, []):
            P = load_proba(arm, c)
            if P is None:
                continue
            grid.setdefault(ctag(c), {})[arm] = {
                "micro_ap": micro_ap(y_for(arm), P),
                "macro_ap": macro_ap(y_for(arm), P),
                "micro_f1": micro_f1(y_for(arm), P),
                "p_at_5": p_at_5(y_for(arm), P),
            }

    # ---- per-arm oracle: primary = highest micro-AP; secondary = highest micro-F1@0.5,
    # kept alongside for comparison, over the SAME per-arm grid.
    best_c_ap, best_c_f1 = {}, {}
    for arm in ARMS:
        cands = []
        for c in avail.get(arm, []):
            g = grid.get(ctag(c), {}).get(arm)
            if g is not None:
                cands.append((c, g["micro_ap"], g["micro_f1"]))
        if cands:
            best_c_ap[arm] = max(cands, key=lambda t: t[1])[0]
            best_c_f1[arm] = max(cands, key=lambda t: t[2])[0]

    # ---- full recompute at each arm's AP-selected C
    selected_proba = {arm: load_proba(arm, c) for arm, c in best_c_ap.items()}
    selected_metrics = {}
    for arm, P in selected_proba.items():
        y_use = y_for(arm)
        t_star, f1_at_t, macro_f1_at_t = best_threshold_micro_f1(y_use, P)
        selected_metrics[arm] = {
            "C": best_c_ap[arm],
            "C_by_micro_f1_secondary": best_c_f1.get(arm),
            "micro_ap": micro_ap(y_use, P),
            "macro_ap": macro_ap(y_use, P),
            "micro_f1_at_0.5": micro_f1(y_use, P),
            "macro_f1_at_0.5": macro_f1(y_use, P),
            "best_threshold": t_star,
            "micro_f1_at_best_threshold": f1_at_t,
            "macro_f1_at_best_threshold": macro_f1_at_t,
            "p_at_5": p_at_5(y_use, P),
            "per_label_ap": per_label_ap(y_use, P),
        }

    # ---- binarization gap: sparseup at its best C minus sparseup_binary at its best C
    gap = None
    if {"sparseup", "sparseup_binary"} <= set(selected_metrics):
        a, b = selected_metrics["sparseup"], selected_metrics["sparseup_binary"]
        gap = {
            "micro_ap": a["micro_ap"] - b["micro_ap"],
            "macro_ap": a["macro_ap"] - b["macro_ap"],
            "best_threshold_micro_f1": a["micro_f1_at_best_threshold"] - b["micro_f1_at_best_threshold"],
            "p_at_5": a["p_at_5"] - b["p_at_5"],
        }

    # ---- literal-mention split on per-label AP: sparseup vs tfidf_word+char vs
    # tfidf_word+char_trunc512 (when present), from results/arm1.json
    literal_split = None
    arm1_path = RESULTS / "arm1.json"
    lit_arms = [a for a in ("sparseup", "tfidf_word+char", "tfidf_word+char_trunc512")
                if a in selected_metrics]
    if arm1_path.exists() and lit_arms:
        arm1 = json.loads(arm1_path.read_text())
        lit_rates = {lab: e.get("literal_mention_rate") for lab, e in arm1.get("per_label", {}).items()
                     if e.get("literal_mention_rate") is not None}
        group_hi = [label_idx[l] for l in labels if lit_rates.get(l, 0.0) >= 0.5]
        group_lo = [label_idx[l] for l in labels if lit_rates.get(l, 0.0) < 0.5]

        def mean_ap(arm, idxs):
            aps = np.array(selected_metrics[arm]["per_label_ap"])
            return float(np.nanmean(aps[idxs])) if idxs else None

        literal_split = {
            "n_labels_high_literal_rate_ge_0.5": len(group_hi),
            "n_labels_low_literal_rate_lt_0.5": len(group_lo),
            "mean_ap": {arm: {"high_literal": mean_ap(arm, group_hi), "low_literal": mean_ap(arm, group_lo)}
                        for arm in lit_arms},
        }

    # ---- paired bootstrap CIs on micro-AP and P@5 (each arm at its own selected C)
    boot = {}
    for a, b in BOOT_COMPARISONS:
        if a in selected_proba and b in selected_proba:
            name = f"{a}_minus_{b}"
            boot[name] = {
                "micro_ap": bootstrap_ci_diff(Y, selected_proba[a], selected_proba[b], micro_ap),
                "p_at_5": bootstrap_ci_diff(Y, selected_proba[a], selected_proba[b], p_at_5),
            }
        else:
            print(f"Skipping bootstrap for sparseup - {b} -- missing selected proba", file=sys.stderr)

    out = {
        "n": n,
        "n_labels": len(labels),
        "available_C_per_arm": avail,
        "selection_method": ("primary: for each arm independently, the C in ITS OWN available grid "
                              "with the highest pooled micro-AP (threshold-free). secondary: the C that "
                              "micro-F1@0.5 would have picked instead, kept for comparison "
                              "(C_by_micro_f1_secondary) -- not used for any of the metrics below, which "
                              "are all computed at the AP-selected C."),
        "selected_C_by_micro_ap": best_c_ap,
        "selected_C_by_micro_f1_secondary": best_c_f1,
        "selected_metrics": selected_metrics,
        "same_C_grid": grid,
        "binarization_gap_sparseup_minus_binary": gap,
        "literal_mention_split": literal_split,
        "bootstrap_ci_diff": boot,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selected.json").write_text(json.dumps(out, indent=1))

    # ---- headline.json: exactly the numbers this round's write-up is expected to quote
    headline = {}
    for arm, m in selected_metrics.items():
        headline[f"{arm}_micro_ap"] = round(m["micro_ap"], 3)
        headline[f"{arm}_macro_ap"] = round(m["macro_ap"], 3)
        headline[f"{arm}_best_thr_micro_f1"] = round(m["micro_f1_at_best_threshold"], 3)
        headline[f"{arm}_p_at_5"] = round(m["p_at_5"], 3)
    for a, b in BOOT_COMPARISONS:
        name = f"{a}_minus_{b}"
        if name in boot:
            headline[f"boot_{name}_micro_ap_diff"] = round(boot[name]["micro_ap"]["mean_diff"], 3)
    arm1_path = RESULTS / "arm1.json"
    if arm1_path.exists():
        arm1 = json.loads(arm1_path.read_text())
        for hd, vals in arm1["micro_recall"].items():
            for k, v in vals.items():
                if v is not None:
                    headline[f"arm1_{hd}_recall_k{k}"] = round(v, 3)
    (RESULTS / "headline.json").write_text(json.dumps(headline, indent=1))

    print("\nper-arm oracle over each arm's own available C grid, selected on micro-AP "
          "(secondary column: what micro-F1@0.5 alone would have picked)\n")
    print(f"{'arm':<28}{'C avail':<28}{'AP-C':>7}{'F1-C':>7}{'microAP':>9}{'macroAP':>9}"
          f"{'F1@.5':>8}{'best-t':>8}{'F1@t':>8}{'P@5':>8}")
    for arm in ARMS:
        cs = avail.get(arm, [])
        cs_str = ",".join(f"{c:g}" for c in cs) if cs else "--"
        if arm in selected_metrics:
            m = selected_metrics[arm]
            print(f"{arm:<28}{cs_str:<28}{m['C']:>7g}{(m['C_by_micro_f1_secondary'] or 0):>7g}"
                  f"{m['micro_ap']:>9.4f}{m['macro_ap']:>9.4f}{m['micro_f1_at_0.5']:>8.4f}"
                  f"{m['best_threshold']:>8.3f}{m['micro_f1_at_best_threshold']:>8.4f}{m['p_at_5']:>8.4f}")
        else:
            print(f"{arm:<28}{cs_str:<28}{'--':>7}{'--':>7}{'n/a':>9}")
    if gap:
        print(f"\nbinarization gap (sparseup@C{best_c_ap.get('sparseup','?'):g} - "
              f"sparseup_binary@C{best_c_ap.get('sparseup_binary','?'):g}): "
              f"micro-AP {gap['micro_ap']:+.4f}, macro-AP {gap['macro_ap']:+.4f}, "
              f"best-thr micro-F1 {gap['best_threshold_micro_f1']:+.4f}, P@5 {gap['p_at_5']:+.4f}")
    for name, d in boot.items():
        b = d["micro_ap"]
        print(f"{name} micro-AP: {b['mean_diff']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")
    print(f"\nWrote {RESULTS / 'selected.json'} and {RESULTS / 'headline.json'}")


if __name__ == "__main__":
    main()
