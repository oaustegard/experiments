"""Reads results/*.json (and data/encode_meta.json) and prints the markdown
tables RESULTS.md will paste, each measured number next to the PLAN.md
prediction it corresponds to, where PLAN.md made one. Predictions below are
copied by hand from PLAN.md's Predictions table (pre-registered 2026-09-21) —
edit them there only if PLAN.md's own table changes.

Order: selected-by-micro-AP table, the C grids (micro-AP, macro-AP, micro-F1@0.5,
P@5), binarization gap, bootstrap CIs, literal-mention split, then arm 1.

Reads results/arm1.json, results/selected.json and results/probe_C*.json (for
run-metadata only — the metrics themselves come from selected.json, which
recomputes from cached proba and so stays consistent with recheck.py). Does
NOT read results/probe.json — it no longer exists.

Usage: python3 make_tables.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = HERE / "data"

# --- predictions, copied by hand from PLAN.md's "Predictions" table ---
PRED_TRUNCATED_PCT = 25.0
PRED_ARM1_EXACT = {16: 0.15, 32: 0.20, 64: 0.30}
PRED_ARM1_ALL_PARTS = {16: 0.22, 32: 0.30, 64: 0.42}
PRED_ARM2_SPARSEUP = {"micro_f1": 0.55, "macro_f1": 0.35, "p_at_5": 0.45}  # F1@0.5, PLAN's own metric
PRED_TFIDF_WORDCHAR_MICRO_F1 = 0.57
PRED_GTE_SMALL_MICRO_F1 = 0.50
PRED_BINARIZATION_GAP_MAX = 0.02  # arm2 - arm3, micro-F1, predicted <= this
PRED_LABELS_WHERE_SPARSEUP_BEATS_TFIDF = ("lexical labels (tag word literal in text), not process "
                                           "tags like correction/session-log/preference")

ARMS_ORDER = ("sparseup", "sparseup_binary", "tfidf_word", "tfidf_char",
              "tfidf_word+char", "tfidf_word+char_trunc512", "gte_small", "prior", "shuffled")


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, spec=".4f"):
    if v is None:
        return "n/a"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def discover_probe_C_files():
    """{C (float): parsed probe_C{c}.json}, for the run-metadata note only —
    the actual metrics tables come from results/selected.json (recomputed
    from cached proba, so consistent with recheck.py's checks)."""
    out = {}
    for p in sorted(RESULTS.glob("probe_C*.json")):
        m = re.match(r"probe_C([0-9.]+)\.json$", p.name)
        if m:
            out[float(m.group(1))] = json.loads(p.read_text())
    return out


def print_selected(selected):
    print("## Selected by micro-AP (per-arm oracle, threshold-free) — pooled out-of-fold\n")
    if not selected or not selected.get("selected_metrics"):
        print("| — | results/selected.json not found — run select.py first | | | | | | | |")
        print()
        return
    print(f"_{selected.get('selection_method', '')}_\n")
    avail = selected.get("available_C_per_arm", {})
    sm = selected["selected_metrics"]
    print("| arm | C avail | C(AP) | C(F1@.5) | micro-AP | macro-AP | F1@.5 (micro/macro) "
          "| best-thr | F1@thr (micro/macro) | P@5 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS_ORDER:
        cs = avail.get(arm, [])
        cs_str = ",".join(f"{c:g}" for c in cs) if cs else "—"
        if arm in sm:
            m = sm[arm]
            f1c = m.get("C_by_micro_f1_secondary")
            f1c_str = f"{f1c:g}" if f1c is not None else "—"
            print(f"| {arm} | {cs_str} | {m['C']:g} | {f1c_str} | "
                  f"{fmt(m['micro_ap'])} | {fmt(m['macro_ap'])} | "
                  f"{fmt(m['micro_f1_at_0.5'])} / {fmt(m['macro_f1_at_0.5'])} | "
                  f"{fmt(m['best_threshold'], '.3f')} | "
                  f"{fmt(m['micro_f1_at_best_threshold'])} / {fmt(m['macro_f1_at_best_threshold'])} | "
                  f"{fmt(m['p_at_5'])} |")
        else:
            print(f"| {arm} | {cs_str} | — | — | n/a | n/a | n/a | — | n/a | n/a |")
    if "sparseup" in sm:
        p = PRED_ARM2_SPARSEUP
        print(f"| _predicted_ sparseup (F1@0.5/P@5 only) | — | — | — | — | — | {p['micro_f1']} / {p['macro_f1']} "
              f"| — | — | {p['p_at_5']} |")
    print(f"| _predicted_ tfidf_word+char (micro-F1@0.5 only) | — | — | — | — | — | {PRED_TFIDF_WORDCHAR_MICRO_F1} / — | — | — | — |")
    print(f"| _predicted_ gte_small (micro-F1@0.5 only) | — | — | — | — | — | {PRED_GTE_SMALL_MICRO_F1} / — | — | — | — |")
    print()


def print_c_grid(selected, probe_c_files):
    if probe_c_files:
        c_list = sorted(probe_c_files)
        meta_bits = [f"C={c:g} ({probe_c_files[c].get('wall_seconds', '?')}s)" for c in c_list]
        print(f"_probe_C*.json sweeps on record: {', '.join(meta_bits)}. An arm's actual "
              f"available C values (used for selection) are in the 'C avail' column above and "
              f"can be a strict subset or superset of this list — a new arm or a new C can lag "
              f"the others. TF-IDF and gte-small rows are L2-normalised (row norm 1.0); SPARSEUP "
              f"rows are not (mean row L2 norm ~32) — one C is a different effective prior per "
              f"arm, hence sweeping and selecting per arm._\n")
    if not selected or not selected.get("same_C_grid"):
        print("## C grid\n")
        print("| — | results/selected.json not found or has no C grid — run select.py first | |")
        print()
        return
    grid = selected["same_C_grid"]
    ctags = sorted(grid.keys(), key=lambda t: float(t[1:]))
    Cs = [float(t[1:]) for t in ctags]
    header = "| arm | " + " | ".join(f"C={c:g}" for c in Cs) + " |"
    sep = "|---|" + "---|" * len(Cs)

    for title, key in (("micro-AP", "micro_ap"), ("macro-AP", "macro_ap"),
                        ("micro-F1@0.5", "micro_f1"), ("P@5", "p_at_5")):
        print(f"## C grid — {title} (arm x C)\n")
        print(header)
        print(sep)
        for arm in ARMS_ORDER:
            row = [fmt(grid[t].get(arm, {}).get(key)) for t in ctags]
            print(f"| {arm} | " + " | ".join(row) + " |")
        print()


def print_binarization(selected):
    print("## Binarization gap (sparseup best-C-by-AP - sparseup_binary best-C-by-AP)\n")
    print("| metric | measured | predicted |")
    print("|---|---|---|")
    gap = selected.get("binarization_gap_sparseup_minus_binary") if selected else None
    if gap:
        print(f"| micro-AP | {fmt(gap['micro_ap'], '+.4f')} | — |")
        print(f"| macro-AP | {fmt(gap['macro_ap'], '+.4f')} | — |")
        print(f"| best-threshold micro-F1 | {fmt(gap['best_threshold_micro_f1'], '+.4f')} | <= {PRED_BINARIZATION_GAP_MAX} (PLAN predicted this on F1@0.5) |")
        print(f"| P@5 | {fmt(gap['p_at_5'], '+.4f')} | — |")
    else:
        print("| — | not available yet (needs sparseup and sparseup_binary selected) | |")
    print()


def print_bootstrap(selected):
    print("## Bootstrap 95% CI, paired differences over documents (each arm at its own AP-selected C)\n")
    print("| comparison | metric | mean diff | 95% CI |")
    print("|---|---|---|---|")
    boot = selected.get("bootstrap_ci_diff") if selected else None
    if boot:
        for cmp_name, metrics in boot.items():
            for metric_name, v in metrics.items():
                print(f"| {cmp_name} | {metric_name} | {fmt(v['mean_diff'], '+.4f')} | "
                      f"[{fmt(v['ci_lo'], '+.4f')}, {fmt(v['ci_hi'], '+.4f')}] |")
    else:
        print("| — | not available yet | | |")
    print()


def print_literal_split(selected):
    print("## Per-label split by literal-mention rate (arm1), mean per-label AP\n")
    ls = selected.get("literal_mention_split") if selected else None
    if not ls:
        print("| — | not available yet | |")
        print()
        return
    arms = list(ls["mean_ap"].keys())
    print("| group | " + " | ".join(arms) + " |")
    print("|---|" + "---|" * len(arms))
    row_hi = [fmt(ls["mean_ap"][a]["high_literal"]) for a in arms]
    row_lo = [fmt(ls["mean_ap"][a]["low_literal"]) for a in arms]
    print(f"| high literal-mention (>= 0.5), n={ls['n_labels_high_literal_rate_ge_0.5']} | " + " | ".join(row_hi) + " |")
    print(f"| low literal-mention (< 0.5), n={ls['n_labels_low_literal_rate_lt_0.5']} | " + " | ".join(row_lo) + " |")
    print(f"\n_predicted_: SPARSEUP beats TF-IDF on {PRED_LABELS_WHERE_SPARSEUP_BEATS_TFIDF}")
    print()


def print_arm1(arm1):
    print("## Arm 1 — direct tag recall (micro, no training)\n")
    if not arm1:
        print("| — | results/arm1.json not found — run arm1_recall.py first | | | | | |")
        print()
        return
    note = " _(smoke test — not the pre-registered run)_" if arm1.get("smoke_test") else ""
    print(f"| k | exact-token | (pred) | all-parts | (pred) | subword |{note}")
    print("|---|---|---|---|---|---|")
    mr = arm1["micro_recall"]
    for k in arm1["ks"]:
        e = mr["exact"].get(str(k))
        a = mr["all_parts"].get(str(k))
        s = mr.get("subword", {}).get(str(k))
        print(f"| {k} | {fmt(e)} | {PRED_ARM1_EXACT.get(k, 'n/a')} | {fmt(a)} | "
              f"{PRED_ARM1_ALL_PARTS.get(k, 'n/a')} | {fmt(s)} |")
    stb = arm1["single_token_bound"]
    print(f"\nSingle-token labels (bound on exact-token recall): "
          f"{stb['n_single_token_either']}/{stb['n_labels']}\n")
    pb = arm1.get("subword_pieces_distribution")
    if pb:
        print(f"Subword pieces per label: 1={pb['1']} 2={pb['2']} 3={pb['3']} 4+={pb['4+']} "
              f"(of {stb['n_labels']})\n")
    sw = arm1.get("subword_recall_k32_spotlight")
    if sw:
        print(f"Highest subword recall@{sw['k']} (n>={sw['n_min']}): " +
              ", ".join(f"{lab} ({r:.2f})" for lab, r in sw["highest"]))
        print(f"\nLowest subword recall@{sw['k']} (n>={sw['n_min']}): " +
              ", ".join(f"{lab} ({r:.2f})" for lab, r in sw["lowest"]))
    print()


def main():
    arm1 = load("arm1.json")
    selected = load("selected.json")
    probe_c_files = discover_probe_C_files()

    print_selected(selected)
    print_c_grid(selected, probe_c_files)
    print_binarization(selected)
    print_bootstrap(selected)
    print_literal_split(selected)
    print_arm1(arm1)


if __name__ == "__main__":
    main()
