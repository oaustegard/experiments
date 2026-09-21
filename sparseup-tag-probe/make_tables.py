"""Reads results/*.json (and data/encode_meta.json) and prints the markdown
tables RESULTS.md will paste, each measured number next to the PLAN.md
prediction it corresponds to. Predictions below are copied by hand from
PLAN.md's Predictions table (pre-registered 2026-09-21) — edit them there
only if PLAN.md's own table changes.

Reads results/arm1.json, results/selected.json and results/probe_C*.json.
Does NOT read results/probe.json — that file no longer exists; probe.py now
writes one results/probe_C{c}.json per swept C, and select.py's oracle pick
over that grid (results/selected.json) is the authoritative per-arm number.

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
PRED_ARM2_SPARSEUP = {"micro_f1": 0.55, "macro_f1": 0.35, "p_at_5": 0.45}
PRED_TFIDF_WORDCHAR_MICRO_F1 = 0.57
PRED_GTE_SMALL_MICRO_F1 = 0.50
PRED_BINARIZATION_GAP_MAX = 0.02  # arm2 - arm3, micro-F1, predicted <= this
PRED_LABELS_WHERE_SPARSEUP_BEATS_TFIDF = ("lexical labels (tag word literal in text), not process "
                                           "tags like correction/session-log/preference")

ARMS_ORDER = ("sparseup", "sparseup_binary", "tfidf_word", "tfidf_char",
              "tfidf_word+char", "gte_small", "prior", "shuffled")


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


def print_truncation():
    meta_path = DATA / "encode_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    print("## Truncation\n")
    print("| quantity | measured | predicted |")
    print("|---|---|---|")
    if meta and meta.get("n"):
        pct = 100.0 * meta.get("n_truncated", 0) / meta["n"]
        print(f"| memories truncated at 512 tokens | {pct:.1f}% ({meta.get('n_truncated')}/{meta['n']}) "
              f"| ~{PRED_TRUNCATED_PCT:.0f}% |")
    else:
        print(f"| memories truncated at 512 tokens | n/a (encode not done) | ~{PRED_TRUNCATED_PCT:.0f}% |")
    print()


def print_arm1(arm1):
    print("## Arm 1 — direct tag recall (micro, no training)\n")
    if not arm1:
        print("| — | results/arm1.json not found — run arm1_recall.py first | | | |")
        print()
        return
    note = " _(smoke test — not the pre-registered run)_" if arm1.get("smoke_test") else ""
    print(f"| k | exact-token measured | exact-token predicted | all-parts measured | all-parts predicted |{note}")
    print("|---|---|---|---|---|")
    mr = arm1["micro_recall"]
    for k in arm1["ks"]:
        e = mr["exact"].get(str(k))
        a = mr["all_parts"].get(str(k))
        print(f"| {k} | {fmt(e)} | {PRED_ARM1_EXACT.get(k, 'n/a')} | {fmt(a)} | {PRED_ARM1_ALL_PARTS.get(k, 'n/a')} |")
    stb = arm1["single_token_bound"]
    print(f"\nSingle-token labels (bound on exact-token recall): "
          f"{stb['n_single_token_either']}/{stb['n_labels']}\n")


def print_c_grid(selected, probe_c_files):
    print("## C grid — micro-F1 (arm x C)\n")
    if probe_c_files:
        c_list = sorted(probe_c_files)
        meta_bits = [f"C={c:g} ({probe_c_files[c].get('wall_seconds', '?')}s)" for c in c_list]
        print(f"_Swept: {', '.join(meta_bits)}. TF-IDF and gte-small rows are L2-normalised "
              f"(row norm 1.0); SPARSEUP rows are not (mean row L2 norm ~32) — one C is a "
              f"different effective prior per arm, hence the sweep._\n")
    if not selected or not selected.get("same_C_grid"):
        print("| — | results/selected.json not found or has no C grid — run select.py first | |")
        print()
        return
    grid = selected["same_C_grid"]
    ctags = sorted(grid.keys(), key=lambda t: float(t[1:]))
    Cs = [float(t[1:]) for t in ctags]
    header = "| arm | " + " | ".join(f"C={c:g}" for c in Cs) + " |"
    sep = "|---|" + "---|" * len(Cs)

    print(header)
    print(sep)
    for arm in ARMS_ORDER:
        row = [fmt(grid[t].get(arm, {}).get("micro_f1")) for t in ctags]
        print(f"| {arm} | " + " | ".join(row) + " |")
    print()

    print("## C grid — P@5 (arm x C)\n")
    print(header)
    print(sep)
    for arm in ARMS_ORDER:
        row = [fmt(grid[t].get(arm, {}).get("p_at_5")) for t in ctags]
        print(f"| {arm} | " + " | ".join(row) + " |")
    print()


def print_selected(selected):
    print("## Selected C (per-arm oracle over the swept grid) — pooled out-of-fold\n")
    if not selected or not selected.get("selected_metrics"):
        print("| — | results/selected.json not found — run select.py first | | | |")
        print()
        return
    print(f"_{selected.get('selection_method', '')}_\n")
    sm = selected["selected_metrics"]
    print("| arm | best C | micro-F1 | macro-F1 | P@5 |")
    print("|---|---|---|---|---|")
    for arm in ARMS_ORDER:
        if arm in sm:
            m = sm[arm]
            print(f"| {arm} | {m['C']:g} | {fmt(m['micro_f1'])} | {fmt(m['macro_f1'])} | {fmt(m['p_at_5'])} |")
        else:
            print(f"| {arm} | — | n/a (no cached proba yet) | — | — |")
    print(f"| _predicted_ sparseup | — | {PRED_ARM2_SPARSEUP['micro_f1']} | "
          f"{PRED_ARM2_SPARSEUP['macro_f1']} | {PRED_ARM2_SPARSEUP['p_at_5']} |")
    print(f"| _predicted_ tfidf_word+char (micro-F1 only) | — | {PRED_TFIDF_WORDCHAR_MICRO_F1} | — | — |")
    print(f"| _predicted_ gte_small (micro-F1 only) | — | {PRED_GTE_SMALL_MICRO_F1} | — | — |")
    print()


def print_binarization(selected):
    print("## Binarization gap (sparseup best-C - sparseup_binary best-C)\n")
    print("| metric | measured | predicted |")
    print("|---|---|---|")
    gap = selected.get("binarization_gap_sparseup_minus_binary") if selected else None
    if gap:
        print(f"| micro-F1 | {fmt(gap['micro_f1'], '+.4f')} | <= {PRED_BINARIZATION_GAP_MAX} |")
        print(f"| macro-F1 | {fmt(gap['macro_f1'], '+.4f')} | — |")
        print(f"| P@5 | {fmt(gap['p_at_5'], '+.4f')} | — |")
    else:
        print("| — | not available yet (needs sparseup and sparseup_binary selected) | |")
    print()


def print_bootstrap(selected):
    print("## Bootstrap 95% CI, paired differences over documents (each arm at its own selected C)\n")
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
    print("## Per-label split by literal-mention rate (arm1)\n")
    print("| group | sparseup mean F1 | tfidf_word+char mean F1 |")
    print("|---|---|---|")
    ls = selected.get("literal_mention_split") if selected else None
    if ls:
        hi = ls["mean_f1"]
        print(f"| high literal-mention (>= 0.5), n={ls['n_labels_high_literal_rate_ge_0.5']} | "
              f"{fmt(hi['sparseup']['high_literal'])} | {fmt(hi['tfidf_word+char']['high_literal'])} |")
        print(f"| low literal-mention (< 0.5), n={ls['n_labels_low_literal_rate_lt_0.5']} | "
              f"{fmt(hi['sparseup']['low_literal'])} | {fmt(hi['tfidf_word+char']['low_literal'])} |")
        print(f"\n_predicted_: SPARSEUP beats TF-IDF on {PRED_LABELS_WHERE_SPARSEUP_BEATS_TFIDF}")
    else:
        print("| — | not available yet | |")


def main():
    arm1 = load("arm1.json")
    selected = load("selected.json")
    probe_c_files = discover_probe_C_files()

    print_truncation()
    print_arm1(arm1)
    print_c_grid(selected, probe_c_files)
    print_selected(selected)
    print_binarization(selected)
    print_bootstrap(selected)
    print_literal_split(selected)


if __name__ == "__main__":
    main()
