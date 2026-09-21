"""Reads results/*.json (and data/encode_meta.json) and prints the markdown
tables RESULTS.md will paste, each measured number next to the PLAN.md
prediction it corresponds to. Predictions below are copied by hand from
PLAN.md's Predictions table (pre-registered 2026-09-21) — edit them there
only if PLAN.md's own table changes.

Usage: python3 make_tables.py
"""
import json
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
PRED_LABELS_WHERE_SPARSEUP_BEATS_TFIDF = "lexical labels (tag word literal in text), not process tags like correction/session-log/preference"


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(v, spec=".4f"):
    if v is None:
        return "n/a"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def main():
    meta = json.loads((DATA / "encode_meta.json").read_text()) if (DATA / "encode_meta.json").exists() else None
    arm1 = load("arm1.json")
    probe = load("probe.json")

    print("## Truncation\n")
    print("| quantity | measured | predicted |")
    print("|---|---|---|")
    if meta and meta.get("n"):
        pct = 100.0 * meta.get("n_truncated", 0) / meta["n"]
        print(f"| memories truncated at 512 tokens | {pct:.1f}% ({meta.get('n_truncated')}/{meta['n']}) | ~{PRED_TRUNCATED_PCT:.0f}% |")
    else:
        print(f"| memories truncated at 512 tokens | n/a (encode not done) | ~{PRED_TRUNCATED_PCT:.0f}% |")
    print()

    print("## Arm 1 — direct tag recall (micro, no training)\n")
    if arm1 and not arm1.get("smoke_test"):
        note = ""
    elif arm1:
        note = " _(smoke test — not the pre-registered run)_"
    else:
        note = ""
    print(f"| k | exact-token measured | exact-token predicted | all-parts measured | all-parts predicted |{note}")
    print("|---|---|---|---|---|")
    if arm1:
        mr = arm1["micro_recall"]
        for k in arm1["ks"]:
            e = mr["exact"].get(str(k))
            a = mr["all_parts"].get(str(k))
            print(f"| {k} | {fmt(e)} | {PRED_ARM1_EXACT.get(k, 'n/a')} | {fmt(a)} | {PRED_ARM1_ALL_PARTS.get(k, 'n/a')} |")
        stb = arm1["single_token_bound"]
        print(f"\nSingle-token labels (bound on exact-token recall): "
              f"{stb['n_single_token_either']}/{stb['n_labels']}\n")
    else:
        print("| — | results/arm1.json not found — run arm1_recall.py first | | | |")
    print()

    print("## Arm 2/3/4 — trained linear map (pooled out-of-fold)\n")
    print("| arm | micro-F1 | macro-F1 | P@5 |")
    print("|---|---|---|---|")
    if probe:
        for arm in ("sparseup", "sparseup_binary", "tfidf_word", "tfidf_char",
                    "tfidf_word+char", "gte_small", "prior", "shuffled"):
            m = probe["arms"][arm]
            print(f"| {arm} | {fmt(m['micro_f1'])} | {fmt(m['macro_f1'])} | {fmt(m['p_at_5'])} |")
        print(f"| _predicted_ sparseup | {PRED_ARM2_SPARSEUP['micro_f1']} | {PRED_ARM2_SPARSEUP['macro_f1']} | {PRED_ARM2_SPARSEUP['p_at_5']} |")
        print(f"| _predicted_ tfidf_word+char (micro-F1 only) | {PRED_TFIDF_WORDCHAR_MICRO_F1} | — | — |")
        print(f"| _predicted_ gte_small (micro-F1 only) | {PRED_GTE_SMALL_MICRO_F1} | — | — |")
        if probe.get("smoke_test"):
            print("\n_(smoke test — not the pre-registered run)_")
    else:
        print("| — | results/probe.json not found — run probe.py first | | |")
    print()

    print("## Binarization gap (sparseup - sparseup_binary)\n")
    print("| metric | measured | predicted |")
    print("|---|---|---|")
    if probe:
        gap = probe["binarization_gap_sparseup_minus_binary"]
        print(f"| micro-F1 | {fmt(gap['micro_f1'], '+.4f')} | <= {PRED_BINARIZATION_GAP_MAX} |")
        print(f"| macro-F1 | {fmt(gap['macro_f1'], '+.4f')} | — |")
        print(f"| P@5 | {fmt(gap['p_at_5'], '+.4f')} | — |")
    else:
        print("| — | results/probe.json not found | |")
    print()

    print("## Bootstrap 95% CI, paired differences over documents\n")
    print("| comparison | metric | mean diff | 95% CI |")
    print("|---|---|---|---|")
    if probe:
        boot = probe["bootstrap_ci_diff"]
        for cmp_name, metrics in boot.items():
            for metric_name, v in metrics.items():
                print(f"| {cmp_name} | {metric_name} | {fmt(v['mean_diff'], '+.4f')} | "
                      f"[{fmt(v['ci_lo'], '+.4f')}, {fmt(v['ci_hi'], '+.4f')}] |")
    else:
        print("| — | results/probe.json not found | | |")
    print()

    print("## Per-label split by literal-mention rate (arm1)\n")
    print("| group | sparseup mean F1 | tfidf_word+char mean F1 |")
    print("|---|---|---|")
    if probe:
        ls = probe["literal_mention_split"]
        hi = ls["mean_f1"]
        print(f"| high literal-mention (>= 0.5), n={ls['n_labels_high_literal_rate_ge_0.5']} | "
              f"{fmt(hi['sparseup']['high_literal'])} | {fmt(hi['tfidf_word+char']['high_literal'])} |")
        print(f"| low literal-mention (< 0.5), n={ls['n_labels_low_literal_rate_lt_0.5']} | "
              f"{fmt(hi['sparseup']['low_literal'])} | {fmt(hi['tfidf_word+char']['low_literal'])} |")
        print(f"\n_predicted_: SPARSEUP beats TF-IDF on {PRED_LABELS_WHERE_SPARSEUP_BEATS_TFIDF}")
    else:
        print("| — | results/probe.json not found | |")


if __name__ == "__main__":
    main()
