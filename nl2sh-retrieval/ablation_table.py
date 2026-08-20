#!/usr/bin/env python3
"""RAG vs non-RAG base, fine-tuned identically — does the source-quoting matter?

The gate model is Pleias-RAG-350M fine-tuned on 600 NL->command rows. It works
(0.923 non-`find`), but its verbatim rate is 0.000 before and after, so the
literal-source-quoting property it was chosen for is not what carries the
result. This ablation fine-tunes the RAG-less sibling — `Pleias-350m-Preview`,
same lab, same architecture (llama L26 h1024), same base data, same tokenizer
vocabulary — on the byte-identical training rows, and scores it on the same gate.

**One confound, stated up front and not correctable here:** the preview
tokenizer lacks the RAG special tokens, so a `<|source_start|>...<|source_end|>`
block costs it 18 tokens instead of 5. The ablation therefore varies two things
— RAG mid-training AND native source delimiters — and its 45.8 min training time
vs the RAG model's 25.5 is that longer encoding. A clean single-variable run
would need a plain-text source format for both; this is the two-variable version.

    python3 ablation_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str) -> list[dict]:
    p = HERE / name
    return json.loads(p.read_text())["rows"] if p.is_file() else []


def main() -> int:
    contaminated = set(json.loads((HERE / "data" / "gate_contaminated_rows.json").read_text())) \
        if (HERE / "data" / "gate_contaminated_rows.json").is_file() else set()

    arms = {"base (RAG, zero-shot)": "results_gate.json",
            "fine-tuned (RAG)": "results_gate_ft.json",
            "fine-tuned (non-RAG)": "results_gate_nonrag.json"}
    n_find = None
    out = {}
    for label, fn in arms.items():
        rows = load(fn)
        if not rows:
            out[label] = None
            continue
        nonfind = [r for r in rows if r["gold_utility"] != "find"]
        clean = [r for i, r in enumerate(rows, 1) if i not in contaminated]
        n_find = sum(r["gold_utility"] == "find" for r in rows)
        out[label] = {
            "n": len(rows),
            "utility_all": round(sum(r["utility_ok"] for r in rows) / len(rows), 3),
            "utility_nonfind": round(sum(r["utility_ok"] for r in nonfind) / len(nonfind), 3) if nonfind else None,
            "utility_uncontaminated": round(sum(r["utility_ok"] for r in clean) / len(clean), 3) if clean else None,
            "command_rate": round(sum(bool(r["command"]) for r in rows) / len(rows), 3),
            "verbatim": round(sum(r["verbatim"] for r in rows) / len(rows), 3),
        }
    result = {"always_find_prior": round(n_find / 40, 3) if n_find else None, "arms": out}
    (HERE / "results_ablation.json").write_text(json.dumps(result, indent=1) + "\n")

    print(f"{'arm':<26}{'util all':>10}{'nonfind':>10}{'cmd rate':>10}{'verbatim':>10}")
    print("-" * 66)
    for label, s in out.items():
        if s is None:
            print(f"{label:<26}{'(pending)':>10}")
            continue
        nf = "  -  " if s["utility_nonfind"] is None else f"{s['utility_nonfind']:.3f}"
        print(f"{label:<26}{s['utility_all']:>10.3f}{nf:>10}"
              f"{s['command_rate']:>10.3f}{s['verbatim']:>10.3f}")
    print(f"\nalways-`find` constant prior: {result['always_find_prior']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
