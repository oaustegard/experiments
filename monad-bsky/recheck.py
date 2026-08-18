#!/usr/bin/env python3
"""Check RESULTS.md against the artifacts. No model, no network, under a second.

Recomputes every headline number from `results_*.json` and the sibling
experiment's results, then matches it against the literal text of the writeup,
so the prose and the data cannot drift apart between rebuilds.

    python3 recheck.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment

NEEDLE = experiment("needle-bsky")

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(f"{label}: {detail}")
    print(f"  {'ok ' if ok else 'FAIL'} {label}{'  ' + detail if detail and not ok else ''}")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    md = (HERE / "RESULTS.md").read_text()

    print("monad arms")
    arms = {}
    for label in ("base", "tuned-e1", "tuned-e2", "tuned-e3"):
        p = HERE / f"results_{label}.json"
        check(f"{label} results present", p.exists())
        if p.exists():
            arms[label] = load(p)

    for label, d in arms.items():
        s = d["summary"]
        check(f"{label} rows = 62", len(d["rows"]) == 62, str(len(d["rows"])))
        check(f"{label} routable {s['tool_acc_routable']:.3f} quoted", f"{s['tool_acc_routable']:.3f}" in md)
        check(f"{label} args {s['args_acc_routable']:.3f} quoted", f"{s['args_acc_routable']:.3f}" in md)
        check(f"{label} parse-ok {s['parse_ok_rate']:.3f} quoted", f"{s['parse_ok_rate']:.3f}" in md)

    print("\nclaims about the base model")
    b = arms.get("base", {}).get("summary", {})
    check("zero-shot routable is exactly 0", b.get("tool_acc_routable") == 0.0, str(b.get("tool_acc_routable")))
    check("zero-shot parsed nothing", b.get("parse_ok_rate") == 0.0, str(b.get("parse_ok_rate")))
    check("the vacuous-refusal caveat is stated", "vacuous" in md)

    print("\nepoch curve")
    e = {k: arms[k]["summary"]["tool_acc_routable"] for k in ("tuned-e1", "tuned-e2", "tuned-e3") if k in arms}
    if len(e) == 3:
        check("epoch 2 is the best Monad arm", e["tuned-e2"] == max(e.values()), str(e))
        check("epoch 3 does not beat epoch 2", e["tuned-e3"] <= e["tuned-e2"], str(e))
    check("val losses quoted", all(x in md for x in ("0.0128", "0.0040", "0.0017")))
    check("val-loss-is-not-accuracy caveat present", "not a held-out task" in md)

    print("\ncopy probe")
    cp_path = HERE / "results_copy_probe.json"
    check("copy probe present", cp_path.exists())
    if cp_path.exists():
        cp = load(cp_path)
        for label in ("needle-base", "needle-lora", "monad-e1", "monad-e3"):
            if label in cp:
                v = cp[label]["copy_accuracy"]
                check(f"{label} copy {v:.3f} quoted", f"{v:.3f}" in md)
        m3 = cp.get("monad-e3", {}).get("copy_accuracy")
        m1 = cp.get("monad-e1", {}).get("copy_accuracy")
        nb = cp.get("needle-base", {}).get("copy_accuracy")
        check("more training copies worse", m3 is not None and m1 is not None and m3 < m1, f"{m1} -> {m3}")
        check("Needle base out-copies tuned Monad", nb is not None and m3 is not None and nb > m3)
        check("all arms scored on the same 41 literal arguments",
              len({v["n_literal_args"] for v in cp.values()}) == 1)

    print("\ntokenizer claim")
    check("the tokenizer explanation is retracted, not stated",
          "is wrong" in md and "8,192-piece vocabularies" in md)
    check("identical segmentation quoted", "['a','ust','eg','ard','.','com']" in md)
    check("111 vs 109 quoted", "111" in md and "109" in md)

    print("\nrepair arm")
    rp = HERE / "results_tuned-e3-repaired.json"
    check("repaired results present", rp.exists())
    if rp.exists():
        r = load(rp)
        base = arms["tuned-e3"]["summary"]
        check("repair improves args", r["summary"]["args_acc_routable"] > base["args_acc_routable"])
        check("repair leaves tool accuracy alone",
              r["summary"]["tool_acc_routable"] == base["tool_acc_routable"])
        check(f"repaired args {r['summary']['args_acc_routable']:.3f} quoted",
              f"{r['summary']['args_acc_routable']:.3f}" in md)
        check("the fitted-extractor caveat is stated", "fitted to this" in md)

    print("\ncross-experiment")
    for label, path in (
        ("needle-base", NEEDLE / "results_tuned-min.json"),
        ("needle-lora", NEEDLE / "results_finetuned.json"),
        ("needle-2stage", NEEDLE / "results_two_stage_heuristic.json"),
    ):
        d = load(path)
        v = d["summary"]["tool_acc_routable"]
        check(f"{label} routable {v:.3f} quoted", f"{v:.3f}" in md)
    check("evalset is the sibling's", (NEEDLE / "evalset.jsonl").exists())
    check("62 queries", sum(1 for _ in (NEEDLE / "evalset.jsonl").read_text().splitlines() if _.strip()) == 62)

    print("\nhygiene")
    check("no absolute container paths", "/workspace/" not in md and "/home/user/" not in md)
    check("caveats section present", "## Caveats" in md)
    check("asymmetries disclosed", "Not size-matched" in md and "Not method-matched" in md)
    check("ERRORS.md present", (HERE / "ERRORS.md").exists())
    check("params.json present", (HERE / "params.json").exists())

    root = HERE.parent
    check("README index row", "monad-bsky/" in (root / "README.md").read_text())
    check("METHODS entry", "monad-bsky" in (root / "METHODS.md").read_text())

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    for f in FAILURES:
        print(f"  FAIL {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
