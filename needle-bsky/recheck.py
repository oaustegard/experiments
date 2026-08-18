#!/usr/bin/env python3
"""Check RESULTS.md against the artifacts. No network, no model, under a second.

Two directions, because either alone lets the writeup drift:

* every headline number in RESULTS.md is recomputed from `results_*.json` and
  `evalset.jsonl`, then matched against the literal text of the file;
* the artifacts are checked for internal consistency (row counts, arm coverage,
  the determinism flag).

    python3 recheck.py          # prints one line per check, non-zero exit on failure
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["auto", "auto-min", "tuned", "tuned-min"]
ORACLE = [f"oracle-{a}" for a in ARMS]

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(f"{label}: {detail}")
    print(f"  {'ok ' if ok else 'FAIL'} {label}{'  ' + detail if detail and not ok else ''}")


def load(arm: str) -> dict:
    return json.loads((HERE / f"results_{arm}.json").read_text())


def main() -> int:
    md = (HERE / "RESULTS.md").read_text()
    items = [json.loads(x) for x in (HERE / "evalset.jsonl").read_text().splitlines() if x.strip()]
    R = {a: load(a) for a in ARMS + ORACLE}

    print("eval set")
    routable = [i for i in items if i["tool"]]
    check("62 queries", len(items) == 62, f"got {len(items)}")
    check("54 routable / 8 off-topic", len(routable) == 54 and len(items) - len(routable) == 8)
    check("ids unique", len({i["id"] for i in items}) == len(items))
    declared = set()
    try:
        sys.path.insert(0, str(HERE))
        from needle_bsky.tools_tuned import SCHEMAS

        declared = {s["name"] for s in SCHEMAS}
    except ImportError as exc:
        FAILURES.append(f"tools_tuned import: {exc}")
    check("18 tools declared", len(declared) == 18, f"got {len(declared)}")
    unknown = {t for i in items for t in i["tool"]} - declared
    check("eval set references only declared tools", not unknown, f"unknown: {sorted(unknown)}")

    print("\nmain table matches results_*.json")
    for arm in ARMS:
        s = R[arm]["summary"]
        check(f"{arm} rows = 62", len(R[arm]["rows"]) == 62)
        for field, label in (
            ("tool_acc", "tool acc"),
            ("tool_acc_routable", "routable"),
            ("refusal_acc", "refusal"),
            ("args_acc_routable", "args"),
            ("invented_rate", "invented"),
        ):
            v = f"{s[field]:.3f}"
            check(f"{arm} {label} {v} in RESULTS.md", v in md)
        ms = f"{s['median_latency_ms']:.0f} ms"
        check(f"{arm} median {ms} in RESULTS.md", ms in md)

    print("\noracle table")
    for arm in ARMS:
        o = R[f"oracle-{arm}"]["summary"]["tool_acc_routable"]
        b = R[arm]["summary"]["tool_acc_routable"]
        check(f"oracle-{arm} routable {o:.3f} in RESULTS.md", f"{o:.3f}" in md)
        delta = o - b
        sign = "+" if delta >= 0 else "−"
        check(f"oracle-{arm} delta {sign}{abs(delta):.3f} in RESULTS.md", f"{sign}{abs(delta):.3f}" in md)

    print("\nclaims")
    best = max(R[f"oracle-{a}"]["summary"]["tool_acc_routable"] for a in ARMS)
    check("81.5% oracle ceiling is the max oracle arm", abs(best - 0.815) < 5e-4, f"max={best}")
    check("determinism reported for all four arms", all(R[a]["deterministic"] for a in ARMS))
    check("determinism claim in RESULTS.md", "byte-identical" in md)

    inv_auto = R["auto"]["summary"]["invented_rate"]
    inv_auto_min = R["auto-min"]["summary"]["invented_rate"]
    inv_tuned = R["tuned"]["summary"]["invented_rate"]
    inv_tuned_min = R["tuned-min"]["summary"]["invented_rate"]
    check("invented drops with arity, both wordings", inv_auto_min < inv_auto and inv_tuned_min < inv_tuned)
    check(
        "invented numbers quoted",
        all(f"{v:.3f}" in md for v in (inv_auto, inv_auto_min, inv_tuned, inv_tuned_min)),
    )

    mc = R["tuned"]["summary"]["mean_conf_correct"]
    mcm = R["tuned-min"]["summary"]["mean_conf_correct"]
    check("tuned mean confidence on correct calls quoted", f"{mc:.3f}" in md)
    check("tuned-min mean confidence on correct calls quoted", f"{mcm:.3f}" in md)
    check("gate: tuned-min beats tuned on mean confidence", mcm > mc, f"{mcm} vs {mc}")

    print("\ngate sweep rows quoted in RESULTS.md")
    for arm in ("tuned", "tuned-min"):
        by_t = {g["threshold"]: g for g in R[arm]["gate_sweep"]}
        for t in (0.0, 0.4, 0.6, 0.8, 0.9):
            g = by_t[t]
            cell = f"{g['coverage']:.2f} / {g['precision_tool']:.3f}"
            check(f"{arm} t={t} -> {cell}", cell in md)

    ap_path = HERE / "results_arity_probe.json"
    if ap_path.exists():
        print("\narity probe")
        ap = json.loads(ap_path.read_text())
        conds = ap["conditions"]
        check("30 probe queries", len(ap["queries"]) == 30, str(len(ap["queries"])))
        means = [conds[k]["mean_confidence"] for k in ("required", "+optional", "+noise")]
        check("confidence falls monotonically with arity", means[0] > means[1] > means[2], str(means))
        for k in ("required", "+optional", "+noise"):
            v = f"{conds[k]['mean_confidence']:.3f}"
            check(f"probe {k} mean {v} quoted", v in md)
        for key, label in (
            ("required_vs_+optional", "handle -> +limit"),
            ("+optional_vs_+noise", "+limit -> +transcribe"),
            ("required_vs_+noise", "handle -> both"),
        ):
            t = ap["sign_tests"][key]
            check(f"{label} sign test p={t['p']} quoted", str(t["p"]) in md or f"{t['p']:.5f}" in md)
            check(f"{label} counts {t['down']}/{t['up']} quoted", f"| {t['down']} | {t['up']} |" in md)
        check("all three probe steps significant", all(ap["sign_tests"][k]["p"] < 0.05 for k in ap["sign_tests"]))

    lat_path = HERE / "results_latency_vs_catalogue.json"
    if lat_path.exists():
        print("\nlatency vs catalogue size")
        lat = {int(k): v for k, v in json.loads(lat_path.read_text()).items()}
        check("five-tool median quoted", f"{lat[5]:.0f} ms" in md, f"{lat[5]}")
        check("six-tool median quoted", f"{lat[6]:.0f} ms" in md, f"{lat[6]}")
        ratio = lat[6] / lat[5]
        check("the sixth tool costs 3.6x", abs(ratio - 3.6) < 0.15, f"ratio={ratio:.2f}")
        check("ratio stated in RESULTS.md", "3.6x" in md)
        check("flat from 6 to 18", abs(lat[18] - lat[6]) / lat[6] < 0.15, f"{lat[6]} -> {lat[18]}")

    print("\nprose hygiene")
    check("no absolute container paths in RESULTS.md", "/workspace/" not in md and "/home/user/" not in md)
    check("caveats section present", "## Caveats" in md)
    check("n=62 caveat present", "n=62" in md)
    ft = re.search(r"## Fine-tune\n+(.+?)(\n## |\Z)", md, re.DOTALL)
    check("fine-tune section filled", bool(ft) and "<!-- FT -->" not in md)

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    for f in FAILURES:
        print(f"  FAIL {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
