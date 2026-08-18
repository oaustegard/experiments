#!/usr/bin/env python3
"""Score a fitted router on the fitted family, the held-out family and the wild set.

The numbers that matter are the *differences* between the three splits:

* family A is what the rules were fitted on — an upper bound and nothing else
* family B is the same 79 intents in a disjoint phrasing from a disjoint entity
  pool — what `monad-bsky` could only check for its hand-written rules once
* `wild.jsonl` is hand-authored, was committed before the fitter existed, and is
  the only split whose sentences were not produced by a template at all

Reported per split: coverage (did it answer), precision on what it answered,
accuracy over all routable rows, and abstention on off-topic rows — which is the
number `monad-bsky` watched collapse from 0.500 to 0.183.

    python3 eval.py rules_schema.json rules_open.json rules_cues.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from handwritten import HandRouter
from router import Router

HERE = Path(__file__).resolve().parent


def load_split(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def score(router: Router, rows: list[dict]) -> dict:
    on = [r for r in rows if r.get("label")]
    off = [r for r in rows if not r.get("label")]
    lat, hits, tool_hits, answered, arg_hits, arg_n = [], 0, 0, 0, 0, 0
    meth_n = meth_hits = 0
    errors = []

    for r in on:
        t0 = time.perf_counter()
        got = router.route(r["query"])
        lat.append((time.perf_counter() - t0) * 1000)
        gold = r["label"]
        if got is not None:
            answered += 1
        ok = got == gold
        hits += ok
        tool_ok = got is not None and got.split("::")[0] == gold.split("::")[0]
        tool_hits += tool_ok
        if "::" in gold and tool_ok:
            meth_n += 1
            meth_hits += ok
        if not ok:
            errors.append({"id": r["id"], "query": r["query"], "gold": gold, "got": got})
        if r.get("args"):
            call = router.call(r["query"])
            arg_n += 1
            arg_hits += bool(
                call and tool_ok
                and all(str(call["args"].get(k)) == str(v) for k, v in r["args"].items())
            )

    off_ok = 0
    for r in off:
        t0 = time.perf_counter()
        got = router.route(r["query"])
        lat.append((time.perf_counter() - t0) * 1000)
        off_ok += got is None
        if got is not None:
            errors.append({"id": r["id"], "query": r["query"], "gold": None, "got": got})

    n_on = len(on) or 1
    return {
        "n_routable": len(on), "n_offtopic": len(off),
        "coverage": round(answered / n_on, 4),
        "precision": round(hits / answered, 4) if answered else 0.0,
        "label_acc": round(hits / n_on, 4),
        "tool_acc": round(tool_hits / n_on, 4),
        "method_acc_given_tool": round(meth_hits / meth_n, 4) if meth_n else None,
        "abstain_acc": round(off_ok / len(off), 4) if off else None,
        "args_acc": round(arg_hits / arg_n, 4) if arg_n else None,
        "median_latency_ms": round(statistics.median(lat), 4),
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rules", nargs="+",
                    help="arm names from the arms.py registry, or paths to rules_*.json")
    ap.add_argument("--dump-errors", type=Path)
    a = ap.parse_args()

    splits = {
        "family A (fitted)": HERE / "data" / "family_a.jsonl",
        "family B (held-out)": HERE / "data" / "family_b.jsonl",
        "wild (hand-authored)": HERE / "wild.jsonl",
    }
    out, all_errors = {}, {}
    hdr = f"{'arm':<24}{'split':<22}{'cov':>7}{'prec':>7}{'acc':>7}{'tool':>7}{'meth':>7}{'abst':>7}{'args':>7}{'ms':>9}"
    print(hdr)
    print("-" * len(hdr))
    import arms as arms_mod
    arms_mod.load_all()

    for spec in a.rules:
        # Three ways to name an arm, so the first-pass command lines still work:
        # a registry name, a rules_*.json path, or the `hand` pseudo-path.
        if spec in arms_mod.REGISTRY:
            r, arm = arms_mod.build(spec), spec
        elif spec in ("hand", "hand+fallback"):
            r = HandRouter(fallback=spec.endswith("fallback"))
            arm = spec
        else:
            rp = Path(spec)
            r, arm = Router(rp), rp.stem.replace("rules_", "")
        n_rules = len(getattr(r, "rules", ()) or ())
        for sname, spath in splits.items():
            s = score(r, load_split(spath))
            all_errors[f"{arm}|{sname}"] = s.pop("errors")
            out.setdefault(arm, {})[sname] = s
            f = lambda k: "  -  " if s[k] is None else f"{s[k]:.3f}"
            tag = f"{arm} ({n_rules}r)" if n_rules else arm
            print(f"{tag:<24}{sname:<22}{f('coverage'):>7}"
                  f"{f('precision'):>7}{f('label_acc'):>7}{f('tool_acc'):>7}"
                  f"{f('method_acc_given_tool'):>7}{f('abstain_acc'):>7}"
                  f"{f('args_acc'):>7}{s['median_latency_ms']:>9.4f}")
        print()

    (HERE / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    if a.dump_errors:
        a.dump_errors.write_text(json.dumps(all_errors, indent=1) + "\n")
    print("wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
