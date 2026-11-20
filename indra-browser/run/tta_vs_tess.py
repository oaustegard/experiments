"""Second Indra pass vs Tesseract as a second reader, on Round 4's test set.

Round 4 (tess_pair.py) paired Indra with the benchmark's published Tesseract
verdicts on the 8,499 text-family tests (present, order, repeat) Tesseract can
attempt. This puts each TTA arm in Tesseract's place on the same tests: how
many base failures does it pass (rescues), how many base passes does it fail,
and what is the text pass rate if every test took the better of base and the
second reader (an oracle no router reaches).

    python3 run/tta_vs_tess.py /tmp/indra-eval/out base ARM ... --results /tmp/opb-data/ds/results.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TEXT = {"present", "order", "repeat"}


def verdicts(run: Path) -> pd.Series:
    rows = []
    for f in (run / "pages").glob("*.json"):
        rows += [(ln["test_id"], ln["result"] == "pass") for ln in json.loads(f.read_text())["lines"]]
    return pd.Series(dict(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("base")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    pub = pd.read_parquet(a.results)
    tess = pub[(pub.system == "Tesseract") & pub.test_type.isin(TEXT) & (pub.result != "unsupported")]
    tess = tess.set_index("test_id").result == "pass"
    base = verdicts(a.out / a.base).reindex(tess.index)
    keep = base.notna()
    tess, base = tess[keep], base[keep].astype(bool)
    fails = int((~base).sum())
    res = {"tests": int(len(base)), "base_text_pass": float(base.mean()), "base_fails": fails, "second": {}}

    tess_resc = ~base & tess

    def row(name, other):
        r = {"rescued": int((~base & other).sum()), "lost": int((base & ~other).sum()),
             "rescued_also_by_tesseract": int((~base & other & tess_resc).sum()),
             "alone": float(other.mean()), "oracle_pair": float((base | other).mean())}
        r["rescued_share_of_base_fails"] = r["rescued"] / fails
        res["second"][name] = r
        print(f"{name:<14} alone {100 * r['alone']:.2f}  rescues {r['rescued']:>4} "
              f"({100 * r['rescued_share_of_base_fails']:.1f}% of fails)  lost {r['lost']:>4}  "
              f"base-or-it {100 * r['oracle_pair']:.2f}")

    print(f"{res['tests']} tests, base text pass {100 * res['base_text_pass']:.2f}, {fails} fails")
    row("Tesseract", tess)
    union = pd.Series(False, index=base.index)
    pages = [arm for arm in a.arms if arm.startswith("p-")]
    for arm in a.arms:
        v = verdicts(a.out / arm).reindex(base.index).astype(bool)
        row(arm, v)
        if arm in pages:
            union |= v
    if pages:
        u = {"arms": pages, "rescued": int((~base & union).sum()),
             "tesseract_rescues_no_page_arm_fixes": int((tess_resc & ~union).sum()),
             "oracle": float((base | union).mean()), "oracle_with_tesseract": float((base | union | tess).mean())}
        res["page_arms_union"] = u
        print(f"any of {len(pages)} page arms: rescues {u['rescued']}, oracle {100 * u['oracle']:.2f}, "
              f"with Tesseract {100 * u['oracle_with_tesseract']:.2f}; Tesseract rescues no page arm fixes: "
              f"{u['tesseract_rescues_no_page_arm_fixes']}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
