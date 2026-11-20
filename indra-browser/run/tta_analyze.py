"""Test-time augmentation on OmniParseBench: how much could combining perturbed
readings recover, and how much does a deployable combiner recover?

Reads opb_eval.py runs (OUT/<arm>/pages/*.json): a base run and page-level
variant runs (the same model on perturbed pages). Reports, against the base:
  - each arm alone (headline, families, tests flipped each way);
  - the test-level oracle: a test passes if any arm passes it. No system can
    do this (it mixes readings within a page), so it is the loosest ceiling;
  - the page-level oracle: each page takes the one arm that scores best on it,
    weighting each test by its family's share of the headline. This is the
    ceiling for any selector that picks one reading per page;
  - a confidence selector: each page takes the arm whose lines have the
    highest mean CTC confidence (weighted by text length), which needs no
    ground truth;
  - greedy arm order: the arm that adds most to the page-level oracle first,
    then the next, to show how many passes the gain needs.
Intervals are 95% bootstrap over documents, paired against the base.

    python3 run/tta_analyze.py /tmp/indra-eval/out base p-gamma2 p-otsu ... --json results/tta.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze import FAMILY, headline, load


def page_conf(rec) -> float:
    lines = rec.get("raw", {}).get("lines", [])
    w = [(len(t), c) for _, t, c in lines if t]
    n = sum(a for a, _ in w)
    return sum(a * c for a, c in w) / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("base")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()

    bt, bp = load(a.out / a.base)
    ids = sorted(bt)
    runs = {a.base: (bt, bp)}
    for arm in a.arms:
        t, p = load(a.out / arm)
        if set(t) != set(bt):
            print(f"{arm}: {len(set(bt) - set(t))} tests missing, skipped")
            continue
        runs[arm] = (t, p)
    names = list(runs)

    fam_n = Counter(bt[t][1] for t in ids)
    page_tests = defaultdict(list)
    for pid, rec in bp.items():
        for ln in rec["lines"]:
            page_tests[pid].append(ln["test_id"])

    def page_value(arm, pid):
        tests = runs[arm][0]
        return sum(tests[t][2] / fam_n[tests[t][1]] for t in page_tests[pid])

    def compose(choice):  # page -> arm  =>  test -> (doc, family, ok)
        return {t: runs[choice[pid]][0][t] for pid in page_tests for t in page_tests[pid]}

    docs = sorted({bt[t][0] for t in ids})
    rng = np.random.default_rng(0)
    weights = []
    for _ in range(a.boot):
        c = np.bincount(rng.choice(len(docs), len(docs)), minlength=len(docs))
        weights.append({docs[i]: float(c[i]) for i in range(len(docs))})
    base_h, base_f = headline(ids, bt)
    base_boot = [headline(ids, bt, w)[0] for w in weights]

    def scored(tests):
        h, f = headline(ids, tests)
        d = np.array([headline(ids, tests, w)[0] - b for w, b in zip(weights, base_boot)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"headline": h, "families": f, "delta": h - base_h, "ci95": [float(lo), float(hi)]}

    def show(label, r, extra=""):
        print(f"{label:<26} headline {100 * r['headline']:.2f} ({100 * r['delta']:+.2f}, "
              f"{100 * r['ci95'][0]:+.2f} to {100 * r['ci95'][1]:+.2f}) "
              + " ".join(f"{k} {100 * v:.2f}" for k, v in sorted(r["families"].items())) + extra)

    res = {"base": a.base, "base_headline": base_h, "base_families": base_f, "arms": {}}
    print(f"{a.base}: headline {100 * base_h:.2f} " + " ".join(f"{k} {100 * v:.2f}" for k, v in sorted(base_f.items())))
    base_fail = [t for t in ids if not bt[t][2]]
    for arm in names[1:]:
        at, ap_ = runs[arm]
        r = scored(at)
        r["fixed"] = sum(at[t][2] and not bt[t][2] for t in ids)
        r["broken"] = sum(bt[t][2] and not at[t][2] for t in ids)
        r["pages_same_markdown"] = sum(ap_[p]["raw"].get("markdown") == bp[p]["raw"].get("markdown")
                                       for p in bp if "raw" in ap_.get(p, {}) and "raw" in bp[p])
        r["ms_median"] = float(np.median([ap_[p]["ms"] for p in ap_]))
        # comparable to Round 4's Tesseract pairing: text family, base or this arm per test
        text = [t for t in ids if bt[t][1] == "text"]
        r["text_oracle_pair"] = sum(bt[t][2] or at[t][2] for t in text) / len(text)
        res["arms"][arm] = r
        show(arm, r, f"; fixed {r['fixed']} broken {r['broken']}; same markdown {r['pages_same_markdown']}/{len(bp)}")

    # test-level oracle
    orc = {t: (bt[t][0], bt[t][1], any(runs[n][0][t][2] for n in names)) for t in ids}
    r = scored(orc)
    r["rescued"] = sum(orc[t][2] for t in base_fail)
    r["rescued_share_of_base_fails"] = r["rescued"] / max(1, len(base_fail))
    by_type = Counter()
    for t in base_fail:
        if orc[t][2]:
            by_type[runs[a.base][0][t][1]] += 1
    r["rescued_by_family"] = dict(by_type)
    res["oracle_test"] = r
    show("oracle (any arm, per test)", r, f"; rescues {r['rescued']} = {100 * r['rescued_share_of_base_fails']:.1f}% of base fails")

    # how often each base failure is fixed by k arms: concentrated or spread?
    nfix = Counter(sum(runs[n][0][t][2] for n in names[1:]) for t in base_fail)
    res["base_fails_fixed_by_n_arms"] = dict(sorted(nfix.items()))

    # page-level oracle and greedy order
    def best_choice(arms):
        return {pid: max(arms, key=lambda n: (page_value(n, pid), n == a.base)) for pid in page_tests}

    ch = best_choice(names)
    r = scored(compose(ch))
    r["pages_choosing"] = dict(Counter(ch.values()).most_common())
    res["oracle_page"] = r
    show("oracle (best arm per page)", r)

    chosen, order = [a.base], []
    left = names[1:]
    while left:
        best = max(left, key=lambda n: headline(ids, compose(best_choice(chosen + [n])))[0])
        chosen.append(best)
        left.remove(best)
        h = headline(ids, compose(best_choice(chosen)))[0]
        order.append({"arm": best, "page_oracle_headline": h, "delta": h - base_h})
    res["greedy_page_oracle"] = order
    print("greedy page-oracle order: " + ", ".join(f"{o['arm']} {100 * o['delta']:+.2f}" for o in order))

    # deployable: the most confident reading per page
    conf = {n: {pid: page_conf(runs[n][1][pid]) for pid in page_tests} for n in names}
    sel = {pid: max(names, key=lambda n: (conf[n][pid], n == a.base)) for pid in page_tests}
    r = scored(compose(sel))
    r["pages_choosing"] = dict(Counter(sel.values()).most_common())
    res["select_conf"] = r
    show("select by confidence", r)
    for n in names[1:]:
        s2 = {pid: max((a.base, n), key=lambda m: (conf[m][pid], m == a.base)) for pid in page_tests}
        r2 = scored(compose(s2))
        res["arms"][n]["select_conf_pair"] = r2
        show(f"  base|{n} by confidence", r2)

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
