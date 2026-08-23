"""Main sweep, with the controls the adversarial review required.

Changes from the pilot:
  - full-distribution probe (rank, entropy, control token), so a flattened
    distribution is not read as suppression
  - sentence-case baseline, plus an all-lowercase arm as a grammaticality control
  - many keywords per capitalisation-token-cost bin, so the cost is not a
    relabelling of which word it is
  - items stratified by baseline pressure
  - markdown bold, markdown italic, alternating case and a capitalised non-word,
    as emphasis markers that vary token count and case independently
"""
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C
from render import render

MARKER_MODES = ["bold", "italic", "alt", "acronym"]
LOWER_SUBSET = ["never", "do not", "avoid", "on no account",
                "under no circumstances", "it is forbidden to"]


def main():
    items = json.load(open(os.path.join(HERE, "items_v2.json")))["items"]
    kwbins = json.load(open(os.path.join(HERE, "keywords.json")))
    keywords = [r for rows in kwbins.values() for r in rows]

    cells = [("none", None, None, None)]
    for r in keywords:
        for mode in ("title", "caps"):
            cells.append((f"{r['keyword']}|{mode}", r["keyword"], r["frame"], mode))
        if r["keyword"] in LOWER_SUBSET:
            cells.append((f"{r['keyword']}|lower", r["keyword"], r["frame"], "lower"))
    never = next(r for r in keywords if r["keyword"] == "never")
    for mode in MARKER_MODES:
        cells.append((f"never|{mode}", never["keyword"], never["frame"], mode))

    prompts, targets, controls, keys, spans = [], [], [], [], {}
    for cname, kw, frame, mode in cells:
        for it in items:
            if frame is None:
                user = it["question"]
            else:
                d, span = render(frame, kw, it["word"], mode)
                spans[cname] = span
                user = d + " " + it["question"]
            prompts.append(C.build_prompt(user, it["prefix"]))
            targets.append(" " + it["word"])
            controls.append(it["control"])
            keys.append((cname, it["id"]))

    print(f"{len(items)} items x {len(cells)} cells = {len(prompts)} forward passes",
          flush=True)
    res = C.probe_full(prompts, targets, controls, batch_size=8)
    by = collections.defaultdict(dict)
    for (cname, iid), r in zip(keys, res):
        by[cname][iid] = r

    ids = [it["id"] for it in items]
    strat = {it["id"]: it["stratum"] for it in items}

    def norm(cname, iid):
        """Suppression in log-odds, with the control token's shift removed."""
        a, b = by["none"][iid], by[cname][iid]
        dt = C.logodds(a["logp"]) - C.logodds(b["logp"])
        dc = C.logodds(a["logp_control"]) - C.logodds(b["logp_control"])
        return dt - dc

    summary = {}
    for cname in by:
        if cname == "none":
            continue
        v = [norm(cname, i) for i in ids]
        lo, hi = C.bootstrap_ci([(x, 0.0) for x in v])
        summary[cname] = dict(
            norm_suppression=C.mean(v), ci=[lo, hi],
            raw_logp=C.mean(by[cname][i]["logp"] for i in ids),
            raw_delta_logp=C.mean(by["none"][i]["logp"] - by[cname][i]["logp"]
                                  for i in ids),
            entropy=C.mean(by[cname][i]["entropy"] for i in ids),
            entropy_delta=C.mean(by[cname][i]["entropy"] - by["none"][i]["entropy"]
                                 for i in ids),
            mean_rank=C.mean(by[cname][i]["rank"] for i in ids),
            span=spans.get(cname))

    # within-keyword case effect, per keyword and pooled per token-delta bin
    caps_rows, bin_rows = [], collections.defaultdict(list)
    for r in keywords:
        k = r["keyword"]
        a = [norm(f"{k}|title", i) for i in ids]
        b = [norm(f"{k}|caps", i) for i in ids]
        d, t, n = C.paired_t(b, a)      # positive = CAPS suppresses more
        lo, hi = C.bootstrap_ci(list(zip(b, a)))
        caps_rows.append(dict(keyword=k, delta_tokens=r["delta"],
                              caps_effect=d, ci=[lo, hi], t=t, n=n))
        bin_rows[r["delta"]].extend([x - y for x, y in zip(b, a)])
    bins = {str(k): dict(mean=C.mean(v), n_keywords=len(v) // len(ids),
                         ci=list(C.bootstrap_ci([(x, 0.0) for x in v])))
            for k, v in sorted(bin_rows.items())}

    # case effect by baseline-pressure stratum, pooled over keywords
    by_stratum = collections.defaultdict(list)
    for r in keywords:
        k = r["keyword"]
        for i in ids:
            by_stratum[strat[i]].append(norm(f"{k}|caps", i) - norm(f"{k}|title", i))
    strata = {s: dict(mean=C.mean(v), n=len(v),
                      ci=list(C.bootstrap_ci([(x, 0.0) for x in v])))
              for s, v in by_stratum.items()}

    C.dump(os.path.join(HERE, "v2.json"),
           dict(n_items=len(items), n_cells=len(cells), summary=summary,
                caps_by_keyword=caps_rows, caps_by_token_delta=bins,
                caps_by_stratum=strata,
                strata_counts=collections.Counter(strat.values()),
                raw={c: by[c] for c in by}))

    print("\n== emphasis markers on the same directive "
          "(log-odds suppression, control-normalised)")
    for c in ["never|title", "never|caps", "never|lower", "never|bold",
              "never|italic", "never|alt", "never|acronym"]:
        s = summary[c]
        print(f"  {c:16} supp={s['norm_suppression']:+7.3f} "
              f"[{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]  "
              f"dH={s['entropy_delta']:+6.3f}  rank={s['mean_rank']:5.2f}")
    print("\n== CAPS effect pooled by what capitalising costs in tokens")
    for d, v in bins.items():
        print(f"  +{d} tokens  n_kw={v['n_keywords']:2d}  effect={v['mean']:+7.4f} "
              f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
    print("\n== CAPS effect by baseline pressure stratum")
    for s, v in strata.items():
        print(f"  {s:8} n={v['n']:5d}  effect={v['mean']:+7.4f} "
              f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")


if __name__ == "__main__":
    main()
