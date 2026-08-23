"""Does the model register what the directive tells it to do, or only that the
word was named?

The pilot found "Always mention X" and "Never mention X" moved log P by nearly the
same amount, which would mean the prohibition's polarity is not being read at all.
That was measured without the control-token normalisation, so it is re-run here
properly, with a neutral mention as the third arm that names the word without
instructing anything.
"""
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C

CELLS = {
    "none":          None,
    "prohibit":      "Never mention the word {W}.",
    "prohibit_caps": "NEVER mention the word {W}.",
    "require":       "Always mention the word {W}.",
    "require_caps":  "ALWAYS mention the word {W}.",
    "neutral":       "The word {W} may be relevant here.",
    "unrelated":     "Never mention the word bicycle.",
}


def main():
    items = json.load(open(os.path.join(HERE, "items_v2.json")))["items"]
    prompts, targets, controls, keys = [], [], [], []
    for name, tmpl in CELLS.items():
        for it in items:
            user = it["question"] if tmpl is None else \
                tmpl.format(W=it["word"]) + " " + it["question"]
            prompts.append(C.build_prompt(user, it["prefix"]))
            targets.append(" " + it["word"])
            controls.append(it["control"])
            keys.append((name, it["id"]))

    print(f"{len(prompts)} forward passes", flush=True)
    res = C.probe_full(prompts, targets, controls, batch_size=8)
    by = collections.defaultdict(dict)
    for (n, iid), r in zip(keys, res):
        by[n][iid] = r

    ids = [it["id"] for it in items]

    def norm(name, iid):
        a, b = by["none"][iid], by[name][iid]
        return ((C.logodds(a["logp"]) - C.logodds(b["logp"])) -
                (C.logodds(a["logp_control"]) - C.logodds(b["logp_control"])))

    out = {}
    for name in CELLS:
        if name == "none":
            continue
        v = [norm(name, i) for i in ids]
        lo, hi = C.bootstrap_ci([(x, 0.0) for x in v])
        out[name] = dict(suppression=C.mean(v), ci=[lo, hi],
                         mean_rank=C.mean(by[name][i]["rank"] for i in ids))
    # the load-bearing contrast: prohibit vs require, same frame, opposite polarity
    a = [norm("prohibit", i) for i in ids]
    b = [norm("require", i) for i in ids]
    d, t, n = C.paired_t(a, b)
    lo, hi = C.bootstrap_ci(list(zip(a, b)))
    out["_polarity_contrast"] = dict(prohibit_minus_require=d, ci=[lo, hi], t=t, n=n)
    C.dump(os.path.join(HERE, "polarity.json"), out)

    print("\n cell            suppression         95% CI        rank")
    for k, v in out.items():
        if k.startswith("_"):
            continue
        print(f"  {k:15} {v['suppression']:+8.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
              f"{v['mean_rank']:6.2f}")
    p = out["_polarity_contrast"]
    print(f"\n  prohibit - require = {p['prohibit_minus_require']:+.4f} "
          f"[{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}]  t={p['t']:.2f}")


if __name__ == "__main__":
    main()
