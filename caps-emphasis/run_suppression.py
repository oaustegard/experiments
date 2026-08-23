"""Q1 + Q2: does a capitalised directive suppress the forbidden word more than
its lowercase twin, and is any effect attributable to case or to token count."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C
from conditions import KEYWORD_FRAMES, MATCHED, CONTROLS, POSITION


def build_conditions():
    """(name, builder(question, word) -> user_content, directive_template)"""
    conds = []
    for name, _, _ in [(c[0], c[1], c[2]) for c in CONTROLS]:
        pass
    for name, tmpl, span in CONTROLS:
        if tmpl is None:
            conds.append((name, lambda q, w: q, None))
        else:
            conds.append((name, (lambda t: lambda q, w: t.format(W=w) + " " + q)(tmpl), tmpl))
    for key, lo, up, _kw in KEYWORD_FRAMES:
        conds.append((f"{key}_lower", (lambda t: lambda q, w: t.format(W=w) + " " + q)(lo), lo))
        conds.append((f"{key}_caps",  (lambda t: lambda q, w: t.format(W=w) + " " + q)(up), up))
    for name, tmpl, _span in MATCHED:
        conds.append((name, (lambda t: lambda q, w: t.format(W=w) + " " + q)(tmpl), tmpl))
    for name, tmpl, _span in POSITION:
        conds.append((name, (lambda t: lambda q, w: q + " " + t.format(W=w))(tmpl), tmpl))
    return conds


def main():
    items = json.load(open(os.path.join(HERE, "items_screened.json")))["kept"]
    conds = build_conditions()
    print(f"{len(items)} items x {len(conds)} conditions "
          f"= {len(items) * len(conds)} forward passes")

    prompts, targets, keys = [], [], []
    for cname, builder, _t in conds:
        for it in items:
            user = builder(it["question"], it["word"])
            prompts.append(C.build_prompt(user, it["prefix"]))
            targets.append(" " + it["word"])
            keys.append((cname, it["id"]))

    lps = C.logprobs(prompts, targets, batch_size=8)
    by = {}
    for (cname, iid), lp in zip(keys, lps):
        by.setdefault(cname, {})[iid] = lp

    # directive token cost, measured not assumed
    costs = {}
    for cname, _b, tmpl in conds:
        if tmpl is None:
            costs[cname] = 0
        else:
            costs[cname] = C.ntok(" " + tmpl.format(W="Paris"))

    ids = [it["id"] for it in items]
    base = [by["none"][i] for i in ids]
    summary = {}
    for cname in by:
        vals = [by[cname][i] for i in ids]
        d, t, n = C.paired_t(base, vals)   # positive d = suppression vs no directive
        summary[cname] = dict(mean_logp=C.mean(vals), suppression_vs_none=d,
                              t=t, n=n, directive_tokens=costs[cname])

    # within-keyword CAPS contrasts: the load-bearing comparison
    caps_effects = []
    for key, lo, up, _kw in KEYWORD_FRAMES:
        a = [by[f"{key}_lower"][i] for i in ids]
        b = [by[f"{key}_caps"][i] for i in ids]
        d, t, n = C.paired_t(a, b)         # positive = CAPS suppresses more
        delta_tok = costs[f"{key}_caps"] - costs[f"{key}_lower"]
        caps_effects.append(dict(keyword=key, caps_extra_tokens=delta_tok,
                                 caps_effect_nats=d, t=t, n=n,
                                 lower_logp=C.mean(a), caps_logp=C.mean(b)))

    matched = {}
    ref = [by["m_lower"][i] for i in ids]
    for name, _t, _s in MATCHED:
        v = [by[name][i] for i in ids]
        d, t, n = C.paired_t(ref, v)
        matched[name] = dict(mean_logp=C.mean(v), vs_m_lower=d, t=t,
                             directive_tokens=costs[name])

    C.dump(os.path.join(HERE, "suppression.json"),
           dict(n_items=len(items), summary=summary, caps_effects=caps_effects,
                matched=matched, raw=by))

    print("\n== condition means (log P of forbidden word; lower = more suppressed)")
    for cname, s in sorted(summary.items(), key=lambda x: x[1]["mean_logp"]):
        print(f"  {cname:16} logp={s['mean_logp']:8.3f}  vs-none={s['suppression_vs_none']:+7.3f}"
              f"  t={s['t']:7.2f}  dtok={s['directive_tokens']}")
    print("\n== within-keyword CAPS effect (positive = CAPS suppresses more)")
    for e in caps_effects:
        print(f"  {e['keyword']:14} +{e['caps_extra_tokens']} tok  "
              f"effect={e['caps_effect_nats']:+7.3f} nats  t={e['t']:7.2f}")
    print("\n== length-matched set (vs 'you must never', lowercase)")
    for k, v in matched.items():
        print(f"  {k:12} logp={v['mean_logp']:8.3f}  delta={v['vs_m_lower']:+7.3f}"
              f"  t={v['t']:7.2f}  dtok={v['directive_tokens']}")


if __name__ == "__main__":
    main()
