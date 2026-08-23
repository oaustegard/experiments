"""Q3: dose-response. Capitalise a rising fraction of the directive and watch
whether suppression peaks and then declines."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C
from conditions import DOSE_BASE, DOSE_LEVELS, caps_fraction


def main():
    items = json.load(open(os.path.join(HERE, "items_screened.json")))["kept"]
    prompts, targets, keys, texts = [], [], [], {}
    for frac in DOSE_LEVELS:
        for it in items:
            d = caps_fraction(DOSE_BASE.format(W=it["word"]), frac)
            texts.setdefault(frac, d if it["id"] == items[0]["id"] else texts[frac])
            prompts.append(C.build_prompt(d + " " + it["question"], it["prefix"]))
            targets.append(" " + it["word"])
            keys.append((frac, it["id"]))
    # no-directive floor
    for it in items:
        prompts.append(C.build_prompt(it["question"], it["prefix"]))
        targets.append(" " + it["word"])
        keys.append(("none", it["id"]))

    print(f"{len(prompts)} forward passes")
    lps = C.logprobs(prompts, targets, batch_size=8)
    by = {}
    for (frac, iid), lp in zip(keys, lps):
        by.setdefault(str(frac), {})[iid] = lp

    ids = [it["id"] for it in items]
    base = [by["none"][i] for i in ids]
    rows = []
    for frac in DOSE_LEVELS:
        v = [by[str(frac)][i] for i in ids]
        d, t, n = C.paired_t(base, v)
        ntok = C.ntok(" " + caps_fraction(DOSE_BASE.format(W="Paris"), frac))
        rows.append(dict(frac=frac, mean_logp=C.mean(v), suppression=d, t=t, n=n,
                         directive_tokens=ntok,
                         example=caps_fraction(DOSE_BASE.format(W="Paris"), frac)))
    C.dump(os.path.join(HERE, "dose.json"), dict(rows=rows, raw=by))
    print("\n frac  dtok   logp    suppression      t   example")
    for r in rows:
        print(f"  {r['frac']:.2f}  {r['directive_tokens']:4d} {r['mean_logp']:8.3f} "
              f"{r['suppression']:+9.3f} {r['t']:7.2f}   {r['example'][:56]}")


if __name__ == "__main__":
    main()
