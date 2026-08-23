"""Q3: dose-response, with the forbidden word's own case held fixed.

Capitalising a rising fraction of "you must never mention the word Paris" ends up
capitalising PARIS, which turns a prohibition manipulation into a priming one —
Rana (arXiv:2601.08070) finds 87.5% of constraint failures already show a
priming signature. The forbidden word is excluded from the capitalisation set
here, and each dose level averages over many random word subsets so that "dose"
is not confounded with which words happened to be picked.
"""
import sys, os, json, random, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C

BASE = "You must never mention the word {W} anywhere in your answer."
LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
DRAWS = 12
SEED = 20260823


def dose_text(word, frac, rng, freeze_word=True):
    words = BASE.format(W=word).split(" ")
    idx = [i for i, w in enumerate(words)
           if not (freeze_word and word.lower() in w.lower())]
    k = round(len(idx) * frac)
    pick = set(rng.sample(idx, k)) if k else set()
    return " ".join(w.upper() if i in pick else w for i, w in enumerate(words))


def main():
    items = json.load(open(os.path.join(HERE, "items_v2.json")))["items"]
    rng = random.Random(SEED)
    prompts, targets, controls, keys = [], [], [], []
    examples = collections.defaultdict(list)
    for frac in LEVELS:
        draws = 1 if frac in (0.0,) else DRAWS
        for d in range(draws):
            for it in items:
                txt = dose_text(it["word"], frac, rng)
                if it["id"] == items[0]["id"]:
                    examples[frac].append(txt)
                prompts.append(C.build_prompt(txt + " " + it["question"], it["prefix"]))
                targets.append(" " + it["word"])
                controls.append(it["control"])
                keys.append((frac, d, it["id"]))
    for it in items:                       # no-directive floor
        prompts.append(C.build_prompt(it["question"], it["prefix"]))
        targets.append(" " + it["word"])
        controls.append(it["control"])
        keys.append(("none", 0, it["id"]))

    print(f"{len(prompts)} forward passes", flush=True)
    res = C.probe_full(prompts, targets, controls, batch_size=8)
    by = collections.defaultdict(dict)
    for (frac, d, iid), r in zip(keys, res):
        by[(str(frac), d)][iid] = r
    none = by[("none", 0)]

    def norm(cell, iid):
        a, b = none[iid], cell[iid]
        return ((C.logodds(a["logp"]) - C.logodds(b["logp"])) -
                (C.logodds(a["logp_control"]) - C.logodds(b["logp_control"])))

    ids = [it["id"] for it in items]
    rows = []
    for frac in LEVELS:
        draws = 1 if frac in (0.0,) else DRAWS
        per_draw = []
        allv = []
        for d in range(draws):
            v = [norm(by[(str(frac), d)], i) for i in ids]
            per_draw.append(C.mean(v))
            allv.extend(v)
        lo, hi = C.bootstrap_ci([(x, 0.0) for x in allv])
        rows.append(dict(frac=frac, mean=C.mean(allv), ci=[lo, hi],
                         between_draw_sd=C.stdev(per_draw) if draws > 1 else 0.0,
                         within_draw_sd=C.stdev(allv),
                         n_draws=draws, example=examples[frac][0]))
    C.dump(os.path.join(HERE, "dose2.json"), dict(rows=rows, base=BASE,
                                                  draws=DRAWS, seed=SEED))
    print("\n frac   suppression        95% CI        between-draw sd   example")
    for r in rows:
        print(f"  {r['frac']:.2f}  {r['mean']:+8.4f}  "
              f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  {r['between_draw_sd']:8.4f}   "
              f"{r['example'][:52]}")


if __name__ == "__main__":
    main()
