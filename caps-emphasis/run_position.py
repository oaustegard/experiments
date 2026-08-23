"""Sentence-initial vs mid-sentence emphasis.

The corpus split says these are different constructions, not one effect at two
positions: in SYNTH, line-initial `**bold**` labels run at 6.78 per 1,000 words
against 0.62 for bold used as mid-sentence emphasis. The pilot and the main sweep
happened to differ on exactly this axis -- the pilot bolded `never` inside
"You must never mention the word X", the main sweep bolded it at the head of
"Never mention the word X" -- and they disagreed on the sign. This measures the
two positions against each other with everything else held fixed.
"""
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C
from render import render

FRAMES = {"initial": "{K} mention the word {W}.",
          "medial":  "You must {K} mention the word {W}."}
KEYWORD = "never"
MODES = ["title", "caps", "bold", "italic"]


def main():
    items = json.load(open(os.path.join(HERE, "items_v2.json")))["items"]
    prompts, targets, controls, keys, texts = [], [], [], [], {}
    for pos, frame in FRAMES.items():
        for mode in MODES:
            for it in items:
                d, _span = render(frame, KEYWORD, it["word"], mode)
                texts[f"{pos}|{mode}"] = d
                prompts.append(C.build_prompt(d + " " + it["question"], it["prefix"]))
                targets.append(" " + it["word"])
                controls.append(it["control"])
                keys.append((f"{pos}|{mode}", it["id"]))
    for it in items:
        prompts.append(C.build_prompt(it["question"], it["prefix"]))
        targets.append(" " + it["word"])
        controls.append(it["control"])
        keys.append(("none", it["id"]))

    print(f"{len(prompts)} forward passes", flush=True)
    res = C.probe_full(prompts, targets, controls, batch_size=8)
    by = collections.defaultdict(dict)
    for (k, iid), r in zip(keys, res):
        by[k][iid] = r

    ids = [it["id"] for it in items]

    def norm(cell, iid):
        a, b = by["none"][iid], cell[iid]
        return ((C.logodds(a["logp"]) - C.logodds(b["logp"])) -
                (C.logodds(a["logp_control"]) - C.logodds(b["logp_control"])))

    out = {}
    for k in by:
        if k == "none":
            continue
        v = [norm(by[k], i) for i in ids]
        lo, hi = C.bootstrap_ci([(x, 0.0) for x in v])
        out[k] = dict(suppression=C.mean(v), ci=[lo, hi],
                      entropy_delta=C.mean(by[k][i]["entropy"] - by["none"][i]["entropy"]
                                           for i in ids),
                      directive_tokens=C.ntok(" " + texts[k]),
                      example=texts[k])
    # marker effect relative to the sentence-case baseline at the same position
    rel = {}
    for pos in FRAMES:
        base = [norm(by[f"{pos}|title"], i) for i in ids]
        for mode in MODES[1:]:
            v = [norm(by[f"{pos}|{mode}"], i) for i in ids]
            d, t, n = C.paired_t(v, base)
            lo, hi = C.bootstrap_ci(list(zip(v, base)))
            rel[f"{pos}|{mode}"] = dict(effect=d, ci=[lo, hi], t=t)
    C.dump(os.path.join(HERE, "position.json"), dict(cells=out, relative=rel))

    print("\n cell            suppression        95% CI          dH   tokens  text")
    for k, v in out.items():
        print(f"  {k:16} {v['suppression']:+7.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] "
              f"{v['entropy_delta']:+6.3f} {v['directive_tokens']:5d}  {v['example'][:40]}")
    print("\n marker effect vs sentence-case at the SAME position "
          "(positive = suppresses more)")
    for k, v in rel.items():
        print(f"  {k:16} {v['effect']:+7.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]  t={v['t']:6.2f}")


if __name__ == "__main__":
    main()
