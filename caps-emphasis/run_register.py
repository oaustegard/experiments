"""Register control for the markdown-bold arm.

93% of bold spans in SYNTH occur in reasoning traces; user turns contain one bold
span in 763,630 words. So a bolded directive in the user turn is off-distribution
for that register while being maximally in-distribution for the register the
model itself writes. If bold's advantage is emphasis, it should survive being
moved into the think block. If it is surprise at an off-register marker, it
should not.
"""
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C
from render import render

FRAME = "{K} mention the word {W}."
KEYWORD = "never"
MODES = ["title", "caps", "bold"]


def build(it, mode, register):
    tok, _ = C.load()
    d, span = render(FRAME, KEYWORD, it["word"], mode)
    if register == "user":
        head = tok.apply_chat_template(
            [{"role": "user", "content": d + " " + it["question"]}],
            tokenize=False, add_generation_prompt=True)
        return head + "</think>\n" + it["prefix"], span
    # think: question in the user turn, directive restated inside the model's
    # own reasoning register, which is where SYNTH puts markdown emphasis
    head = tok.apply_chat_template(
        [{"role": "user", "content": it["question"]}],
        tokenize=False, add_generation_prompt=True)
    return head + "Constraint: " + d + "\n</think>\n" + it["prefix"], span


def main():
    items = json.load(open(os.path.join(HERE, "items_v2.json")))["items"]
    prompts, targets, controls, keys = [], [], [], []
    for register in ("user", "think"):
        for mode in MODES:
            for it in items:
                p, _span = build(it, mode, register)
                prompts.append(p)
                targets.append(" " + it["word"])
                controls.append(it["control"])
                keys.append((register, mode, it["id"]))
    for register in ("user", "think"):     # matching no-directive floors
        for it in items:
            tok, _ = C.load()
            head = tok.apply_chat_template(
                [{"role": "user", "content": it["question"]}],
                tokenize=False, add_generation_prompt=True)
            p = head + "</think>\n" + it["prefix"] if register == "user" else \
                head + "Constraint: none.\n</think>\n" + it["prefix"]
            prompts.append(p)
            targets.append(" " + it["word"])
            controls.append(it["control"])
            keys.append((register, "none", it["id"]))

    print(f"{len(prompts)} forward passes", flush=True)
    res = C.probe_full(prompts, targets, controls, batch_size=8)
    by = collections.defaultdict(dict)
    for (reg, mode, iid), r in zip(keys, res):
        by[(reg, mode)][iid] = r

    ids = [it["id"] for it in items]
    out = {}
    for reg in ("user", "think"):
        base = by[(reg, "none")]
        for mode in MODES:
            cell = by[(reg, mode)]
            v = []
            for i in ids:
                a, b = base[i], cell[i]
                v.append((C.logodds(a["logp"]) - C.logodds(b["logp"])) -
                         (C.logodds(a["logp_control"]) - C.logodds(b["logp_control"])))
            lo, hi = C.bootstrap_ci([(x, 0.0) for x in v])
            out[f"{reg}|{mode}"] = dict(
                suppression=C.mean(v), ci=[lo, hi],
                entropy_delta=C.mean(cell[i]["entropy"] - base[i]["entropy"]
                                     for i in ids),
                mean_rank=C.mean(cell[i]["rank"] for i in ids))
    C.dump(os.path.join(HERE, "register.json"), out)
    print("\n register|marker    suppression        95% CI         dH     rank")
    for k, v in out.items():
        print(f"  {k:16} {v['suppression']:+8.4f}  "
              f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]  {v['entropy_delta']:+6.3f} "
              f"{v['mean_rank']:6.2f}")


if __name__ == "__main__":
    main()
