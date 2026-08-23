"""Behavioural check: does the forbidden word actually appear in the answer?

Two arms. `framed` forces the assistant turn past the think block and lets the
model write the answer freely (cheap, all items). `free` lets the model produce
its own think block and answer (expensive, subset) so the framed arm can be
checked against unconstrained behaviour.
"""
import sys, os, json, re, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import caps_lib as C
from conditions import MATCHED

CONDS = [("none", None)] + [(n, t) for n, t, _s in MATCHED]


def violates(text, word):
    return re.search(r"\b" + re.escape(word) + r"\b", text, re.I) is not None


@torch.no_grad()
def framed_answer(user, prefix, max_new=48):
    tok, m = C.load()
    p = C.build_prompt(user, prefix)
    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
    out = m.generate(ids, max_new_tokens=max_new, do_sample=False,
                     pad_token_id=C.pad_id(tok))
    return prefix + tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["framed", "free"], default="framed")
    ap.add_argument("--n-items", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args()

    items = json.load(open(os.path.join(HERE, "items_screened.json")))["kept"]
    if a.n_items:
        items = items[:a.n_items]
    rows, t0 = [], time.time()
    for cname, tmpl in CONDS:
        for it in items:
            user = it["question"] if tmpl is None else \
                tmpl.format(W=it["word"]) + " " + it["question"]
            if a.arm == "framed":
                text = framed_answer(user, it["prefix"], a.max_new)
                answer, closed = text, True
            else:
                text = C.generate(user, max_new_tokens=a.max_new)
                closed = "</think>" in text
                answer = text.split("</think>", 1)[1] if closed else ""
            rows.append(dict(cond=cname, id=it["id"], word=it["word"],
                             violated=violates(answer, it["word"]),
                             think_closed=closed, text=text[:900]))
            print(f"  {cname:12} {it['id']:14} viol={rows[-1]['violated']} "
                  f"closed={closed} {time.time()-t0:6.0f}s", flush=True)
        C.dump(os.path.join(HERE, f"generation_{a.arm}.json"),
               dict(arm=a.arm, max_new=a.max_new, n_items=len(items), rows=rows))

    print("\n== violation rate (word appears in the answer)")
    for cname, _t in CONDS:
        r = [x for x in rows if x["cond"] == cname]
        scored = [x for x in r if x["think_closed"]]
        v = sum(x["violated"] for x in scored)
        print(f"  {cname:12} {v}/{len(scored)} = "
              f"{v / len(scored) if scored else float('nan'):.3f}"
              f"   (unclosed think: {len(r) - len(scored)})")


if __name__ == "__main__":
    main()
