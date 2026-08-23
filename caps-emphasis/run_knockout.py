"""Causal check on the attention story.

Correlating attention mass with case does not show the attention is doing the
work; the faithfulness literature (Jain & Wallace 2019, Wiegreffe & Pinter 2019)
rules that out explicitly. This zeroes the final query position's attention onto
the directive span, one layer at a time, and re-reads log P of the forbidden
word. If the span's attention carries the effect, knocking it out removes it.
"""
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import caps_lib as C
from render import render

FRAME = "{K} mention the word {W}."
KEYWORD = "never"
MODES = ["title", "caps", "bold"]


def knockout_logp(prompt, target, a, b, layer_idx):
    """Re-run with layer `layer_idx`'s attention from the last query position
    onto tokens [a,b) masked out. Implemented as an additive -inf bias on those
    key positions for that layer only."""
    tok, m = C.load(eager=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    layers = m.model.layers
    handle = {}

    def hook(mod, args, kwargs):
        mask = kwargs.get("attention_mask")
        n = ids.shape[1]
        bias = torch.zeros(1, 1, n, n)
        bias[0, 0, -1, a:b] = float("-inf")
        kwargs["attention_mask"] = bias if mask is None else mask + bias
        return args, kwargs

    h = layers[layer_idx].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            lp = torch.log_softmax(m(ids).logits[0, -1].float(), -1)
    finally:
        h.remove()
    return lp[tok.encode(target, add_special_tokens=False)[0]].item()


def main():
    items = json.load(open(os.path.join(HERE, "items_v2.json")))["items"][:8]
    tok, m = C.load(eager=True)
    n_layers = len(m.model.layers)
    print(f"{n_layers} layers x {len(MODES)} modes x {len(items)} items")

    out = {}
    for mode in MODES:
        base, per_layer = [], collections.defaultdict(list)
        att_mass = []
        for it in items:
            d, span = render(FRAME, KEYWORD, it["word"], mode)
            prompt = C.build_prompt(d + " " + it["question"], it["prefix"])
            rng = C.span_range(prompt, span)
            if rng is None:
                print(f"  span {span!r} unresolved for {it['id']}")
                continue
            a, b = rng
            r = C.attention_to_span(prompt, span)
            att_mass.append(dict(span_tokens=r[2], total=sum(r[0]),
                                 per_token=sum(r[1])))
            tgt = " " + it["word"]
            clean = C.logprobs([prompt], [tgt])[0]
            base.append(clean)
            for L in range(n_layers):
                per_layer[L].append(knockout_logp(prompt, tgt, a, b, L) - clean)
            print(f"  {mode:6} {it['id']:14} done", flush=True)
        out[mode] = dict(
            span=span, n_items=len(base), base_logp=C.mean(base),
            span_tokens=C.mean(x["span_tokens"] for x in att_mass),
            attention_total=C.mean(x["total"] for x in att_mass),
            attention_per_token=C.mean(x["per_token"] for x in att_mass),
            knockout_by_layer=[C.mean(per_layer[L]) for L in range(n_layers)])
        eff = out[mode]["knockout_by_layer"]
        top = sorted(range(n_layers), key=lambda L: -abs(eff[L]))[:5]
        print(f"  {mode}: strongest layers {[(L, round(eff[L],3)) for L in top]}")
        C.dump(os.path.join(HERE, "knockout.json"), out)


if __name__ == "__main__":
    main()
