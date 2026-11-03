"""Premise check for a follow-up: run adjacent layer pairs of Monad in parallel (Layer Parallelism,
Calvo González et al. 2025, arXiv 2502.02790), in the form that folds into one standard wider layer:

    h = x + A_k(n_k(x)) + A_k+1(n_k+1(x))          both attention blocks read the same input
    y = h + F_k(m_k(h)) + F_k+1(m_k+1(h))          both MLPs read the same input

On a dispatch-bound GPU a pair costs about one layer, so sequential depth drops by one per pair.
Measured exactly as layers.py measures layer removal (KL to the full model's next-token
distribution on its own dev outputs, same 12 sequences x 320 tokens), so the two are comparable.

    python3 lp_probe.py --out results/lp_probe.json
"""
import argparse
import json

import torch
import torch.nn as nn

from common import HERE, load
from layers import calib, kl


class Pair(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, x, attention_mask=None, position_embeddings=None, **kw):
        A, B = self.a, self.b
        h = (x + A.self_attn(A.input_layernorm(x), position_embeddings, attention_mask)[0]
             + B.self_attn(B.input_layernorm(x), position_embeddings, attention_mask)[0])
        return h + A.mlp(A.post_attention_layernorm(h)) + B.mlp(B.post_attention_layernorm(h))


def build(layers, pairs):
    starts = {p for p in pairs}
    out, l = [], 0
    while l < len(layers):
        if l in starts:
            out.append(Pair(layers[l], layers[l + 1]))
            l += 2
        else:
            out.append(layers[l])
            l += 1
    return nn.ModuleList(out)


@torch.no_grad()
def run(model, seqs, layers, pairs):
    model.model.layers = build(layers, pairs)
    outs = []
    for ids, p in seqs:
        lg = model(torch.tensor([ids])).logits[0, p - 1:-1]
        outs.append(torch.log_softmax(lg.float(), -1))
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--len", type=int, default=320)
    ap.add_argument("--out", default="results/lp_probe.json")
    a = ap.parse_args()
    torch.set_num_threads(4)
    tok, model = load()
    layers = list(model.model.layers)
    seqs = calib(tok, a.n, a.len)
    base = run(model, seqs, layers, [])
    single = {k: kl(base, run(model, seqs, layers, [k])) for k in range(len(layers) - 1)}
    print("single pairs done", flush=True)
    order = sorted(single, key=single.get)
    chosen, cum = [], {}
    for k in order:                       # one-shot: lowest single-pair KL first, no overlaps
        if any(abs(k - c) < 2 for c in chosen):
            continue
        chosen.append(k)
        if len(chosen) in (4, 8, 12, 16, 20, 24, 28, 32):
            cum[len(chosen)] = {"pairs": sorted(chosen), "kl": kl(base, run(model, seqs, layers, chosen))}
            print(len(chosen), round(cum[len(chosen)]["kl"], 4), flush=True)
    rem = json.loads((HERE / "results/layers.json").read_text())
    out = {"single_pair_kl": single, "min_single_pair": min(single.values()),
           "median_single_pair": sorted(single.values())[len(single) // 2],
           "oneshot": cum, "removal_greedy_kl": {g["k"]: g["kl"] for g in rem["greedy"]},
           "removal_single_min": min(rem["single_kl"])}
    (HERE / a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
