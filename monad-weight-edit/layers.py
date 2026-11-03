"""How much each of Monad's 64 layers changes the output, measured on Monad's own generations.

    python3 layers.py [--n 16] [--len 384] --out results/layers.json

Per layer l:
  bi    block influence (ShortGPT, Men et al. 2024): 1 - mean cos(h_in, h_out) of the residual stream
  kl    mean KL(full || full-without-layer-l) of the next-token distribution over the generated
        tokens: what removing that one layer costs, measured where it matters
Then the KL of removing the k lowest-kl layers together, k = 4, 8, ..., 32, both by one-shot
ranking and by greedy re-ranking (remove one, re-measure, repeat).
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import HERE, load, prompt
from run_eval import question

FX = json.loads((HERE / "data/fixture.json").read_text())


def calib(tok, n, L):
    """Dev prompts + the base model's own dev generations, cut to L tokens. Returns ids and the
    index where the generation starts in each row."""
    recs = [json.loads(l) for l in open(HERE / "runs/base__dev.jsonl")]
    fx = {x["uid"]: x for x in FX["dev"]}
    seqs = []
    per_task = {}
    for r in recs:
        per_task.setdefault(r["task"], []).append(r)
    pick = []
    while len(pick) < n:
        for t in ("mmlu", "gsm8k", "ifeval"):
            if per_task.get(t):
                pick.append(per_task[t].pop(0))
    for r in pick[:n]:
        p = tok.encode(prompt(question(fx[r["uid"]])), add_special_tokens=False)
        g = tok.encode(r["text"], add_special_tokens=False)
        ids = (p + g)[:L]
        seqs.append((ids, len(p)))
    return seqs


@torch.no_grad()
def run(model, seqs, skip=()):
    """Log-probs at generated positions (concatenated), and per-layer cos stats."""
    keep = [i for i in range(len(model.model.layers_all)) if i not in skip]
    model.model.layers = torch.nn.ModuleList([model.model.layers_all[i] for i in keep])
    outs = []
    for ids, p in seqs:
        lg = model(torch.tensor([ids])).logits[0, p - 1:-1]
        outs.append(F.log_softmax(lg.float(), -1))
    return torch.cat(outs)


def kl(base, other):
    return float((base.exp() * (base - other)).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--len", type=int, default=384)
    ap.add_argument("--out", default="results/layers.json")
    ap.add_argument("--greedy-to", type=int, default=16)
    ap.add_argument("--pool", type=int, default=24)
    a = ap.parse_args()
    torch.set_num_threads(4)
    tok, model = load()
    model.model.layers_all = model.model.layers
    nL = len(model.model.layers_all)
    seqs = calib(tok, a.n, a.len)
    # block influence
    bi = torch.zeros(nL)
    cnt = 0
    with torch.no_grad():
        for ids, p in seqs:
            hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
            for l in range(nL):
                bi[l] += (1 - F.cosine_similarity(hs[l][0], hs[l + 1][0], dim=-1)).sum()
            cnt += len(ids)
    bi = (bi / cnt).tolist()
    base = run(model, seqs)
    single = [kl(base, run(model, seqs, skip=(l,))) for l in range(nL)]
    print("single-layer KL done", flush=True)
    order = sorted(range(nL), key=lambda l: single[l])
    oneshot = {k: kl(base, run(model, seqs, skip=tuple(order[:k]))) for k in range(4, 33, 4)}
    removed, greedy = [], []
    for step in range(a.greedy_to):
        best = None
        for l in order[:a.pool]:     # candidates: the lowest single-layer KLs
            if l in removed:
                continue
            v = kl(base, run(model, seqs, skip=tuple(removed + [l])))
            if best is None or v < best[1]:
                best = (l, v)
        removed.append(best[0])
        greedy.append({"k": len(removed), "layer": best[0], "kl": best[1]})
        print(greedy[-1], flush=True)
        Path(HERE / a.out).write_text(json.dumps({"bi": bi, "single_kl": single, "order": order, "oneshot": oneshot,
                                                  "greedy": greedy, "n_seqs": len(seqs),
                                                  "n_tokens": int(base.shape[0])}, indent=1))


if __name__ == "__main__":
    main()
