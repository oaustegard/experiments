"""Two cheap CPU probes of using the GPU's idle width to cut Monad's sequential depth further.

1. groups: run k consecutive layers in parallel (Layer Parallelism generalized from pairs: all k
   attention blocks read x, all k MLPs read the summed result). KL at equal depth for k = 2, 3, 4.

2. parareal: layer-level speculation that converges to the exact sequential result. Split the 64
   layers into S segments of m layers. A coarse propagator C_s (the segment's m layers run in
   parallel, one fused step) guesses each segment boundary sequentially; the fine propagator G_s
   (the segment's m layers in order) then runs from every guessed boundary at once, and the
   correction U[s+1] <- C_s(U_new[s]) + G_s(U_old[s]) - C_s(U_old[s]) sweeps forward. Parareal
   (Lions, Maday & Turinici 2001) is exact after S iterations; the question is how small the error
   is after 1 or 2. Cost per iteration on a width-free GPU: m fused fine steps + S coarse steps,
   against 64 sequential layers.

Teacher-forced on 8 dev answers of the loop-edited model (A8p), 256 tokens each; KL is to the exact
sequential model's next-token distribution at the answer positions.

    python3 depth_probe.py --out results/depth_probe.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MWE = HERE.parent / "monad-weight-edit"
sys.path.insert(0, str(MWE))

import torch  # noqa: E402

from common import load  # noqa: E402
from fold import calib  # noqa: E402
from layers import kl  # noqa: E402


class Net:
    def __init__(self, model, ids):
        self.m = model
        T = ids.shape[1]
        self.h0 = model.model.embed_tokens(ids)
        self.pe = model.model.rotary_emb(self.h0, torch.arange(T)[None])
        self.mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
        self.L = model.model.layers

    def delta_attn(self, l, x):
        A = self.L[l]
        return A.self_attn(A.input_layernorm(x), self.pe, self.mask)[0]

    def delta_mlp(self, l, h):
        A = self.L[l]
        return A.mlp(A.post_attention_layernorm(h))

    def seq(self, x, ls):
        for l in ls:
            h = x + self.delta_attn(l, x)
            x = h + self.delta_mlp(l, h)
        return x

    def par(self, x, ls):
        h = x + sum(self.delta_attn(l, x) for l in ls)
        return h + sum(self.delta_mlp(l, h) for l in ls)

    def logp(self, x, p):
        lg = self.m.lm_head(self.m.model.norm(x))[0, p - 1:-1]
        return torch.log_softmax(lg.float(), -1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/depth_probe.json")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(4)
    tok, model = load(MWE / "models/A8p")
    seqs = calib(tok, "A8p__gpu_dev.jsonl", n=a.n, L=256)
    nets = []
    for ids, p in seqs:
        net = Net(model, torch.tensor([ids]))
        exact = net.seq(net.h0, range(64))
        ref = model(torch.tensor([ids])).logits[0, p - 1:-1]
        assert float((model.lm_head(model.model.norm(exact))[0, p - 1:-1] - ref).abs().max()) < 1e-3
        nets.append((net, p, net.logp(exact, p)))
    base = torch.cat([r for _, _, r in nets])

    def score(fn):
        return kl(base, torch.cat([fn(net, p) for net, p, _ in nets]))

    out = {"groups": {}, "parareal": {}}
    # 1. groups of k: single-group KL, then lowest non-overlapping groups to a depth saving of 16
    for k in (2, 3, 4):
        single = {}
        for s in range(0, 64 - k + 1):
            def f(net, p, s=s):
                x = net.seq(net.h0, range(s))
                x = net.par(x, range(s, s + k))
                return net.logp(net.seq(x, range(s + k, 64)), p)
            single[s] = score(f)
        chosen = []
        for s in sorted(single, key=single.get):
            if any(not (s + k <= c or c + k <= s) for c in chosen):
                continue
            chosen.append(s)
            if len(chosen) * (k - 1) >= 16:
                break
        chosen.sort()

        def g(net, p, chosen=chosen, k=k):
            x, l = net.h0, 0
            while l < 64:
                if l in chosen:
                    x, l = net.par(x, range(l, l + k)), l + k
                else:
                    x, l = net.seq(x, [l]), l + 1
            return net.logp(x, p)
        out["groups"][k] = {"starts": chosen, "depth": 64 - len(chosen) * (k - 1), "kl": score(g),
                            "best_single": min(single.values())}
        print("group", k, out["groups"][k], flush=True)
    # 2. parareal over depth
    for m in (4, 8, 16):
        S = 64 // m
        seg = [range(s * m, (s + 1) * m) for s in range(S)]
        res = {}
        for K in (0, 1, 2, 3):
            def f(net, p, K=K):
                U = [net.h0]
                for s in range(S):                       # initial coarse sweep
                    U.append(net.par(U[s], seg[s]))
                for _ in range(K):
                    G = [net.seq(U[s], seg[s]) for s in range(S)]          # parallel across segments
                    Cold = [net.par(U[s], seg[s]) for s in range(S)]
                    V = [net.h0]
                    for s in range(S):                                     # correction sweep
                        V.append(net.par(V[s], seg[s]) + G[s] - Cold[s])
                    U = V
                return net.logp(U[S], p)
            res[K] = {"kl": score(f), "cost_layer_steps": S + K * (m + S)}
            print("parareal m", m, "K", K, res[K], flush=True)
        out["parareal"][m] = res
        (HERE / a.out).write_text(json.dumps(out, indent=1))
    (HERE / a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
