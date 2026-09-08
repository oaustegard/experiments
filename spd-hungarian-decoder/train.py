"""Train the three student arms of PLAN.md on MSLR-WEB10K slates.

M1  self-attention head (L=2) -> M in R^{50x50}, Sinkhorn CE   (the paper's winner)
M2  linear probe            -> M in R^{50x50}, Sinkhorn CE     (their Fig. 2a)
M3  self-attention head (L=2) -> one score per item, ListMLE   (the control: no matrix, no solver)

All three share the per-item encoder that stands in for the LLM prefill
hidden state, and all three distill the same pre-computed teacher permutation.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SEED = 20260908
CKPT_EPOCHS = (1, 2, 5, 10, 20)


class Encoder(nn.Module):
    """Stands in for the LLM backbone's per-item readout hidden state."""
    def __init__(self, d_in=136, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, d), nn.LayerNorm(d)
        )

    def forward(self, x):
        return self.net(x)


class AttnHead(nn.Module):
    def __init__(self, d=128, K=50, layers=2, out=None):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=256,
                                         batch_first=True, dropout=0.0,
                                         norm_first=True)
        self.attn = nn.TransformerEncoder(enc, num_layers=layers)
        self.proj = nn.Linear(d, out if out is not None else K)

    def forward(self, h):
        return self.proj(self.attn(h))


class ProbeHead(nn.Module):
    """No cross-item path at all: each item scored on its own."""
    def __init__(self, d=128, K=50):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, K))

    def forward(self, h):
        return self.net(h)


class Student(nn.Module):
    def __init__(self, head, d_in=136, d=128, K=50):
        super().__init__()
        self.enc = Encoder(d_in, d)
        if head == "attn":
            self.head = AttnHead(d, K)
        elif head == "probe":
            self.head = ProbeHead(d, K)
        elif head == "scalar":
            self.head = AttnHead(d, K, out=1)
        else:
            raise ValueError(head)
        self.kind = head

    def forward(self, x):
        out = self.head(self.enc(x))
        return out.squeeze(-1) if self.kind == "scalar" else out


def log_sinkhorn(M, n_iter=20, tau=1.0):
    """Log-space Sinkhorn: alternating row/column log-normalisation (Eq. 2)."""
    logS = M / tau
    for _ in range(n_iter):
        logS = logS - torch.logsumexp(logS, dim=-1, keepdim=True)
        logS = logS - torch.logsumexp(logS, dim=-2, keepdim=True)
    return logS


def sinkhorn_ce(M, teacher_perm, n_iter=20, tau=1.0):
    """Eq. 3: -sum_ij P^teacher_ij log S_ij, with P a permutation matrix.

    teacher_perm[b, r] = index of the item the teacher places at rank r,
    so the target position of item teacher_perm[b, r] is r.
    """
    B, N, K = M.shape
    logS = log_sinkhorn(M, n_iter, tau)
    target_pos = torch.empty(B, N, dtype=torch.long, device=M.device)
    ranks = torch.arange(K, device=M.device).expand(B, K)
    target_pos.scatter_(1, teacher_perm.long(), ranks)
    return -logS.gather(2, target_pos.unsqueeze(-1)).squeeze(-1).mean()


def listmle(scores, teacher_perm):
    """Plackett-Luce NLL of the teacher permutation under the student's scores."""
    s = scores.gather(1, teacher_perm.long())          # scores in teacher order
    rev_lse = torch.logcumsumexp(s.flip(-1), dim=-1).flip(-1)
    return (rev_lse - s).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["M1", "M2", "M3"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--sinkhorn-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    z = np.load(HERE / "data" / "slates.npz")
    mu, sd = z["mu"], z["sd"]
    X = torch.from_numpy((z["Xtr"] - mu) / sd)
    P = torch.from_numpy(z["teacher_perm_tr"].astype(np.int64))
    n, N, _ = X.shape

    head = {"M1": "attn", "M2": "probe", "M3": "scalar"}[a.arm]
    model = Student(head, d_in=X.shape[-1], K=N)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"{a.arm}: head={head}  params={nparam/1e6:.2f}M  slates={n}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((n + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ck = HERE / "checkpoints"; ck.mkdir(exist_ok=True)
    g = torch.Generator().manual_seed(a.seed)

    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            out = model(X[idx])
            loss = (listmle(out, P[idx]) if a.arm == "M3"
                    else sinkhorn_ce(out, P[idx], a.sinkhorn_iters, a.tau))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep:3d}  loss {tot/nb:.4f}")
        if ep in CKPT_EPOCHS:
            torch.save({"arm": a.arm, "head": head, "epoch": ep, "K": N,
                        "d_in": X.shape[-1], "state": model.state_dict()},
                       ck / f"{a.arm}{a.tag}_ep{ep}.pt")
    print("done")


if __name__ == "__main__":
    main()
