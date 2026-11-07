"""Set-transformer denoiser for splat lists (rectified flow, class-conditional, classifier-free guidance).

Input: K splats x D = 10 normalised features (see splats.py). No positional encoding, so the network
is equivariant to the order of the splats, as the representation requires. Each block is DiT-style
adaLN-Zero: LayerNorm without affine, shift/scale/gate from the conditioning vector.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

D = 10
WIDTH = 256
DEPTH = 8
HEADS = 4
TDIM = 128
NCLASS = 47


def timestep_embedding(t, dim=TDIM):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=1)


class Block(nn.Module):
    def __init__(self, d=WIDTH, heads=HEADS):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, e):
        B, K, d = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(e)[:, None].chunk(6, dim=-1)
        h = F.layer_norm(x, (d,)) * (1 + sc1) + sh1
        q, k, v = self.qkv(h).reshape(B, K, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, K, d)
        x = x + g1 * self.proj(a)
        h = F.layer_norm(x, (d,)) * (1 + sc2) + sh2
        x = x + g2 * self.fc2(F.gelu(self.fc1(h), approximate="tanh"))
        return x


class SplatDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(D, WIDTH)
        self.t1 = nn.Linear(TDIM, WIDTH)
        self.t2 = nn.Linear(WIDTH, WIDTH)
        self.cls = nn.Embedding(NCLASS + 1, WIDTH)
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.ada_out = nn.Linear(WIDTH, 2 * WIDTH)
        self.out = nn.Linear(WIDTH, D)
        for m in (self.ada_out, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def embed(self, t, y):
        return F.silu(self.t2(F.silu(self.t1(timestep_embedding(t)))) + self.cls(y))   # consumers see silu(e)

    def forward(self, x, t, y):
        e = self.embed(t, y)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, e)
        sh, sc = self.ada_out(e)[:, None].chunk(2, dim=-1)
        return self.out(F.layer_norm(h, (WIDTH,)) * (1 + sc) + sh)


@torch.no_grad()
def sample(model, y, K, steps=24, cfg=3.0, seed=0, noise=None, device="cpu"):
    """Euler rectified-flow sampler in normalised feature space, t: 1 -> 0."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (noise.clone() if noise is not None else torch.randn(len(y), K, D, generator=g)).to(device)
    y = y.to(device)
    null = torch.full_like(y, NCLASS)
    ts = torch.linspace(1, 0, steps + 1)
    for i in range(steps):
        t = torch.full((len(y),), float(ts[i]), device=device)
        if cfg == 1.0:
            v = model(x, t, y)
        else:
            vc, vu = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])).chunk(2)
            v = vu + cfg * (vc - vu)
        x = x + (float(ts[i + 1]) - float(ts[i])) * v
    return x


if __name__ == "__main__":
    m = SplatDiT()
    print(f"params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    print(m(torch.randn(2, 512, D), torch.rand(2), torch.tensor([3, 47])).shape)
