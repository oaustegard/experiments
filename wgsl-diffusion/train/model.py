"""Small class-conditional conv U-Net for rectified-flow texture generation.

No attention, so the network is translation-equivariant and runs at any size that is
a multiple of 8. Training uses zero padding; at inference the padding can be switched
to circular, which makes the output tile seamlessly.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

CH = (64, 128, 192, 256)
EMB = 256
TDIM = 128
GROUPS = 16
NCLASS = 47          # DTD; index NCLASS is the null (unconditional) class
PAD = {"mode": "zeros"}   # module-level switch read by Conv3


class Conv3(nn.Conv2d):
    """3x3 conv whose padding mode is chosen at call time (zeros or circular)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__(cin, cout, 3, stride=stride, padding=0)

    def forward(self, x):
        if PAD["mode"] == "circular":
            x = F.pad(x, (1, 1, 1, 1), mode="circular")
        else:
            x = F.pad(x, (1, 1, 1, 1))
        return F.conv2d(x, self.weight, self.bias, self.stride)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1 = nn.GroupNorm(GROUPS, cin)
        self.c1 = Conv3(cin, cout)
        self.emb = nn.Linear(EMB, 2 * cout)
        self.n2 = nn.GroupNorm(GROUPS, cout)
        self.c2 = Conv3(cout, cout)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.emb(e)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.c2(F.silu(h))
        return h + (self.skip(x) if self.skip is not None else x)


def timestep_embedding(t, dim=TDIM):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=1)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        c0, c1, c2, c3 = CH
        self.t1 = nn.Linear(TDIM, EMB)
        self.t2 = nn.Linear(EMB, EMB)
        self.cls = nn.Embedding(NCLASS + 1, EMB)
        self.conv_in = Conv3(3, c0)
        self.e0 = ResBlock(c0, c0)
        self.d0 = Conv3(c0, c0, stride=2)
        self.e1 = ResBlock(c0, c1)
        self.d1 = Conv3(c1, c1, stride=2)
        self.e2 = ResBlock(c1, c2)
        self.d2 = Conv3(c2, c2, stride=2)
        self.e3 = ResBlock(c2, c3)
        self.mid = ResBlock(c3, c3)
        self.u3 = ResBlock(c3 + c3, c3)
        self.u2 = ResBlock(c3 + c2, c2)
        self.u1 = ResBlock(c2 + c1, c1)
        self.u0 = ResBlock(c1 + c0, c0)
        self.n_out = nn.GroupNorm(GROUPS, c0)
        self.conv_out = Conv3(c0, 3)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def embed(self, t, y):
        e = self.t2(F.silu(self.t1(timestep_embedding(t)))) + self.cls(y)
        return F.silu(e)       # every consumer applies silu first, so do it once here

    def forward(self, x, t, y):
        e = self.embed(t, y)
        h0 = self.e0(self.conv_in(x), e)
        h1 = self.e1(self.d0(h0), e)
        h2 = self.e2(self.d1(h1), e)
        h3 = self.e3(self.d2(h2), e)
        h = self.mid(h3, e)
        h = self.u3(torch.cat([h, h3], 1), e)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.u2(torch.cat([h, h2], 1), e)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.u1(torch.cat([h, h1], 1), e)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.u0(torch.cat([h, h0], 1), e)
        return self.conv_out(F.silu(self.n_out(h)))


@torch.no_grad()
def sample(model, y, H, W, steps=24, cfg=3.0, seed=0, pad="zeros", device="cpu", noise=None):
    """Euler rectified-flow sampler, t: 1 -> 0. y: LongTensor of class ids."""
    old = PAD["mode"]
    PAD["mode"] = pad
    try:
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = noise.clone() if noise is not None else torch.randn(len(y), 3, H, W, generator=g)
        x = x.to(device)
        y = y.to(device)
        null = torch.full_like(y, NCLASS)
        ts = torch.linspace(1, 0, steps + 1)
        for i in range(steps):
            t = torch.full((len(y),), float(ts[i]), device=device)
            if cfg == 1.0:
                v = model(x, t, y)
            else:
                v2 = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))
                vc, vu = v2.chunk(2)
                v = vu + cfg * (vc - vu)
            x = x + (float(ts[i + 1]) - float(ts[i])) * v
        return x.clamp(-1, 1)
    finally:
        PAD["mode"] = old


if __name__ == "__main__":
    m = UNet()
    n = sum(p.numel() for p in m.parameters())
    print(f"params {n/1e6:.2f}M")
    x = torch.randn(2, 3, 64, 64)
    print(m(x, torch.rand(2), torch.tensor([0, 47])).shape)
