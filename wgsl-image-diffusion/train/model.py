"""Class-conditional U-Net with self-attention for rectified-flow image generation (AFHQv2, 64x64).

The wgsl-diffusion texture U-Net plus what images need: two ResBlocks per level, wider channels, and
multi-head self-attention after every block at 16x16 and 8x8 so distant parts of the picture can agree
(two eyes, one face). Every op is one the WGSL runtime has: 3x3 and 1x1 convs, GroupNorm, FiLM, SiLU,
nearest upsample, concat and attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

CH = (128, 192, 256, 256)   # channels at 64, 32, 16, 8
ATTN = (False, False, True, True)
HEAD_DIM = 64
EMB = 256
TDIM = 128
GROUPS = 16
NCLASS = 3                  # cat, dog, wild; index NCLASS is the null class
NAMES = ["cat", "dog", "wild"]


def conv3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1 = nn.GroupNorm(GROUPS, cin)
        self.c1 = conv3(cin, cout)
        self.emb = nn.Linear(EMB, 2 * cout)
        self.n2 = nn.GroupNorm(GROUPS, cout)
        self.c2 = conv3(cout, cout)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.emb(e)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.c2(F.silu(h))
        return h + (self.skip(x) if self.skip is not None else x)


class Attn(nn.Module):
    """x + proj(MHA(GroupNorm(x))); q, k, v from one 1x1 conv, channels [q | k | v], head-major."""

    def __init__(self, c):
        super().__init__()
        self.heads = c // HEAD_DIM
        self.norm = nn.GroupNorm(GROUPS, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, HEAD_DIM, H * W).transpose(3, 4)
        o = F.scaled_dot_product_attention(qkv[:, 0], qkv[:, 1], qkv[:, 2])   # B, heads, L, D
        return x + self.proj(o.transpose(2, 3).reshape(B, C, H, W))


def timestep_embedding(t, dim=TDIM):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=1)


def layout():
    """The forward pass as a flat list the WGSL runtime mirrors: (kind, name, cin, cout).

    kinds: conv_in, res, attn, down, upcat (concat the skip of that level, upsampling first when up=1),
    out. Kept here so model.py and web/diffusion.js can be checked against one list.
    """
    ops = [("conv_in", "conv_in", 3, CH[0])]
    c = CH[0]
    for lv, ch in enumerate(CH):
        for b in "ab":
            ops.append(("res", f"e{lv}{b}", c, ch)); c = ch
            if ATTN[lv]:
                ops.append(("attn", f"e{lv}{b}_at", ch, ch))
        ops.append(("skip", f"h{lv}", ch, ch))
        if lv < len(CH) - 1:
            ops.append(("down", f"d{lv}", ch, ch))
    ops += [("res", "m1", c, c), ("attn", "m_at", c, c), ("res", "m2", c, c)]
    for lv in reversed(range(len(CH))):
        ch = CH[lv]
        ops.append(("upcat", f"h{lv}", c, c + ch))
        ops.append(("res", f"u{lv}a", c + ch, ch)); c = ch
        if ATTN[lv]:
            ops.append(("attn", f"u{lv}a_at", ch, ch))
        ops.append(("res", f"u{lv}b", ch, ch))
        if ATTN[lv]:
            ops.append(("attn", f"u{lv}b_at", ch, ch))
    ops.append(("out", "out", c, 3))
    return ops


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = nn.Linear(TDIM, EMB)
        self.t2 = nn.Linear(EMB, EMB)
        self.cls = nn.Embedding(NCLASS + 1, EMB)
        self.ops = layout()
        for kind, name, cin, cout in self.ops:
            if kind == "conv_in":
                self.conv_in = conv3(cin, cout)
            elif kind == "res":
                setattr(self, name, ResBlock(cin, cout))
            elif kind == "attn":
                setattr(self, name, Attn(cout))
            elif kind == "down":
                setattr(self, name, conv3(cin, cout, stride=2))
            elif kind == "out":
                self.n_out = nn.GroupNorm(GROUPS, cin)
                self.conv_out = conv3(cin, cout)
                nn.init.zeros_(self.conv_out.weight)
                nn.init.zeros_(self.conv_out.bias)

    def embed(self, t, y):
        e = self.t2(F.silu(self.t1(timestep_embedding(t)))) + self.cls(y)
        return F.silu(e)       # every consumer applies silu first, so do it once here

    def forward(self, x, t, y):
        e = self.embed(t, y)
        skips = {}
        h = x
        for kind, name, _, _ in self.ops:
            if kind == "conv_in":
                h = self.conv_in(h)
            elif kind == "res":
                h = getattr(self, name)(h, e)
            elif kind in ("attn", "down"):
                h = getattr(self, name)(h)
            elif kind == "skip":
                skips[name] = h
            elif kind == "upcat":
                s = skips[name]
                if h.shape[-1] != s.shape[-1]:
                    h = F.interpolate(h, scale_factor=2, mode="nearest")
                h = torch.cat([h, s], 1)
            elif kind == "out":
                h = self.conv_out(F.silu(self.n_out(h)))
        return h


@torch.no_grad()
def sample(model, y, H=64, W=64, steps=20, cfg=2.0, seed=0, device="cpu", noise=None):
    """Euler rectified-flow sampler, t: 1 -> 0. y: LongTensor of class ids."""
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


if __name__ == "__main__":
    m = UNet()
    n = sum(p.numel() for p in m.parameters())
    print(f"params {n/1e6:.2f}M")
    for op in m.ops:
        print(op)
    x = torch.randn(2, 3, 64, 64)
    print(m(x, torch.rand(2), torch.tensor([0, NCLASS])).shape)
