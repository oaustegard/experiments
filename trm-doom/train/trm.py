"""Tiny Recursive Model (Jolicoeur-Martineau, arXiv 2510.04871) for path planning on DOOM grids.

One small network, applied recursively: z <- net(x + y + z) n times, then y <- net(y + z), the whole
thing T times (the first T-1 without gradients), under deep supervision with a learned halting head.
Post-norm RMSNorm, SwiGLU and bidirectional attention, no biases, as in the paper; positions are a
learned table plus a learned 2D relative-position bias per head (Swin-style) rather than RoPE: the bias
gives attention the grid's locality from the start, and both are a table lookup in WGSL.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


FLEX = [None]  # set to a compiled flex_attention to use it (training on GPU)


def use_flex():
    from torch.nn.attention.flex_attention import flex_attention
    FLEX[0] = torch.compile(flex_attention, dynamic=False)


def rms(x, eps=1e-5):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def rel_index(n):
    """[L, L] index into a (2n-1)^2 table of (dy, dx) offsets for an n×n grid, row-major."""
    j, i = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    j, i = j.reshape(-1), i.reshape(-1)
    return (j[:, None] - j[None, :] + n - 1) * (2 * n - 1) + (i[:, None] - i[None, :] + n - 1)


class Block(nn.Module):
    def __init__(s, d, heads, mult=4, grid=32):
        super().__init__()
        s.h = heads
        s.rel = nn.Parameter(torch.zeros(heads, (2 * grid - 1) ** 2))
        s.register_buffer("ridx", rel_index(grid), persistent=False)
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.o = nn.Linear(d, d, bias=False)
        hid = mult * d
        s.gu = nn.Linear(d, 2 * hid, bias=False)
        s.down = nn.Linear(hid, d, bias=False)

    def forward(s, x):
        B, L, D = x.shape
        q, k, v = s.qkv(x).view(B, L, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        if FLEX[0] is not None:
            rel = s.rel
            def mod(score, b, h, qi, ki):
                return score + rel[h, (qi // 32 - ki // 32 + 31) * 63 + (qi % 32 - ki % 32 + 31)]
            a = FLEX[0](q, k, v, score_mod=mod)
        else:
            bias = s.rel[:, s.ridx].to(q.dtype)  # [H, L, L]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None])
        a = a.transpose(1, 2).reshape(B, L, D)
        x = rms(x + s.o(a))
        g, u = s.gu(x).chunk(2, -1)
        return rms(x + s.down(F.silu(g) * u))


class ConvBlock(nn.Module):
    """The same post-norm block with a 3×3 convolution over the grid as the token mixer, the way the TRM
    paper swaps attention for an MLP on small fixed grids. Local mixing is what BFS needs, and recurrent
    conv nets are known to learn maze solving quickly (Schwarzschild et al. 2021)."""
    def __init__(s, d, mult=4, grid=32):
        super().__init__()
        s.g = grid
        s.conv = nn.Conv2d(d, d, 3, padding=1, bias=False)
        hid = mult * d
        s.gu = nn.Linear(d, 2 * hid, bias=False)
        s.down = nn.Linear(hid, d, bias=False)

    def forward(s, x):
        B, L, D = x.shape
        c = s.conv(x.transpose(1, 2).reshape(B, D, s.g, s.g)).reshape(B, D, L).transpose(1, 2)
        x = rms(x + c)
        g, u = s.gu(x).chunk(2, -1)
        return rms(x + s.down(F.silu(g) * u))


class TRM(nn.Module):
    def __init__(s, vocab=7, out=2, L=1024, d=128, heads=4, layers=2, n=4, T=3, mixer="attn"):
        super().__init__()
        s.n, s.T, s.d = n, T, d
        s.emb = nn.Embedding(vocab, d)
        s.pos = nn.Parameter(torch.randn(L, d) * (0.5 if mixer == "attn" else 0.0))
        s.blocks = nn.ModuleList((Block(d, heads) if mixer == "attn" else ConvBlock(d)) for _ in range(layers))
        s.head = nn.Linear(d, out, bias=False)
        s.qhead = nn.Linear(d, 1, bias=True)
        s.y0 = nn.Parameter(torch.randn(d) * 0.02)
        s.z0 = nn.Parameter(torch.randn(d) * 0.02)
        nn.init.zeros_(s.qhead.weight)
        nn.init.constant_(s.qhead.bias, -5.0)

    def net(s, h):
        for b in s.blocks:
            h = b(h)
        return h

    def embed(s, tokens):
        return s.emb(tokens) + s.pos

    def init_state(s, B, L, device, dtype):
        y = s.y0.to(dtype).expand(B, L, -1).clone()
        z = s.z0.to(dtype).expand(B, L, -1).clone()
        return y, z

    def latent(s, x, y, z):
        for _ in range(s.n):
            z = s.net(x + y + z)
        y = s.net(y + z)
        return y, z

    def step(s, x, y, z):
        """One supervision step: T-1 recursions without gradients, then one with."""
        with torch.no_grad():
            for _ in range(s.T - 1):
                y, z = s.latent(x, y, z)
        y, z = s.latent(x, y, z)
        logits = s.head(y)
        q = s.qhead(y.mean(1)).squeeze(-1)
        return y.detach(), z.detach(), logits, q
