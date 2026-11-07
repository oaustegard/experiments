"""Additive 2D Gaussian splats on a torus: the representation the diffusion model generates.

A splat is (mu_x, mu_y) in [0, 1) on the unit torus, a lower-triangular precision factor
A = [[e^a, 0], [b, e^c]] in tile units, and an RGB colour c (any sign). The image is
    I(p) = BG + sum_i c_i * exp(-0.5 * |A_i d_i(p)|^2),   d_i(p) = wrap(p - mu_i) in [-0.5, 0.5)^2
Blending is additive, so the order of splats does not matter, and the wrap makes every tile seamless.

For diffusion each splat is a 10-vector: cos/sin of 2*pi*mu_x, cos/sin of 2*pi*mu_y, a, b, c, rgb.
"""
from __future__ import annotations

import math

import torch

BG = 0.0          # images live in [-1, 1]; additive splats sit on a mid-grey background
D = 10            # diffusion features per splat


def render(mu, A, col, H, W, pix=None, chunk=None):
    """mu [B,K,2], A [B,K,3] (a, b, c), col [B,K,3] -> [B,3,H,W] (or [B,3,P] at pixel subset pix [P,2])."""
    if pix is None:
        ys, xs = torch.meshgrid(torch.arange(H, device=mu.device), torch.arange(W, device=mu.device), indexing="ij")
        pix = torch.stack([(xs.reshape(-1) + 0.5) / W, (ys.reshape(-1) + 0.5) / H], -1)
        full = True
    else:
        full = False
    P = pix.shape[0]
    out = []
    step = chunk or P
    for s in range(0, P, step):
        p = pix[s:s + step]                                       # [P', 2]
        d = p[None, None] - mu[:, :, None]                        # [B,K,P',2]
        d = d - torch.round(d)                                    # torus wrap to [-0.5, 0.5]
        a, b, c = A[..., 0:1].exp(), A[..., 1:2], A[..., 2:3].exp()
        u = a * d[..., 0]
        v = b * d[..., 0] + c * d[..., 1]
        g = torch.exp(-0.5 * (u * u + v * v))                     # [B,K,P']
        out.append(BG + torch.einsum("bkp,bkc->bcp", g, col))
    img = torch.cat(out, -1)
    return img.reshape(img.shape[0], 3, H, W) if full else img


def _render_pix(mu, A, col, pix):
    d = pix[None, None] - mu[:, :, None]
    d = d - torch.round(d)
    u = A[..., 0:1].exp() * d[..., 0]
    v = A[..., 1:2] * d[..., 0] + A[..., 2:3].exp() * d[..., 1]
    return BG + torch.einsum("bkp,bkc->bcp", torch.exp(-0.5 * (u * u + v * v)), col)


_compiled = None


def render_pix(mu, A, col, pix):
    """Pixel-subset render for fitting; uses torch.compile on CUDA (one fused kernel), eager elsewhere."""
    global _compiled
    if mu.is_cuda:
        if _compiled is None:
            try:
                _compiled = torch.compile(_render_pix, dynamic=False)
            except Exception:                      # noqa: BLE001 -- fall back to eager on any compile failure
                _compiled = _render_pix
        return _compiled(mu, A, col, pix)
    return _render_pix(mu, A, col, pix)


def to_features(mu, A, col):
    t = 2 * math.pi * mu
    return torch.cat([torch.cos(t[..., :1]), torch.sin(t[..., :1]), torch.cos(t[..., 1:]), torch.sin(t[..., 1:]),
                      A, col], -1)


def from_features(f):
    mx = torch.atan2(f[..., 1], f[..., 0]) / (2 * math.pi) % 1.0
    my = torch.atan2(f[..., 3], f[..., 2]) / (2 * math.pi) % 1.0
    return torch.stack([mx, my], -1), f[..., 4:7], f[..., 7:10]


def fit(target, K, steps=600, n_pix=4096, lr=0.02, seed=0, log=None):
    """Fit K splats to each image in target [B,3,H,W] (values in [-1,1]). Returns mu, A, col and per-image PSNR."""
    B, _, H, W = target.shape
    dev = target.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu = torch.rand(B, K, 2, generator=g).to(dev)
    s0 = math.log(math.sqrt(K) / 1.5)                             # sigma ~ 1.5 / sqrt(K) of the tile
    A = torch.stack([torch.full((B, K), s0), torch.zeros(B, K), torch.full((B, K), s0)], -1).to(dev)
    # initial colour: the target at the splat centre, shared over the ~2 splats expected to overlap
    ix = (mu[..., 0] * W).long().clamp(0, W - 1)
    iy = (mu[..., 1] * H).long().clamp(0, H - 1)
    col = target[torch.arange(B, device=dev)[:, None], :, iy, ix] * 0.4
    mu, A, col = (t.clone().requires_grad_(True) for t in (mu, A, col))
    opt = torch.optim.Adam([{"params": [mu], "lr": lr * 0.1}, {"params": [A], "lr": lr}, {"params": [col], "lr": lr}])
    flat = target.reshape(B, 3, -1)
    for i in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = gp.setdefault("lr0", gp["lr"]) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * i / steps)))
        idx = torch.randint(H * W, (n_pix,), device=dev)
        pix = torch.stack([((idx % W).float() + 0.5) / W, ((idx // W).float() + 0.5) / H], -1)
        pred = render_pix(mu, A, col, pix)
        loss = ((pred - flat[:, :, idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu.remainder_(1.0)
            A[..., 0].clamp_(max=math.log(H * 1.0)); A[..., 2].clamp_(max=math.log(H * 1.0))   # no splat below ~1 px
        if log and (i % 100 == 0 or i == steps - 1):
            log(i, float(loss))
    with torch.no_grad():
        full = render(mu, A, col, H, W, chunk=4096).clamp(-1, 1)
        mse = ((full - target) ** 2).mean((1, 2, 3))
        psnr = 10 * torch.log10(4 / mse)
    return mu.detach(), A.detach(), col.detach(), psnr
