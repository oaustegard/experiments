"""Opaque, layered 2D Gaussian splats on a torus: the representation the diffusion model generates (v2).

A splat has a position mu in [0, 1)^2 on the unit torus, a lower-triangular precision factor
A = [[e^a, 0], [b, e^c]] (tile units), an RGB colour in [-1, 1], an opacity logit o, an edge-sharpness
s and a depth z. Its footprint is  g = exp(-0.5 * q^p),  q = |A d|^2,  p = 1 + softplus(s), so p = 1 is a
Gaussian and larger p gives a flat top with a crisp edge. Splats are composited front to back in order
of depth (largest z in front):
    C(x) = sum_i T_i(x) w_i(x) c_i + T_end(x) BG,   w_i = 0.99 sigmoid(o_i) g_i,   T_i = prod_{j in front} (1 - w_j)
The depth is an ordinary feature, so the model still sees an unordered set; only the renderer sorts.
Distances wrap around the tile, so every texture tiles without a seam.

For diffusion each splat is D = 13 numbers: cos/sin of 2 pi mu_x and mu_y, a, b, c, rgb, o, s, z.
In this module splats are kept back-to-front along K (index K-1 in front); z = (index + 0.5) / K.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BG = 0.0
D = 13
SIGMA_CAP = 0.2       # fitting penalises a splat's long axis beyond this fraction of the tile (no needles)


def _weights(mu, A, o, s, pix):
    d = pix[None, None] - mu[:, :, None]                          # [B,K,P,2]
    d = d - torch.round(d)
    u = A[..., 0:1].exp() * d[..., 0]
    v = A[..., 1:2] * d[..., 0] + A[..., 2:3].exp() * d[..., 1]
    q = u * u + v * v
    p = 1.0 + F.softplus(s)[..., None]
    g = torch.exp(-0.5 * torch.pow(q + 1e-8, p))
    return 0.99 * torch.sigmoid(o)[..., None] * g                 # [B,K,P]


def _render_pix(mu, A, col, o, s, pix):
    """Splats ordered back-to-front along K. Returns [B,3,P]."""
    w = _weights(mu, A, o, s, pix)
    lt = torch.log1p(-w)                                          # log(1 - w)
    incl = torch.flip(torch.cumsum(torch.flip(lt, [1]), 1), [1])  # sum over j >= i (i and everything in front)
    T = torch.exp(incl - lt)                                      # transmittance in front of splat i
    out = torch.einsum("bkp,bkc->bcp", T * w, col)
    return out + BG * torch.exp(incl[:, :1])


_compiled = None


def render_pix(mu, A, col, o, s, pix):
    global _compiled
    if mu.is_cuda:
        if _compiled is None:
            try:
                _compiled = torch.compile(_render_pix, dynamic=False)
            except Exception:                      # noqa: BLE001 -- fall back to eager on any compile failure
                _compiled = _render_pix
        return _compiled(mu, A, col, o, s, pix)
    return _render_pix(mu, A, col, o, s, pix)


def grid(H, W, device):
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    return torch.stack([(xs.reshape(-1) + 0.5) / W, (ys.reshape(-1) + 0.5) / H], -1)


def render(mu, A, col, o, s, H, W, chunk=2048):
    """Full image [B,3,H,W] from back-to-front ordered splats."""
    pix = grid(H, W, mu.device)
    out = [_render_pix(mu, A, col, o, s, pix[i:i + chunk]) for i in range(0, pix.shape[0], chunk)]
    return torch.cat(out, -1).reshape(mu.shape[0], 3, H, W)


def sort_by_depth(mu, A, col, o, s, z):
    """Reorder generated splats back-to-front by z (ascending)."""
    idx = torch.argsort(z, dim=1)
    take = lambda t: torch.gather(t, 1, idx[..., None].expand(-1, -1, t.shape[-1])) if t.dim() == 3 else torch.gather(t, 1, idx)
    return take(mu), take(A), take(col), take(o), take(s), take(z)


def sigma_max(A):
    """Long-axis standard deviation of each splat's footprint (tile units)."""
    a, b, c = A[..., 0].exp(), A[..., 1], A[..., 2].exp()
    tr = a * a + b * b + c * c
    det = a * c
    smin2 = tr / 2 - torch.sqrt(torch.clamp(tr * tr / 4 - det * det, min=0.0))
    return 1.0 / torch.sqrt(torch.clamp(smin2, min=1e-8))


def to_features(mu, A, col, o, s, z):
    t = 2 * math.pi * mu
    return torch.cat([torch.cos(t[..., :1]), torch.sin(t[..., :1]), torch.cos(t[..., 1:]), torch.sin(t[..., 1:]),
                      A, col, o[..., None], s[..., None], z[..., None]], -1)


def from_features(f):
    mx = torch.atan2(f[..., 1], f[..., 0]) / (2 * math.pi) % 1.0
    my = torch.atan2(f[..., 3], f[..., 2]) / (2 * math.pi) % 1.0
    return torch.stack([mx, my], -1), f[..., 4:7], f[..., 7:10], f[..., 10], f[..., 11], f[..., 12]


def fit(target, K, steps=1000, n_pix=4096, lr=0.03, seed=0, cap=SIGMA_CAP, cap_weight=1.0):
    """Fit K opaque layered splats to each image in target [B,3,H,W] in [-1,1].

    Splats start back-to-front with sizes shrinking toward the front (coarse layers behind fine ones).
    Returns (mu, A, col, o, s, z, psnr), splats ordered back-to-front.
    """
    B, _, H, W = target.shape
    dev = target.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu = torch.rand(B, K, 2, generator=g).to(dev)
    frac = (torch.arange(K, dtype=torch.float32) + 0.5) / K
    sig = (2.5 - 1.8 * frac) / math.sqrt(K)                       # back splats larger, front ones smaller
    la = (-torch.log(sig))[None].expand(B, K).to(dev)
    A = torch.stack([la, torch.zeros(B, K, device=dev), la], -1).clone()
    ix = (mu[..., 0] * W).long().clamp(0, W - 1)
    iy = (mu[..., 1] * H).long().clamp(0, H - 1)
    col = target[torch.arange(B, device=dev)[:, None], :, iy, ix].clone()
    o = torch.full((B, K), 0.5, device=dev)
    s = torch.full((B, K), -1.0, device=dev)
    params = [mu, A, col, o, s]
    for t in params:
        t.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [mu], "lr": lr * 0.1}, {"params": [A, col, o, s], "lr": lr}])
    flat = target.reshape(B, 3, -1)
    for i in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = gp.setdefault("lr0", gp["lr"]) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * i / steps)))
        idx = torch.randint(H * W, (n_pix,), device=dev)
        pix = torch.stack([((idx % W).float() + 0.5) / W, ((idx // W).float() + 0.5) / H], -1)
        pred = render_pix(mu, A, col, o, s, pix)
        loss = ((pred - flat[:, :, idx]) ** 2).mean()
        loss = loss + cap_weight * (F.relu(sigma_max(A) - cap) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu.remainder_(1.0)
            A[..., 0].clamp_(max=math.log(H * 1.0)); A[..., 2].clamp_(max=math.log(H * 1.0))
            col.clamp_(-1.5, 1.5)
            s.clamp_(-4.0, 4.0)
    z = frac[None].expand(B, K).to(dev).clone()
    with torch.no_grad():
        full = render(mu, A, col, o, s, H, W).clamp(-1, 1)
        psnr = 10 * torch.log10(4 / ((full - target) ** 2).mean((1, 2, 3)))
    return mu.detach(), A.detach(), col.detach(), o.detach(), s.detach(), z, psnr
