"""image-to-wgsl: an image compressed into one WGSL fragment shader.

The image is an ordered list of soft, rotated ellipse splats painted over a
bilinear background, plus an optional tiny sine-activated MLP residual. The
splat parameters are quantized to a fixed bit layout and written into the
shader as a const array; the fragment shader decodes and paints them per pixel.
The compressed size is the gzip size of the minified shader source.

Training is ordinary gradient descent in PyTorch on randomly sampled pixels,
growing the splat list coarse to fine. With prefix training the loss is the
mean log-MSE over every power-of-two prefix of the list, so any truncation of
the list is itself a usable image (and the shader can paint itself in).

    python3 codec.py fit data/kodak/kodim23.png --n 1024 --prefix --out results/runs/k23
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
FORGE = HERE.parent / "wgsl-forge"
torch.set_num_threads(4)


# ----------------------------------------------------------------------------- layout

@dataclass
class Layout:
    """Bit depth of every field; the same numbers drive training, packing and the shader."""
    bx: int = 10      # centre x over [0, aspect]
    by: int = 9       # centre y over [0, 1]
    bs: int = 6       # log2 sigma over [s_lo, s_hi], per axis
    ba: int = 6       # angle over [0, pi)
    bc: int = 6       # colour channel over [0, 1]
    bal: int = 4      # alpha: (q + 1) / 2^bal
    bm: int = 2       # shape: exponent 2^q on the squared radius (1 = gaussian, 8 = near-flat top)
    bbg: int = 6      # background corner colour channel
    s_lo: float = -9.0
    s_hi: float = 0.0

    FIELDS = ("x", "y", "sx", "sy", "th", "r", "g", "b", "al", "m")

    def widths(self):
        return [self.bx, self.by, self.bs, self.bs, self.ba, self.bc, self.bc, self.bc, self.bal, self.bm]

    @property
    def bits(self):
        return sum(self.widths())

    def maxes(self):
        return torch.tensor([float(2 ** b - 1) for b in self.widths()])


def ste_round(t):
    return t + (torch.round(t) - t).detach()


def physical(P, L: Layout, A: float, quant: bool):
    """Quantization units -> physical splat parameters. Mirrors the WGSL decoder."""
    M = L.maxes()
    P = torch.minimum(torch.clamp(P, min=0.0), M)
    if quant:
        P = ste_round(P)
    qx, qy, qsx, qsy, qth, qr, qg, qb, qal, qm = P.unbind(1)
    x = qx * (A / M[0])
    y = qy * (1.0 / M[1])
    span = (L.s_hi - L.s_lo) / M[2]
    sx = torch.exp2(L.s_lo + qsx * span)
    sy = torch.exp2(L.s_lo + qsy * span)
    th = qth * (math.pi / 2 ** L.ba)
    col = torch.stack([qr, qg, qb], 1) / M[5]
    al = (qal + 1.0) / 2 ** L.bal
    m = torch.exp2(qm)
    return x, y, sx, sy, th, col, al, m


def bg_colors(bgp, L: Layout, quant: bool):
    Mb = float(2 ** L.bbg - 1)
    b = torch.clamp(bgp, 0.0, Mb)
    if quant:
        b = ste_round(b)
    return (b / Mb).view(4, 3)


def weights(x, y, sx, sy, th, al, m, X, Y):
    """Splat opacity w[g, p] at pixel coordinates X, Y (height units)."""
    dx = X[None, :] - x[:, None]
    dy = Y[None, :] - y[:, None]
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    a = (dx * c + dy * s) / sx[:, None]
    b = (dy * c - dx * s) / sy[:, None]
    e = torch.clamp(a * a + b * b, max=40.0)
    e = (e + 1e-12) ** m[:, None]
    return al[:, None] * torch.exp(-0.5 * e)


def composite(c, w, col):
    """Paint G splats over c in order, closed form: c*prod(1-w) + sum_j col_j w_j prod_{l>j}(1-w_l)."""
    om = 1.0 - w
    cp = torch.cumprod(torch.flip(om, [0]), 0)
    after = torch.cat([torch.flip(cp, [0])[1:], torch.ones_like(cp[:1])], 0)
    return c * cp[-1][:, None] + torch.einsum("gp,gc->pc", w * after, col)


def background(bg, X, Y, A):
    h = (X / A)[:, None]
    v = Y[:, None]
    top = bg[0] * (1 - h) + bg[1] * h
    bot = bg[2] * (1 - h) + bg[3] * h
    return top * (1 - v) + bot * v


def render(P, bgp, X, Y, A, L: Layout, quant: bool, checkpoints=(), chunk=256):
    """Colours at the given pixels after all splats; also snapshots after each checkpoint count."""
    x, y, sx, sy, th, col, al, m = physical(P, L, A, quant)
    c = background(bg_colors(bgp, L, quant), X, Y, A)
    n = P.shape[0]
    edges = sorted({k for k in checkpoints if 0 < k < n} | {n})
    snaps = {}
    start = 0
    for e in edges:
        for s0 in range(start, e, chunk):
            s1 = min(e, s0 + chunk)
            w = weights(x[s0:s1], y[s0:s1], sx[s0:s1], sy[s0:s1], th[s0:s1], al[s0:s1], m[s0:s1], X, Y)
            c = composite(c, w, col[s0:s1])
        snaps[e] = c
        start = e
    return c, snaps


# ----------------------------------------------------------------------------- residual MLP

class Siren(torch.nn.Module):
    """r(X, Y) = W3 sin(W2 sin(w0 (W1 [X, Y] + b1)) + b2) + b3, added after the splats."""

    def __init__(self, h=16, w0=30.0, A=1.5):
        super().__init__()
        self.h, self.w0, self.A = h, w0, A
        self.l1 = torch.nn.Linear(2, h)
        self.l2 = torch.nn.Linear(h, h)
        self.l3 = torch.nn.Linear(h, 3)
        with torch.no_grad():
            self.l1.weight.uniform_(-1 / 2, 1 / 2)
            self.l1.bias.uniform_(-1, 1)
            bound = math.sqrt(6 / h) / 1.0
            self.l2.weight.uniform_(-bound, bound)
            self.l3.weight.mul_(0.01)
            self.l3.bias.zero_()

    def tensors(self):
        return [self.l1.weight, self.l1.bias, self.l2.weight, self.l2.bias, self.l3.weight, self.l3.bias]

    @staticmethod
    def q8(t, quant):
        if not quant:
            return t, None
        s = t.detach().abs().max().clamp(min=1e-8) / 127.0
        return ste_round(t / s).clamp(-127, 127) * s, s

    def forward(self, X, Y, quant=False):
        W = [self.q8(t, quant)[0] for t in self.tensors()]
        z = torch.stack([X, Y], 1)
        z = torch.sin(self.w0 * (z @ W[0].T + W[1]))
        z = torch.sin(z @ W[2].T + W[3])
        return z @ W[4].T + W[5]

    def export(self):
        """int8 values and one f32 scale per tensor."""
        out = []
        for t in self.tensors():
            s = t.detach().abs().max().clamp(min=1e-8) / 127.0
            q = torch.round(t.detach() / s).clamp(-127, 127).to(torch.int64)
            out.append((q.flatten().tolist(), float(s)))
        return out


# ----------------------------------------------------------------------------- fitting

def load_target(path):
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.asarray(img, np.float32) / 255.0)
    return img, t


def pixel_grid(Hh, Ww, A):
    ys, xs = torch.meshgrid(torch.arange(Hh, dtype=torch.float32), torch.arange(Ww, dtype=torch.float32), indexing="ij")
    X = (xs.flatten() + 0.5) / Ww * A
    Y = (ys.flatten() + 0.5) / Hh
    return X, Y


@torch.no_grad()
def render_image(P, bgp, Hh, Ww, A, L, quant=True, n=None, residual=None, pix_chunk=32768):
    X, Y = pixel_grid(Hh, Ww, A)
    out = torch.empty(X.shape[0], 3)
    Pn = P if n is None else P[:n]
    for i in range(0, X.shape[0], pix_chunk):
        c, _ = render(Pn, bgp, X[i:i + pix_chunk], Y[i:i + pix_chunk], A, L, quant)
        if residual is not None:
            c = c + residual(X[i:i + pix_chunk], Y[i:i + pix_chunk], quant=quant)
        out[i:i + pix_chunk] = c
    return out.view(Hh, Ww, 3).clamp(0, 1)


def psnr8(a, b):
    """PSNR on 8-bit-rounded images (a, b float in [0,1] or uint8)."""
    a8 = np.clip(np.rint(np.asarray(a, np.float64) * (255 if np.asarray(a).dtype != np.uint8 else 1)), 0, 255)
    b8 = np.clip(np.rint(np.asarray(b, np.float64) * (255 if np.asarray(b).dtype != np.uint8 else 1)), 0, 255)
    mse = np.mean((a8 - b8) ** 2)
    return float(10 * math.log10(255 ** 2 / max(mse, 1e-12)))


def checkpoints_for(n, lo=16):
    ks, k = [], lo
    while k < n:
        ks.append(k)
        k *= 2
    return ks + [n]


def init_splats(n_new, target, current, A, L: Layout, n_total, gen):
    """New splats at pixels drawn in proportion to squared current error, sized for n_total splats."""
    Hh, Ww, _ = target.shape
    err = ((current - target) ** 2).sum(-1).flatten() + 1e-8
    idx = torch.multinomial(err, n_new, replacement=False, generator=gen)
    iy, ix = idx // Ww, idx % Ww
    sigma = 0.6 * math.sqrt(A / n_total)
    blur_px = max(0.5, sigma * Hh * 0.5)
    tb = np.asarray(Image.fromarray((target.numpy() * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_px)), np.float32) / 255
    col = torch.from_numpy(tb[iy.numpy(), ix.numpy()])
    M = L.maxes()
    P = torch.zeros(n_new, 10)
    P[:, 0] = (ix.float() + 0.5) / Ww * M[0]
    P[:, 1] = (iy.float() + 0.5) / Hh * M[1]
    ls = math.log2(sigma) + torch.empty(n_new).uniform_(-0.3, 0.3, generator=gen)
    qs = (ls - L.s_lo) / (L.s_hi - L.s_lo) * M[2]
    P[:, 2] = qs
    P[:, 3] = qs + torch.empty(n_new).uniform_(-2, 2, generator=gen)
    P[:, 4] = torch.empty(n_new).uniform_(0, float(2 ** L.ba), generator=gen)
    P[:, 5:8] = col * M[5]
    P[:, 8] = 0.75 * 2 ** L.bal - 1
    P[:, 9] = 0.0
    return torch.minimum(P.clamp(min=0), M)


# Adam steps in units of each field's range (fraction of 2^b - 1 per step at lr = 1).
STEP_FRAC = torch.tensor([0.004, 0.004, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03])


def optimize(P, bgp, target_flat, X, Y, A, L, steps, lr0, lr1, prefix, quant, n_pix, gen, residual=None,
             train_splats=True, log=None, tag=""):
    """Adam on reparametrised splats (base = P / scale) with cosine lr; returns updated P, bgp."""
    M = L.maxes()
    scale = (M * STEP_FRAC)[None, :]
    bscale = float(2 ** L.bbg - 1) * 0.01
    base = (P / scale).clone().requires_grad_(train_splats)
    bbase = (bgp / bscale).clone().requires_grad_(train_splats)
    params = ([base, bbase] if train_splats else []) + (list(residual.parameters()) if residual is not None else [])
    groups = []
    if train_splats:
        groups.append({"params": [base, bbase], "lr": lr0})
    if residual is not None:
        groups.append({"params": list(residual.parameters()), "lr": 1e-3, "res": True})
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    n = P.shape[0]
    ks = checkpoints_for(n) if prefix else [n]
    npx = X.shape[0]
    for it in range(steps):
        f = it / max(1, steps - 1)
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * f))
        for g in opt.param_groups:
            g["lr"] = (1e-3 * (0.1 + 0.9 * (1 - f))) if g.get("res") else lr
        sel = torch.randint(0, npx, (n_pix,), generator=gen)
        Xs, Ys, T = X[sel], Y[sel], target_flat[sel]
        Pc = base * scale
        bc = bbase * bscale
        c, snaps = render(Pc, bc, Xs, Ys, A, L, quant, checkpoints=ks)
        if residual is not None:
            r = residual(Xs, Ys, quant=quant)
            snaps = {k: v + r for k, v in snaps.items()}
        losses = [torch.mean((snaps[k] - T) ** 2) for k in ks]
        loss = torch.stack([torch.log(l) for l in losses]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            base.copy_(torch.minimum(torch.clamp(base * scale, min=0), M) / scale)
        if log is not None and (it % 100 == 0 or it == steps - 1):
            log.append({"tag": tag, "it": it, "n": n, "psnr_full_sample": float(-10 * math.log10(losses[-1].item())),
                        "lr": lr})
    return (base * scale).detach(), (bbase * bscale).detach()


def fit(path, n=1024, prefix=True, steps_stage=150, steps_final=600, steps_qat=300, n_pix=16384, seed=0,
        L: Layout | None = None, start=16, verbose=True):
    L = L or Layout()
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    img, target = load_target(path)
    Hh, Ww, _ = target.shape
    A = Ww / Hh
    X, Y = pixel_grid(Hh, Ww, A)
    tflat = target.view(-1, 3)
    M = L.maxes()
    # background: mean colours of the four quadrants
    q = [target[:Hh // 2, :Ww // 2], target[:Hh // 2, Ww // 2:], target[Hh // 2:, :Ww // 2], target[Hh // 2:, Ww // 2:]]
    bgp = torch.cat([t.reshape(-1, 3).mean(0) for t in q]) * float(2 ** L.bbg - 1)
    P = torch.zeros(0, 10)
    log, t0 = [], time.time()
    sizes = []
    k = start
    while k < n:
        sizes.append(k)
        k *= 2
    sizes.append(n)
    small = (Hh // 4, Ww // 4)
    tsmall = torch.from_numpy(np.asarray(img.resize((small[1], small[0]), Image.BOX), np.float32) / 255)
    for si, size in enumerate(sizes):
        cur = render_image(P, bgp, small[0], small[1], A, L, quant=False)
        newP = init_splats(size - P.shape[0], tsmall, cur, A, L, size, gen)
        P = torch.cat([P, newP], 0)
        final = si == len(sizes) - 1
        steps = steps_final if final else steps_stage
        P, bgp = optimize(P, bgp, tflat, X, Y, A, L, steps, 1.0, 0.1 if not final else 0.03, prefix, False,
                          n_pix, gen, log=log, tag=f"stage{size}")
        if verbose:
            print(f"  stage {size:5d}: {log[-1]['psnr_full_sample']:.2f} dB sample  {time.time() - t0:.0f}s", flush=True)
    P, bgp = optimize(P, bgp, tflat, X, Y, A, L, steps_qat, 0.3, 0.02, prefix, True, n_pix, gen, log=log, tag="qat")
    Pq = torch.round(torch.minimum(P.clamp(min=0), M))
    bgq = torch.round(bgp.clamp(0, float(2 ** L.bbg - 1)))
    if verbose:
        print(f"  qat: {log[-1]['psnr_full_sample']:.2f} dB sample  {time.time() - t0:.0f}s", flush=True)
    return {"P": Pq, "bg": bgq, "A": A, "H": Hh, "W": Ww, "layout": L, "log": log, "seconds": time.time() - t0,
            "target": target}


def fit_residual(res_run, h=16, steps=1500, n_pix=16384, seed=0, quant_frac=0.3):
    """Train a SIREN residual on top of frozen quantized splats."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    target, A, L = res_run["target"], res_run["A"], res_run["layout"]
    Hh, Ww, _ = target.shape
    X, Y = pixel_grid(Hh, Ww, A)
    # cache the splat render once (frozen)
    base = render_image(res_run["P"], res_run["bg"], Hh, Ww, A, L, quant=True).view(-1, 3)
    resid_target = target.view(-1, 3) - base
    net = Siren(h=h, A=A)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    nq = int(steps * quant_frac)
    for it in range(steps):
        f = it / max(1, steps - 1)
        for g in opt.param_groups:
            g["lr"] = 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * f)))
        sel = torch.randint(0, X.shape[0], (n_pix,), generator=gen)
        r = net(X[sel], Y[sel], quant=it >= steps - nq)
        loss = torch.mean((base[sel] + r - target.view(-1, 3)[sel]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net


# ----------------------------------------------------------------------------- WGSL emission

VS = "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n" \
     "  return vec4f(vec2f(f32(i % 2u), f32(i / 2u)) * 4. - 1., 0., 1.);\n}\n"


def fnum(v):
    s = f"{float(np.float32(v)):.9g}"
    if "e" not in s and "." not in s:
        s += "."
    return s


# WebKit (Safari, and every browser on iOS) refuses `const` arrays with more than 2047 elements;
# Tint and naga accept them. Found on the GPU bridge, iPhone, 2026-11-02 (results/bridge_i2w-render-1.json).
MAX_CONST_ARRAY = 2047


def pack_bits(fields):
    """[(value, nbits)] -> u32 words, LSB first, plus one zero pad word for the straddling read."""
    acc, off = 0, 0
    for v, b in fields:
        v = int(v)
        assert 0 <= v < (1 << b), (v, b)
        acc |= v << off
        off += b
    nw = (off + 31) // 32
    return [(acc >> (32 * i)) & 0xFFFFFFFF for i in range(nw)] + [0], off


def emit_wgsl(run, n=None, encoding="packed", residual=None, paint_seconds=6.0, hexlit=False, store="auto",
              chunk=None):
    """One self-contained shader: vs + fs, uniform vec4f(width, height, time, 0) at group 0 binding 0.

    time <= 0 paints everything; time > 0 paints the first N * min(1, (t / paint_seconds)^2) splats,
    so the image paints itself in when driven by a clock.

    store (packed only): "u32" one array<u32>; "vec4u" one array<vec4u>, four words per element;
    "split" several array<u32> of `chunk` words; "auto" = u32 while it fits WebKit's limit, else vec4u.
    """
    chunk = chunk or MAX_CONST_ARRAY
    L: Layout = run["layout"]
    P = run["P"].to(torch.int64).tolist()
    bg = run["bg"].to(torch.int64).tolist()
    if n is not None:
        P = P[:n]
    N = len(P)
    A = run["A"]
    W = L.widths()
    Mx, My, Ms = 2 ** L.bx - 1, 2 ** L.by - 1, 2 ** L.bs - 1
    Mc, Mb = 2 ** L.bc - 1, 2 ** L.bbg - 1
    lit = (lambda v: hex(v)) if hexlit else (lambda v: str(v))
    out = ["@group(0) @binding(0) var<uniform> u: vec4f;\n", f"const A = {fnum(A)};\n"]
    if encoding == "packed":
        fields = [(v, L.bbg) for v in bg]
        for p in P:
            fields += list(zip(p, W))
        words, nbits = pack_bits(fields)
        if store == "auto":
            store = "u32" if len(words) <= MAX_CONST_ARRAY else "vec4u"
        if store == "u32":
            out.append(f"const D = array<u32, {len(words)}>({','.join(lit(w) for w in words)});\n")
            at = "D[{}]"
        elif store == "split":
            # several plain u32 arrays of at most `chunk` words; dw() picks the array per read
            parts = [words[i:i + chunk] for i in range(0, len(words), chunk)]
            for j, part in enumerate(parts):
                out.append(f"const D{j} = array<u32, {len(part)}>({','.join(lit(w) for w in part)});\n")
            sel = "".join(f"  if (i < {chunk * (j + 1)}u) {{ return D{j}[i - {chunk * j}u]; }}\n" for j in range(len(parts) - 1))
            out.append(f"fn dw(i: u32) -> u32 {{\n{sel}  return D{len(parts) - 1}[i - {chunk * (len(parts) - 1)}u];\n}}\n")
            at = "dw({})"
        else:
            # WebKit rejects const arrays over 2047 elements; four words per vec4u element.
            words += [0] * (-len(words) % 4)
            vecs = [f"vec4u({','.join(lit(w) for w in words[i:i + 4])})" for i in range(0, len(words), 4)]
            if len(vecs) > MAX_CONST_ARRAY:
                raise ValueError(f"{len(words)} words exceed {4 * MAX_CONST_ARRAY} even as vec4u")
            out.append(f"const D = array<vec4u, {len(vecs)}>({','.join(vecs)});\n")
            out.append("fn dw(i: u32) -> u32 { return D[i >> 2u][i & 3u]; }\n")
            at = "dw({})"
        out.append("fn rd(o: u32, n: u32) -> f32 {\n  let w = o >> 5u;\n  let s = o & 31u;\n"
                   f"  var v = {at.format('w')} >> s;\n"
                   f"  if (s + n > 32u) {{ v |= {at.format('w + 1u')} << (32u - s); }}\n"
                   "  return f32(v & ((1u << n) - 1u));\n}\n")
        offs = np.cumsum([0] + W).tolist()
        hb = 12 * L.bbg
        B = L.bits

        def fld(j):
            return f"rd(o + {offs[j]}u, {W[j]}u)"

        def bgv(j):
            return f"vec3f(rd({(3 * j) * L.bbg}u, {L.bbg}u), rd({(3 * j + 1) * L.bbg}u, {L.bbg}u), rd({(3 * j + 2) * L.bbg}u, {L.bbg}u))"
        head = f"    let o = {hb}u + i * {B}u;\n"
    elif encoding == "columnar":
        cols = list(zip(*P)) if N else [[] for _ in W]
        names = ["X", "Y", "S", "U", "R", "C", "G", "B", "L", "M"]
        for nm, col in zip(names, cols):
            out.append(f"const {nm} = array<u32, {max(1, N)}>({','.join(lit(v) for v in col) or '0'});\n")
        out.append(f"const K = array<u32, 12>({','.join(lit(v) for v in bg)});\n")

        def fld(j):
            return f"f32({names[j]}[i])"

        def bgv(j):
            return f"vec3f(f32(K[{3 * j}]), f32(K[{3 * j + 1}]), f32(K[{3 * j + 2}]))"
        head = ""
    else:
        raise ValueError(encoding)
    if residual is not None:
        out.append(residual_wgsl(residual))
    out.append(VS)
    body = [
        "@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {\n",
        "  let k = min(u.x / A, u.y);\n",
        "  let q = (p.xy - (u.xy - vec2f(A, 1.) * k) * .5) / k;\n",
        "  if (q.x < 0. || q.y < 0. || q.x > A || q.y > 1.) { return vec4f(1.); }\n",
        "  let h = q.x / A;\n",
        f"  var c = mix(mix({bgv(0)}, {bgv(1)}, h), mix({bgv(2)}, {bgv(3)}, h), q.y) / {fnum(Mb)};\n",
        f"  var n = {N}u;\n",
        f"  if (u.z > 0.) {{ n = min(n, u32(f32(n) * min(1., u.z * u.z / {fnum(paint_seconds ** 2)}))); }}\n",
        "  for (var i = 0u; i < n; i++) {\n",
        head,
        f"    let d = q - vec2f({fld(0)} * {fnum(A / Mx)}, {fld(1)} * {fnum(1 / My)});\n",
        f"    let t = {fld(4)} * {fnum(math.pi / 2 ** L.ba)};\n",
        "    let cs = cos(t);\n    let sn = sin(t);\n",
        f"    let a = (d.x * cs + d.y * sn) / exp2({fnum(L.s_lo)} + {fld(2)} * {fnum((L.s_hi - L.s_lo) / Ms)});\n",
        f"    let b = (d.y * cs - d.x * sn) / exp2({fnum(L.s_lo)} + {fld(3)} * {fnum((L.s_hi - L.s_lo) / Ms)});\n",
        "    var e = min(a * a + b * b, 40.);\n",
        f"    for (var j = 0u; j < u32({fld(9)}); j++) {{ e *= e; }}\n",
        f"    let w = ({fld(8)} + 1.) * {fnum(1 / 2 ** L.bal)} * exp(-.5 * e);\n",
        f"    c = mix(c, vec3f({fld(5)}, {fld(6)}, {fld(7)}) * {fnum(1 / Mc)}, w);\n",
        "  }\n",
    ]
    if residual is not None:
        body.append("  c += res(q);\n")
    body.append("  return vec4f(c, 1.);\n}\n")
    out += body
    return "".join(out)


def residual_wgsl(net: Siren):
    """SIREN weights as int8 packed four per u32, one f32 scale per tensor."""
    ex = net.export()
    h = net.h
    vals, scales, offs = [], [], []
    for q, s in ex:
        offs.append(len(vals))
        vals += q
        scales.append(s)
    while len(vals) % 4:
        vals.append(0)
    words = []
    for i in range(0, len(vals), 4):
        w = 0
        for j in range(4):
            w |= (vals[i + j] & 0xFF) << (8 * j)
        words.append(w)
    o1w, o1b, o2w, o2b, o3w, o3b = offs
    s1w, s1b, s2w, s2b, s3w, s3b = [fnum(s) for s in scales]
    return (
        f"const Q = array<u32, {len(words)}>({','.join(str(w) for w in words)});\n"
        "fn qv(i: u32) -> f32 { return f32(i32(Q[i >> 2u] << (24u - 8u * (i & 3u))) >> 24u); }\n"
        f"fn res(q: vec2f) -> vec3f {{\n"
        f"  var z: array<f32, {h}>;\n"
        f"  for (var i = 0u; i < {h}u; i++) {{\n"
        f"    z[i] = sin({fnum(net.w0)} * (qv({o1w}u + 2u * i) * {s1w} * q.x + qv({o1w + 1}u + 2u * i) * {s1w} * q.y + qv({o1b}u + i) * {s1b}));\n"
        f"  }}\n"
        f"  var c = vec3f(qv({o3b}u), qv({o3b + 1}u), qv({o3b + 2}u)) * {s3b};\n"
        f"  for (var i = 0u; i < {h}u; i++) {{\n"
        f"    var s = qv({o2b}u + i) * {s2b};\n"
        f"    for (var j = 0u; j < {h}u; j++) {{ s += qv({o2w}u + i * {h}u + j) * {s2w} * z[j]; }}\n"
        f"    let y = sin(s);\n"
        f"    c += vec3f(qv({o3w}u + i), qv({o3w + h}u + i), qv({o3w + 2 * h}u + i)) * {s3w} * y;\n"
        f"  }}\n"
        f"  return c;\n}}\n"
    )


def minify(src: str) -> str:
    sys.path.insert(0, str(FORGE))
    from forge.minify import minify as fm
    return fm(src)


def rate(src: str) -> dict:
    m = minify(src)
    return {"raw": len(src.encode()), "min": len(m.encode()), "gz": len(gzip.compress(m.encode(), 9, mtime=0)),
            "wgsl": m}


# ----------------------------------------------------------------------------- persistence

def save_run(run, path, residual=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {"P": run["P"].to(torch.int64).tolist(), "bg": run["bg"].to(torch.int64).tolist(), "A": run["A"],
         "H": run["H"], "W": run["W"], "layout": asdict(run["layout"]), "seconds": run["seconds"],
         "log": run["log"]}
    if residual is not None:
        d["residual"] = {"h": residual.h, "w0": residual.w0, "state": {k: v.tolist() for k, v in residual.state_dict().items()}}
    path.write_text(json.dumps(d))


def load_run(path, image=None):
    d = json.loads(Path(path).read_text())
    run = {"P": torch.tensor(d["P"], dtype=torch.float32), "bg": torch.tensor(d["bg"], dtype=torch.float32),
           "A": d["A"], "H": d["H"], "W": d["W"], "layout": Layout(**d["layout"]), "seconds": d["seconds"],
           "log": d["log"]}
    if image is not None:
        run["target"] = load_target(image)[1]
    res = None
    if "residual" in d:
        r = d["residual"]
        res = Siren(h=r["h"], w0=r["w0"], A=d["A"])
        res.load_state_dict({k: torch.tensor(v) for k, v in r["state"].items()})
    return run, res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("image")
    f.add_argument("--n", type=int, default=1024)
    f.add_argument("--prefix", action="store_true")
    f.add_argument("--steps-stage", type=int, default=150)
    f.add_argument("--steps-final", type=int, default=600)
    f.add_argument("--steps-qat", type=int, default=300)
    f.add_argument("--n-pix", type=int, default=16384)
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        run = fit(a.image, n=a.n, prefix=a.prefix, steps_stage=a.steps_stage, steps_final=a.steps_final,
                  steps_qat=a.steps_qat, n_pix=a.n_pix)
        img = render_image(run["P"], run["bg"], run["H"], run["W"], run["A"], run["layout"])
        print("full-res PSNR", psnr8(img.numpy(), run["target"].numpy()))
        save_run(run, a.out + ".json")
        src = emit_wgsl(run)
        r = rate(src)
        Path(a.out + ".wgsl").write_text(r["wgsl"])
        print({k: v for k, v in r.items() if k != "wgsl"})


if __name__ == "__main__":
    main()
