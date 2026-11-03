"""Check web/gputrain.js's loss and gradient against PyTorch autograd on the same model.

    python3 bridge/gradcheck_gpu.py [W H N [focus|bgimg]]   (runs gpu_gradcheck.mjs in headless Chromium, SwiftShader)
With "focus", pixels carry Gaussian weights (normalised to mean 1) in both losses, as a focus map does.
With "bgimg", the splats paint over a per-pixel background (another image), as a deep-zoom detail tile does.
Model: splats in quantization units, weight clamped at 0.9999, painted in order over the fixed
background, loss = mean over prefixes k of log(mean squared error of the k-splat prefix).
"""
import json, math, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import codec  # noqa: E402

L = codec.Layout()
M = L.maxes()

def ckpts(n):
    k, out = 16, []
    while k < n: out.append(k); k *= 2
    return out + [n]

def torch_loss(P, bg, T, W, H, wpix=None, bgimg=None):
    A = W / H
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    X = ((xs + .5) / H).flatten(); Y = ((ys + .5) / H).flatten()
    q = P.clamp(min=0) ; q = torch.minimum(q, M.double())
    qx, qy, qsx, qsy, qth, qr, qg, qb, qal, qm = q.unbind(1)
    x = qx * A / M[0]; y = qy / M[1]
    span = (L.s_hi - L.s_lo) / M[2]
    sx = torch.exp2(L.s_lo + qsx * span); sy = torch.exp2(L.s_lo + qsy * span)
    th = qth * math.pi / 2 ** L.ba; col = torch.stack([qr, qg, qb], 1) / M[5]
    al = (qal + 1) / 2 ** L.bal; m = torch.exp2(qm)
    bgc = (bg / 63.).view(4, 3).double()
    h = (X / A)[:, None]; v = Y[:, None]
    c = (bgc[0] * (1 - h) + bgc[1] * h) * (1 - v) + (bgc[2] * (1 - h) + bgc[3] * h) * v
    if bgimg is not None:
        c = bgimg
    ks = ckpts(P.shape[0]); snaps = {}
    for j in range(P.shape[0]):
        if j in ks: snaps[j] = c
        dx = X - x[j]; dy = Y - y[j]; cs, sn = torch.cos(th[j]), torch.sin(th[j])
        a = (dx * cs + dy * sn) / sx[j]; b = (dy * cs - dx * sn) / sy[j]
        e = (torch.clamp(a * a + b * b, max=40.) + 1e-12) ** m[j]
        w = torch.clamp(al[j] * torch.exp(-.5 * e), max=.9999)[:, None]
        c = c * (1 - w) + col[j] * w
    snaps[P.shape[0]] = c
    wcol = torch.ones(T.shape[0], 1, dtype=torch.float64) if wpix is None else wpix[:, None]
    mse = [(((snaps[k] - T) ** 2) * wcol).mean() for k in ks]
    return torch.stack([torch.log(s) for s in mse]).mean(), [float(s) for s in mse]

def main():
    W, H, N = int(sys.argv[1]) if len(sys.argv) > 1 else 48, int(sys.argv[2]) if len(sys.argv) > 2 else 32, int(sys.argv[3]) if len(sys.argv) > 3 else 40
    img = Image.open(HERE.parent / "data" / "kodak" / "kodim23.png").convert("RGB").resize((W, H), Image.LANCZOS)
    rgba = np.array(img.convert("RGBA"), np.uint8)
    T = torch.tensor(np.asarray(img, np.float64).reshape(-1, 3) / 255)
    g = torch.Generator().manual_seed(3)
    P = torch.rand(N, 10, generator=g, dtype=torch.float64) * M.double()
    P[:, 2:4] = 36 + 10 * torch.rand(N, 2, generator=g, dtype=torch.float64)     # sigma ~ 2^-3.9..-2.5 of height
    P[:, 8] = torch.rand(N, generator=g, dtype=torch.float64) * 13                 # alpha <= 0.875: away from the clamp
    P[:, 9] = torch.rand(N, generator=g, dtype=torch.float64) * 2                  # shape exponent 1..4
    bg = torch.tensor([30, 40, 20, 50, 45, 35, 25, 30, 20, 40, 35, 30])
    focus = None
    if len(sys.argv) > 4 and sys.argv[4] == "focus":   # Gaussian around (0.3, 0.4), floored at 0.05
        ys, xs = np.mgrid[0:H, 0:W]
        f = np.maximum(np.exp(-0.5 * (((xs + .5) / W - .3) ** 2 + ((ys + .5) / H - .4) ** 2) / .15 ** 2), 0.05).flatten()
        focus = f.tolist()
    wpix = None if focus is None else torch.tensor(focus, dtype=torch.float64) * (W * H) / sum(focus)
    bgimg = None
    if len(sys.argv) > 4 and sys.argv[4] == "bgimg":   # kodim05 at the same size, as float32 (what the GPU holds)
        b = np.asarray(Image.open(HERE.parent / "data" / "kodak" / "kodim05.png").convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255
        bgimg = torch.tensor(b.reshape(-1, 3), dtype=torch.float64)
    P.requires_grad_(True)
    loss, mse = torch_loss(P, bg, T, W, H, wpix, bgimg); loss.backward()
    spec = {"W": W, "H": H, "rgba": rgba.flatten().tolist(), "P": P.detach().flatten().tolist(), "bg": bg.tolist(), "focus": focus,
            "bgimg": None if bgimg is None else np.concatenate([bgimg.numpy(), np.ones((W * H, 1))], 1).flatten().tolist()}
    (HERE.parent / "scratch").mkdir(exist_ok=True)
    sp = HERE.parent / "scratch" / "gpu_gc_in.json"; op = HERE.parent / "scratch" / "gpu_gc_out.json"
    sp.write_text(json.dumps(spec))
    r = subprocess.run(["node", str(HERE / "gpu_gradcheck.mjs"), str(sp), str(op)], capture_output=True, text=True,
                       env={**__import__("os").environ, "NODE_PATH": subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()})
    print(r.stdout[-2000:], r.stderr[-2000:])
    out = json.loads(op.read_text())
    gg = np.array(out["grad"]).reshape(N, 10); gt = P.grad.numpy()
    print(f"loss torch {loss.item():.6f} gpu {out['loss']:.6f}; mse torch {[round(x, 6) for x in mse]} gpu {[round(x, 6) for x in out['mse']]}")
    names = ["x", "y", "sx", "sy", "th", "r", "g", "b", "al", "m"]
    worst = 0
    for f2, nm in enumerate(names):
        rel = np.abs(gg[:, f2] - gt[:, f2]).max() / max(np.abs(gt[:, f2]).max(), 1e-30)
        corr = np.corrcoef(gg[:, f2], gt[:, f2])[0, 1]
        worst = max(worst, rel)
        print(f"  {nm:3} max rel err {rel:.2e}  corr {corr:.6f}  |g| max {np.abs(gt[:, f2]).max():.3e}")
    print("PASS" if worst < 2e-3 else "CHECK", f"worst {worst:.2e}")

if __name__ == "__main__":
    main()
