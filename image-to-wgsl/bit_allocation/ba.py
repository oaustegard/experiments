import sys, json, math, dataclasses
from pathlib import Path
import torch
sys.path.insert(0, "/home/user/experiments/image-to-wgsl")
import codec
from codec import Layout
ROOT = Path("/home/user/experiments/image-to-wgsl")
OUT = ROOT / "results/bit_allocation_raw.json"
IMS = ["kodim01", "kodim05", "kodim07", "kodim15", "kodim20", "kodim23"]
NS = (256, 1024)
GROUPS = {"xy": ["bx", "by"], "sigma": ["bs"], "angle": ["ba"], "colour": ["bc"], "alpha": ["bal"], "shape": ["bm"]}
COLS = {"xy": [0, 1], "sigma": [2, 3], "angle": [4], "colour": [5, 6, 7], "alpha": [8], "shape": [9]}

def coarsen(run, g, k):
    L = run["layout"]
    L2 = dataclasses.replace(L, **{a: getattr(L, a) - k for a in GROUPS[g]})
    W, W2 = L.widths(), L2.widths()
    P = run["P"].clone()
    for c in COLS[g]:
        M, M2 = 2 ** W[c] - 1, 2 ** W2[c] - 1
        q = P[:, c]
        if g == "angle":
            q2 = torch.round(q / 2 ** k) % (2 ** W2[c])
        elif g == "alpha":
            q2 = torch.clamp(torch.round((q + 1) / 2 ** k - 1), 0, M2)
        elif g == "shape":
            # exponent q -> nearest representable exponent on the coarser grid (levels 0..M2); 0 bits -> gaussian
            q2 = torch.clamp(q, 0, M2) if M2 > 0 else torch.zeros_like(q)
        else:
            q2 = torch.round(q * M2 / M)
        P[:, c] = q2
    r2 = dict(run); r2["P"] = P; r2["layout"] = L2
    return r2

def measure(run, tgt):
    out = {}
    for n in NS:
        img = codec.render_image(run["P"], run["bg"], run["H"], run["W"], run["A"], run["layout"], n=n)
        out[f"psnr{n}"] = codec.psnr8(img.numpy(), tgt)
        out[f"gz{n}"] = codec.rate(codec.emit_wgsl(run, n=n))["gz"]
    out["bits"] = run["layout"].bits
    return out

res = json.loads(OUT.read_text()) if OUT.exists() else {}
for im in IMS:
    if im in res and len(res[im]) == 13: continue
    run, _ = codec.load_run(ROOT / f"results/runs/{im}_prefix1024.json", ROOT / f"data/kodak/{im}.png")
    tgt = run["target"].numpy()
    d = {"base": measure(run, tgt)}
    for g in GROUPS:
        for k in (1, 2):
            d[f"{g}-{k}"] = measure(coarsen(run, g, k), tgt)
    res[im] = d
    OUT.write_text(json.dumps(res)); print(im, "done", flush=True)
