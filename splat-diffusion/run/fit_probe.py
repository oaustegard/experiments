"""How many additive torus splats does a 128x128 DTD texture tile need? Fit a few crops on CPU."""
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import pyarrow.parquet as pq
from splats import fit

raw = urllib.request.urlopen("https://huggingface.co/datasets/tanganke/dtd/resolve/main/data/test-00000-of-00001.parquet", timeout=300).read()
tab = pq.read_table(io.BytesIO(raw)).to_pydict()
rng = np.random.default_rng(0)
ims = []
for k in rng.choice(len(tab["label"]), 8, replace=False):
    im = Image.open(io.BytesIO(tab["image"][k]["bytes"])).convert("RGB")
    s = 256 / min(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - 128) // 2, (im.height - 128) // 2
    ims.append(np.asarray(im.crop((l, t, l + 128, t + 128))))
x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float() / 127.5 - 1
torch.set_num_threads(4)
for K in (256, 512, 1024):
    t0 = time.time()
    mu, A, col, psnr = fit(x, K, steps=int(sys.argv[1]) if len(sys.argv) > 1 else 400)
    print(K, f"psnr median {float(psnr.median()):.2f}", [round(float(p), 1) for p in psnr], f"{time.time() - t0:.0f}s", flush=True)
    np.savez(f"/tmp/claude-0/probe_K{K}.npz", x=x.numpy(), mu=mu.numpy(), A=A.numpy(), col=col.numpy())
