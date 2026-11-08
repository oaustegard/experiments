"""PSNR of a GPU-bridge sample against PyTorch run on the same exported f16 weights and the same noise.

    python3 run/compare_mac.py RESULT_DIR NOISE.npy [OUT.json] [MODEL_DIR=web/model]

RESULT_DIR is `gpu_bridge.py fetch --out` output (img.npy); the job sampled class 0 (cat), 64x64,
24 steps, guidance 2, from NOISE. The torch reference loads web/model/weights.bin itself
(export.load_bin), so it cannot drift from what the GPU ran (wgsl-diffusion ERRORS.md #1).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

here = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(here / "train"))
from export import load_bin
from model import sample

res, noise_path = Path(sys.argv[1]), sys.argv[2]
gpu = np.load(res / "img.npy").reshape(3, 64, 64).clip(-1, 1)
noise = torch.from_numpy(np.load(noise_path).reshape(1, 3, 64, 64))
md = here / (sys.argv[4] if len(sys.argv) > 4 else "web/model")
m = load_bin(md / "weights.bin", md / "manifest.json")
ref = sample(m, torch.tensor([0]), 64, 64, steps=24, cfg=2.0, noise=noise)[0].numpy()
mse = float(((gpu - ref) ** 2).mean())
out = {"psnr_db": 10 * np.log10(4 / mse), "max_abs": float(np.abs(gpu - ref).max()), "mse": mse}
print(out)
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
