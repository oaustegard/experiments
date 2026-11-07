"""Load a UNet from the exported weights.bin/manifest.json, so torch references use exactly the GPU's weights."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from model import UNet


def load(bin_path, man_path):
    w = np.fromfile(bin_path, np.float16).astype(np.float32)
    man = json.loads(Path(man_path).read_text())["tensors"]
    m = UNet().eval()
    m.load_state_dict({k: torch.from_numpy(w[o:o + int(np.prod(s))].reshape(s).copy()) for k, (o, s) in man.items()})
    return m
