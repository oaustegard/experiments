"""Load the SplatDiT from the exported weights.bin/manifest.json, so torch references use the GPU's weights."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from model import SplatDiT


def load(bin_path, man_path):
    man = json.loads(Path(man_path).read_text())
    w = np.fromfile(bin_path, np.float16).astype(np.float32)
    m = SplatDiT().eval()
    m.load_state_dict({k: torch.from_numpy(w[o:o + int(np.prod(s))].reshape(s).copy()) for k, (o, s) in man["tensors"].items()})
    return m, man
