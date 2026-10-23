"""Round 4: remex (random rotation + Lloyd-Max levels + one fp16 norm per row) on the cut's trained
table, against int-g32 at more bytes. -> results/r4_remex_cut.json"""
import json
import sys
from pathlib import Path

import numpy as np
import remex
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, LS, Benches, cut_student  # noqa: E402

torch.set_num_threads(4)
sd = torch.load(LS / ".work/mac/out/l1/mac_k4_v32k/final.pt", map_location="cpu")
X = sd["tm.embed_tokens.weight"].numpy()
B = Benches()
out = {}
for b in (3, 2):
    q = remex.Quantizer(512, bits=b, seed=0, rotation="haar")
    R = q.decode(q.encode(X)).astype(np.float32)
    r = B.run(cut_student(sd, R), "cut")
    r["table_mb"] = round(len(X) * (512 * b / 8 + 2) / 1e6, 3)
    out[f"remex{b}"] = r
    print(f"remex{b}", r, flush=True)
(HERE / "results" / "r4_remex_cut.json").write_text(json.dumps(out, indent=1))
