"""Export a checkpoint's EMA weights for the WGSL planner: OUT.bin (f32) + OUT.json (offsets, config)."""
import json, sys
import numpy as np, torch
ck = torch.load(sys.argv[1], map_location="cpu"); out = sys.argv[2]
sd = ck["ema"]; a = ck["args"]
order = ["emb.weight", "pos", "y0", "z0", "head.weight", "qhead.weight", "qhead.bias"]
order += [k for k in sd if k.startswith("blocks.")]
parts, tensors, off = [], {}, 0
for k in order:
    v = sd[k].float().numpy().ravel()
    tensors[k] = {"offset": off, "shape": list(sd[k].shape)}
    parts.append(v); off += v.size
np.concatenate(parts).astype("<f4").tofile(out + ".bin")
cfg = {k: a[k] for k in ("d", "heads", "layers", "n", "T", "nsup")}; cfg["L"] = 1024; cfg["step"] = ck.get("step")
json.dump({"config": cfg, "tensors": tensors}, open(out + ".json", "w"))
print(out, off, "floats", cfg)
