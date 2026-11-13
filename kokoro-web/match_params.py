"""Name every large ONNX initializer by the PyTorch parameter it holds (matched by value, as ettin-native's
webexport does), so build.py can tell the duration path from the parts that only shape the sound. Node scopes are
ambiguous: kmodel.text_encoder and kmodel.predictor.text_encoder both export as /text_encoder/.

    HF_HOME=<Kokoro-82M cache> /opt/kokoro-env/bin/python match_params.py MODEL.onnx  -> param_map.json

Needs torch, kokoro and onnx (onnx only to read the file; it is pure protobuf, no graph is run).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from kokoro import KModel

HERE = Path(__file__).resolve().parent
model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
params = {}
for name, p in model.named_parameters():
    params[name] = p.detach().numpy()
for name, mod in model.named_modules():  # weight_norm keeps weight as a computed attribute: export folds it
    w = getattr(mod, "weight", None)
    if w is not None and hasattr(mod, "weight_v"):
        params[name + ".weight"] = w.detach().numpy()

index = defaultdict(list)
for name, a in params.items():
    if a.size >= 4096:
        index[(round(float(np.abs(a).sum()), 1), a.size)].append(name)

out, unmatched = {}, []
g = onnx.load(sys.argv[1]).graph
for t in g.initializer:
    a = numpy_helper.to_array(t)
    if a.size < 4096 or a.dtype != np.float32:
        continue
    key = (round(float(np.abs(a).sum()), 1), a.size)
    names = index.get(key, [])
    if not names:  # LSTM gates are reordered (iofc -> iofc / ifgo) and the two directions stacked: match by sorted values
        flat = np.sort(a.ravel())
        names = [n for n, b in params.items() if b.size == a.size and np.allclose(np.sort(b.ravel()), flat, atol=1e-6)]
        if not names:  # an ONNX LSTM tensor holds both directions: look for the forward and reverse params together
            for n, b in params.items():
                if n.endswith("_l0") and (n + "_reverse") in params and 2 * b.size == a.size:
                    both = np.sort(np.concatenate([b.ravel(), params[n + "_reverse"].ravel()]))
                    if np.allclose(both, flat, atol=1e-6):
                        names = [n, n + "_reverse"]
                        break
    if names:
        out[t.name] = names
    else:
        unmatched.append((t.name, list(a.shape)))
(HERE / "param_map.json").write_text(json.dumps(out, indent=1))
print(len(out), "matched;", len(unmatched), "unmatched:", unmatched[:8])
