"""Write an 8-bit export's tensors into the layout of the page's shared graph, so a new weight set
loads with the exact model.onnx already on R2 (same graph_sha256).

    python3 assemble.py export/ft_q8 export/r2/model.onnx export/ft_q8_page

The graph comes from export/r2/model.onnx (downloaded from models/monad-chat/v1/model.onnx). A
local onnxruntime-genai 0.18.0 build produces byte-identical tensors but names two things
differently (the tied embedding is `lm_head.MatMul.weight` here and `model.embed_tokens.weight`
there, stored transposed, and constant-node names), so tensors are matched by name with that one rename, and every
shape and dtype is checked.
"""
import json, shutil, sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T

from export import split

HERE = Path(__file__).resolve().parent
RENAME = {"model.embed_tokens.weight": "lm_head.MatMul.weight"}


def external(m, data):
    out = {}
    for t in m.graph.initializer:
        if t.data_location == T.EXTERNAL:
            kv = {e.key: e.value for e in t.external_data}
            o, l = int(kv["offset"]), int(kv["length"])
            out[t.name] = (t.data_type, list(t.dims), o, l, data[o:o + l] if data is not None else None)
    return out


def main():
    src, ref, dst = (HERE / p if not Path(p).is_absolute() else Path(p) for p in sys.argv[1:4])
    ms = onnx.load(str(src / "model.onnx"), load_external_data=False)
    s = external(ms, np.fromfile(src / "model.onnx.data", np.uint8))
    mr = onnx.load(str(ref), load_external_data=False)
    r = external(mr, None)
    total = max(o + l for _, _, o, l, _ in r.values())
    buf = np.zeros(total, np.uint8)
    for name, (dt, dims, o, l, _) in r.items():
        sd, sdims, _, sl, sb = s[RENAME.get(name, name)]
        if name in RENAME and sdims == dims[::-1]:   # stored as the lm_head's [256, 8192] here
            assert sd == dt == T.FLOAT
            sb = np.ascontiguousarray(sb.view(np.float32).reshape(sdims).T).view(np.uint8).ravel()
            sdims = dims
        assert (sd, sdims, sl) == (dt, dims, l), (name, sd, sdims, sl, dt, dims, l)
        buf[o:o + l] = sb
    assert len(r) == len(s), (len(r), len(s))
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    shutil.copy(ref, dst / "model.onnx")
    buf.tofile(dst / "model.onnx.data")
    for f in ("genai_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(src / f, dst / f)
    print(json.dumps({k: v for k, v in split(dst).items() if k != "regions"}))


if __name__ == "__main__":
    main()
