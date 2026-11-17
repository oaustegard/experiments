"""Pack the tunable tiny-voca graph's matmul weights smaller, as kokoro-web/build.py does.

    fp32       as exported
    fp16       every large MatMul weight fp16 + Cast
    int8       every large MatMul weight int8 per output column + DequantizeLinear
    int8-dur   int8 except the text encoder and duration predictor (fp16 there)

    python pack.py SRC.onnx OUTDIR [variant ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

MIN_SIZE = 4096
DURATION_PATH = ("/encoder/", "/duration/")


def build(src, variant):
    model = onnx.load(src)
    g = model.graph
    cons = defaultdict(list)
    for node in g.node:
        for i, name in enumerate(node.input):
            cons[name].append((node, i))
    keep, new_nodes, report = [], [], defaultdict(int)
    for t in g.initializer:
        w = numpy_helper.to_array(t)
        users = cons.get(t.name, [])
        weight = users and all(n.op_type == "MatMul" and i == 1 for n, i in users)
        if variant == "fp32" or w.dtype != np.float32 or w.size < MIN_SIZE or not weight or w.ndim != 2:
            keep.append(t)
            continue
        on_duration_path = any(n.name.startswith(DURATION_PATH) for n, _ in users)
        if variant == "int8" or (variant == "int8-dur" and not on_duration_path):
            scale = np.abs(w).max(0) / 127
            scale[scale == 0] = 1.0
            q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
            keep += [numpy_helper.from_array(q, t.name + "__q"), numpy_helper.from_array(scale.astype(np.float32), t.name + "__s"),
                     numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), t.name + "__z")]
            new_nodes.append(helper.make_node("DequantizeLinear", [t.name + "__q", t.name + "__s", t.name + "__z"],
                                              [t.name], name=t.name + "__dq", axis=1))
            report["int8"] += w.size
        else:
            keep.append(numpy_helper.from_array(w.astype(np.float16), t.name + "__h"))
            new_nodes.append(helper.make_node("Cast", [t.name + "__h"], [t.name], name=t.name + "__cast", to=onnx.TensorProto.FLOAT))
            report["fp16"] += w.size
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    return model, dict(report)


if __name__ == "__main__":
    src, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    for variant in sys.argv[3:] or ["fp32", "fp16", "int8", "int8-dur"]:
        model, report = build(src, variant)
        path = out / f"{stem}-{variant}.onnx"
        onnx.save(model, str(path))
        onnx.checker.check_model(str(path))
        print(variant, path.stat().st_size, json.dumps(report), flush=True)
