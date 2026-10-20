"""Where the bytes of each shipped q4 file live: token table, transformer layers, PLE
projection, head, graph, tokenizer.

Usage: python3 -I run/sizes.py   (reads out/export/l2_*/model_q4.onnx; writes results/sizes.json)
"""
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ROOT  # noqa: E402


def part(name):
    if "embed_tokens" in name:
        return "embed_tokens"
    if "layers" in name:
        return "layers"
    if "ple" in name:
        return "ple_projection"
    return "head_and_norms"


def main():
    res = {}
    for d in sorted((ROOT / "out" / "export").glob("l2_*")):
        p = d / "model_q4.onnx"
        if not p.exists():
            continue
        m = onnx.load(str(p))
        owner = {}  # initializer -> the node that reads it (torch names MatMul weights onnx::MatMul_N)
        for n in m.graph.node:
            for x in n.input:
                owner.setdefault(x, n.name)
        b = collections.Counter()
        for i in m.graph.initializer:
            a = numpy_helper.to_array(i)
            nbytes = (a.size + 1) // 2 if i.data_type in (21, 22) else a.nbytes  # (u)int4: two per byte
            b[part(i.name + " " + owner.get(i.name, ""))] += nbytes
        total = p.stat().st_size + (d / "model_q4.onnx_data").stat().st_size
        tok = len(gzip.compress((d / "tokenizer.json").read_bytes()))
        r = {k: v for k, v in b.items()} | {"graph_and_other": total - sum(b.values()),
                                             "tokenizer_gz": tok, "total": total + tok}
        res[d.name] = {k: {"bytes": int(v), "mb": round(v / 1e6, 2)} for k, v in r.items()}
        print(d.name, {k: v["mb"] for k, v in res[d.name].items()})
    (ROOT / "results" / "sizes.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
