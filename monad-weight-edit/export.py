"""Export a Monad HF directory to the ONNX form the bridge job runs on WebGPU.

    python3 export.py models/monad export/base

1. onnxruntime-genai's model builder, fp32, target webgpu: one graph, fused GroupQueryAttention
   with rotary and KV cache, ~11 ops per layer.
2. Surgery: logits only where they are needed. A new int64 input `logits_index` [B] picks one
   row of the flattened [B*T, 256] final hidden state per sequence, so a 32 x 200-token prefill
   returns 32 x 8192 logits instead of 32 x 200 x 8192 (210 MB) to read back.
3. Packing (pack()): bf16-exact fp32 tensors travel as bf16, then the packed file is cut into
   parts under 95 MB (git and the bridge carry files up to 100 MB). meta.json lists the regions,
   the parts and the sha256 of the fp32 data file the job must rebuild.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent
PART = 95 * 2**20


def build(src: Path, out: Path):
    tmp = out.with_name(out.name + "_raw")
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "onnxruntime_genai.models.builder", "-i", str(src), "-o", str(tmp),
                    "-p", "fp32", "-e", "webgpu", "-c", str(HERE / "export/cache"),
                    "--extra_options", "hf_token=false"], check=True, stdout=subprocess.DEVNULL)
    return tmp


def surgery(raw: Path, out: Path):
    m = onnx.load(str(raw / "model.onnx"), load_external_data=False)
    g = m.graph
    lm = next(n for n in g.node if n.name == "/lm_head/MatMul")
    hid = lm.input[0]
    g.input.append(helper.make_tensor_value_info("logits_index", TensorProto.INT64, ["batch_size"]))
    shape = helper.make_tensor("/sel/shape", TensorProto.INT64, [2], [-1, 256])
    axes = helper.make_tensor("/sel/axes", TensorProto.INT64, [1], [1])
    g.initializer.extend([shape, axes])
    new = [
        helper.make_node("Reshape", [hid, "/sel/shape"], ["/sel/flat"], name="/sel/Reshape"),
        helper.make_node("Gather", ["/sel/flat", "logits_index"], ["/sel/rows"], name="/sel/Gather", axis=0),
        helper.make_node("Unsqueeze", ["/sel/rows", "/sel/axes"], ["/sel/out"], name="/sel/Unsqueeze"),
    ]
    idx = list(g.node).index(lm)
    for k, n in enumerate(new):
        g.node.insert(idx + k, n)
    lm.input[0] = "/sel/out"
    for o in g.output:
        if o.name == "logits":
            d = o.type.tensor_type.shape.dim
            d[1].ClearField("dim_param")
            d[1].dim_value = 1
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(out / "model.onnx"))
    shutil.copy(raw / "model.onnx.data", out / "model.onnx.data")
    for f in ("genai_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(raw / f, out / f)
    shutil.rmtree(raw)


def pack(out: Path):
    """model.onnx.data -> model.packed: each fp32 initializer whose values are all exact in bf16
    (Monad is stored in bf16, so every weight is) travels as its upper 16 bits, anything else
    (the rotary cos/sin caches) as fp32. The job rebuilds the identical fp32 file. Halves the bytes
    pushed through git and downloaded by the page."""
    m = onnx.load(str(out / "model.onnx"), load_external_data=False)
    data = np.frombuffer((out / "model.onnx.data").read_bytes(), dtype=np.uint8)
    regions = []
    for t in m.graph.initializer:
        if t.data_location != TensorProto.EXTERNAL:
            continue
        kv = {e.key: e.value for e in t.external_data}
        off, ln = int(kv["offset"]), int(kv["length"])
        regions.append((off, ln, t.data_type))
    regions.sort()
    chunks, manifest, pos = [], [], 0
    for off, ln, dt in regions:
        raw = data[off:off + ln]
        kind = "f32"
        if dt == TensorProto.FLOAT and ln % 4 == 0:
            u = raw.view(np.uint32)
            if not (u & 0xFFFF).any():
                kind = "bf16"
                raw = (u >> 16).astype(np.uint16).view(np.uint8)
        manifest.append([off, ln, kind, pos])
        chunks.append(raw.tobytes())
        pos += len(chunks[-1])
    packed = b"".join(chunks)
    (out / "model.packed").write_bytes(packed)
    covered = sum(ln for _, ln, _ in regions)
    return manifest, len(packed), covered


def split(out: Path):
    manifest, packed_bytes, covered = pack(out)
    data = (out / "model.packed").read_bytes()
    parts = []
    for i in range(0, len(data), PART):
        p = out / f"model.packed.{len(parts):03d}"
        p.write_bytes(data[i:i + PART])
        parts.append({"file": p.name, "bytes": len(data[i:i + PART])})
    full = (out / "model.onnx.data").read_bytes()
    n_layers = json.loads((out / "genai_config.json").read_text())["model"]["decoder"]["num_hidden_layers"]
    meta = {"layers": n_layers, "data_bytes": len(full), "data_sha256": hashlib.sha256(full).hexdigest(),
            "covered_bytes": covered, "packed_bytes": packed_bytes,
            "graph_sha256": hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest(), "parts": parts,
            "regions": manifest}
    (out / "meta.json").write_text(json.dumps(meta))
    return {k: v for k, v in meta.items() if k != "regions"}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    src = src if src.is_absolute() else HERE / src
    out = out if out.is_absolute() else HERE / out
    raw = build(src, out)
    surgery(raw, out)
    print(json.dumps(split(out)))


if __name__ == "__main__":
    main()
