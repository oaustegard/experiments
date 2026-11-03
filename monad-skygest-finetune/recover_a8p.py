"""Rebuild A8p (monad-weight-edit's loop edit) from the published 8-bit weights.

    python3 recover_a8p.py            # -> models/A8p (bf16 HF dir) + models/A8p/edit.json

A8p = base Monad with W_o <- (I - u u^T) W_o in the attention output projection of layers 27-34. The
direction u was computed ad hoc in that experiment and never committed (models/ is gitignored), so it
is recovered here: dequantize those eight o_proj matrices from the A8p_q8 export on R2, subtract
them from the base weights, and take the top left singular vector of the stacked differences
(each difference is u (u^T W) plus int8 rounding noise). The rebuilt model is then re-exported and
re-quantized, and its 8-bit bytes are compared with the published A8p_q8.
"""
import json, shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto as T
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
LAYERS = range(27, 35)


def unpack(d):
    meta = json.loads((d / "meta.json").read_text())
    P = np.fromfile(d / "model.packed", np.uint8)
    out = np.zeros(meta["data_bytes"], np.uint8)
    for off, ln, kind, pos in meta["regions"]:
        if kind == "bf16":
            out[off:off + ln] = (P[pos:pos + ln // 2].view(np.uint16).astype(np.uint32) << 16).view(np.uint8)
        else:
            out[off:off + ln] = P[pos:pos + ln]
    return out


def tensors(graph, data):
    m = onnx.load(str(graph), load_external_data=False)
    r = {}
    for t in m.graph.initializer:
        if t.data_location == T.EXTERNAL:
            kv = {e.key: e.value for e in t.external_data}
            o, l = int(kv["offset"]), int(kv["length"])
            r[t.name] = (t.data_type, list(t.dims), data[o:o + l])
    return r


def dequant(ts, layer):
    """MatMulNBits, 8 bits, block 32, symmetric: B stored [N, K/32, 32] as uint8 with zero point 128."""
    _, dims, q = ts[f"model.layers.{layer}.attn.o_proj.MatMul.weight_Q8"]
    sdt, _, s = ts[f"model.layers.{layer}.attn.o_proj.MatMul.weight_scales"]
    q = q.reshape(dims).astype(np.float32) - 128.0
    s = s.view(np.float32 if sdt == T.FLOAT else np.float16).astype(np.float32).reshape(dims[0], dims[1], 1)
    return (q * s).reshape(dims[0], -1)             # [N = out = residual, K = in]


def main():
    torch.set_num_threads(4)
    graph = HERE / "export/r2/model.onnx"
    tq = tensors(graph, unpack(HERE / "export/r2/A8p_q8"))
    tb = tensors(graph, unpack(HERE / "export/r2/base_q8"))
    base = AutoModelForCausalLM.from_pretrained(str(HERE / "models/monad"), dtype=torch.bfloat16)
    W = {l: base.model.layers[l].self_attn.o_proj.weight.detach().float().numpy() for l in LAYERS}
    # sanity: dequantized base_q8 is the base weight up to rounding
    err_b = max(np.abs(dequant(tb, l) - W[l]).max() for l in LAYERS)
    D = np.concatenate([W[l] - dequant(tq, l) for l in LAYERS], axis=1)
    U, S, _ = np.linalg.svd(D, full_matrices=False)
    u = U[:, 0].astype(np.float64)
    # orient u as A8p's file did not record; projection is sign-invariant
    print(json.dumps({"base_dequant_max_err": float(err_b), "sv": S[:4].round(3).tolist()}))
    tok = AutoTokenizer.from_pretrained(str(HERE / "models/monad"))
    ut = torch.tensor(u, dtype=torch.float32)
    P = torch.eye(256) - torch.outer(ut, ut)
    with torch.no_grad():
        for l in LAYERS:
            w = base.model.layers[l].self_attn.o_proj.weight
            w.copy_((P @ w.float()).to(w.dtype))
    out = HERE / "models/A8p"
    shutil.rmtree(out, ignore_errors=True)
    base.save_pretrained(out)
    tok.save_pretrained(out)
    for f in ("chat_template.json", "generation_config.json"):
        shutil.copy(HERE / "models/monad" / f, out / f)
    (out / "edit.json").write_text(json.dumps({"oproj": list(LAYERS), "u": u.tolist(), "recovered_from": "R2 models/monad-chat/v1/A8p_q8",
                                               "singular_values": S[:4].tolist()}))
    print("saved", out)


if __name__ == "__main__":
    main()
