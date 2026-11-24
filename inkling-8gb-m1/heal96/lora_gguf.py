#!/usr/bin/env python3
"""Write heal.py expert LoRA adapters as a llama.cpp LoRA GGUF for the inkling2_loop sliced model.

GGUF expects, per expert tensor W (ggml ne = [in, out, n_expert]): lora_a ne [in, r, n_expert], lora_b ne
[r, out, n_expert] (numpy [n_expert, r, in] and [n_expert, out, r]). heal.py's gate/up LoRA output rows follow
the checkpoint's interleaved w13 (g0, u0, g1, u1...); the GGUF ffn_gate_up_exps rows are [gate; up], so B13 rows
are reordered. llama.cpp adds the delta after the per-expert .scale multiply, i.e. in true weight units, which is
what heal.py trained. Scale = alpha / rank, same as heal.py's lora_scale.
usage: lora_gguf.py adapters.pt out.gguf"""
import re, sys
import numpy as np, torch
from gguf import GGUFWriter, GGUFType, Keys, GGMLQuantizationType
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ad, rank, alpha = sd["adapters"], sd["rank"], sd["alpha"]
w = GGUFWriter(sys.argv[2], "inkling2_loop")
w.add_type(GGUFType.ADAPTER)
w.add_string(Keys.Adapter.TYPE, "lora")
w.add_float32(Keys.Adapter.LORA_ALPHA, float(alpha))
n = 0
for li in range(24):
    get = lambda k: ad[f"model.layers.{li}.mlp.experts.lora.{k}"].float()
    a13, b13, a2, b2 = get("A13"), get("B13"), get("A2"), get("B2")
    b13 = torch.cat([b13[:, 0::2], b13[:, 1::2]], dim=1)  # interleaved gate/up rows -> [gate; up]
    for name, a, b in ((f"blk.{li}.ffn_gate_up_exps.weight", a13, b13), (f"blk.{li}.ffn_down_exps.weight", a2, b2)):
        w.add_tensor(name + ".lora_a", np.ascontiguousarray(a.numpy().astype(np.float16)), raw_dtype=GGMLQuantizationType.F16)
        w.add_tensor(name + ".lora_b", np.ascontiguousarray(b.numpy().astype(np.float16)), raw_dtype=GGMLQuantizationType.F16)
        n += 1
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"{n} LoRA pairs, rank {rank}, alpha {alpha} -> {sys.argv[2]}")
