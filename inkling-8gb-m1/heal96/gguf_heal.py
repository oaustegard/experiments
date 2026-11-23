#!/usr/bin/env python3
"""Splice healed experts into an Inkling 2 Mini GGUF.

  inspect  model.gguf                         tensor names, types, shapes of layer 0 + kv expert_count
  check    model.gguf export_dir layer        correlation of the GGUF's expert weights with exported ones
  splice   model.gguf export_dir out.gguf     keep only exported experts; expert tensors replaced by the
                                              exported (healed) weights as Q8_0, router rows replaced as F32
  typeargs model.gguf                         --tensor-type args reproducing the file's per-tensor types

Run with PYTHONPATH=<llama.cpp>/gguf-py (pip's gguf does not know NVFP4).
Exported layer files come from `heal.py export`: gate/up [K, I, H], down [K, H, I], router [K, H], bias [K], ids.
"""
import os, re, sys
import numpy as np

EXPERT_T = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down", "ffn_gate_up_exps": "gate_up"}
GATE_UP_ORDER = os.environ.get("GATE_UP_ORDER", "concat")  # how the GGUF stacks gate and up rows: concat | interleave


def expert_array(e, key):
    if key != "gate_up":
        return e[key]
    g, u = e["gate"], e["up"]
    if GATE_UP_ORDER == "concat":
        return np.concatenate([g, u], axis=1)
    out = np.empty((g.shape[0], 2 * g.shape[1], g.shape[2]), dtype=g.dtype)
    out[:, 0::2], out[:, 1::2] = g, u
    return out


def load_layer(d, li):
    from safetensors.numpy import load_file
    import torch  # bf16 is not a numpy dtype: load with torch
    from safetensors.torch import load_file as lt
    t = lt(os.path.join(d, f"layer_{li:02d}.safetensors"))
    return {k: (v.float().numpy() if v.dtype == torch.bfloat16 else v.numpy()) for k, v in t.items()}


def cmd_inspect(path):
    from gguf import GGUFReader
    r = GGUFReader(path)
    for name, f in r.fields.items():
        if "expert" in name or "general.architecture" in name:
            print(name, f.contents())
    for t in r.tensors:
        if t.name.startswith("blk.0.") or not t.name.startswith("blk."):
            print(t.name, t.tensor_type.name, [int(x) for x in t.shape])


def dequant_tensor(t):
    from gguf import quants
    return quants.dequantize(t.data, t.tensor_type)


def cmd_check(path, export_dir, layer):
    from gguf import GGUFReader
    li = int(layer)
    r = GGUFReader(path)
    ex = load_layer(export_dir, li)
    ids = ex["ids"].astype(int)
    for t in r.tensors:
        m = re.match(rf"blk\.{li}\.(\w+)\.weight$", t.name)
        if not m or m.group(1) not in EXPERT_T:
            continue
        w = dequant_tensor(t).reshape([int(x) for x in reversed(list(t.shape))])  # numpy order [E, rows, cols]
        global GATE_UP_ORDER
        for order in (["concat", "interleave"] if EXPERT_T[m.group(1)] == "gate_up" else [GATE_UP_ORDER]):
            GATE_UP_ORDER = order
            mine = expert_array(ex, EXPERT_T[m.group(1)])
            cs = [np.corrcoef(w[ids[j]].reshape(-1), mine[j].reshape(-1))[0, 1] for j in range(4)]
            ratio = [float(np.abs(w[ids[j]]).mean() / np.abs(mine[j]).mean()) for j in range(4)]
            print(f"{t.name} {t.tensor_type.name} order={order} gguf{list(w.shape)} export{list(mine.shape)} "
                  f"corr {np.round(cs, 4).tolist()} mag ratio {np.round(ratio, 4).tolist()}")
    for t in r.tensors:
        if t.name in (f"blk.{li}.ffn_gate_up_exps.scale", f"blk.{li}.ffn_down_exps.scale"):
            print(t.name, np.asarray(t.data)[ids[:4]].tolist())
    for t in r.tensors:
        if t.name in (f"blk.{li}.ffn_gate_inp.weight", f"blk.{li}.exp_probs_b.bias"):
            w = np.asarray(t.data, dtype=np.float32)
            key = "router" if "gate_inp" in t.name else "bias"
            print(t.name, t.tensor_type.name, list(w.shape), "max abs diff vs export",
                  float(np.abs(w[ids] - ex[key]).max()))


def cmd_splice(path, export_dir, out):
    from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType, quants
    r = GGUFReader(path)
    class _Lazy(dict):  # one exported layer in memory at a time (all 24 as float32 would be ~73 GB)
        def __missing__(self, li):
            self.clear()
            self[li] = load_layer(export_dir, li)
            return self[li]
    ex = _Lazy()
    k = len(ex[0]["ids"])
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(out, arch)
    n_expert = None
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        val = f.contents()
        if name.endswith(".expert_count"):
            n_expert = int(val)
            val = k
        vt = f.types[0]
        sub = f.types[-1] if vt == GGUFValueType.ARRAY else None
        w.add_key_value(name, val, vt, sub_type=sub)
    assert n_expert == 128, n_expert
    replaced = sliced = 0
    scales = {t.name[:-len(".scale")]: np.asarray(t.data, dtype=np.float32) for t in r.tensors
              if t.name.endswith("_exps.scale")}
    for t in r.tensors:
        shape = [int(x) for x in t.shape]  # ggml order
        parts = t.name.split(".")
        if parts[0] == "blk" and shape and shape[-1] == n_expert:
            li = int(parts[1])
            e = ex[li]
            kind = parts[2]
            if kind in EXPERT_T and parts[3] == "weight":
                arr = expert_array(e, EXPERT_T[kind]).astype(np.float32)  # [K, rows, cols]
                sc = scales.get(t.name[:-len(".weight")])
                if sc is not None:  # GGUF stores W / scale and multiplies by the per-expert scale at run time
                    arr = arr / sc[e["ids"].astype(int)][:, None, None]
                arr = np.ascontiguousarray(arr)
                q = quants.quantize(arr, GGMLQuantizationType.Q8_0)
                # int8 view: the writer then takes raw_shape as the element shape (a uint8 array is read as bytes)
                w.add_tensor(t.name, q.reshape(-1).view(np.int8), raw_shape=list(arr.shape),
                             raw_dtype=GGMLQuantizationType.Q8_0)
                replaced += 1
                continue
            if kind in EXPERT_T and parts[3] == "scale":  # keep the original per-expert factors of kept experts
                w.add_tensor(t.name, np.ascontiguousarray(np.asarray(t.data, dtype=np.float32)[e["ids"].astype(int)]),
                             raw_dtype=GGMLQuantizationType.F32)
                replaced += 1
                continue
            if kind in ("ffn_gate_inp", "exp_probs_b") and t.tensor_type == GGMLQuantizationType.F32:
                arr = np.ascontiguousarray(e["router" if kind == "ffn_gate_inp" else "bias"].astype(np.float32))
                w.add_tensor(t.name, arr, raw_dtype=GGMLQuantizationType.F32)
                replaced += 1
                continue
            raw = np.asarray(t.data).reshape(-1).view(np.uint8)
            per = raw.size // n_expert
            raw = raw.reshape(n_expert, per)[e["ids"].astype(int)].reshape(-1)
            shape[-1] = k
            sliced += 1
            print("sliced (not replaced):", t.name, t.tensor_type.name, flush=True)
            w.add_tensor(t.name, raw.view(np.int8), raw_shape=list(reversed(shape)), raw_dtype=t.tensor_type)
            continue
        raw = np.asarray(t.data).reshape(-1).view(np.uint8)
        w.add_tensor(t.name, raw.view(np.int8), raw_shape=list(reversed(shape)), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"replaced {replaced} tensors, sliced {sliced}, {k} experts -> {out}")


def cmd_typeargs(path, out):
    from gguf import GGUFReader
    r = GGUFReader(path)
    with open(out, "w") as f:
        for t in r.tensors:
            f.write(f"^{re.escape(t.name)}$={t.tensor_type.name.lower()}\n")
    print("wrote", out)


if __name__ == "__main__":
    {"inspect": cmd_inspect, "check": cmd_check, "splice": cmd_splice, "typeargs": cmd_typeargs}[sys.argv[1]](*sys.argv[2:])
