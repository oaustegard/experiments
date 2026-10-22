"""Package an ONNX variant for the browser: one weights file shared by the WGSL engine and onnxruntime-web.

Usage: .venv/bin/python -I run/webexport.py NAME VARIANT CKPT     e.g. e17c t4-w8L63 out/contrast/e17-cap-c/epoch1

Writes web/models/NAME-VARIANT/{model.onnx, tokenizer.json, meta.json, manifest.json}.
model.onnx is the single-file graph onnxruntime-web loads; web/engine.js reads the same bytes
through manifest.json (byte offset and length of each tensor inside model.onnx).

The ONNX exporter names MatMul weights anonymously (onnx::MatMul_1417), so the manifest is built by
value, not by node name: each named parameter of the SentenceTransformer checkpoint is matched to the
initializer that reproduces it. Linear weights: int8 codes x per-column scales ~= W^T (shape match,
then smallest error). 1-D tensors (norm weights, biases) and BERT's position/type tables: exact
fp32 match. Token table: the t4 (packed uint8 + fp16 scales) or t3 (vendored eg2-layer-student
codes/scales/zp) initializers. Config values (layers, heads, eps, RoPE theta, window) come from the
checkpoint's config.json.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent


def main():
    name, variant, ckpt = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    src = ROOT / "out" / "models" / name / variant
    dst = ROOT / "web" / "models" / f"{name}-{variant}"
    dst.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(src / "model.onnx"))
    arrs = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}  # before save: it moves the data out
    # One file for both paths. onnxruntime-web gets the single-file model.onnx (with the weights in
    # external data it skips its DequantizeLinear+MatMul fusion and runs ~4x slower); the engine reads
    # the same bytes, each tensor located by its raw bytes inside the file (identical bytes are
    # identical data, so the first match is always right) and copied to an aligned GPU buffer.
    (dst / "model.onnx_data").unlink(missing_ok=True)
    shutil.copy(src / "model.onnx", dst / "model.onnx")
    blob = (dst / "model.onnx").read_bytes()
    for f in ("tokenizer.json", "meta.json"):
        shutil.copy(src / f, dst / f)
    def ext(n):
        raw = arrs[n].tobytes()
        off = blob.find(raw)
        assert off >= 0 and len(raw) > 0, ("tensor bytes not found in model.onnx", n)
        return {"offset": off, "length": len(raw), "dims": list(arrs[n].shape), "dtype": str(arrs[n].dtype)}

    cfg = json.loads((ckpt / "config.json").read_text())
    sd = load_file(str(ckpt / "model.safetensors"))
    dense_dir = next(p for p in ckpt.iterdir() if p.name.endswith("_Dense"))
    dsd = load_file(str(dense_dir / "model.safetensors"))
    dcfg = json.loads((dense_dir / "config.json").read_text())

    q8 = {n[:-3]: (arrs[n].astype(np.float32) * arrs[n[:-3] + "_w8s"]) for n in arrs if n.endswith("_w8")}
    used = set()

    def linear(W):  # torch Linear weight [out, in] -> int8 [in, out] initializer pair
        Wt = W.T.astype(np.float32)
        cands = [(np.abs(D - Wt).max() / (np.abs(Wt).max() + 1e-12), k) for k, D in q8.items()
                 if D.shape == Wt.shape and k not in used]
        if not cands:  # left in fp32 (e.g. exported as Gemm, which the int8 pass skips)
            for k, a in arrs.items():
                if a.dtype == np.float32 and a.shape == Wt.shape and np.array_equal(a, Wt):
                    return {"f32": ext(k), "K": Wt.shape[0], "N": Wt.shape[1]}             # [K, N]
                if a.dtype == np.float32 and a.shape == W.shape and np.array_equal(a, W):
                    return {"f32_t": ext(k), "K": Wt.shape[0], "N": Wt.shape[1]}           # [N, K]
            raise KeyError(("no match for linear", W.shape))
        best = min(cands)
        assert best[0] < 0.02, ("no int8 match", W.shape, best[0])
        used.add(best[1])
        return {"q": ext(best[1] + "_w8"), "s": ext(best[1] + "_w8s"), "K": Wt.shape[0], "N": Wt.shape[1]}

    def exact(v):
        v = v.astype(np.float32)
        for k, a in arrs.items():
            if a.dtype == np.float32 and a.shape == v.shape and np.array_equal(a, v):
                return ext(k)
        raise KeyError(("no exact match", v.shape))

    man = {"source": f"{name}/{variant}", "checkpoint": str(ckpt), "dim": dcfg["out_features"]}
    tab = [k for k in arrs if k.endswith("_q4")]
    if tab:
        base = tab[0][:-3]
        V, H2 = arrs[tab[0]].shape
        man["table"] = {"bits": 4, "V": V, "H": H2 * 2, "codes": ext(base + "_q4"), "scales": ext(base + "_s16")}
    else:
        p = next(k for k in arrs if k.endswith("_codes"))[:-5]
        V = arrs[p + "codes"].shape[0]
        H = arrs[p + "scales"].shape[1] * 32
        man["table"] = {"bits": int(p.rstrip("_")[-1]), "V": V, "H": H,
                        "codes": ext(p + "codes"), "scales": ext(p + "scales"), "zp": ext(p + "zp")}
    def theta(kind, old):  # transformers 5 nests RoPE under rope_parameters; 4.x had flat keys
        rp = cfg.get("rope_parameters") or {}
        return float(rp[kind]["rope_theta"]) if kind in rp else float(cfg[old])

    pre = "model." if any(k.startswith("model.") for k in sd) else ""
    if cfg["model_type"] == "modernbert":
        man |= {"arch": "modernbert", "hidden": cfg["hidden_size"], "heads": cfg["num_attention_heads"],
                "inter": cfg["intermediate_size"], "eps": cfg["norm_eps"], "window": cfg["local_attention"],
                "global_every": cfg["global_attn_every_n_layers"], "theta_global": theta("full_attention", "global_rope_theta"),
                "theta_local": theta("sliding_attention", "local_rope_theta")}
        man["emb_norm"] = exact(sd[pre + "embeddings.norm.weight"])
        man["final_norm"] = exact(sd[pre + "final_norm.weight"])
        man["layers"] = []
        for i in range(cfg["num_hidden_layers"]):
            L = f"{pre}layers.{i}."
            man["layers"].append({
                "attn_norm": exact(sd[L + "attn_norm.weight"]) if L + "attn_norm.weight" in sd else None,
                "Wqkv": linear(sd[L + "attn.Wqkv.weight"]), "Wo": linear(sd[L + "attn.Wo.weight"]),
                "mlp_norm": exact(sd[L + "mlp_norm.weight"]),
                "Wi": linear(sd[L + "mlp.Wi.weight"]), "Wo_mlp": linear(sd[L + "mlp.Wo.weight"]),
                "global": i % cfg["global_attn_every_n_layers"] == 0})
    else:
        assert cfg["model_type"] == "bert"
        man |= {"arch": "bert", "hidden": cfg["hidden_size"], "heads": cfg["num_attention_heads"],
                "inter": cfg["intermediate_size"], "eps": cfg["layer_norm_eps"]}
        E = pre + "embeddings."
        man["pos"] = exact(sd[E + "position_embeddings.weight"])
        man["type0"] = exact(sd[E + "token_type_embeddings.weight"])  # row 0 is read
        man["emb_ln"] = {"w": exact(sd[E + "LayerNorm.weight"]), "b": exact(sd[E + "LayerNorm.bias"])}
        man["layers"] = []
        for i in range(cfg["num_hidden_layers"]):
            L = f"{pre}encoder.layer.{i}."
            lin = lambda k: linear(sd[L + k + ".weight"]) | {"b": exact(sd[L + k + ".bias"])}
            man["layers"].append({
                "q": lin("attention.self.query"), "k": lin("attention.self.key"), "v": lin("attention.self.value"),
                "o": lin("attention.output.dense"),
                "ln1": {"w": exact(sd[L + "attention.output.LayerNorm.weight"]), "b": exact(sd[L + "attention.output.LayerNorm.bias"])},
                "up": lin("intermediate.dense"), "down": lin("output.dense"),
                "ln2": {"w": exact(sd[L + "output.LayerNorm.weight"]), "b": exact(sd[L + "output.LayerNorm.bias"])}})
    man["dense"] = linear(dsd["linear.weight"])
    if dcfg.get("bias"):
        man["dense"]["b"] = exact(dsd["linear.bias"])
    left = set(q8) - used
    assert not left, ("unmatched int8 matmuls", left)
    (dst / "manifest.json").write_text(json.dumps(man, indent=1))
    print(dst.relative_to(ROOT), {f.name: f.stat().st_size for f in dst.iterdir()})


if __name__ == "__main__":
    main()
