"""Convert a ModernBERT-base SPLADE Hugging Face checkpoint into a browser weight pack.

Format: sparseup-pack-1
- weights.bin: 256-byte aligned regions ("main", "layers"), 16-byte aligned tensors within each region.
- manifest.json: metadata, region offsets/lengths, and tensor/linear descriptors.
- tokenizer.json, tokenizer_config.json: copied from the HF directory.

    python3 pack.py --hf hf --out packs/int8 --quant int8
    python3 pack.py --hf hf --out packs/q4 --quant q4

Written by gemini-4-hydrogen from a spec (DELEGATION.md #1), one unused variable removed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


class Region:
    """Sequential byte buffer with 16-byte tensor alignment."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def add(self, arr: np.ndarray) -> dict[str, int]:
        pad = (-len(self.buf)) % 16
        if pad:
            self.buf.extend(b"\x00" * pad)
        offset = len(self.buf)
        raw = np.ascontiguousarray(arr).tobytes()
        self.buf.extend(raw)
        return {"offset": offset, "length": len(raw)}


def quant_i8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.max(np.abs(w), axis=1) / 127.0
    s = np.where(s == 0.0, 1.0, s).astype(np.float32)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s


def dequant_i8(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * s[:, None]).astype(np.float32)


def quant_q4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = w.shape
    if k % 32 != 0:
        raise ValueError(f"q4 requires K % 32 == 0, got {k}")
    n_blocks = k // 32
    blocks = w.reshape(n, n_blocks, 32)
    s = np.max(np.abs(blocks), axis=2) / 7.0
    s = np.where(s == 0.0, 1.0, s).astype(np.float32)
    c = (np.clip(np.rint(blocks / s[:, :, None]), -7, 7) + 8).astype(np.uint8).reshape(n, k)
    q = (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8)
    zp = np.full((n, (n_blocks + 1) // 2), 0x88, dtype=np.uint8)
    return q, s, zp


def dequant_q4(q: np.ndarray, s: np.ndarray, zp: np.ndarray, n: int, k: int) -> np.ndarray:
    n_blocks = k // 32
    c = np.empty((n, k), dtype=np.float32)
    c[:, 0::2] = q & 0x0F
    c[:, 1::2] = (q >> 4) & 0x0F

    zp_unpacked = np.empty((n, zp.shape[1] * 2), dtype=np.float32)
    zp_unpacked[:, 0::2] = zp & 0x0F
    zp_unpacked[:, 1::2] = (zp >> 4) & 0x0F
    zp_blocks = zp_unpacked[:, :n_blocks]

    w = (c.reshape(n, n_blocks, 32) - zp_blocks[:, :, None]) * s[:, :, None]
    return w.reshape(n, k).astype(np.float32)


def _add_linear(region: Region, region_name: str, w: np.ndarray, fmt: str) -> tuple[dict, float]:
    n, k = w.shape
    if fmt == "i8":
        q, s = quant_i8(w)
        err = float(np.max(np.abs(w - dequant_i8(q, s))))
        lin = {"fmt": "i8", "N": n, "K": k, "region": region_name, "q": region.add(q), "s": region.add(s)}
        return lin, err
    if fmt == "q4":
        q, s, zp = quant_q4(w)
        err = float(np.max(np.abs(w - dequant_q4(q, s, zp, n, k))))
        lin = {"fmt": "q4", "N": n, "K": k, "region": region_name,
               "q": region.add(q), "s": region.add(s), "zp": region.add(zp)}
        return lin, err
    raise ValueError(f"Unknown quant format: {fmt}")


def _validate_keys(sd: dict[str, np.ndarray], n_layers: int) -> None:
    expected = {
        "vocab_fold_index",
        "decoder.bias",
        "head.dense.weight",
        "head.norm.weight",
        "model.embeddings.norm.weight",
        "model.embeddings.tok_embeddings.weight",
        "model.final_norm.weight",
    }
    for i in range(n_layers):
        expected.add(f"model.layers.{i}.attn.Wqkv.weight")
        expected.add(f"model.layers.{i}.attn.Wo.weight")
        if i > 0:
            expected.add(f"model.layers.{i}.attn_norm.weight")
        expected.add(f"model.layers.{i}.mlp.Wi.weight")
        expected.add(f"model.layers.{i}.mlp.Wo.weight")
        expected.add(f"model.layers.{i}.mlp_norm.weight")

    actual = set(sd.keys())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise KeyError(f"Checkpoint key mismatch. Missing: {missing}; Unexpected: {unexpected}")


def load_pack(pack_dir: str | Path) -> tuple[dict, bytes]:
    pack_path = Path(pack_dir)
    manifest = json.loads((pack_path / "manifest.json").read_text(encoding="utf-8"))
    weights = (pack_path / "weights.bin").read_bytes()
    return manifest, weights


def dequant_state_dict(pack_dir: str | Path) -> dict[str, torch.Tensor]:  # noqa: F821
    import torch

    manifest, weights = load_pack(pack_dir)
    regions = manifest["regions"]

    def read_arr(region_name: str, ref: dict[str, int], dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        start = regions[region_name]["offset"] + ref["offset"]
        return np.frombuffer(weights, dtype=dtype, count=int(np.prod(shape)), offset=start).reshape(shape)

    def read_linear(lin: dict) -> np.ndarray:
        n, k, reg, fmt = lin["N"], lin["K"], lin["region"], lin["fmt"]
        if fmt == "i8":
            q = read_arr(reg, lin["q"], np.int8, (n, k))
            s = read_arr(reg, lin["s"], np.float32, (n,))
            return dequant_i8(q, s)
        if fmt == "q4":
            q = read_arr(reg, lin["q"], np.uint8, (n, k // 2))
            s = read_arr(reg, lin["s"], np.float32, (n, k // 32))
            zp = read_arr(reg, lin["zp"], np.uint8, (n, (k // 32 + 1) // 2))
            return dequant_q4(q, s, zp, n, k)
        raise ValueError(f"Unknown linear format: {fmt}")

    vocab = manifest["vocab"]
    vpad = manifest["vpad"]
    hidden = manifest["hidden"]

    def vec(region_name: str, ref: dict[str, int]) -> torch.Tensor:
        return torch.from_numpy(read_arr(region_name, ref, np.float32, (hidden,)).copy())

    out: dict[str, torch.Tensor] = {}
    tok_q = read_arr("main", manifest["table"]["codes"], np.int8, (vpad, hidden))
    tok_s = read_arr("main", manifest["table"]["scales"], np.float32, (vpad,))
    out["model.embeddings.tok_embeddings.weight"] = torch.from_numpy(dequant_i8(tok_q, tok_s)[:vocab].copy())
    dec_b = read_arr("main", manifest["decoder_bias"], np.float32, (vpad,))
    out["decoder.bias"] = torch.from_numpy(dec_b[:vocab].copy())
    out["model.embeddings.norm.weight"] = vec("main", manifest["emb_norm"])
    out["model.final_norm.weight"] = vec("main", manifest["final_norm"])
    out["head.dense.weight"] = torch.from_numpy(read_linear(manifest["head"]["dense"]).copy())
    out["head.norm.weight"] = vec("main", manifest["head"]["norm"])

    for i, layer in enumerate(manifest["layers"]):
        if layer["attn_norm"] is not None:
            out[f"model.layers.{i}.attn_norm.weight"] = vec("main", layer["attn_norm"])
        out[f"model.layers.{i}.attn.Wqkv.weight"] = torch.from_numpy(read_linear(layer["Wqkv"]).copy())
        out[f"model.layers.{i}.attn.Wo.weight"] = torch.from_numpy(read_linear(layer["Wo"]).copy())
        out[f"model.layers.{i}.mlp_norm.weight"] = vec("main", layer["mlp_norm"])
        out[f"model.layers.{i}.mlp.Wi.weight"] = torch.from_numpy(read_linear(layer["Wi"]).copy())
        out[f"model.layers.{i}.mlp.Wo.weight"] = torch.from_numpy(read_linear(layer["Wo_mlp"]).copy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack HF ModernBERT-SPLADE checkpoint for browser inference.")
    parser.add_argument("--hf", type=Path, required=True, help="Input Hugging Face directory")
    parser.add_argument("--out", type=Path, required=True, help="Output pack directory")
    parser.add_argument("--quant", choices=["int8", "q4"], required=True, help="Quantization mode")
    args = parser.parse_args()

    cfg = json.loads((args.hf / "config.json").read_text(encoding="utf-8"))
    tok = json.loads((args.hf / "tokenizer.json").read_text(encoding="utf-8"))
    sd = load_file(str(args.hf / "model.safetensors"))

    n_layers = int(cfg["num_hidden_layers"])
    _validate_keys(sd, n_layers)

    vocab = int(cfg["vocab_size"])
    hidden = int(cfg["hidden_size"])
    vpad = math.ceil(vocab / 32) * 32

    query_prefix = cfg["query_prefix"]
    document_prefix = cfg["document_prefix"]
    added_ids = {t["content"]: int(t["id"]) for t in tok.get("added_tokens", [])}

    main_reg = Region()
    layers_reg = Region()
    max_err: dict[str, float] = {}

    def record_err(kind: str, err: float) -> None:
        max_err[kind] = max(max_err.get(kind, 0.0), err)

    # Token table (always i8, padded to vpad); also the tied decoder.
    tok_w = sd["model.embeddings.tok_embeddings.weight"].astype(np.float32)
    tok_q, tok_s = quant_i8(tok_w)
    record_err("tok_embeddings", float(np.max(np.abs(tok_w - dequant_i8(tok_q, tok_s)))))
    tok_q_pad = np.zeros((vpad, hidden), dtype=np.int8)
    tok_q_pad[:vocab] = tok_q
    tok_s_pad = np.ones((vpad,), dtype=np.float32)
    tok_s_pad[:vocab] = tok_s
    table_ref = {"codes": main_reg.add(tok_q_pad), "scales": main_reg.add(tok_s_pad)}

    dec_b_pad = np.zeros((vpad,), dtype=np.float32)
    dec_b_pad[:vocab] = sd["decoder.bias"].astype(np.float32)
    decoder_bias_ref = main_reg.add(dec_b_pad)

    fold_pad = np.arange(vpad, dtype=np.uint32)
    fold_pad[:vocab] = sd["vocab_fold_index"].astype(np.uint32)
    fold_ref = main_reg.add(fold_pad)

    emb_norm_ref = main_reg.add(sd["model.embeddings.norm.weight"].astype(np.float32))
    final_norm_ref = main_reg.add(sd["model.final_norm.weight"].astype(np.float32))

    head_dense_ref, err = _add_linear(main_reg, "main", sd["head.dense.weight"].astype(np.float32), "i8")
    record_err("head.dense", err)
    head_norm_ref = main_reg.add(sd["head.norm.weight"].astype(np.float32))

    layer_fmt = "q4" if args.quant == "q4" else "i8"
    layer_types = cfg["layer_types"]
    layers_manifest = []
    for i in range(n_layers):
        attn_norm_ref = main_reg.add(sd[f"model.layers.{i}.attn_norm.weight"].astype(np.float32)) if i > 0 else None
        mlp_norm_ref = main_reg.add(sd[f"model.layers.{i}.mlp_norm.weight"].astype(np.float32))
        refs = {}
        for name, key, fmt in (("Wqkv", "attn.Wqkv", layer_fmt), ("Wo", "attn.Wo", layer_fmt),
                               ("Wi", "mlp.Wi", layer_fmt), ("Wo_mlp", "mlp.Wo", "i8")):
            refs[name], err = _add_linear(layers_reg, "layers", sd[f"model.layers.{i}.{key}.weight"].astype(np.float32), fmt)
            record_err(key, err)
        layers_manifest.append({"global": layer_types[i] == "full_attention", "attn_norm": attn_norm_ref,
                                "Wqkv": refs["Wqkv"], "Wo": refs["Wo"], "mlp_norm": mlp_norm_ref,
                                "Wi": refs["Wi"], "Wo_mlp": refs["Wo_mlp"]})

    weights_bin = bytearray(main_reg.buf)
    main_len = len(main_reg.buf)
    weights_bin.extend(b"\x00" * ((-len(weights_bin)) % 256))
    layers_offset = len(weights_bin)
    weights_bin.extend(layers_reg.buf)
    layers_len = len(layers_reg.buf)

    manifest = {
        "format": "sparseup-pack-1",
        "quant": args.quant,
        "hidden": hidden,
        "heads": int(cfg["num_attention_heads"]),
        "inter": int(cfg["intermediate_size"]),
        "n_layers": n_layers,
        "eps": float(cfg["norm_eps"]),
        "theta_global": float(cfg["rope_parameters"]["full_attention"]["rope_theta"]),
        "theta_local": float(cfg["rope_parameters"]["sliding_attention"]["rope_theta"]),
        "window": int(cfg["local_attention"]),
        "vocab": vocab,
        "vpad": vpad,
        "shift": float(cfg["logit_shift"]),
        "sigma": float(cfg["logit_sigma"]),
        "top_k": int(cfg["position_top_k"]),
        "cls": int(cfg["cls_token_id"]),
        "sep": int(cfg["sep_token_id"]),
        "query_prefix": query_prefix,
        "document_prefix": document_prefix,
        "query_prefix_id": added_ids[query_prefix],
        "document_prefix_id": added_ids[document_prefix],
        "add_prefix_space": bool(cfg["add_prefix_space"]),
        "regions": {"main": {"offset": 0, "length": main_len},
                    "layers": {"offset": layers_offset, "length": layers_len}},
        "table": table_ref,
        "decoder_bias": decoder_bias_ref,
        "fold": fold_ref,
        "emb_norm": emb_norm_ref,
        "final_norm": final_norm_ref,
        "head": {"dense": head_dense_ref, "norm": head_norm_ref},
        "layers": layers_manifest,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (args.out / "weights.bin").write_bytes(weights_bin)
    shutil.copy2(args.hf / "tokenizer.json", args.out / "tokenizer.json")
    shutil.copy2(args.hf / "tokenizer_config.json", args.out / "tokenizer_config.json")

    print(f"main:   {main_len:,} bytes")
    print(f"layers: {layers_len:,} bytes")
    print(f"total:  {len(weights_bin):,} bytes")
    for kind, e in max_err.items():
        print(f"max abs err [{kind}]: {e:.6f}")


if __name__ == "__main__":
    main()
