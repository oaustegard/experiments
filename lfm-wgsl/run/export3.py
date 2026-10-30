"""Export LFM3-1.7B gq4 (results/quant3.json) as one streamable file plus a manifest for web/lfm3.js.

  python3 lfm-wgsl/run/export3.py      -> model3/web/lfm3-gq4.bin, web/manifest3.json

The file is the engine's GPU buffers back to back, each 256-byte aligned, so the page can stream it
and write every chunk straight into the buffer that owns those bytes (manifest `buffers`: name,
file offset, size). Per layer:
  L{i}.norms  f32: operator, post-operator, ffn, post-ffn norms (2048 each), then the conv taps
              [2048 x 7] or the q and k head norms (128 each)
  L{i}.qkvg   attention: q (2048) | k (512) | v (512) | gate (16) rows, zero-padded to a multiple of 32
  L{i}.in     conv in_proj (6144 rows)
  L{i}.out    conv out_proj / attention o_proj
  L{i}.w13    w1 (7680 rows) | w3 (7680 rows)
  L{i}.w2     w2
A gq4 matrix [N, K] is N*K/2 bytes of codes (row-major, two per byte, low nibble first; code c means
Lloyd-Max level LEVELS[c]) followed by N*K/32 fp16 scales. The matrix acts on activations rotated
in 256-wide blocks: x' = H diag(SIGNS) x / 16 per block, H the Sylvester Hadamard matrix.
The token table (also lm_head) is 4-bit round-to-nearest in MatMulNBits layout: codes, f32 scales,
packed zero points, block 32.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3 import M3, ROOT, bf16, config  # noqa: E402
import quant3 as q  # noqa: E402

ALIGN = 256


def gq_bytes(P, names, pad_to=32):
    """Concatenate gq4 matrices by rows (same K), pad rows to a multiple of pad_to; return bytes, N, K."""
    codes = torch.cat([P[n + ".codes"] for n in names])
    scales = torch.cat([P[n + ".scales"] for n in names])
    N, K = codes.shape[0], codes.shape[1] * 2
    pad = (-N) % pad_to
    if pad:
        codes = torch.cat([codes, torch.zeros(pad, codes.shape[1], dtype=torch.uint8)])
        scales = torch.cat([scales, torch.zeros(pad, scales.shape[1], dtype=torch.float16)])
    return codes.numpy().tobytes() + scales.numpy().view(np.uint8).tobytes(), N + pad, K, N


def rtn4_table(w):
    """MatMulNBits-layout 4-bit RTN of the token table, the same math as quant3.rtn."""
    V, K = w.shape
    W = w.float().reshape(V, K // 32, 32)
    lo, hi = W.min(-1, keepdim=True).values, W.max(-1, keepdim=True).values
    s = ((hi - lo) / 15).clamp(min=1e-12)
    z = torch.round(-lo / s).clamp(0, 15)
    c = torch.clamp(torch.round(W / s) + z, 0, 15).to(torch.uint8).reshape(V, K)
    codes = (c[:, 0::2] | (c[:, 1::2] << 4)).numpy().tobytes()
    zz = z[..., 0].to(torch.uint8)
    zp = (zz[:, 0::2] | (zz[:, 1::2] << 4)).numpy().tobytes()
    return codes, s[..., 0].numpy().astype(np.float32).tobytes(), zp


def main():
    cfg = config()
    P = load_file(str(M3 / "gq4_codes.safetensors"))
    W = bf16()
    out_dir = M3 / "web"; out_dir.mkdir(exist_ok=True)
    blobs, man = [], {"buffers": [], "layers": []}
    off = 0

    def add(name, data):
        nonlocal off
        data = bytes(data)
        pad = (-len(data)) % ALIGN
        blobs.append(data + b"\0" * pad)
        man["buffers"].append({"name": name, "offset": off, "size": len(data)})
        off += len(data) + pad

    f32 = lambda t: t.float().numpy().astype(np.float32).tobytes()  # noqa: E731
    codes, scales, zp = rtn4_table(W["model.embed_tokens.weight"])
    add("embed", codes + scales + zp)
    V, D = W["model.embed_tokens.weight"].shape
    man["embed"] = {"buf": "embed", "q": 0, "s": len(codes) // 4, "z": (len(codes) + len(scales)) // 4, "K": D, "N": V}
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}."
        L = {"kind": kind}
        norms = [W[p + n + ".weight"] for n in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm")]
        if kind == "conv":
            norms.append(W[p + "conv.conv.weight"].reshape(-1))          # [2048, 1, 7] -> c * 7 + j
        else:
            norms += [W[p + "self_attn.q_norm.weight"], W[p + "self_attn.k_norm.weight"]]
        add(f"L{i}.norms", b"".join(f32(t) for t in norms))
        L["norms"] = f"L{i}.norms"
        if kind == "conv":
            b, N, K, n0 = gq_bytes(P, [p + "conv.in_proj.weight"]); add(f"L{i}.in", b); L["in"] = {"buf": f"L{i}.in", "N": N, "K": K}
        else:
            b, N, K, n0 = gq_bytes(P, [p + f"self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "gate_proj")])
            add(f"L{i}.qkvg", b); L["qkvg"] = {"buf": f"L{i}.qkvg", "N": N, "K": K, "rows": n0}
        o = p + ("conv.out_proj.weight" if kind == "conv" else "self_attn.o_proj.weight")
        b, N, K, _ = gq_bytes(P, [o]); add(f"L{i}.out", b); L["out"] = {"buf": f"L{i}.out", "N": N, "K": K}
        b, N, K, _ = gq_bytes(P, [p + "feed_forward.w1.weight", p + "feed_forward.w3.weight"]); add(f"L{i}.w13", b)
        L["w13"] = {"buf": f"L{i}.w13", "N": N, "K": K}
        b, N, K, _ = gq_bytes(P, [p + "feed_forward.w2.weight"]); add(f"L{i}.w2", b); L["w2"] = {"buf": f"L{i}.w2", "N": N, "K": K}
        man["layers"].append(L)
    add("final", f32(W["model.embedding_norm.weight"]))
    rp = cfg["rope_parameters"]
    man.update({
        "source": "LiquidAI/LFM3-1.7B bf16, rotated GPTQ 4-bit (lfm-wgsl/run/quant3.py gq4)",
        "file": "lfm3-gq4.bin", "file_bytes": off,
        "hidden": D, "ffn": cfg["intermediate_size"], "vocab": V, "heads": cfg["num_attention_heads"],
        "kv_heads": cfg["num_key_value_heads"], "head_dim": cfg["head_dim"], "eps": cfg["norm_eps"],
        "conv_L": cfg["conv_L_cache"], "window": cfg["sliding_window"],
        "rope": {k: {"theta": v["rope_theta"], "angles": int(v.get("partial_rotary_factor", 1.0) * cfg["head_dim"]) // 2}
                 for k, v in rp.items()},
        "bos": cfg["bos_token_id"], "eos": cfg["eos_token_id"],
        "levels": q.lloyd_max(4).float().tolist(),
        "signs": [float(x) for x in torch.from_numpy(np.random.default_rng(0).choice([-1.0, 1.0], q.ROT)).tolist()],
        "rot": q.ROT,
    })
    with open(out_dir / "lfm3-gq4.bin", "wb") as fh:
        for b in blobs:
            fh.write(b)
    (ROOT / "web" / "manifest3.json").write_text(json.dumps(man, separators=(",", ":")))
    print(f"{off / 1e6:.1f} MB, {len(man['buffers'])} buffers, largest {max(b['size'] for b in man['buffers']) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
