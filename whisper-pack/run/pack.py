"""Pack an OpenAI Whisper checkpoint into weights.bin + manifest.json for the browser engine.

    python3 run/pack.py --model openai/whisper-base --scheme "tab=g32r7,enc=r31,dec=r31,conv=f16" \
        --out ../out/packs/<name>

Specs per part (grammar of quant.py, restricted to what the engine reads):
  r<L>   L <= 127, int8 codes + scale per output row, scale = fp16(max|w| / L) exactly as
         quant.quantize computes it, so q * s equals quant.apply_scheme bit for bit
  g32r7  `tab` only: nibble-packed 4-bit table, stored code = value + 8, fp16 scale per 32 columns
  f16 | f32   any part

Record layouts (all offsets absolute and 4-byte aligned, one weights.bin):
  matmul   {"q": int8 [N,K] (one row per output, read by the GPU as is), "s": float32 [N], "K", "N", "b": float16 [N] or absent}
           f16/f32 parts: {"f16_t" | "f32_t": [N,K] (torch layout), "K", "N", "b"}
  conv     weights stored [out, k, in]; {"kind": "f16"|"f32", "w", "b"} or
           {"kind": "int8", "q": int8 [out,k,in], "s": float32 [out], "b"}; plus in/out/k
  table    {"bits": 4, "V", "H", "codes", "scales"}   (the decider's format)
           {"bits": 8, "V", "H", "codes": int8 [V,H], "scales": float32 [V]}
           {"bits": 16|32, "V", "H", "f16"|"f32": [V,H]}
  norms/biases/dec.pos: float16 records; {"w", "b"} for layer norms.
Encoder positions are not stored; the reader regenerates the sinusoids.
"""
import argparse
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quant  # noqa: E402

N_AUDIO_CTX = 1500


def sinusoids(length, channels):
    """Whisper's encoder position table."""
    inc = np.log(10000) / (channels // 2 - 1)
    inv = np.exp(-inc * np.arange(channels // 2, dtype=np.float64))
    t = np.arange(length, dtype=np.float64)[:, None] * inv[None]
    # the checkpoints hold this table rounded float64 -> float32 -> fp16 (unrounded differs by 2.4e-4)
    return np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32).astype(np.float16).astype(np.float32)


class Packer:
    def __init__(self):
        self.parts, self.off = [], 0

    def add(self, arr, dtype):
        a = np.ascontiguousarray(np.asarray(arr).astype(dtype))
        pad = (-self.off) % 4
        if pad:
            self.parts.append(b"\0" * pad)
            self.off += pad
        rec = {"offset": self.off, "length": a.nbytes, "dims": list(a.shape), "dtype": str(a.dtype)}
        self.parts.append(a.tobytes())
        self.off += a.nbytes
        return rec

    def bytes(self):
        return b"".join(self.parts)


def check_spec(part, spec):
    if spec in ("f16", "f32"):
        return
    m = re.fullmatch(r"r(\d+)", spec)
    if m and 1 <= int(m[1]) <= 127:
        return
    if part == "tab" and spec == "g32r7":
        return
    raise ValueError(f"unsupported spec {spec!r} for part {part!r}: allowed r<L> (L<=127), f16, f32"
                     + (", g32r7" if part == "tab" else ""))


def row_codes(w, L):
    """Per-row int8 codes and fp32 scale, same arithmetic as quant.quantize (no clip search)."""
    flat = w.reshape(w.shape[0], -1).to(torch.float32)
    amax = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    s = quant._f16(amax * 1.0 / L).clamp_min(1e-12)
    c = torch.round(flat / s).clamp_(-L, L)
    return c, s


def build(model_id, scheme, out):
    from huggingface_hub import hf_hub_download
    from transformers import WhisperForConditionalGeneration

    specs = quant.parse(scheme)
    for part, sp in specs.items():
        check_spec(part, sp)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).float().eval()  # turbo ships fp16 and loads as fp16
    cfg, gc = model.config, model.generation_config
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    d = cfg.d_model
    assert torch.equal(sd["proj_out.weight"], sd["model.decoder.embed_tokens.weight"]), "output projection not tied"

    pos = sinusoids(N_AUDIO_CTX, d)
    diff = float(np.abs(pos - sd["model.encoder.embed_positions.weight"].numpy()).max())
    if diff > 1e-5:
        raise SystemExit(f"regenerated encoder sinusoids differ from the checkpoint by {diff:.3g} (> 1e-5)")

    P = Packer()
    f16 = lambda t: P.add(t.to(torch.float32).numpy(), np.float16)  # noqa: E731

    def exact(codes, s, w, spec):
        ref = quant.quantize(w, spec)[0]
        got = (codes * s).reshape(w.shape)
        assert torch.equal(got, ref), f"pack decodes differently from quant.quantize for spec {spec}"

    def lin(wkey, bkey=None):
        w = sd[wkey]
        spec = specs[quant.part_of(wkey)]
        N, K = w.shape
        if spec == "f16":
            r = {"f16_t": P.add(quant._f16(w).numpy(), np.float16), "K": K, "N": N}
        elif spec == "f32":
            r = {"f32_t": P.add(w.numpy(), np.float32), "K": K, "N": N}
        else:
            c, s = row_codes(w, int(spec[1:]))
            exact(c, s, w, spec)
            r = {"q": P.add(c.numpy(), np.int8), "s": P.add(s.reshape(-1).numpy(), np.float32), "K": K, "N": N}
        if bkey:
            r["b"] = f16(sd[bkey])
        return r

    def conv(i):
        key = f"model.encoder.conv{i}.weight"
        w = sd[key]
        spec = specs["conv"]
        out_c, in_c, k = w.shape
        r = {"in": in_c, "out": out_c, "k": k}
        wt = w.permute(0, 2, 1).contiguous()  # [out, k, in]
        if spec == "f16":
            r.update(kind="f16", w=P.add(quant._f16(wt).numpy(), np.float16))
        elif spec == "f32":
            r.update(kind="f32", w=P.add(wt.numpy(), np.float32))
        else:
            c, s = row_codes(w, int(spec[1:]))
            exact(c, s, w, spec)
            c = c.reshape(w.shape).permute(0, 2, 1).contiguous()
            r.update(kind="int8", q=P.add(c.numpy(), np.int8), s=P.add(s.reshape(-1).numpy(), np.float32))
        r["b"] = f16(sd[f"model.encoder.conv{i}.bias"])
        return r

    def norm(prefix):
        return {"w": f16(sd[prefix + ".weight"]), "b": f16(sd[prefix + ".bias"])}

    def table():
        T = sd["model.decoder.embed_tokens.weight"]
        V, H = T.shape
        spec = specs["tab"]
        r = {"V": V, "H": H}
        if spec == "g32r7":
            assert H % 32 == 0
            g = T.reshape(-1, 32).to(torch.float32)
            amax = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            s = quant._f16(amax * 1.0 / 7).clamp_min(1e-12)
            c = torch.round(g / s).clamp_(-7, 7)
            exact(c, s, T, "g32r7")
            q = (c.to(torch.int16) + 8).to(torch.uint8).reshape(V, H).numpy()
            packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
            r.update(bits=4, codes=P.add(packed, np.uint8),
                     scales=P.add(s.reshape(V, H // 32, 1).numpy(), np.float16))
        elif spec == "f16":
            r.update(bits=16, f16=P.add(quant._f16(T).numpy(), np.float16))
        elif spec == "f32":
            r.update(bits=32, f32=P.add(T.numpy(), np.float32))
        else:
            c, s = row_codes(T, int(spec[1:]))
            exact(c, s, T, spec)
            r.update(bits=8, codes=P.add(c.numpy(), np.int8), scales=P.add(s.reshape(-1).numpy(), np.float32))
        return r

    m = {"format": "whisper-pack/1", "model": model_id, "scheme": scheme, "specs": specs,
         "dims": {"n_mels": cfg.num_mel_bins, "d_model": d, "enc_layers": cfg.encoder_layers,
                  "dec_layers": cfg.decoder_layers, "heads": cfg.encoder_attention_heads,
                  "vocab": cfg.vocab_size, "n_audio_ctx": N_AUDIO_CTX, "n_text_ctx": cfg.max_target_positions,
                  "ffn": cfg.encoder_ffn_dim, "ln_eps": 1e-5}}
    assert cfg.encoder_attention_heads == cfg.decoder_attention_heads
    m["conv1"], m["conv2"] = conv(1), conv(2)
    E = "model.encoder."
    enc = []
    for i in range(cfg.encoder_layers):
        p = f"{E}layers.{i}."
        enc.append({"ln1": norm(p + "self_attn_layer_norm"),
                    "q": lin(p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias"),
                    "k": lin(p + "self_attn.k_proj.weight"),
                    "v": lin(p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias"),
                    "o": lin(p + "self_attn.out_proj.weight", p + "self_attn.out_proj.bias"),
                    "ln2": norm(p + "final_layer_norm"),
                    "fc1": lin(p + "fc1.weight", p + "fc1.bias"),
                    "fc2": lin(p + "fc2.weight", p + "fc2.bias")})
    m["enc"] = {"layers": enc, "ln_post": norm(E + "layer_norm")}
    D = "model.decoder."
    m["dec"] = {"pos": f16(sd[D + "embed_positions.weight"]), "table": table()}
    dec = []
    for i in range(cfg.decoder_layers):
        p = f"{D}layers.{i}."
        L = {"ln1": norm(p + "self_attn_layer_norm")}
        for n, a in (("s", "self_attn"), ("c", "encoder_attn")):
            L[n + "q"] = lin(f"{p}{a}.q_proj.weight", f"{p}{a}.q_proj.bias")
            L[n + "k"] = lin(f"{p}{a}.k_proj.weight")
            L[n + "v"] = lin(f"{p}{a}.v_proj.weight", f"{p}{a}.v_proj.bias")
            L[n + "o"] = lin(f"{p}{a}.out_proj.weight", f"{p}{a}.out_proj.bias")
            if n == "s":
                L["ln2"] = norm(p + "encoder_attn_layer_norm")
        L["ln3"] = norm(p + "final_layer_norm")
        L["fc1"] = lin(p + "fc1.weight", p + "fc1.bias")
        L["fc2"] = lin(p + "fc2.weight", p + "fc2.bias")
        dec.append(L)
    m["dec"]["layers"] = dec
    m["dec"]["ln"] = norm(D + "layer_norm")

    eot = gc.eos_token_id
    m["generation"] = {
        "multilingual": bool(getattr(gc, "is_multilingual", False)),
        "sot": gc.decoder_start_token_id, "eot": eot[0] if isinstance(eot, list) else eot,
        "no_timestamps": gc.no_timestamps_token_id,
        "task_to_id": dict(getattr(gc, "task_to_id", None) or {}),
        "lang_to_id": dict(getattr(gc, "lang_to_id", None) or {}),
        "suppress_tokens": list(gc.suppress_tokens), "begin_suppress_tokens": list(gc.begin_suppress_tokens)}

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = P.bytes()
    (out / "weights.bin").write_bytes(data)
    mj = json.dumps(m, indent=1).encode()
    (out / "manifest.json").write_bytes(mj)
    shutil.copyfile(hf_hub_download(model_id, "tokenizer.json"), out / "tokenizer.json")
    tok = (out / "tokenizer.json").read_bytes()
    return {"weights_bytes": len(data), "gzip_bytes": len(gzip.compress(data, 6)),
            "total_bytes": len(data) + len(mj) + len(tok),
            "total_gzip_bytes": len(gzip.compress(data, 6)) + len(gzip.compress(mj, 6)) + len(gzip.compress(tok, 6)),
            "pos_diff": diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        r = build(a.model, a.scheme, a.out)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    print(f"{a.out}: weights.bin {r['weights_bytes']} bytes, gzip-6 {r['gzip_bytes']} bytes; "
          f"whole pack (+manifest, tokenizer) {r['total_bytes']} / gzip-6 {r['total_gzip_bytes']}; "
          f"encoder sinusoid diff {r['pos_diff']:.2e}")


if __name__ == "__main__":
    main()
