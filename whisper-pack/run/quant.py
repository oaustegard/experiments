"""Weight quantization schemes for Whisper packs.

apply_scheme(model, scheme) rewrites a transformers WhisperForConditionalGeneration
in place with the values a pack would decode to, and returns the bytes that pack
would occupy. The arithmetic is the pack's own (integer codes times an fp16 scale),
so a WER measured on the rewritten model is the WER of the packed weights, up to
the float error of the runtime.

Scheme grammar: comma-separated `part=spec`, later entries override earlier ones.

  parts   tab        decoder token table (tied to the output projection, stored once)
          enc        encoder layer matrices (q, k, v, out, fc1, fc2)
          dec        decoder layer matrices (self-attn, cross-attn, fc1, fc2)
          conv       the two conv1d stem weights
          enc.attn enc.fc dec.self dec.cross dec.fc    finer overrides
          all        tab + enc + dec + conv

  specs   f32 | f16
          r<L>       one scale per row (output channel); codes in [-L, L]
          g<G>r<L>   one scale per group of G consecutive weights within a row
          ...m       suffix: pick each scale by a clip search that minimises squared
                     error instead of using max-abs

Everything not named by a part (biases, layer norms, decoder position table) is
fp16 in any pack; the bare scheme "f32" leaves the whole model untouched and
counts 4 bytes per parameter. The encoder position table is the fixed sinusoid
and is regenerated at load, so it costs 0 bytes.

Byte counts per quantized tensor:
  raw      codes bit-packed at ceil(log2(2L+1)) bits, plus 2 bytes per scale
  entropy  order-0 entropy of the codes (what gzip or brotli approach), plus scales
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch

PARTS = ("tab", "enc.attn", "enc.fc", "dec.self", "dec.cross", "dec.fc", "conv")
ALIASES = {
    "all": PARTS,
    "enc": ("enc.attn", "enc.fc"),
    "dec": ("dec.self", "dec.cross", "dec.fc"),
}
CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4)


def part_of(name: str) -> str | None:
    """Map a state-dict key to its part, or None for tensors that stay fp16."""
    if name == "model.decoder.embed_tokens.weight":
        return "tab"
    if name == "proj_out.weight":
        return "tied"
    if re.fullmatch(r"model\.encoder\.conv[12]\.weight", name):
        return "conv"
    m = re.fullmatch(r"model\.(encoder|decoder)\.layers\.\d+\.(fc1|fc2)\.weight", name)
    if m:
        return ("enc" if m[1] == "encoder" else "dec") + ".fc"
    m = re.fullmatch(r"model\.(encoder|decoder)\.layers\.\d+\.(self_attn|encoder_attn)\.\w+_proj\.weight", name)
    if m:
        if m[1] == "encoder":
            return "enc.attn"
        return "dec.self" if m[2] == "self_attn" else "dec.cross"
    return None


def parse(scheme: str) -> dict:
    spec = {p: "f32" for p in PARTS}
    if scheme == "f32":
        return spec
    for item in scheme.split(","):
        part, _, s = item.partition("=")
        if not s:
            raise ValueError(f"bad scheme item {item!r}")
        for p in ALIASES.get(part, (part,)):
            if p not in spec:
                raise ValueError(f"unknown part {p!r}")
            spec[p] = s
    return spec


def _f16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16).to(torch.float32)


def quantize(w: torch.Tensor, spec: str):
    """Return (dequantized weight, info) for one tensor. Rows are dim 0."""
    n = w.numel()
    if spec == "f32":
        return w, {"raw_bytes": 4 * n, "entropy_bytes": 4 * n, "bits": 32.0}
    if spec == "f16":
        return _f16(w), {"raw_bytes": 2 * n, "entropy_bytes": 2 * n, "bits": 16.0}
    m = re.fullmatch(r"(?:g(\d+))?r(\d+)(m?)", spec)
    if not m:
        raise ValueError(f"bad spec {spec!r}")
    group = int(m[1]) if m[1] else None
    L = int(m[2])
    search = bool(m[3])
    rows = w.shape[0]
    flat = w.reshape(rows, -1).to(torch.float32)
    cols = flat.shape[1]
    pad = (-cols) % group if group else 0
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    g = flat.reshape(-1, group) if group else flat
    amax = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)

    def run(clip):
        s = _f16(amax * clip / L).clamp_min(1e-12)
        c = torch.round(g / s).clamp_(-L, L)
        return s, c

    if search:
        best_err = torch.full((g.shape[0],), float("inf"))
        scale = torch.empty_like(amax)
        codes = torch.empty_like(g)
        for clip in CLIPS:
            s, c = run(clip)
            err = ((c * s - g) ** 2).sum(dim=1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            scale[better] = s[better]
            codes[better] = c[better]
    else:
        scale, codes = run(1.0)
    deq = (codes * scale).reshape(flat.shape)[:, :cols].reshape(w.shape)
    used = codes.reshape(flat.shape)[:, :cols]
    counts = np.bincount((used.to(torch.int64) + L).flatten().numpy(), minlength=2 * L + 1)
    p = counts[counts > 0] / counts.sum()
    ent_bits = float(-(p * np.log2(p)).sum())
    bits = math.ceil(math.log2(2 * L + 1))
    scale_bytes = 2 * scale.numel()
    return deq.to(w.dtype), {
        "raw_bytes": math.ceil(bits * n / 8) + scale_bytes,
        "entropy_bytes": math.ceil(ent_bits * n / 8) + scale_bytes,
        "bits": bits,
        "entropy_bits": round(ent_bits, 3),
        "rel_err": float(((deq - w).norm() / w.norm()).item()),
    }


@torch.no_grad()
def apply_scheme(model, scheme: str) -> dict:
    spec = parse(scheme)
    parts: dict = {}

    def add(part, params, raw, ent, rel=None):
        d = parts.setdefault(part, {"params": 0, "raw_bytes": 0, "entropy_bytes": 0, "rel_err": []})
        d["params"] += params
        d["raw_bytes"] += raw
        d["entropy_bytes"] += ent
        if rel is not None:
            d["rel_err"].append(rel)

    dev = next(model.parameters()).device
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        part = part_of(name)
        n = p.numel()
        if part == "tied":  # only reached if the tie was broken; count it with the table
            part = "tab"
        if name == "model.encoder.embed_positions.weight":
            b = 4 * n if scheme == "f32" else 0
            add("enc.pos", n, b, b)
            continue
        if part is None:
            if scheme == "f32":
                add("other", n, 4 * n, 4 * n)
            else:
                p.copy_(_f16(p))
                add("other", n, 2 * n, 2 * n)
            continue
        deq, info = quantize(p.data.cpu(), spec[part])
        p.copy_(deq.to(dev))
        add(part, n, info["raw_bytes"], info["entropy_bytes"], info.get("rel_err"))
    for d in parts.values():
        r = d.pop("rel_err")
        if r:
            d["rel_err_mean"] = round(float(np.mean(r)), 5)
    return {
        "scheme": scheme,
        "raw_bytes": sum(d["raw_bytes"] for d in parts.values()),
        "entropy_bytes": sum(d["entropy_bytes"] for d in parts.values()),
        "gzip_bytes": None,
        "parts": parts,
    }


if __name__ == "__main__":
    import sys

    from transformers import WhisperForConditionalGeneration

    mid = sys.argv[1]
    for sch in sys.argv[2:]:
        mdl = WhisperForConditionalGeneration.from_pretrained(mid)
        r = apply_scheme(mdl, sch)
        print(f"{mid} {sch}: raw {r['raw_bytes'] / 1e6:.1f} MB, entropy {r['entropy_bytes'] / 1e6:.1f} MB", flush=True)
        for k, d in r["parts"].items():
            print(f"   {k:10s} {d['params'] / 1e6:8.2f}M  raw {d['raw_bytes'] / 1e6:8.2f}  ent {d['entropy_bytes'] / 1e6:8.2f}  rel {d.get('rel_err_mean', '')}")
