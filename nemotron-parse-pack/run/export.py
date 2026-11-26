"""Pack NVIDIA Nemotron Parse 2.0 or Nemotron 3.5 Parse weights for the WebGPU engine (web/engine/nemotron.js).
The model is recognised from the tensor names (2.0 has cross-attention; manifest.arch "np2" or "np35").

    python3 run/export.py --scheme scheme.json --out web/models/<name>/ [--weights model.safetensors]
                          [--tokenizer tokenizer.json] [--shard-mb 47] [--no-stats]

Writes manifest.json, tokenizer.json (slim: token strings + special ids) and gzip shards
weights-NNN.bin.gz, each at most --shard-mb MiB before compression (so at most that after).

scheme.json maps tensor-name regexes (re.search against the safetensors name, first match wins, in
file order) to a kind, either as a string or as {"kind": ..., "group": G}:

    {"group": 32,
     "radio_model\\.model\\.blocks\\..*\\.weight$": "int4g",
     "^decoder\\..*(proj|fc[12])\\.weight$": "int8",
     "embed_tokens": {"kind": "int8"}}

The top-level "group" key (an integer) is the default int4g group size. Unmatched tensors, and every
tensor with fewer than 2 dims (norms, biases), are f16.

Quantization math (pinned with the Kaggle accuracy study, so q * s here is bit-identical there):
  shape   ndim > 2 with a leading 1 drops it (pos_embed [1, 16384, 1280] -> rows are positions);
          any other ndim > 2 flattens to [out, in*kh*kw] (conv1, conv2).
  int8    per output row: s = f16(max|w_row| / 127), q = clamp(round(w / f32(s)), -127, 127)
  int4g   per output row per G consecutive inputs: s = f16(max|w_group| / 7),
          q = clamp(round(w / f32(s)), -7, 7), stored as q + 8 in a nibble (low nibble = even index);
          when K is not a multiple of G the last group is the K % G columns left over (scales [N, ceil(K/G)])
  round is torch.round (half to even); a zero max gives s = 1 (as does a max so small that the
  f16 scale underflows to 0); dequant = q * f32(s).
  f16     w.half()
lm_head is tied to decoder.embed_tokens.weight and is stored once under that name.

Byte stream: all records back to back, each 4-byte aligned, cut into shards at fixed raw size.
manifest.tensors[name] = {kind, shape: [N, K] (or the 1-D shape), group?, q|w: [offset, length],
s: [offset, length]} with offsets into the unsharded stream. q for int4g is N*K/2 bytes, s is f16.
"""
import argparse
import gzip
import hashlib
import json
import re
import zlib
from pathlib import Path

import numpy as np
import torch

FORMAT = "nemotron-pack/1"
SKIP = re.compile(r"summary_idxs|input_conditioner|^lm_head\.|^decoder\.extra_(heads|proj)\.")   # extra_*: 3.5's multi-token-prediction heads, unused in plain decoding
DEFAULT_SCHEME = {
    "group": 32,
    r"radio_model\.model\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])\.weight$": "int4g",
    r"^decoder\.layers\.\d+\..*(_proj|fc[12])\.weight$": "int8",
    r"^decoder\.embed_tokens\.weight$": "int8",
}


def spec_for(name, shape, scheme):
    """(kind, group) for a tensor name under a scheme dict."""
    if len(shape) < 2:
        return "f16", None
    default_group = scheme.get("group", 32) if isinstance(scheme.get("group", 32), int) else 32
    for pat, val in scheme.items():
        if pat == "group" and isinstance(val, int):
            continue
        if re.search(pat, name):
            if isinstance(val, str):
                val = {"kind": val}
            kind = val["kind"]
            if kind not in ("f16", "int8", "int4g"):
                raise ValueError(f"{pat}: unknown kind {kind!r}")
            return kind, (int(val.get("group", default_group)) if kind == "int4g" else None)
    return "f16", None


def as_rows(w):
    """The 2-D [rows, inputs] view quantization runs over."""
    if w.ndim > 2 and w.shape[0] == 1:
        w = w[0]
    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)
    return w


def _scale(mx, levels):
    s = (mx / levels).half()
    s = torch.where((mx == 0) | (s == 0), torch.ones_like(s), s)
    return s


def quantize(w, kind, group=None):
    """w: float32 tensor (any shape). Returns (codes int8 [N, K] or None, scales f16 or None, dequantized float32 in w's shape)."""
    w = w.float()
    if kind == "f16" or w.ndim < 2:
        h = w.half()
        if not torch.isfinite(h).all():
            raise ValueError("f16 overflow")
        return None, None, h.float()
    r = as_rows(w)
    N, K = r.shape
    if kind == "int8":
        s = _scale(r.abs().amax(dim=1), 127)
        q = torch.clamp(torch.round(r / s.float()[:, None]), -127, 127)
        d = q * s.float()[:, None]
    elif kind == "int4g":
        if K % 8 or group % 8:
            raise ValueError(f"int4g needs K and group to be multiples of 8 (K={K}, group={group})")
        pad = (-K) % group   # a last group shorter than `group` (3.5's fc2, K = 4304) takes its scale from the columns it has
        g = torch.nn.functional.pad(r, (0, pad)).reshape(N, -1, group)
        s = _scale(g.abs().amax(dim=2), 7)
        q = torch.clamp(torch.round(g / s.float()[:, :, None]), -7, 7)
        d = (q * s.float()[:, :, None]).reshape(N, -1)[:, :K]
        q = q.reshape(N, -1)[:, :K]
    else:
        raise ValueError(kind)
    return q.to(torch.int8), s, d.reshape(w.shape)


def dequantized(name, w, scheme):
    """The float32 values the engine computes with for tensor `name` under `scheme`."""
    kind, group = spec_for(name, tuple(w.shape), scheme)
    return quantize(w, kind, group)[2]


def component(name):
    if name == "decoder.embed_tokens.weight":
        return "embedding"
    if name.startswith("decoder."):
        return "decoder"
    if "model_encoder" in name:
        return "encoder"
    return "neck"


class Stream:
    """Appends 4-byte-aligned records, cutting gzip shards at a fixed raw size."""

    def __init__(self, out, shard_bytes, stats):
        self.out, self.size, self.off = out, shard_bytes, 0
        self.buf, self.shards = bytearray(), []
        self.stats = {} if stats else None
        self.sha = hashlib.sha256()

    def add(self, arr, comp):
        b = np.ascontiguousarray(arr).tobytes()
        b += b"\0" * ((-len(b)) % 4)
        rec = [self.off, arr.nbytes]
        self.off += len(b)
        self.buf += b
        self.sha.update(b)
        if self.stats is not None:
            st = self.stats.setdefault(comp, {"raw": 0, "gzip": 0, "z": zlib.compressobj(6, zlib.DEFLATED, 31)})
            st["raw"] += len(b)
            st["gzip"] += len(st["z"].compress(b))
        while len(self.buf) >= self.size:
            self._cut(self.size)
        return rec

    def _cut(self, n):
        name = f"weights-{len(self.shards):03d}.bin.gz"
        with open(self.out / name, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", compresslevel=6, mtime=0) as g:
            g.write(bytes(self.buf[:n]))
        del self.buf[:n]
        self.shards.append({"file": name, "bytes": n, "gzip": (self.out / name).stat().st_size})
        print(f"  {name}: {n} -> {self.shards[-1]['gzip']}", flush=True)

    def close(self):
        if self.buf:
            self._cut(len(self.buf))
        if self.stats is not None:
            for st in self.stats.values():
                st["gzip"] += len(st.pop("z").flush())


def slim_tokenizer(path):
    t = json.loads(Path(path).read_text())
    vocab = t["model"]["vocab"]
    n = max(max(vocab.values()), max(a["id"] for a in t["added_tokens"])) + 1
    tokens = [""] * n
    for s, i in vocab.items():
        tokens[i] = s
    for a in t["added_tokens"]:
        tokens[a["id"]] = a["content"]
    return {"format": "nemotron-tok/1", "decoder": t["decoder"]["type"], "tokens": tokens,
            "special": sorted(a["id"] for a in t["added_tokens"] if a["special"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", help="scheme.json (default: the built-in test scheme)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/home/user/np-work/hf/model.safetensors")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (default: next to the weights)")
    ap.add_argument("--shard-mb", type=float, default=47)
    ap.add_argument("--no-stats", action="store_true", help="skip the per-component gzip sizes")
    a = ap.parse_args()
    from safetensors import safe_open

    scheme = json.loads(Path(a.scheme).read_text()) if a.scheme else DEFAULT_SCHEME
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("weights-*.bin.gz"):
        old.unlink()
    st = Stream(out, int(a.shard_mb * 1024 * 1024) // 4 * 4, not a.no_stats)
    tensors, kinds = {}, {}
    with safe_open(a.weights, "pt") as f:
        names = [k for k in f.keys() if not SKIP.search(k)]
        names.sort(key=lambda k: (["encoder", "neck", "decoder", "embedding"].index(component(k)), k))
        for name in names:
            w = f.get_tensor(name).float()
            kind, group = spec_for(name, tuple(w.shape), scheme)
            q, s, _ = quantize(w, kind, group)
            comp = component(name)
            rec = {"kind": kind}
            if kind == "f16":
                rec["shape"] = list(as_rows(w).shape) if w.ndim >= 2 else list(w.shape)
                rec["w"] = st.add((as_rows(w) if w.ndim >= 2 else w).half().numpy(), comp)
            else:
                N, K = q.shape
                rec["shape"] = [N, K]
                if kind == "int8":
                    rec["q"] = st.add(q.numpy(), comp)
                else:
                    u = (q.numpy() + 8).astype(np.uint8)
                    rec["group"] = group
                    rec["q"] = st.add(u[:, 0::2] | (u[:, 1::2] << 4), comp)
                rec["s"] = st.add(s.numpy(), comp)
            tensors[name] = rec
            kinds[kind] = kinds.get(kind, 0) + w.numel()
    st.close()
    tok = slim_tokenizer(a.tokenizer or Path(a.weights).with_name("tokenizer.json"))
    (out / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False, separators=(",", ":")))
    arch = "np2" if "decoder.layers.0.encoder_attn.q_proj.weight" in tensors else "np35"
    shape = lambda n: tensors[n]["shape"]
    R = "encoder.model_encoder.radio_model.model."
    E, F = shape(R + "blocks.0.mlp.fc1.weight")[1], shape(R + "blocks.0.mlp.fc1.weight")[0]
    layers = lambda pre: 1 + max(int(k[len(pre):].split(".")[0]) for k in tensors if k.startswith(pre))
    prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_" + ("no_text_in_pic>" if arch == "np2" else "text_in_pic>")
    manifest = {
        "format": FORMAT,
        "id": st.sha.hexdigest()[:16],
        "model": "nvidia/NVIDIA-Nemotron-Parse-2.0" if arch == "np2" else "nvidia/Nemotron-3.5-Parse",
        "arch": arch,
        "scheme": scheme,
        "dims": {"patch": 16, "enc_width": E, "enc_heads": 16, "enc_mlp": F, "enc_layers": layers(R + "blocks."),
                 "enc_prefix": shape(R + "patch_generator.cls_token.token")[0], "enc_summary": shape("encoder.sum_proj.weight")[1] // E,
                 "pos_grid": 128, "enc_eps": 1e-6, "neck_eps": 1e-6, "d_model": 1024, "dec_heads": 16,
                 "dec_kv_heads": shape("decoder.layers.0.self_attn.k_proj.weight")[0] // 64,
                 "dec_ffn": 4096, "dec_layers": layers("decoder.layers."), "dec_eps": 1e-5, "vocab": len(tok["tokens"]), "embed_scale": 32.0},
        "image": {"height": 2048, "width": 1664, "mean": [0.48145466, 0.4578275, 0.40821073],
                  "std": [0.26862954, 0.26130258, 0.27577711]},
        "generation": {"eos": 2, "repetition_penalty": 1.1, "max_new_tokens": 9000 if arch == "np2" else 8744, "prompt": prompt},
        "bytes": st.off,
        "shards": st.shards,
        "tensors": tensors,
    }
    if st.stats is not None:
        manifest["sizes"] = st.stats
    (out / "manifest.json").write_text(json.dumps(manifest))
    gz = sum(s["gzip"] for s in st.shards)
    print(f"{out}: {len(st.shards)} shards, {st.off} bytes raw, {gz} gzip; params by kind {kinds}")
    if st.stats is not None:
        for c, v in st.stats.items():
            print(f"  {c:10s} raw {v['raw'] / 1e6:8.1f} MB  gzip {v['gzip'] / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
