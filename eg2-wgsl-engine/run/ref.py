"""NumPy forward pass of the k-layer EG2 cut, read through web/manifest.json, against onnxruntime.

Usage: python3 -I run/ref.py

This is the engine's algorithm in float32 NumPy, step for step (pool before the output
projection, as the engine does), so a mismatch in the browser can be bisected against it. It
also writes out/queries.json: the test queries tokenized exactly as the page does, and their
onnxruntime CPU vectors, which run/check.py compares the browser against.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "out" / "model"
QUERY = "task: search result | query: "
TEXTS = ["a cat asleep in a sunbeam", "foggy mountains at dawn", "watercolor of a city street", "a bird in the snow",
         "neon signs at night", "moss on a tree trunk", "a dog at the beach", "an old film camera",
         "who won the match in the last minute", "x",
         "Ørsted's offshore wind farm off the coast of Norway, photographed from a helicopter at sunset 🌅",
         "the quick brown fox jumps over the lazy dog " * 12]


def tensor(data, t):
    raw = data[t["offset"]:t["offset"] + t["length"]]
    dt = {"UINT8": np.uint8, "FLOAT": np.float32, "FLOAT16": np.float16}[t["dtype"]]
    return np.frombuffer(raw, dt).reshape(t["dims"])


def q4(data, w):
    """MatMulNBits weight -> float32 [N, K]."""
    N, K, B = w["N"], w["K"], w["block"]
    nb = K // B
    q = tensor(data, w["q"]).reshape(N, nb, B // 2)
    v = np.stack([q & 15, q >> 4], -1).reshape(N, nb, B).astype(np.float32)
    z = tensor(data, w["zp"]).reshape(N, -1)
    z = np.stack([z & 15, z >> 4], -1).reshape(N, -1)[:, :nb].astype(np.float32)
    s = tensor(data, w["scales"]).reshape(N, nb)
    return ((v - z[..., None]) * s[..., None]).reshape(N, K)


def table_rows(data, tbl, ids):
    codes = tensor(data, tbl["codes"])[ids]  # [S, 192]: 64 groups of 8 values in 3 bytes
    S = len(ids)
    b = codes.reshape(S, 64, 3).astype(np.uint32)
    word = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    v = ((word[..., None] >> (np.arange(8, dtype=np.uint32) * 3)) & 7).reshape(S, 16, 32).astype(np.float32)
    zp = tensor(data, tbl["zp"])[ids].astype(np.float32)
    sc = tensor(data, tbl["scales"])[ids].astype(np.float32)
    return ((v - zp[..., None]) * sc[..., None]).reshape(S, 512)


def rms(x, w=None, eps=1e-6):
    y = x * np.power((x * x).mean(-1, keepdims=True) + np.float32(eps), -0.5)
    return y if w is None else y * w


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.float32(0.7978845608028654) * (x + np.float32(0.044715) * x ** 3)))


class Ref:
    def __init__(self, man, data):
        self.m, self.d = man, data
        self.W = {}

    def w(self, key, spec):
        if key not in self.W:
            self.W[key] = q4(self.d, spec) if "N" in spec else tensor(self.d, spec).astype(np.float32)
        return self.W[key]

    def __call__(self, ids):
        m, eps = self.m, self.m["eps"]
        S = len(ids)
        x = table_rows(self.d, m["table"], np.asarray(ids)) * np.float32(m["embed_scale"])
        ple = (x @ self.w("ple", m["ple"]).T) * np.float32(m["ple_scale"])
        ple = rms(ple.reshape(S, m["ple_layers"], m["ple_dim"]), self.w("pln", m["per_layer_projection_norm"]), eps)
        pos = np.arange(S, dtype=np.float32)[:, None] * np.asarray(m["inv_freq"], np.float32)
        ang = np.concatenate([pos, pos], -1)
        cos, sin = np.cos(ang), np.sin(ang)
        h = x
        for i, L in enumerate(m["layers"]):
            W = lambda n: self.w(f"{i}.{n}", L[n])  # noqa: E731
            hd, nh, nkv = L["head_dim"], L["n_heads"], L["n_kv_heads"]
            a = rms(h, W("input_layernorm"), eps)
            q = (a @ W("q_proj").T).reshape(S, nh, hd)
            k = (a @ W("k_proj").T).reshape(S, nkv, hd)
            v = (a @ W("v_proj").T).reshape(S, nkv, hd)
            q, k, v = rms(q, W("q_norm"), eps), rms(k, W("k_norm"), eps), rms(v, None, eps)
            rot = lambda t: np.concatenate([-t[..., hd // 2:], t[..., :hd // 2]], -1)  # noqa: E731
            q = q * cos[:, None] + rot(q) * sin[:, None]
            k = k * cos[:, None] + rot(k) * sin[:, None]
            o = np.empty((S, nh, hd), np.float32)
            for hh in range(nh):
                kv = hh // (nh // nkv)
                sc = (q[:, hh] @ k[:, kv].T) * np.float32(L["attn_scale"])
                p = np.exp(sc - sc.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                o[:, hh] = p @ v[:, kv]
            t = o.reshape(S, nh * hd) @ W("o_proj").T
            h = h + rms(t, W("post_attention_layernorm"), eps)
            a = rms(h, W("pre_feedforward_layernorm"), eps)
            f = gelu(a @ W("gate_proj").T) * (a @ W("up_proj").T)
            h = h + rms(f @ W("down_proj").T, W("post_feedforward_layernorm"), eps)
            p = gelu(h @ W("per_layer_input_gate").T) * ple[:, i]
            h = (h + rms(p @ W("per_layer_projection").T, W("post_per_layer_input_norm"), eps)) * np.float32(L["layer_scalar"])
        pooled = rms(h, self.w("norm", m["norm"]), eps).mean(0)
        e = self.w("proj", m["embedding_projection"]) @ pooled
        return e / max(np.linalg.norm(e), 1e-12)


def main():
    man = json.loads((ROOT / "web" / "manifest.json").read_text())
    data = (MODEL / man["data"]).read_bytes()
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    specials = sorted((t.content for t in tok.get_added_tokens_decoder().values() if t.special), key=len, reverse=True)

    def clean(t):
        for sp in specials:
            t = t.replace(sp, " ")
        return t
    ids = [tok.encode(QUERY + clean(t)).ids[:128] for t in TEXTS]
    sess = ort.InferenceSession(str(MODEL / man["source"]), providers=["CPUExecutionProvider"])
    R = Ref(man, data)
    out = []
    for t, x in zip(TEXTS, ids):
        a = np.asarray(x, np.int64)[None]
        o = sess.run(["embedding"], {"input_ids": a, "attention_mask": np.ones_like(a)})[0][0]
        r = R(x)
        out.append({"text": t, "ids": x, "ort": o.tolist()})
        print(f"S={len(x):3d} cos(ref, ort)={float(r @ o / np.linalg.norm(o)):.7f}  {t[:40]!r}")
    (ROOT / "web" / "queries.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
