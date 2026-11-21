"""NumPy reference reader for a whisper-pack/1 directory: the spec the WGSL engine follows.

    p = load(pack_dir); states = p.encode(mel [n_mels, 3000]); ids, logits = p.decode(states, prompt_ids)

No torch. Weights are decoded to float32 at load (q * s, (nibble - 8) * scale). The only non-NumPy
import is scipy.special.erf for the exact GELU.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import erf

HERE = Path(__file__).resolve().parent


def sinusoids(length, channels):
    inc = np.log(10000) / (channels // 2 - 1)
    inv = np.exp(-inc * np.arange(channels // 2, dtype=np.float64))
    t = np.arange(length, dtype=np.float64)[:, None] * inv[None]
    # the checkpoints hold this table rounded float64 -> float32 -> fp16 (unrounded differs by 2.4e-4)
    return np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32).astype(np.float16).astype(np.float32)


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def layer_norm(x, n, eps):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * n["w"] + n["b"]


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class Pack:
    """A loaded pack: .encode(mel) and .decode(states, prompt_ids, max_new)."""

    def encode(self, mel):
        return encode(self, mel)

    def decode(self, states, prompt_ids, max_new=224):
        return decode(self, states, prompt_ids, max_new)


def load(pack_dir):
    pack_dir = Path(pack_dir)
    m = json.loads((pack_dir / "manifest.json").read_text())
    if m["format"] != "whisper-pack/1":
        raise ValueError(f"unknown format {m['format']!r}")
    data = (pack_dir / "weights.bin").read_bytes()

    def t(r):
        dt = np.dtype(r["dtype"])
        return np.frombuffer(data, dtype=dt, count=r["length"] // dt.itemsize, offset=r["offset"]).reshape(r["dims"])

    def f(r):
        return t(r).astype(np.float32)

    def lin(r):
        """-> (W [K, N] float32, bias or None)"""
        if "q" in r:
            W = (t(r["q"]).astype(np.float32) * t(r["s"])[:, None]).T
        else:
            W = f(r["f16_t"] if "f16_t" in r else r["f32_t"]).T
        return np.ascontiguousarray(W), (f(r["b"]) if "b" in r else None)

    def norm(r):
        return {"w": f(r["w"]), "b": f(r["b"])}

    def conv(r):
        w = t(r["q"]).astype(np.float32) * t(r["s"])[:, None, None] if r["kind"] == "int8" else f(r["w"])
        return {"w": w.reshape(r["out"], -1), "b": f(r["b"]), "k": r["k"]}

    def table(r):
        V, H, b = r["V"], r["H"], r["bits"]
        if b == 4:
            p = t(r["codes"])
            q = np.stack([p & 15, p >> 4], -1).reshape(V, H).astype(np.float32) - 8
            s = t(r["scales"]).astype(np.float32).reshape(V, H // 32, 1)
            return (q.reshape(V, H // 32, 32) * s).reshape(V, H)
        if b == 8:
            return t(r["codes"]).astype(np.float32) * t(r["scales"])[:, None]
        return f(r["f16"] if b == 16 else r["f32"])

    p = Pack()
    p.m, p.dims, p.gen = m, m["dims"], m["generation"]
    p.eps = m["dims"]["ln_eps"]
    p.conv1, p.conv2 = conv(m["conv1"]), conv(m["conv2"])
    p.enc_pos = sinusoids(m["dims"]["n_audio_ctx"], m["dims"]["d_model"])
    p.enc = [{k: (norm(v) if k.startswith("ln") else lin(v)) for k, v in L.items()} for L in m["enc"]["layers"]]
    p.ln_post = norm(m["enc"]["ln_post"])
    p.dec_pos = f(m["dec"]["pos"])
    p.table = table(m["dec"]["table"])
    p.dec = [{k: (norm(v) if k.startswith("ln") else lin(v)) for k, v in L.items()} for L in m["dec"]["layers"]]
    p.ln = norm(m["dec"]["ln"])
    return p


def _mm(x, wb):
    W, b = wb
    y = x @ W
    return y if b is None else y + b


def _heads(x, h):
    return x.reshape(x.shape[0], h, -1).transpose(1, 0, 2)  # [h, T, hd]


def _attend(q, k, v, mask=None):
    s = q @ k.transpose(0, 2, 1)
    if mask is not None:
        s = s + mask
    o = softmax(s) @ v
    return o.transpose(1, 0, 2).reshape(q.shape[1], -1)


def _conv(x, c, stride):
    k = c["k"]
    T = (x.shape[0] + 2 - k) // stride + 1
    xp = np.pad(x, ((1, 1), (0, 0)))
    cols = np.concatenate([xp[j: j + stride * (T - 1) + 1: stride] for j in range(k)], axis=1)  # [T, k*in]
    return cols @ c["w"].T + c["b"]


def encode(p, mel):
    """mel [n_mels, 3000] log-mel -> encoder states [1500, d]."""
    h = p.dims["heads"]
    x = gelu(_conv(np.asarray(mel, np.float32).T, p.conv1, 1))
    x = gelu(_conv(x, p.conv2, 2))
    x = x + p.enc_pos[: x.shape[0]]
    for L in p.enc:
        y = layer_norm(x, L["ln1"], p.eps)
        hd = y.shape[1] // h
        q = _heads(_mm(y, L["q"]) * hd ** -0.5, h)
        k, v = _heads(_mm(y, L["k"]), h), _heads(_mm(y, L["v"]), h)
        x = x + _mm(_attend(q, k, v), L["o"])
        y = layer_norm(x, L["ln2"], p.eps)
        x = x + _mm(gelu(_mm(y, L["fc1"])), L["fc2"])
    return layer_norm(x, p.ln_post, p.eps)


def decode(p, states, prompt_ids, max_new=224):
    """Greedy decode with a per-layer KV cache. Returns (ids, logits): ids are the generated tokens
    (the end-of-text token included when it stopped there), logits the raw [vocab] vector of each
    step before suppression (suppress_tokens / begin_suppress_tokens are applied to the argmax only)."""
    h, d = p.dims["heads"], p.dims["d_model"]
    hd = d // h
    nL, n_ctx = len(p.dec), p.dims["n_text_ctx"]
    eot = p.gen["eot"]
    sup = np.array(p.gen["suppress_tokens"], dtype=np.int64)
    begin = np.array(p.gen["begin_suppress_tokens"], dtype=np.int64)
    cross = [(_heads(_mm(states, L["ck"]), h), _heads(_mm(states, L["cv"]), h)) for L in p.dec]
    kc = [np.zeros((h, n_ctx, hd), np.float32) for _ in range(nL)]
    vc = [np.zeros((h, n_ctx, hd), np.float32) for _ in range(nL)]

    def run(ids, pos0):
        T = len(ids)
        x = p.table[ids] + p.dec_pos[pos0: pos0 + T]
        mask = np.triu(np.full((T, pos0 + T), -np.inf, np.float32), k=pos0 + 1) if T > 1 else None
        for i, L in enumerate(p.dec):
            y = layer_norm(x, L["ln1"], p.eps)
            q = _heads(_mm(y, L["sq"]) * hd ** -0.5, h)
            kc[i][:, pos0: pos0 + T] = _heads(_mm(y, L["sk"]), h)
            vc[i][:, pos0: pos0 + T] = _heads(_mm(y, L["sv"]), h)
            x = x + _mm(_attend(q, kc[i][:, : pos0 + T], vc[i][:, : pos0 + T], mask), L["so"])
            y = layer_norm(x, L["ln2"], p.eps)
            x = x + _mm(_attend(_heads(_mm(y, L["cq"]) * hd ** -0.5, h), *cross[i]), L["co"])
            y = layer_norm(x, L["ln3"], p.eps)
            x = x + _mm(gelu(_mm(y, L["fc1"])), L["fc2"])
        return layer_norm(x[-1:], p.ln, p.eps) @ p.table.T

    prompt = list(prompt_ids)
    out, logits = [], []
    lg = run(np.array(prompt), 0)[0]
    pos = len(prompt)
    for step in range(max_new):
        logits.append(lg)
        z = lg.copy()
        z[sup] = -np.inf
        if step == 0:
            z[begin] = -np.inf
        tok = int(np.argmax(z))
        out.append(tok)
        if tok == eot or pos >= n_ctx:
            break
        lg = run(np.array([tok]), pos)[0]
        pos += 1
    return out, logits
