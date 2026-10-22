"""NumPy reference forward pass from web/models/<pkg>/{manifest.json, model.onnx}, checked
against onnxruntime on the same package. It is the spec web/engine.js implements in WGSL.

Usage: .venv/bin/python -I run/ref.py PKG [PKG ...]      e.g. e17c-t4-w8L63 leafirc-t4-w8L63
Prints cosine (reference vs onnxruntime, graph optimizations off and on) over test strings of 1 to
128 tokens, and writes web/models/PKG/fixtures.json (texts, token ids, reference vectors) for the
browser parity check. With optimizations off onnxruntime matches the reference to cosine 1.0;
its default optimizations fuse DequantizeLinear+MatMul into kernels that quantize activations.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TEXTS = ["hi", "claude coding agent costs", "do LLMs actually reason or just pattern match?",
         "posts about copyright lawsuits against AI labs " * 3, "Ünïcödé émoji 🤖 test",
         "a long post about retrieval, embeddings, quantization and browsers " * 12]


class Model:
    def __init__(self, pkg):
        d = ROOT / "web" / "models" / pkg
        self.m = json.loads((d / "manifest.json").read_text())
        self.buf = (d / "model.onnx").read_bytes()  # tensors located by manifest byte offsets
        meta = json.loads((d / "meta.json").read_text())
        self.qp = (meta.get("prompts") or {}).get("query", "")
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(128)
        self.dir = d

    def t(self, e, dtype):
        return np.frombuffer(self.buf, dtype, e["length"] // np.dtype(dtype).itemsize, e["offset"]).reshape(e["dims"])

    def lin(self, x, w):
        if "q" in w:
            W = self.t(w["q"], np.int8).astype(np.float32) * self.t(w["s"], np.float32)
        elif "f32" in w:
            W = self.t(w["f32"], np.float32)
        else:
            W = self.t(w["f32_t"], np.float32).T
        y = x @ W
        return y + self.t(w["b"], np.float32) if "b" in w else y

    def ln(self, x, w, b=None):
        mu = x.mean(-1, keepdims=True)
        v = ((x - mu) ** 2).mean(-1, keepdims=True)
        y = (x - mu) / np.sqrt(v + self.m["eps"]) * self.t(w, np.float32)
        return y + self.t(b, np.float32) if b is not None else y

    def table(self, ids):
        T = self.m["table"]
        H = T["H"]
        if T["bits"] == 4:
            p = self.t(T["codes"], np.uint8)[ids]                      # [S, H/2], lo nibble = even column
            q = np.stack([p & 15, p >> 4], -1).reshape(len(ids), H).astype(np.float32) - 8
            s = self.t(T["scales"], np.float16)[ids].astype(np.float32).reshape(len(ids), H // 32, 1)
            return (q.reshape(len(ids), H // 32, 32) * s).reshape(len(ids), H)
        bits = T["bits"]
        c = self.t(T["codes"], np.uint8)[ids].reshape(len(ids), H // 8, bits).astype(np.uint32)
        word = sum(c[..., b] << (8 * b) for b in range(bits))
        q = ((word[..., None] >> (np.arange(8, dtype=np.uint32) * bits)) & (2 ** bits - 1)).reshape(len(ids), H // 32, 32)
        z = self.t(T["zp"], np.uint8)[ids].astype(np.float32)[..., None]
        s = self.t(T["scales"], np.float16)[ids].astype(np.float32)[..., None]
        return ((q - z) * s).reshape(len(ids), H)

    def attend(self, q, k, v, mask=None):
        sc = np.einsum("hid,hjd->hij", q, k) / math.sqrt(q.shape[-1])
        if mask is not None:
            sc = np.where(mask, sc, -np.inf)
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        return np.einsum("hij,hjd->hid", sc / sc.sum(-1, keepdims=True), v)

    def forward(self, ids):
        m, S = self.m, len(ids)
        H, nh = m["hidden"], m["heads"]
        D = H // nh
        gelu = lambda x: 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
        if m["arch"] == "modernbert":
            h = self.ln(self.table(ids), m["emb_norm"])
            pos = np.arange(S)[:, None]
            dist = np.abs(np.arange(S)[None, :] - np.arange(S)[:, None])
            for L in m["layers"]:
                a = self.ln(h, L["attn_norm"]) if L["attn_norm"] else h
                qkv = self.lin(a, L["Wqkv"]).reshape(S, 3, nh, D).transpose(1, 2, 0, 3)
                theta = m["theta_global"] if L["global"] else m["theta_local"]
                inv = 1.0 / theta ** (np.arange(0, D, 2) / D)
                ang = np.concatenate([pos * inv, pos * inv], -1)
                rot = lambda x: x * np.cos(ang) + np.concatenate([-x[..., D // 2:], x[..., :D // 2]], -1) * np.sin(ang)
                o = self.attend(rot(qkv[0]), rot(qkv[1]), qkv[2], None if L["global"] else dist <= m["window"] // 2)
                h = h + self.lin(o.transpose(1, 0, 2).reshape(S, H), L["Wo"])
                u = self.lin(self.ln(h, L["mlp_norm"]), L["Wi"])
                h = h + self.lin(gelu(u[:, :m["inter"]]) * u[:, m["inter"]:], L["Wo_mlp"])
            h = self.ln(h, m["final_norm"])
        else:
            h = self.table(ids) + self.t(m["pos"], np.float32)[:S] + self.t(m["type0"], np.float32)[0]
            h = self.ln(h, m["emb_ln"]["w"], m["emb_ln"]["b"])
            for L in m["layers"]:
                sp = lambda x: x.reshape(S, nh, D).transpose(1, 0, 2)
                o = self.attend(sp(self.lin(h, L["q"])), sp(self.lin(h, L["k"])), sp(self.lin(h, L["v"])))
                h = self.ln(h + self.lin(o.transpose(1, 0, 2).reshape(S, H), L["o"]), L["ln1"]["w"], L["ln1"]["b"])
                h = self.ln(h + self.lin(gelu(self.lin(h, L["up"])), L["down"]), L["ln2"]["w"], L["ln2"]["b"])
        e = self.lin(h.mean(0, keepdims=True), m["dense"])[0]
        return e / np.linalg.norm(e)


def main():
    for pkg in sys.argv[1:]:
        M = Model(pkg)
        sess = {}
        for tag, lvl in (("noopt", ort.GraphOptimizationLevel.ORT_DISABLE_ALL), ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)):
            so = ort.SessionOptions()
            so.graph_optimization_level = lvl
            sess[tag] = ort.InferenceSession(str(M.dir / "model.onnx"), so, providers=["CPUExecutionProvider"])
        cos, lens, fx = {"noopt": [], "default": []}, [], []
        for t in TEXTS:
            ids = M.tok.encode(M.qp + t).ids
            r = M.forward(np.array(ids))
            for tag, s_ in sess.items():
                o = s_.run(None, {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64)})[0][0]
                cos[tag].append(round(float(r @ (o / np.linalg.norm(o))), 7))
            lens.append(len(ids))
            fx.append({"text": t, "ids": ids, "vec": [round(float(x), 7) for x in r]})
        (M.dir / "fixtures.json").write_text(json.dumps(fx))
        print(pkg, "tokens", lens, "cos ref vs ORT", cos)


if __name__ == "__main__":
    main()
