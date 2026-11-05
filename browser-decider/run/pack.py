"""Pack a trained decider into the browser format, and check it against PyTorch.

    python3 -I run/pack.py CKPT_NAME [--variant t4-w8L63|fp32] [--check N] [--score]

Writes out/web/<CKPT_NAME>-<variant>/{weights.bin, manifest.json, tokenizer.json, decider.json}.
The manifest uses ettin-native's schema (each tensor's byte range in one file) so the WGSL engine
reads it the same way; matmuls are int8 per output column with codes in +-63, the token table 4-bit
in groups of 32 with an fp16 scale (ettin-native's t4-w8L63, the 10.2 MB build). The decision head
stays f32. A NumPy forward pass from the packed file alone (`Ref`) is the spec the browser follows:
--check N compares it with the PyTorch model on N test questions, --score scores the packed model on
the whole test split with the dataset's scorer (the number the shipped configuration produces).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORKFLOWS, examples, load_cases, score_examples  # noqa: E402


class Packer:
    def __init__(self):
        self.parts, self.off = [], 0

    def add(self, arr, dtype):
        a = np.ascontiguousarray(arr.astype(dtype))
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


def pack(sd, cfg, variant, temps):
    P = Packer()
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    q8 = variant != "fp32"

    def lin(W):  # torch [out=N, in=K]
        W = W.astype(np.float32)
        N, K = W.shape
        if not q8:
            return {"f32_t": P.add(W, np.float32), "K": K, "N": N}
        Wk = W.T  # [K, N], the file layout the engine transposes once
        s = (np.abs(Wk).max(0) / 63 + 1e-12).astype(np.float32)
        q = np.clip(np.round(Wk / s), -63, 63).astype(np.int8)
        return {"q": P.add(q, np.int8), "s": P.add(s, np.float32), "K": K, "N": N}

    T = sd["enc.embeddings.tok_embeddings.weight"].astype(np.float32)
    V = T.shape[0]
    if q8:
        G = H // 32
        Tg = T.reshape(V, G, 32)
        s = np.abs(Tg).max(2, keepdims=True) / 7 + 1e-12
        q = (np.clip(np.round(Tg / s), -8, 7).astype(np.int16) + 8).astype(np.uint8).reshape(V, H)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        table = {"bits": 4, "V": V, "H": H, "codes": P.add(packed, np.uint8),
                 "scales": P.add(s.reshape(V, G, 1), np.float16)}
    else:
        table = {"bits": 32, "V": V, "H": H, "f32": P.add(T, np.float32)}
    rp = cfg.get("rope_parameters")   # transformers 5 nests RoPE bases by attention type; 4.x has flat keys
    rope = ({"global": rp["full_attention"]["rope_theta"], "local": rp["sliding_attention"]["rope_theta"]} if rp else
            {"global": cfg["global_rope_theta"], "local": cfg["local_rope_theta"] or cfg["global_rope_theta"]})
    m = {"arch": "modernbert", "hidden": H, "heads": cfg["num_attention_heads"], "inter": I,
         "eps": cfg["norm_eps"], "window": cfg["local_attention"], "global_every": cfg["global_attn_every_n_layers"],
         "theta_global": rope["global"], "theta_local": rope["local"],
         "table": table, "emb_norm": P.add(sd["enc.embeddings.norm.weight"], np.float32), "layers": []}
    for i in range(cfg["num_hidden_layers"]):
        p = f"enc.layers.{i}."
        m["layers"].append({
            "attn_norm": P.add(sd[p + "attn_norm.weight"], np.float32) if p + "attn_norm.weight" in sd else None,
            "Wqkv": lin(sd[p + "attn.Wqkv.weight"]), "Wo": lin(sd[p + "attn.Wo.weight"]),
            "mlp_norm": P.add(sd[p + "mlp_norm.weight"], np.float32),
            "Wi": lin(sd[p + "mlp.Wi.weight"]), "Wo_mlp": lin(sd[p + "mlp.Wo.weight"]),
            "global": i % cfg["global_attn_every_n_layers"] == 0})
    m["final_norm"] = P.add(sd["enc.final_norm.weight"], np.float32)
    m["head"] = {"w0": P.add(sd["head.0.weight"], np.float32), "b0": P.add(sd["head.0.bias"], np.float32),
                 "ln_w": P.add(sd["head.2.weight"], np.float32), "ln_b": P.add(sd["head.2.bias"], np.float32),
                 "w1": P.add(sd["head.3.weight"].reshape(-1), np.float32), "b1": float(sd["head.3.bias"][0])}
    m["temps"] = temps
    m["variant"] = variant
    return m, P.bytes()


def gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))


class Ref:
    """NumPy forward pass from manifest + weights.bin only."""

    def __init__(self, m, data):
        self.m, self.d = m, data

    def t(self, r, dtype=np.float32):
        return np.frombuffer(self.d, dtype=dtype, count=r["length"] // np.dtype(dtype).itemsize,
                             offset=r["offset"]).reshape(r["dims"])

    def lin(self, x, w):
        if "f32_t" in w:
            return x @ self.t(w["f32_t"]).T
        return (x @ self.t(w["q"], np.int8).astype(np.float32)) * self.t(w["s"])

    def ln(self, x, w, b=None):
        mu = x.mean(-1, keepdims=True)
        v = ((x - mu) ** 2).mean(-1, keepdims=True)
        y = (x - mu) / np.sqrt(v + self.m["eps"]) * w
        return y if b is None else y + b

    def table(self, ids):
        T = self.m["table"]
        H = T["H"]
        if T["bits"] == 32:
            return self.t(T["f32"])[ids].copy()
        p = self.t(T["codes"], np.uint8)[ids]
        q = np.stack([p & 15, p >> 4], -1).reshape(len(ids), H).astype(np.float32) - 8
        s = self.t(T["scales"], np.float16)[ids].astype(np.float32).reshape(len(ids), H // 32, 1)
        return (q.reshape(len(ids), H // 32, 32) * s).reshape(len(ids), H)

    def rope(self, x, theta):  # x [nh, S, D]
        D = x.shape[-1]
        inv = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float32) / D))
        ang = np.arange(x.shape[1], dtype=np.float32)[:, None] * inv[None]
        cos, sin = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))
        rot = np.concatenate([-x[..., D // 2:], x[..., : D // 2]], -1)
        return x * cos + rot * sin

    def hidden(self, ids):
        m = self.m
        H, nh = m["hidden"], m["heads"]
        D = H // nh
        S = len(ids)
        x = self.ln(self.table(ids), self.t(m["emb_norm"]))
        pos = np.arange(S)
        local = np.abs(pos[:, None] - pos[None]) <= m["window"] // 2
        for L in m["layers"]:
            h = x if L["attn_norm"] is None else self.ln(x, self.t(L["attn_norm"]))
            qkv = self.lin(h, L["Wqkv"]).reshape(S, 3, nh, D).transpose(1, 2, 0, 3)
            th = m["theta_global"] if L["global"] else m["theta_local"]
            q, k, v = self.rope(qkv[0], th), self.rope(qkv[1], th), qkv[2]
            sc = np.einsum("hid,hjd->hij", q, k) / math.sqrt(D)
            if not L["global"]:
                sc = np.where(local, sc, -np.inf)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            o = np.einsum("hij,hjd->hid", sc, v).transpose(1, 0, 2).reshape(S, H)
            x = x + self.lin(o, L["Wo"])
            u = self.lin(self.ln(x, self.t(L["mlp_norm"])), L["Wi"])
            I = m["inter"]
            x = x + self.lin(gelu(u[:, :I]) * u[:, I:], L["Wo_mlp"])
        return self.ln(x, self.t(m["final_norm"]))

    def logits(self, ids, pos):
        hd = self.m["head"]
        h = self.hidden(np.array(ids))[pos]
        h = gelu(h @ self.t(hd["w0"]).T + self.t(hd["b0"]))
        h = self.ln(h, self.t(hd["ln_w"]), self.t(hd["ln_b"]))
        return h @ self.t(hd["w1"]) + hd["b1"]


def main():
    import torch
    from train_enc import Decider, Enc
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--variant", default="t4-w8L63", choices=["t4-w8L63", "fp32"])
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--fixtures", type=int, default=0, help="write N cases (5 questions each) with reference logits")
    a = ap.parse_args()
    ck = ROOT / "out" / "ckpt" / a.name
    meta = json.loads((ck / "meta.json").read_text())
    sd = {k: v.numpy() for k, v in torch.load(ck / "best.pt", map_location="cpu").items()}
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(meta["init"]).to_dict()
    man, data = pack(sd, cfg, a.variant, meta["temps"])
    out = ROOT / "out" / "web" / f"{a.name}-{a.variant}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(data)
    (out / "manifest.json").write_text(json.dumps(man))
    tok_src = Path(meta["init"]) / "tokenizer.json"
    (out / "tokenizer.json").write_bytes(tok_src.read_bytes())
    print(f"packed {out.name}: {len(data) / 1e6:.2f} MB weights")

    ref = Ref(man, data)
    tok = Enc(tok_src)
    te = examples([c for w in WORKFLOWS for c in load_cases(w, "test")])
    for e in te:
        e["ids"], e["pos"] = tok.build(e)
    if a.check:
        torch.set_num_threads(2)
        model = Decider(meta["init"])
        model.load_state_dict(torch.load(ck / "best.pt", map_location="cpu"))
        model.eval()
        diffs = []
        for e in te[: a.check]:
            with torch.no_grad():
                ids = torch.tensor([e["ids"]])
                zt = model(ids, torch.ones_like(ids), torch.zeros(len(e["pos"]), dtype=torch.long),
                           torch.tensor(e["pos"])).numpy()
            zr = ref.logits(e["ids"], e["pos"])
            diffs.append(float(np.abs(zt - zr).max()))
        print(f"check {a.check} questions: max |logit torch - ref| = {max(diffs):.2e}, median {np.median(diffs):.2e}")
    if a.fixtures:
        import pandas as pd
        cases = []
        for w in WORKFLOWS:
            df = pd.read_parquet(ROOT / "data" / "td" / w / "test.parquet")
            for r in df.head(math.ceil(a.fixtures / len(WORKFLOWS))).itertuples():
                cases.append({"id": r.id, "state": r.state, "questions": json.loads(r.questions)})
        by_case = {}
        for e in te:
            by_case.setdefault(e["case"], []).append(e)
        fx = {"cases": [], "variant": a.variant}
        for c in cases:
            qs = []
            for e in by_case[c["id"]]:
                qs.append({"name": e["qname"], "type": e["type"], "ids": e["ids"], "pos": e["pos"],
                           "ref_logits": [float(x) for x in ref.logits(e["ids"], e["pos"])]})
            fx["cases"].append({**c, "seqs": qs})
        (out / "fixtures.json").write_text(json.dumps(fx))
        print(f"fixtures: {len(fx['cases'])} cases")
    if a.score:
        probs = []
        for e in te:
            z = ref.logits(e["ids"], e["pos"]) / man["temps"][e["type"]]
            z = np.exp(z - z.max())
            probs.append((z / z.sum()).tolist())
        summ, _ = score_examples(te, probs)
        (ROOT / "results" / f"packed_{a.name}-{a.variant}.json").write_text(json.dumps(summ, indent=1))
        print(f"packed {a.variant} TEST {json.dumps(summ)}")


if __name__ == "__main__":
    main()
