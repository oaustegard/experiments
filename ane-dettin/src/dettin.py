"""dEttin in Python: load a packed build, decide, embed, swap spikes. PyTorch on CUDA or CPU.

    from dettin import Dettin
    m = Dettin.load("out/packs/ettin68-base-t4-w8L63")           # manifest.json, weights.bin, tokenizer.json
    m.load_spike("out/colab/adapters68/S-lora-invoice_processing-s0.pt")
    m.decide(state_text, {"urgent": {"type": "noul", "instructions": "..."}})   # -> {"urgent": {"true": p, "false": 1 - p}}
    m.unload_spike()

    python3 run/dettin.py pack-base jhu-clsp/ettin-encoder-68m out/packs/ettin68-base-t4-w8L63
    python3 run/dettin.py check PACK_DIR [--spike S.pt --workflow W] [--n 40]    # Typed Decisions test cases

The file format is browser-decider's (`run/pack.py`): a 4-bit token table in groups of 32 with fp16
scales, int8 matmul weights per output column with a float scale, fp32 norms and head. Loading decodes
it to fp32 once, so compute is ordinary PyTorch (bf16 autocast on CUDA). A spike is a rank-8 LoRA
adapter plus a decision head, in adapter-library's `adapter_state` layout; loading one adds B·A to the
decoded base weights in place and keeps the originals for `unload_spike`. A fully fine-tuned build
(one made by pack.py from a trained checkpoint) carries its own head and needs no spike.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn
from transformers import ModernBertConfig, ModernBertModel

LIN = {"Wqkv": "attn.Wqkv", "Wo": "attn.Wo", "Wi": "mlp.Wi", "Wo_mlp": "mlp.Wo"}
LORA_SCALE = 2.0   # alpha 16 / rank 8, as trained
MAX_LEN = 1024


def _t(data, rec, dtype=np.float32):
    return np.frombuffer(data, dtype=dtype, count=rec["length"] // np.dtype(dtype).itemsize,
                         offset=rec["offset"]).reshape(rec["dims"])


def _lin(data, w):
    if "f32_t" in w:
        return _t(data, w["f32_t"]).copy()
    return np.ascontiguousarray((_t(data, w["q"], np.int8).astype(np.float32) * _t(data, w["s"])).T)


def _table(data, T):
    if T["bits"] == 32:
        return _t(data, T["f32"]).copy()
    V, H = T["V"], T["H"]
    p = _t(data, T["codes"], np.uint8)
    q = np.stack([p & 15, p >> 4], -1).reshape(V, H).astype(np.float32) - 8
    s = _t(data, T["scales"], np.float16).astype(np.float32).reshape(V, H // 32, 1)
    return (q.reshape(V, H // 32, 32) * s).reshape(V, H)


class Dettin:
    def __init__(self, enc, head, tok, temps, device):
        self.enc, self.head, self.tok, self.temps, self.dev = enc, head, tok, temps or {}, device
        self.ids = {t: tok.token_to_id(t) for t in ("[CLS]", "[SEP]", "[MASK]", "[PAD]")}
        self._base = None
        self.embed_head = None

    @classmethod
    def load(cls, pack_dir, device=None):
        pack_dir = Path(pack_dir)
        m = json.loads((pack_dir / "manifest.json").read_text())
        data = (pack_dir / "weights.bin").read_bytes()
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = ModernBertConfig(
            vocab_size=m["table"]["V"], hidden_size=m["hidden"], num_hidden_layers=len(m["layers"]),
            num_attention_heads=m["heads"], intermediate_size=m["inter"], norm_eps=m["eps"],
            local_attention=m["window"], global_attn_every_n_layers=m["global_every"],
            global_rope_theta=m["theta_global"], local_rope_theta=m["theta_local"], max_position_embeddings=7999,
            attn_implementation="sdpa", reference_compile=False)
        enc = ModernBertModel(cfg)
        sd = {"embeddings.tok_embeddings.weight": _table(data, m["table"]),
              "embeddings.norm.weight": _t(data, m["emb_norm"]), "final_norm.weight": _t(data, m["final_norm"])}
        for i, L in enumerate(m["layers"]):
            for k, name in LIN.items():
                sd[f"layers.{i}.{name}.weight"] = _lin(data, L[k])
            sd[f"layers.{i}.mlp_norm.weight"] = _t(data, L["mlp_norm"])
            if L["attn_norm"] is not None:
                sd[f"layers.{i}.attn_norm.weight"] = _t(data, L["attn_norm"])
        missing, unexpected = enc.load_state_dict({k: torch.tensor(np.array(v)) for k, v in sd.items()}, strict=False)
        assert not unexpected and not [k for k in missing if "position_ids" not in k], (missing, unexpected)
        H = m["hidden"]
        head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H), nn.Linear(H, 1))
        if "head" in m:
            hd = m["head"]
            head.load_state_dict({"0.weight": torch.tensor(_t(data, hd["w0"]).copy()), "0.bias": torch.tensor(_t(data, hd["b0"]).copy()),
                                  "2.weight": torch.tensor(_t(data, hd["ln_w"]).copy()), "2.bias": torch.tensor(_t(data, hd["ln_b"]).copy()),
                                  "3.weight": torch.tensor(_t(data, hd["w1"]).copy()).reshape(1, H), "3.bias": torch.tensor([hd["b1"]])})
        tok = Tokenizer.from_file(str(pack_dir / "tokenizer.json"))
        tok.no_truncation()
        tok.no_padding()
        self = cls(enc.to(device).eval(), head.to(device).eval(), tok, m.get("temps"), device)
        if "embed_head" in m:
            self.embed_head = torch.tensor(_t(data, m["embed_head"]).copy(), device=device)
        return self

    # ---- spikes
    def load_spike(self, path, temps=None):
        st = torch.load(path, map_location="cpu")
        self.unload_spike()
        self._base = {}
        sd = self.enc.state_dict()
        for k in [k for k in st if k.endswith(".A")]:
            name = k[:-2] + ".weight"          # e.g. layers.3.attn.Wqkv.weight
            self._base[name] = sd[name].clone()
            delta = (st[k[:-2] + ".B"] @ st[k]) * LORA_SCALE
            sd[name].add_(delta.to(sd[name]))
        self._base["head"] = {k: v.clone() for k, v in self.head.state_dict().items()}
        self.head.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith("head.")})
        self.temps = temps or {}
        return self

    def unload_spike(self):
        if self._base:
            sd = self.enc.state_dict()
            for name, w in self._base.items():
                if name == "head":
                    self.head.load_state_dict(w)
                else:
                    sd[name].copy_(w)
        self._base = None

    # ---- decisions
    @staticmethod
    def _options(q):
        if q["type"] == "noul":
            crit = q.get("criteria") or {"true": "The statement is true for this case.",
                                          "false": "The statement is not true for this case."}
            return [(k, crit[k]) for k in ("true", "false")]
        crit = q["criteria"]
        if isinstance(crit, list):
            return [(str(i), d) for i, d in enumerate(crit)]
        return list(crit.items())

    def _build(self, state, q):
        e = lambda s: self.tok.encode(s, add_special_tokens=False).ids
        ids = [self.ids["[CLS]"]] + e(f"{q['type']}: {q['instructions']}") + [self.ids["[SEP]"]]
        pos, opts = [], self._options(q)
        for name, desc in opts:
            pos.append(len(ids))
            ids += [self.ids["[MASK]"]] + e(f" {name}: {desc}")
        ids.append(self.ids["[SEP]"])
        ids += e(state)[: MAX_LEN - len(ids) - 1] + [self.ids["[SEP]"]]
        return ids, pos, [o for o, _ in opts]

    @torch.no_grad()
    def decide(self, state, questions):
        """One state, a dict of typed questions -> {question: {option: probability}}. One batch on the device."""
        built = {n: self._build(state, q) for n, q in questions.items()}
        L = max(len(b[0]) for b in built.values())
        ids = torch.full((len(built), L), self.ids["[PAD]"], dtype=torch.long)
        att = torch.zeros((len(built), L), dtype=torch.long)
        for i, (b, _, _) in enumerate(built.values()):
            ids[i, : len(b)] = torch.tensor(b)
            att[i, : len(b)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            hs = self.enc(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).last_hidden_state
        out = {}
        for i, (n, (_, pos, opts)) in enumerate(built.items()):
            z = self.head(hs[i, pos].float()).squeeze(-1) / self.temps.get(questions[n]["type"], 1.0)
            out[n] = dict(zip(opts, torch.softmax(z, -1).tolist()))
        return out

    # ---- embeddings (a build packed with an embed_head)
    @torch.no_grad()
    def embed(self, texts, max_len=1024, bs=32):
        assert self.embed_head is not None, "this build has no embedding head"
        out = np.zeros((len(texts), self.embed_head.shape[0]), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for i in range(0, len(order), bs):
            idx = order[i: i + bs]
            enc = [[self.ids["[CLS]"]] + self.tok.encode(texts[j], add_special_tokens=False).ids[: max_len - 2] + [self.ids["[SEP]"]] for j in idx]
            L = max(map(len, enc))
            ids = torch.full((len(enc), L), self.ids["[PAD]"], dtype=torch.long)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            for r, e in enumerate(enc):
                ids[r, : len(e)] = torch.tensor(e)
                att[r, : len(e)] = 1
            ids, att = ids.to(self.dev), att.to(self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
            m = att.unsqueeze(-1).to(h.dtype)
            v = ((h * m).sum(1) / m.sum(1)).float() @ self.embed_head.T
            out[idx] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        return out


def _pack_mod():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "browser-decider" / "run"))
    import pack
    return pack


def fold_head(W):
    """A bias-free H -> 768 head W has rank at most H, so W = U S Vt and the H x H head S Vt gives vectors
    with the same lengths and inner products as W's (U has orthonormal columns). Search results are identical."""
    _, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    return (S[:, None] * Vt).astype(np.float32)


def pack_base(model_id, out_dir, variant="t4-w8L63", emb_ckpt=None, fold=False):
    """Pack a stock encoder (spikes bring the head), or an embedder checkpoint from run/embed.py with its
    768-d head, or with --fold that head folded to the encoder's own width."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModel
    pack = _pack_mod()
    path = snapshot_download(model_id, allow_patterns=["*.json", "*.bin", "*.safetensors"])
    cfg = AutoConfig.from_pretrained(path).to_dict()
    H = cfg["hidden_size"]
    if emb_ckpt:
        st = torch.load(emb_ckpt, map_location="cpu")
        sd = {k: v.numpy() for k, v in st.items() if k.startswith("enc.")}
    else:
        sd = {f"enc.{k}": v.numpy() for k, v in AutoModel.from_pretrained(path).state_dict().items()}
    z = np.zeros
    sd.update({"head.0.weight": z((H, H), np.float32), "head.0.bias": z(H, np.float32), "head.2.weight": np.ones(H, np.float32),
               "head.2.bias": z(H, np.float32), "head.3.weight": z((1, H), np.float32), "head.3.bias": z(1, np.float32)})
    man, data = pack.pack(sd, cfg, variant, {})
    if emb_ckpt:
        del man["head"]                      # the zero decision head stays in the file but is not read
        w = np.ascontiguousarray(st["head.weight"].numpy().astype(np.float32))
        if fold:
            w = np.ascontiguousarray(fold_head(w))
        pad = (-len(data)) % 4
        man["embed_head"] = {"offset": len(data) + pad, "length": w.nbytes, "dims": list(w.shape), "dtype": "float32"}
        data = data + b"\0" * pad + w.tobytes()
    else:
        del man["head"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(data)
    (out / "manifest.json").write_text(json.dumps(man))
    (out / "tokenizer.json").write_bytes((Path(path) / "tokenizer.json").read_bytes())
    print(f"packed {out}: {len(data) / 1e6:.2f} MB")


def check(pack_dir, spike=None, workflow=None, n=40):
    """Accuracy and ms per case on the first n Typed Decisions test cases (5 questions each)."""
    import time
    _pack_mod()
    from common import WORKFLOWS, load_cases
    m = Dettin.load(pack_dir)
    if spike:
        m.load_spike(spike)
    ok = tot = 0
    ts = []
    for w in ([workflow] if workflow else WORKFLOWS):
        for c in load_cases(w, "test")[: n if workflow else max(1, n // 4)]:
            t0 = time.perf_counter()
            res = m.decide(c["state"], c["questions"])
            ts.append(time.perf_counter() - t0)
            for name, p in res.items():
                g = c["gold"][name]["probabilities"]
                ok += max(p, key=p.get) == max(g, key=g.get)
                tot += 1
    out = {"pack": str(pack_dir), "spike": spike, "workflow": workflow, "device": str(m.dev), "decisions": tot,
           "accuracy": ok / tot, "ms_per_case_median": float(np.median(ts) * 1000)}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack-base")
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--variant", default="t4-w8L63")
    p.add_argument("--emb-ckpt")
    p.add_argument("--fold", action="store_true", help="fold the 768-d embedding head to the encoder width")
    c = sub.add_parser("check")
    c.add_argument("pack")
    c.add_argument("--spike")
    c.add_argument("--workflow")
    c.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "pack-base":
        pack_base(a.model, a.out, a.variant, a.emb_ckpt, a.fold)
    else:
        check(a.pack, a.spike, a.workflow, a.n)
