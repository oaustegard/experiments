"""Monad (Llama, 64 layers x 256 wide) in plain PyTorch, with an Inkling-style loop over a band of layers.

Layers [0, s) are the prelude, [s, e) the band, [e, 64) the coda. With K passes the band runs K times
with shared weights. Pass 1 reads the prelude output x_p. Pass k > 1 reads

    x_p + g_k * (h_{k-1} - x_p)

where h_{k-1} is the band output of the previous pass (Inkling's "message") and g_k is a per-channel
gate. g = 0 makes every pass identical to pass 1, so the looped model equals the base model exactly;
g = 1 is a plain replay of the band on its own output. Training-free arms set g to a constant alpha;
retrofit arms learn it from zero. Inkling mixes the message through attention residuals (arXiv
2603.15031); the gated interpolation here is a simplification.

Every pass of a band layer keeps its own KV cache (virtual layer index), so attention in pass k sees
pass-k keys of earlier tokens, as a model that was really that deep would.

No transformers dependency: weights come straight from model.safetensors, so the same file runs on
Kaggle, Colab and CPU whatever transformers version each one has.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, hd, ff = cfg["hidden_size"], cfg["num_attention_heads"], cfg["head_dim"], cfg["intermediate_size"]
        self.h, self.hd = h, hd
        self.q_proj = nn.Linear(d, h * hd, bias=False)
        self.k_proj = nn.Linear(d, h * hd, bias=False)
        self.v_proj = nn.Linear(d, h * hd, bias=False)
        self.o_proj = nn.Linear(h * hd, d, bias=False)
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)
        self.input_layernorm = RMSNorm(d, cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(d, cfg["rms_norm_eps"])

    def forward(self, x, cos, sin, mask, cache=None, slot=None, pos=None):
        """cache: dict slot -> (K, V) static tensors [B, H, Lmax, hd]; pos: start index of x in the cache."""
        B, T, _ = x.shape
        y = self.input_layernorm(x)
        q = self.q_proj(y).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(y).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v_proj(y).view(B, T, self.h, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if cache is not None:
            K, V = cache[slot]
            K[:, :, pos:pos + T] = k
            V[:, :, pos:pos + T] = v
            k, v = K[:, :, :pos + T], V[:, :, :pos + T]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and cache is None)
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, T, -1))
        y = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(y)) * self.up_proj(y))


class LoopedMonad(nn.Module):
    def __init__(self, cfg, band=None, kmax=1):
        super().__init__()
        self.cfg = cfg
        d = cfg["hidden_size"]
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], d)
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(d, cfg["rms_norm_eps"])
        self.band = tuple(band) if band else None
        self.kmax = kmax
        # gate[k-2] is g_k for pass k = 2..kmax; zero = exact base model
        self.gate = nn.Parameter(torch.zeros(max(kmax - 1, 1), d))
        inv = 1.0 / (cfg.get("rope_theta", 10000.0) ** (torch.arange(0, cfg["head_dim"], 2).float() / cfg["head_dim"]))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.grad_ckpt = False

    # ------------------------------------------------------------------ structure
    def schedule(self, K):
        """List of (layer index, virtual slot, mix pass or 0). mix = k marks the first band layer of pass k > 1."""
        n = len(self.layers)
        if not self.band or K <= 1:
            return [(i, i, 0) for i in range(n)]
        s, e = self.band
        out, slot = [], 0
        for i in range(s):
            out.append((i, slot, 0)); slot += 1
        for k in range(1, K + 1):
            for i in range(s, e):
                out.append((i, slot, k if (i == s and k > 1) else 0)); slot += 1
        for i in range(e, n):
            out.append((i, slot, 0)); slot += 1
        return out

    def n_slots(self, K):
        return len(self.schedule(K))

    def rope(self, pos_ids, dtype):
        f = pos_ids.float()[..., None] * self.inv_freq          # [B, T, hd/2]
        emb = torch.cat((f, f), dim=-1)
        return emb.cos()[:, None].to(dtype), emb.sin()[:, None].to(dtype)

    # ------------------------------------------------------------------ forward
    def forward(self, ids, K=1, alpha=None, mask=None, pos_ids=None, cache=None, pos=0, return_hidden=False):
        """ids [B, T]. alpha: float to override the learned gate with a constant (training-free arms).
        mask: additive/bool attention mask [B, 1, T, Lk] or None for plain causal. Returns logits [B, T, V]."""
        B, T = ids.shape
        x = self.embed_tokens(ids)
        if pos_ids is None:
            pos_ids = torch.arange(pos, pos + T, device=ids.device)[None].expand(B, T)
        cos, sin = self.rope(pos_ids, x.dtype)
        sched = self.schedule(K)
        s = self.band[0] if self.band else None
        e = self.band[1] if self.band else None
        xp = None
        prev = None
        for (i, slot, mix) in sched:
            if self.band and K > 1 and i == s and mix == 0 and xp is None:
                xp = x                                         # prelude output, input of pass 1
            if mix:
                g = (torch.full_like(self.gate[0], alpha) if alpha is not None else self.gate[mix - 2]).to(x.dtype)
                x = xp + g * (x - xp)                          # x here is h_{k-1}
            layer = self.layers[i]
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, mask, cache, slot, pos, use_reentrant=False)
            else:
                x = layer(x, cos, sin, mask, cache, slot, pos)
        h = self.norm(x)
        if return_hidden:
            return h
        return F.linear(h, self.embed_tokens.weight)          # tied embeddings

    # ------------------------------------------------------------------ io
    @classmethod
    def from_pretrained(cls, path, band=None, kmax=1, dtype=torch.float32):
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        m = cls(cfg, band=band, kmax=kmax)
        sd = load_file(str(path / "model.safetensors"))
        own = {}
        for k, v in sd.items():
            k2 = k[len("model."):] if k.startswith("model.") else k
            k2 = k2.replace("_orig_mod.", "").replace("self_attn.", "").replace("mlp.", "")   # _orig_mod: torch.compile wrapper
            own[k2] = v
        own.pop("lm_head.weight", None)
        missing, unexpected = m.load_state_dict(own, strict=False)
        missing = [k for k in missing if k != "gate"]
        assert not missing and not unexpected, (missing, unexpected)
        gp = path / "loop_gate.pt"
        if gp.exists():
            g = torch.load(gp, map_location="cpu")
            m.band, m.kmax = tuple(g["band"]), g["kmax"]
            m.gate = nn.Parameter(g["gate"].float())
        return m.to(dtype)

    def save(self, path, src):
        """Write a Monad-format checkpoint: HF-named safetensors (bf16, as Monad ships) + loop_gate.pt."""
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sd = {}
        for k, v in self.state_dict().items():
            k = k.replace("_orig_mod.", "")
            if k == "gate":
                continue
            if k.startswith("layers."):
                _, i, name = k.split(".", 2)
                sub = "self_attn." if name.split(".")[0] in ("q_proj", "k_proj", "v_proj", "o_proj") else \
                      "mlp." if name.split(".")[0] in ("gate_proj", "up_proj", "down_proj") else ""
                sd[f"model.layers.{i}.{sub}{name}"] = v.detach().to(torch.bfloat16).cpu().contiguous()
            else:
                sd[f"model.{k}"] = v.detach().to(torch.bfloat16).cpu().contiguous()
        save_file(sd, str(path / "model.safetensors"), metadata={"format": "pt"})
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "chat_template.json", "generation_config.json"):
            p = Path(src) / f
            if p.exists():
                (path / f).write_bytes(p.read_bytes())
        if self.band:
            torch.save({"band": list(self.band), "kmax": self.kmax, "gate": self.gate.detach().float().cpu()},
                       path / "loop_gate.pt")


# ---------------------------------------------------------------------- generation
def new_cache(model, K, B, Lmax, device, dtype):
    h, hd = model.cfg["num_attention_heads"], model.cfg["head_dim"]
    return {s: (torch.zeros(B, h, Lmax, hd, device=device, dtype=dtype),
                torch.zeros(B, h, Lmax, hd, device=device, dtype=dtype)) for s in range(model.n_slots(K))}


@torch.no_grad()
def generate(model, tok, prompts, K=1, alpha=None, max_new=1024, batch=32, stop="<|im_end|>", device="cpu", dtype=torch.float32):
    """Greedy batched decoding with left padding and a static per-slot KV cache.
    Returns [{ids, stopped}] with ids cut before the stop string (decoded check) or EOS."""
    out = []
    eos = tok.eos_token_id
    for b0 in range(0, len(prompts), batch):
        chunk = prompts[b0:b0 + batch]
        enc = [tok.encode(p, add_special_tokens=False) for p in chunk]
        B, P = len(enc), max(len(e) for e in enc)
        Lmax = P + max_new
        ids = torch.full((B, P), tok.pad_token_id or 0, dtype=torch.long)
        valid = torch.zeros(B, Lmax, dtype=torch.bool)
        for r, e in enumerate(enc):
            ids[r, P - len(e):] = torch.tensor(e)
            valid[r, P - len(e):P] = True
        ids, valid = ids.to(device), valid.to(device)
        start = (P - torch.tensor([len(e) for e in enc], device=device))     # first real position per row
        cache = new_cache(model, K, B, Lmax, device, dtype)
        # prefill
        causal = torch.ones(P, P, dtype=torch.bool, device=device).tril()
        m = causal[None, None] & valid[:, None, None, :P]
        m = m | torch.eye(P, dtype=torch.bool, device=device)[None, None]        # padded rows attend to self
        pos_ids = (torch.arange(P, device=device)[None] - start[:, None]).clamp(min=0)
        logits = model(ids, K=K, alpha=alpha, mask=m, pos_ids=pos_ids, cache=cache, pos=0)[:, -1]
        gen = torch.zeros(B, max_new, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        n_gen = max_new
        for t in range(max_new):
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            gen[:, t] = nxt
            done |= nxt == eos
            if (t + 1) % 16 == 0:
                for r in range(B):
                    if not done[r] and stop in tok.decode(gen[r, max(0, t - 31):t + 1].tolist()):
                        done[r] = True
            if bool(done.all()):
                n_gen = t + 1
                break
            p = P + t
            valid[:, p] = True
            m = valid[:, None, None, :p + 1]
            pos_ids = (p - start)[:, None]
            logits = model(nxt[:, None], K=K, alpha=alpha, mask=m, pos_ids=pos_ids, cache=cache, pos=p)[:, -1]
        for r in range(B):
            out.append(cut(tok, gen[r, :n_gen].tolist(), stop))
    return out


def cut(tok, row, stop):
    eos = tok.eos_token_id
    if eos in row:
        row = row[:row.index(eos)]
        stopped = True
    else:
        stopped = False
    if stop not in tok.decode(row):
        return {"ids": row, "stopped": stopped}
    lo, hi = 1, len(row)
    while lo < hi:
        mid = (lo + hi) // 2
        if stop in tok.decode(row[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return {"ids": row[:lo], "stopped": True}
