"""Minimal pure-torch Llama forward for SmolLM2-135M, split at layer k.

Explicit past key/value tensors in and out -- no HF cache objects, no python
control flow that depends on tensor values -- so the two halves export cleanly
to ONNX and can be driven step by step from JavaScript.

Layout
------
lower(input_ids[1,T] int64, position_ids[1,T] int64, attn_mask[1,1,T,S] float,
      past_k[L1,1,KV,P,D], past_v[L1,1,KV,P,D])
    -> hidden[1,T,H]  (output of decoder layer k == input of layer k+1)
       present_k[L1,1,KV,S,D], present_v[...]        L1 = k+1 layers (0..k)

upper(hidden[1,T,H], position_ids[1,T], attn_mask[1,1,T,S],
      past_k[L2,1,KV,P,D], past_v[L2,1,KV,P,D])
    -> logits[1,T,V], present_k[L2,...], present_v[...]   L2 = N-k-1 (k+1..N-1)

S = P + T.  attn_mask is 1.0 where attention is allowed and 0.0 elsewhere;
the graph turns that into an additive -1e4, which is representable in fp16 and
is exactly zero after softmax in fp32.
"""


import torch
from torch import nn

MASK_NEG = -1.0e4


def rms_norm(x, weight, eps):
    dt = x.dtype
    x = x.float()
    v = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(v + eps)
    return (weight.float() * x).to(dt)


class RMSNorm(nn.Module):
    def __init__(self, h, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(h))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def rope_cos_sin(position_ids, head_dim, theta, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32)
                           / head_dim))
    f = position_ids.float().unsqueeze(-1) * inv.view(1, 1, -1)   # [1,T,D/2]
    emb = torch.cat([f, f], dim=-1)                                # [1,T,D]
    return emb.cos().to(dtype).unsqueeze(1), emb.sin().to(dtype).unsqueeze(1)


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, nh, nkv, hd = (cfg["hidden_size"], cfg["num_attention_heads"],
                          cfg["num_key_value_heads"], cfg["head_dim"])
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.rep = nh // nkv
        self.scaling = hd ** -0.5
        self.q_proj = nn.Linear(h, nh * hd, bias=False)
        self.k_proj = nn.Linear(h, nkv * hd, bias=False)
        self.v_proj = nn.Linear(h, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, h, bias=False)

    def forward(self, x, cos, sin, mask, pk, pv):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k = torch.cat([pk, k], dim=2)
        v = torch.cat([pv, v], dim=2)
        present_k, present_v = k, v
        kk = k.repeat_interleave(self.rep, dim=1)
        vv = v.repeat_interleave(self.rep, dim=1)
        att = torch.matmul(q, kk.transpose(2, 3)) * self.scaling
        att = att + (1.0 - mask) * MASK_NEG
        att = torch.softmax(att.float(), dim=-1).to(q.dtype)
        out = torch.matmul(att, vv).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), present_k, present_v


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, i = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))
                              * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg["rms_norm_eps"]
        self.input_layernorm = RMSNorm(cfg["hidden_size"], eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask, pk, pv):
        a, k, v = self.self_attn(self.input_layernorm(x), cos, sin, mask, pk, pv)
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k, v


class Lower(nn.Module):
    """Embeddings + decoder layers 0..k."""

    def __init__(self, cfg, k):
        super().__init__()
        self.cfg, self.k = cfg, k
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(k + 1)])

    def forward(self, input_ids, position_ids, attn_mask, past_k, past_v):
        x = self.embed_tokens(input_ids)
        cos, sin = rope_cos_sin(position_ids, self.cfg["head_dim"],
                                self.cfg["rope_theta"], x.dtype)
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, cos, sin, attn_mask, past_k[i], past_v[i])
            ks.append(k)
            vs.append(v)
        return x, torch.stack(ks, 0), torch.stack(vs, 0)


class Upper(nn.Module):
    """Decoder layers k+1..N-1 + final norm + lm_head (tied embeddings)."""

    def __init__(self, cfg, k):
        super().__init__()
        self.cfg, self.k = cfg, k
        n = cfg["num_hidden_layers"] - k - 1
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(n)])
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        # SmolLM2 ties lm_head to the input embeddings, and the export puts one
        # copy of the 49152x576 table in each half.  Keeping the weight in
        # [vocab, hidden] orientation and applying it to a 2-D activation makes
        # the exporter emit Gemm(transB=1) over an initializer whose BYTES are
        # identical to the embedding table, so export_onnx.py can point both
        # graphs at a single external file instead of shipping it twice.
        self.lm_head_weight = nn.Parameter(
            torch.zeros(cfg["vocab_size"], cfg["hidden_size"]))

    def forward(self, hidden, position_ids, attn_mask, past_k, past_v):
        x = hidden
        cos, sin = rope_cos_sin(position_ids, self.cfg["head_dim"],
                                self.cfg["rope_theta"], x.dtype)
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, cos, sin, attn_mask, past_k[i], past_v[i])
            ks.append(k)
            vs.append(v)
        h = self.norm(x)
        b, t, hs = h.shape
        logits = torch.nn.functional.linear(
            h.reshape(b * t, hs), self.lm_head_weight).reshape(b, t, -1)
        return logits, torch.stack(ks, 0), torch.stack(vs, 0)


# ------------------------------------------------------------------ loading
def config_from_hf(hf):
    c = hf.config
    return {
        "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size,
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "num_hidden_layers": c.num_hidden_layers,
        "vocab_size": c.vocab_size,
        "rms_norm_eps": c.rms_norm_eps,
        "rope_theta": float(getattr(c, "rope_theta", None)
                            or c.rope_parameters["rope_theta"]),
        "head_dim": getattr(c, "head_dim", None) or
                    (c.hidden_size // c.num_attention_heads),
    }


def _copy_layer(dst, src):
    dst.input_layernorm.weight.data.copy_(src.input_layernorm.weight.data)
    dst.post_attention_layernorm.weight.data.copy_(
        src.post_attention_layernorm.weight.data)
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(dst.self_attn, n).weight.data.copy_(
            getattr(src.self_attn, n).weight.data)
    for n in ("gate_proj", "up_proj", "down_proj"):
        getattr(dst.mlp, n).weight.data.copy_(getattr(src.mlp, n).weight.data)


def split_from_hf(hf, k):
    cfg = config_from_hf(hf)
    lo, up = Lower(cfg, k), Upper(cfg, k)
    lo.embed_tokens.weight.data.copy_(hf.model.embed_tokens.weight.data)
    for i in range(k + 1):
        _copy_layer(lo.layers[i], hf.model.layers[i])
    for j, i in enumerate(range(k + 1, cfg["num_hidden_layers"])):
        _copy_layer(up.layers[j], hf.model.layers[i])
    up.norm.weight.data.copy_(hf.model.norm.weight.data)
    up.lm_head_weight.data.copy_(hf.lm_head.weight.data)
    lo.eval()
    up.eval()
    for p in list(lo.parameters()) + list(up.parameters()):
        p.requires_grad_(False)
    return lo, up, cfg


# ------------------------------------------------------------------ helpers
def empty_past(cfg, n_layers, dtype=torch.float32):
    kv, hd = cfg["num_key_value_heads"], cfg["head_dim"]
    z = torch.zeros(n_layers, 1, kv, 0, hd, dtype=dtype)
    return z, z.clone()


def causal_mask(t, past_len, dtype=torch.float32):
    """[1,1,T,P+T] float, 1.0 where allowed."""
    s = past_len + t
    q = torch.arange(past_len, s).view(t, 1)
    kk = torch.arange(s).view(1, s)
    return (kk <= q).to(dtype).view(1, 1, t, s)


class Runner:
    """Torch reference driver for the two-half pipeline (batch 1)."""

    def __init__(self, lower, upper, cfg, k):
        self.lower, self.upper, self.cfg, self.k = lower, upper, cfg, k
        self.n_lo, self.n_up = k + 1, cfg["num_hidden_layers"] - k - 1

    def reset(self):
        self.lk, self.lv = empty_past(self.cfg, self.n_lo)
        self.uk, self.uv = empty_past(self.cfg, self.n_up)

    @torch.no_grad()
    def lower_step(self, ids, pos):
        t = ids.shape[1]
        m = causal_mask(t, self.lk.shape[3])
        h, self.lk, self.lv = self.lower(ids, pos, m, self.lk, self.lv)
        return h

    @torch.no_grad()
    def upper_step(self, hidden, pos):
        t = hidden.shape[1]
        m = causal_mask(t, self.uk.shape[3])
        lg, self.uk, self.uv = self.upper(hidden, pos, m, self.uk, self.uv)
        return lg


# ------------------------------------------------------------------ verify
def _verify():
    """Check llama_min against the HF model and against eval.py's stream arm."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import data as D
    import eval as E
    import model_utils as mu

    K = 16
    hf, tok = mu.load_model("smol")
    lo, up, cfg = split_from_hf(hf, K)
    run = Runner(lo, up, cfg, K)

    prompts = ["12 + 3 =", "What is 4567 plus 89?", "Compare 12 and 7:",
               "Subtract 9 from 100.", "What is 23 multiplied by 45?"]
    worst = 0.0
    for p in prompts:
        ids = torch.tensor([tok(p)["input_ids"]])
        pos = torch.arange(ids.shape[1]).view(1, -1)
        with torch.no_grad():
            ref = hf(input_ids=ids, attention_mask=torch.ones_like(ids),
                     use_cache=False).logits
        run.reset()
        h = run.lower_step(ids, pos)
        got = run.upper_step(h, pos)
        worst = max(worst, float((got - ref).abs().max()))
    print(f"logits max |delta| vs HF over {len(prompts)} prompts: {worst:.3e}")
    assert worst < 1e-3, worst

    # --- stream arm, token for token against eval.generate
    enc = mu.ResultEncoder(cfg["hidden_size"], n_steps=mu.N_STREAM_STEPS)
    ckpt = os.path.join(mu.repo_dir(), "ckpt", "smol_stream-left_final.pt")
    enc.load_state_dict(torch.load(ckpt)["state_dict"])
    enc.eval()
    hook = mu.attach_hook(hf, K)
    rows = D.load_split("test_in", 5)
    newline = tok("\n", add_special_tokens=False)["input_ids"][-1]
    n_match = 0
    for r in rows:
        res = D.compute(r["op"], r["a"], r["b"])[0]
        syms = mu.result_symbols([res], align="left")
        ref_txt, _, ref_ids = E.generate(hf, tok, [r["prompt"]], "stream", K,
                                         bs=1, enc=enc, syms=syms)
        # two-half pipeline
        ids = torch.tensor([tok(r["prompt"])["input_ids"]])
        t = ids.shape[1] - 1
        run.reset()
        with torch.no_grad():
            h = run.lower_step(ids, torch.arange(ids.shape[1]).view(1, -1))
            h[:, t, :] = h[:, t, :] + enc(syms, step=-1)
            lg = run.upper_step(h, torch.arange(ids.shape[1]).view(1, -1))
            nxt = int(lg[0, -1].argmax(-1))
            gen = []
            for step in range(E.MAX_NEW):
                gen.append(nxt)
                if (nxt in (newline, tok.eos_token_id)
                        and tok.decode(gen, skip_special_tokens=True).strip()):
                    break
                if step == E.MAX_NEW - 1:
                    break
                pos = torch.tensor([[t + 1 + step]])
                h = run.lower_step(torch.tensor([[nxt]]), pos)
                h[:, 0, :] = h[:, 0, :] + enc(syms, step=step)
                lg = run.upper_step(h, pos)
                nxt = int(lg[0, -1].argmax(-1))
        txt = tok.decode(gen, skip_special_tokens=True).strip().split("\n")[0].strip()
        ok = (gen == ref_ids[0]) and (txt == ref_txt[0])
        n_match += int(ok)
        print(f"  {r['prompt']!r:44} ref={ref_txt[0]!r} mine={txt!r} "
              f"ids_equal={gen == ref_ids[0]}")
    hook.remove()
    print(f"stream-arm token-for-token match: {n_match}/{len(rows)}")
    assert n_match == len(rows)
    print("DONE llama_min verify")


if __name__ == "__main__":
    _verify()
