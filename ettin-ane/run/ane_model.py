"""dEttin-17M (ModernBERT-style decision cross-encoder) rewritten for the Apple Neural Engine.

Layout follows Apple's ane-transformers recipe: activations are (1, C, 1, S), every Linear is a 1x1 Conv2d,
attention is computed per head with einsum, sequence length is fixed, and RoPE tables and the local-window
mask are constants baked in for that length. The decision head runs at every position; the caller reads the
logits at the [MASK] positions.
"""
import math, torch
from torch import nn

class LN(nn.Module):
    """LayerNorm over the channel axis of (B, C, 1, S)."""
    def __init__(self, w, b=None, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(w.reshape(1, -1, 1, 1)); self.b = None if b is None else nn.Parameter(b.reshape(1, -1, 1, 1)); self.eps = eps
    def forward(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        x = x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + self.eps) * self.w
        return x if self.b is None else x + self.b

def conv(w, b=None):
    o, i = w.shape
    c = nn.Conv2d(i, o, 1, bias=b is not None)
    c.weight.data = w.reshape(o, i, 1, 1).clone()
    if b is not None: c.bias.data = b.clone()
    return c

class Layer(nn.Module):
    def __init__(self, sd, i, heads, is_global, S, window, theta, eps):
        super().__init__()
        g = lambda k: sd[f"layers.{i}.{k}"]
        self.attn_norm = LN(g("attn_norm.weight"), eps=eps) if f"layers.{i}.attn_norm.weight" in sd else None
        self.qkv, self.o = conv(g("attn.Wqkv.weight")), conv(g("attn.Wo.weight"))
        self.mlp_norm = LN(g("mlp_norm.weight"), eps=eps)
        self.wi, self.wo = conv(g("mlp.Wi.weight")), conv(g("mlp.Wo.weight"))
        H = self.o.weight.shape[0]; self.heads, self.d = heads, H // heads
        inv = 1.0 / (theta ** (torch.arange(0, self.d, 2).float() / self.d))
        f = torch.outer(torch.arange(S).float(), inv); emb = torch.cat([f, f], -1).T          # (d, S)
        self.register_buffer("cos", emb.cos().reshape(1, self.d, 1, S)); self.register_buffer("sin", emb.sin().reshape(1, self.d, 1, S))
        if is_global: self.local = None
        else:
            idx = torch.arange(S); far = (idx[:, None] - idx[None, :]).abs() > window // 2      # (key, query)
            self.register_buffer("local", (far.float() * -1e4).reshape(1, S, 1, S))
    def rope(self, x):
        a, b = x[:, : self.d // 2], x[:, self.d // 2:]
        return x * self.cos + torch.cat([-b, a], 1) * self.sin
    def forward(self, x, key_mask):                                                           # key_mask (1, S, 1, 1), 0 or -1e4
        h = x if self.attn_norm is None else self.attn_norm(x)
        q, k, v = self.qkv(h).split(self.heads * self.d, dim=1)
        outs = []
        for qh, kh, vh in zip(q.split(self.d, 1), k.split(self.d, 1), v.split(self.d, 1)):
            qh, kh = self.rope(qh) * (self.d ** -0.5), self.rope(kh)
            w = torch.einsum("bchq,bkhc->bkhq", qh, kh.transpose(1, 3)) + key_mask             # (1, S_key, 1, S_query)
            if self.local is not None: w = w + self.local
            outs.append(torch.einsum("bkhq,bchk->bchq", w.softmax(dim=1), vh))
        x = x + self.o(torch.cat(outs, 1))
        a, gate = self.wi(self.mlp_norm(x)).chunk(2, dim=1)
        return x + self.wo(nn.functional.gelu(a) * gate)

class DettinANE(nn.Module):
    def __init__(self, enc, head, S):
        super().__init__()
        c, sd = enc.config, enc.state_dict()
        self.S = S
        self.emb = nn.Embedding.from_pretrained(sd["embeddings.tok_embeddings.weight"].clone())
        self.emb_norm = LN(sd["embeddings.norm.weight"], eps=c.norm_eps)
        self.layers = nn.ModuleList(Layer(sd, i, c.num_attention_heads, i % c.global_attn_every_n_layers == 0, S, c.local_attention,
                                          c.global_rope_theta if i % c.global_attn_every_n_layers == 0 else c.local_rope_theta, c.norm_eps)
                                    for i in range(c.num_hidden_layers))
        self.final_norm = LN(sd["final_norm.weight"], eps=c.norm_eps)
        hs = head.state_dict()
        self.h0, self.hln, self.h1 = conv(hs["0.weight"], hs["0.bias"]), LN(hs["2.weight"], hs["2.bias"], eps=head[2].eps), conv(hs["3.weight"], hs["3.bias"])
    def forward(self, input_ids, pad_mask):
        """input_ids (1, S) int; pad_mask (1, S) float, 1 for real tokens. Returns head logits (1, S)."""
        x = self.emb_norm(self.emb(input_ids).transpose(1, 2).unsqueeze(2))                   # (1, H, 1, S)
        key_mask = ((1.0 - pad_mask) * -1e4).reshape(1, self.S, 1, 1)
        for l in self.layers: x = l(x, key_mask)
        x = self.final_norm(x)
        return self.h1(self.hln(nn.functional.gelu(self.h0(x)))).reshape(1, self.S)
