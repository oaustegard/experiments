"""mdbr-leaf-ir (6-layer BERT, hidden 384, 12 heads, mean pool, Dense 384->768, L2) laid out for the Apple Neural Engine:
(1, C, 1, S) activations, 1x1 Conv2d for every Linear, per-head einsum attention, fixed sequence length."""
import torch
from torch import nn

class LN(nn.Module):
    def __init__(self, w, b, eps):
        super().__init__()
        self.w = nn.Parameter(w.reshape(1, -1, 1, 1)); self.b = nn.Parameter(b.reshape(1, -1, 1, 1)); self.eps = max(eps, 1e-7)  # 1e-12 underflows in fp16
    def forward(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        return x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + self.eps) * self.w + self.b

def conv(w, b):
    o, i = w.shape
    c = nn.Conv2d(i, o, 1); c.weight.data = w.reshape(o, i, 1, 1).clone(); c.bias.data = b.clone()
    return c

class Layer(nn.Module):
    def __init__(self, sd, i, heads, eps):
        super().__init__()
        g = lambda k: (sd[f"encoder.layer.{i}.{k}.weight"], sd[f"encoder.layer.{i}.{k}.bias"])
        self.q, self.k, self.v = conv(*g("attention.self.query")), conv(*g("attention.self.key")), conv(*g("attention.self.value"))
        self.o, self.ln1 = conv(*g("attention.output.dense")), LN(*g("attention.output.LayerNorm"), eps)
        self.i, self.o2, self.ln2 = conv(*g("intermediate.dense")), conv(*g("output.dense")), LN(*g("output.LayerNorm"), eps)
        self.heads = heads; self.d = self.o.weight.shape[0] // heads
    def forward(self, x, key_mask):
        q, k, v = self.q(x) * (self.d ** -0.5), self.k(x), self.v(x)
        outs = []
        for qh, kh, vh in zip(q.split(self.d, 1), k.split(self.d, 1), v.split(self.d, 1)):
            w = torch.einsum("bchq,bkhc->bkhq", qh, kh.transpose(1, 3)) + key_mask
            outs.append(torch.einsum("bkhq,bchk->bchq", w.softmax(dim=1), vh))
        x = self.ln1(x + self.o(torch.cat(outs, 1)))
        return self.ln2(x + self.o2(nn.functional.gelu(self.i(x))))

class LeafANE(nn.Module):
    def __init__(self, sd, dense_w, dense_b, cfg, S):
        super().__init__()
        self.S = S; eps = cfg["layer_norm_eps"]
        self.word = nn.Embedding.from_pretrained(sd["embeddings.word_embeddings.weight"].clone())
        pos = sd["embeddings.position_embeddings.weight"][:S] + sd["embeddings.token_type_embeddings.weight"][0]
        self.register_buffer("pos", pos.T.reshape(1, -1, 1, S).clone())
        self.emb_ln = LN(sd["embeddings.LayerNorm.weight"], sd["embeddings.LayerNorm.bias"], eps)
        self.layers = nn.ModuleList(Layer(sd, i, cfg["num_attention_heads"], eps) for i in range(cfg["num_hidden_layers"]))
        self.dense = conv(dense_w, dense_b)
    def forward(self, input_ids, pad_mask):
        """input_ids (1, S) int, pad_mask (1, S) float with 1 for real tokens -> unit vector (1, 768)."""
        x = self.emb_ln(self.word(input_ids).transpose(1, 2).unsqueeze(2) + self.pos)
        key_mask = ((1.0 - pad_mask) * -1e4).reshape(1, self.S, 1, 1)
        for l in self.layers: x = l(x, key_mask)
        m = pad_mask.reshape(1, 1, 1, self.S)
        pooled = (x * m).sum(dim=3, keepdim=True) / m.sum(dim=3, keepdim=True)       # (1, 384, 1, 1)
        v = self.dense(pooled).reshape(1, -1)
        return v * torch.rsqrt((v * v).sum(dim=1, keepdim=True) + 1e-6)

def load(S, hf_dir):
    import json
    from safetensors.torch import load_file
    cfg = json.load(open(f"{hf_dir}/config.json"))
    sd = load_file(f"{hf_dir}/model.safetensors"); sd = {k.removeprefix("bert."): v for k, v in sd.items()}
    d = load_file(f"{hf_dir}/2_Dense/model.safetensors")
    return LeafANE(sd, d["linear.weight"], d["linear.bias"], cfg, S).eval(), cfg
