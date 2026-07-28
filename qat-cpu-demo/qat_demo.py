#!/usr/bin/env python3
"""
QAT vs PTQ on a tiny char-level transformer — CPU only.

Reproduces, at toy scale, the central claim of Google's Gemma 4 QAT post:
post-training quantization (PTQ) to int4 degrades quality, while
quantization-aware training (QAT) — simulating the int4 grid *during*
training with a straight-through estimator — recovers most of it.

No GPU, no network, no external data. Runs in ~1-2 min on 4 CPU cores.
"""
import math, time, json, random
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(4)

# --- self-contained corpus: combinatorially generated so it has REAL entropy.
# Random subject/verb/object sentences -> the char-LM learns word + grammar
# structure but genuinely cannot predict which token comes next, so the
# perplexity floor stays well above 1 and there is real quality for low-bit
# quantization to destroy. No tiling, no memorization-to-ppl-1.
_rng = random.Random(1)
_subj = ["the corvid", "a quantized model", "the optimizer", "each weight",
         "the int four grid", "memory", "precision", "the gradient",
         "this experiment", "the noisy floor", "a stubborn tensor", "the loss"]
_verb = ["weighs", "rounds", "recovers", "degrades", "simulates", "compresses",
         "learns", "stumbles on", "pays for", "wears", "forgets", "sharpens"]
_obj = ["the evidence", "the low bit floor", "most of the quality",
        "a documented constraint", "four bits per weight", "the same penalty",
        "ground truth", "the int four glove", "precision twice",
        "the rounding error", "every spare decibel", "the cold start"]
_conj = [". ", ", and ", ", but ", ", so ", "; then ", ", because "]
CORPUS = "".join(
    f"{_rng.choice(_subj)} {_rng.choice(_verb)} {_rng.choice(_obj)}{_rng.choice(_conj)}"
    for _ in range(4000)
)  # ~120 KB of varied text

chars = sorted(set(CORPUS))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

BLOCK = 64
BATCH = 32

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])
    return x, y

# --- fake quantization with straight-through estimator ---------------------
class FakeQuant(torch.autograd.Function):
    """Per-output-channel symmetric weight quant. STE: gradient passes through."""
    @staticmethod
    def forward(ctx, w, n_bits):
        qmax = 2 ** (n_bits - 1) - 1            # e.g. int4 -> 7
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        wq = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
        return wq * scale
    @staticmethod
    def backward(ctx, g):
        return g, None                          # straight-through

class QuantLinear(nn.Linear):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.quant = False
        self.n_bits = 4
    def forward(self, x):
        w = FakeQuant.apply(self.weight, self.n_bits) if self.quant else self.weight
        return F.linear(x, w, self.bias)

# --- tiny transformer ------------------------------------------------------
D, H, L = 128, 4, 2

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = QuantLinear(D, 3 * D, bias=False)
        self.proj = QuantLinear(D, D, bias=False)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = QuantLinear(D, 4 * D)
        self.fc2 = QuantLinear(4 * D, D)
    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // H)
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(BLOCK, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

def set_quant(model, on, n_bits=4):
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.quant = on
            m.n_bits = n_bits

@torch.no_grad()
def evaluate(model, iters=40):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch("val")
        logits = model(x)
        losses.append(F.cross_entropy(logits.view(-1, V), y.view(-1)).item())
    model.train()
    l = sum(losses) / len(losses)
    return l, math.exp(l)          # loss, perplexity

def train(model, steps, lr=3e-3, label=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for s in range(steps):
        x, y = get_batch("train")
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % (steps // 4) == 0 or s == steps - 1:
            vl, vp = evaluate(model)
            print(f"  [{label}] step {s:4d}  val_loss {vl:.3f}  ppl {vp:6.2f}")
    print(f"  [{label}] {steps} steps in {time.time()-t0:.1f}s")

def linear_param_count(model):
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, QuantLinear))

# --- run -------------------------------------------------------------------
def mb(params, bits):
    return params * bits / 8 / 1e6      # theoretical footprint, linear weights only

print(f"vocab={V}  corpus={len(data)} chars  device=cpu  threads=4")
model = GPT()
nq = linear_param_count(model)
print(f"quantizable linear weights: {nq:,}\n")

print("1) FP32 baseline training")
train(model, 2000, label="fp32")
fp32_loss, fp32_ppl = evaluate(model, iters=120)

# keep a pristine copy of the fp32 weights to re-init each QAT run
import copy
fp32_state = copy.deepcopy(model.state_dict())

results = {"vocab": V, "corpus_chars": len(data), "quantizable_params": nq,
           "fp32": {"bits": 32, "ppl": fp32_ppl, "loss": fp32_loss, "mb": mb(nq, 32)}}

print("\n2) PTQ — round the trained fp32 weights to N bits, no further training")
for b in [8, 4, 3, 2]:
    set_quant(model, True, b)
    l, p = evaluate(model, iters=120)
    set_quant(model, False)
    results[f"ptq_int{b}"] = {"bits": b, "ppl": p, "loss": l, "mb": mb(nq, b)}
    print(f"  PTQ int{b}: ppl {p:7.2f}")

print("\n3) QAT — re-init from fp32, train with N-bit fake-quant in the loop")
for b in [4, 3, 2]:
    model.load_state_dict(fp32_state)
    set_quant(model, True, b)
    train(model, 1000, lr=1e-3, label=f"qat-int{b}")
    l, p = evaluate(model, iters=120)
    results[f"qat_int{b}"] = {"bits": b, "ppl": p, "loss": l, "mb": mb(nq, b)}
    set_quant(model, False)

print("\n" + "=" * 64)
print(f"{'method':<12}{'bits':>5}{'ppl':>11}{'lin-MB':>10}{'vs fp32':>12}")
print("-" * 64)
for k in ["fp32", "ptq_int8", "ptq_int4", "qat_int4", "ptq_int3", "qat_int3",
          "ptq_int2", "qat_int2"]:
    r = results[k]
    delta = "" if k == "fp32" else f"+{r['ppl']-fp32_ppl:.2f}"
    print(f"{k:<12}{r['bits']:>5}{r['ppl']:>11.2f}{r['mb']:>10.3f}{delta:>12}")
print("=" * 64)
for b in [4, 3, 2]:
    pt, qt = results[f"ptq_int{b}"]["ppl"], results[f"qat_int{b}"]["ppl"]
    gap = pt - fp32_ppl
    rec = (pt - qt) / gap * 100 if gap > 1e-6 else 0.0
    print(f"int{b}: PTQ penalty +{gap:6.2f} ppl  ->  QAT recovers {rec:5.0f}% "
          f"at the same {mb(nq, b):.3f} MB footprint")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote results.json")
