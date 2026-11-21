"""Parity checks for loopmodel.py on CPU.

1. K=1 logits match transformers' LlamaForCausalLM on the same weights.
2. K=2/3 with gate 0 match K=1 exactly (the loop starts as a no-op).
3. Cached greedy decoding matches uncached argmax decoding for a looped arm (alpha 0.5), batched with
   left padding, so the per-slot KV cache and padding masks are right.
4. Greedy text at K=1 matches transformers' generate for one prompt.
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loopmodel import LoopedMonad, generate  # noqa: E402

M = HERE / "models" / "monad"
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(str(M))
if tok.pad_token is None:
    tok.pad_token = "[PAD]"
P = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"
prompts = [P("What is the capital of Norway?"), P("If a train travels 60 km in 45 minutes, what is its average speed in km/h?")]

hf = AutoModelForCausalLM.from_pretrained(str(M), dtype=torch.float32).eval()
lm = LoopedMonad.from_pretrained(M, band=(12, 20), kmax=3).eval()

ids = torch.tensor([tok.encode(prompts[1], add_special_tokens=False)])
with torch.no_grad():
    a = hf(ids).logits
    b = lm(ids, K=1)
    b2 = lm(ids, K=2)
    b3 = lm(ids, K=3)
    b_a = lm(ids, K=2, alpha=0.5)
r = {"hf_vs_k1": (a - b).abs().max().item(), "k1_vs_k2_gate0": (b - b2).abs().max().item(),
     "k1_vs_k3_gate0": (b - b3).abs().max().item(), "k1_vs_k2_alpha05": (b - b_a).abs().max().item()}
print(r)
assert r["hf_vs_k1"] < 1e-3 and r["k1_vs_k2_gate0"] < 1e-4 and r["k1_vs_k3_gate0"] < 1e-4 and r["k1_vs_k2_alpha05"] > 1e-2

# 3. cached vs uncached, batched with left padding, looped arm
N = 24
out = generate(lm, tok, prompts, K=2, alpha=0.5, max_new=N, batch=2)
for p, o in zip(prompts, out):
    x = torch.tensor([tok.encode(p, add_special_tokens=False)])
    ref = []
    with torch.no_grad():
        for _ in range(len(o["ids"])):
            nxt = lm(x, K=2, alpha=0.5)[0, -1].argmax().item()
            ref.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], 1)
    print("cached==uncached", ref == o["ids"], len(ref))
    assert ref == o["ids"]

# 4. greedy text at K=1 vs transformers generate
o = generate(lm, tok, prompts[:1], K=1, max_new=40, batch=1)[0]
enc = tok(prompts[0], return_tensors="pt", add_special_tokens=False)
g = hf.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)[0, enc["input_ids"].shape[1]:].tolist()
print("hf_generate==ours", g[:len(o["ids"])] == o["ids"], repr(tok.decode(o["ids"]))[:200])
assert g[:len(o["ids"])] == o["ids"]
print("parity ok")
