"""Harvest EAGLE training data from Baguettotron on CPU.

An EAGLE head predicts the token *after* the one the target just produced,
given the target's own final hidden state. Training data is therefore triples
taken from a single forward pass over text:

    input   h_t          the target's post-norm feature at position t
            token_{t+1}  the token the target is about to emit
    label   argmax_{t+1} the target's own greedy choice for position t+2

Labelling against the target's argmax rather than the raw text token is hard
self-distillation: acceptance in speculative decoding is defined as matching the
target's choice, so this trains the metric directly and needs no slow
autoregressive generation to produce the labels.
"""
import glob, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
SEQ = 512
TARGET_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
OUT = "/tmp/specdec/eagle_data"

tok = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
model = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron",
                                             dtype=torch.float32).eval()
H = model.config.hidden_size

texts = []
for pat in ["/workspace/experiments/*/RESULTS.md", "/workspace/experiments/*/README.md",
            "/workspace/experiments/*.md", "/workspace/experiments/*/*.py"]:
    for p in sorted(glob.glob(pat)):
        try:
            texts.append(open(p, errors="ignore").read())
        except OSError:
            pass
blob = "\n\n".join(texts)
ids = tok(blob, add_special_tokens=False).input_ids
print(f"corpus: {len(ids)} tokens available, harvesting {TARGET_TOKENS}", flush=True)

chunks = [ids[i:i + SEQ] for i in range(0, min(len(ids), TARGET_TOKENS), SEQ)]
chunks = [c for c in chunks if len(c) == SEQ]

H_buf, tok_buf, lab_buf = [], [], []
t0 = time.perf_counter()
with torch.no_grad():
    for n, c in enumerate(chunks):
        x = torch.tensor([c])
        out = model(x, output_hidden_states=True)
        h = out.hidden_states[-1][0]              # (SEQ, H) feeds the LM head
        am = out.logits[0].argmax(-1)             # am[t] = target's pick for t+1
        # Sample at t: (h_t, token_{t+1}) -> label am[t+1] (its pick for t+2)
        H_buf.append(h[:-1].to(torch.float16).numpy())
        tok_buf.append(np.array(c[1:], dtype=np.int32))
        lab_buf.append(am[1:].to(torch.int32).numpy())
        if n % 20 == 0:
            done = (n + 1) * SEQ
            el = time.perf_counter() - t0
            print(f"  {n+1}/{len(chunks)} chunks  {done} tok  "
                  f"{done/el:.0f} tok/s  {el:.0f}s", flush=True)

Hs = np.concatenate(H_buf); Ts = np.concatenate(tok_buf); Ls = np.concatenate(lab_buf)
np.savez(OUT + ".npz", h=Hs, tok=Ts, lab=Ls, seq=SEQ)
el = time.perf_counter() - t0
print(f"saved {Hs.shape[0]} samples, {Hs.nbytes/1e6:.0f} MB, "
      f"{el:.0f}s at {Hs.shape[0]/el:.0f} tok/s", flush=True)
