#!/usr/bin/env python3
"""Validate the runtime choices for the local-llm layer, against real models.

Three capabilities that justify the layer (vs. the harness Agent tool):
  1. embeddings        -> local vector index, no API        (EmbeddingGemma 300M)
  2. logprob scoring   -> zero-shot classification, no API  (Gemma 3 270M)
  3. gemma-4 support   -> does the pip llama_cpp load it, or do we need the CLI?
"""
import sys, time, math
from llama_cpp import Llama

MODELS = "models"

def hr(t): print(f"\n=== {t} ===")

# --- 1. embeddings ---------------------------------------------------------
hr("1. embeddings (EmbeddingGemma 300M)")
emb = Llama(model_path=f"{MODELS}/embeddinggemma-300M-Q8_0.gguf",
            embedding=True, n_ctx=512, verbose=False)
docs = ["The cat sat on the mat.",
        "A feline rested on the rug.",
        "Quarterly revenue rose 12 percent."]
t0 = time.time()
vecs = [emb.embed(d) for d in docs]
dt = time.time() - t0
def cos(a, b):
    s = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return s/(na*nb)
print(f"dim={len(vecs[0])}  embedded {len(docs)} docs in {dt*1000:.0f}ms")
print(f"cos(cat, feline-paraphrase) = {cos(vecs[0], vecs[1]):.3f}  (expect HIGH)")
print(f"cos(cat, revenue)           = {cos(vecs[0], vecs[2]):.3f}  (expect LOW)")
del emb

# --- 2. logprob zero-shot classification -----------------------------------
hr("2. logprob scoring / zero-shot classification (Gemma 3 270M)")
gen = Llama(model_path=f"{MODELS}/gemma-3-270m-Q8_0.gguf",
            n_ctx=1024, logits_all=True, verbose=False)

def seq_logprob(context: str, choice: str) -> float:
    """Sum log P(choice tokens | context) via echo'd completion."""
    full = context + choice
    out = gen.create_completion(full, max_tokens=0, echo=True, logprobs=1, temperature=0.0)
    lp = out["choices"][0]["logprobs"]
    toks, tok_lps, offs = lp["tokens"], lp["token_logprobs"], lp["text_offset"]
    cut = len(context)
    return sum(l for o, l in zip(offs, tok_lps) if o >= cut and l is not None)

ctx = "Review: This movie was an absolute masterpiece, I loved every minute.\nSentiment:"
for label in [" positive", " negative"]:
    print(f"  logP({label!r}) = {seq_logprob(ctx, label):.2f}")
ctx2 = "Review: Worst two hours of my life, total garbage.\nSentiment:"
for label in [" positive", " negative"]:
    print(f"  logP({label!r}) = {seq_logprob(ctx2, label):.2f}")
del gen

# --- 3. gemma-4 support in the pip-installed llama_cpp ----------------------
hr("3. gemma-4 E2B load test (pip llama_cpp bundled llama.cpp)")
try:
    g4 = Llama(model_path="gemma-4-E2B_q4_0-it.gguf", n_ctx=512, verbose=False)
    out = g4.create_completion("The capital of France is", max_tokens=8, temperature=0.0)
    print("LOADED. sample:", repr(out["choices"][0]["text"]))
    print("=> pip llama_cpp is sufficient for gemma-4; no separate CLI build needed.")
except Exception as e:
    print("FAILED to load gemma-4:", type(e).__name__, str(e)[:200])
    print("=> layer must also ship llama.cpp built from source for gemma-4.")
