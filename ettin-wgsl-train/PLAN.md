# PLAN: LoRA training of Ettin-17m in WGSL

Registered 2026-11-05, before any kernel ran.

## Question

Oskar, after `bsky-account-decider`'s thread-aware LoRA was trained on the container's CPUs: "build the
WGSL backward pass as its own experiment". Can a hand-written WebGPU trainer fine-tune a LoRA adapter on
the packed (int8 / 4-bit) Ettin-17m page model in the browser, and give the same model as PyTorch?

## Prior art (3 searches, 2026-11-05)

Components exist; this composition was not found. webgpu-llm trains small decoders from scratch with
hand-written WGSL backward passes and AdamW. rullama has WGSL LoRA backward kernels over a frozen Q4_K
base (an inference engine first). ferrotensor has a two-pass, atomics-free flash-attention backward in
WGSL. Not found: LoRA training of a pretrained, quantized ModernBERT-style encoder (alternating local and
global attention, RoPE, GeGLU, bias-free LayerNorm) in the browser, checked against PyTorch.

## Fixture

The bsky-account-decider page model (`web/models/account/`): Ettin-17m fine-tuned, 7 layers, hidden 256,
4 heads, GeGLU 384, local window 128 on layers 1, 2, 4, 5; int8 matmuls, 4-bit token table. Data: its
thread-aware "about AI" set (`data/thread_lora`, 7,914 Sonnet labels; 6,194 train, 566 val, 1,154 test
by account) and the 376-reply whole-thread set.

PyTorch reference: `bsky-account-decider/run/thread_lora.py`, the page model rebuilt from its packed
weights (logits match the NumPy reference to 3.6e-6), rank-8 LoRA on Wqkv / Wo / Wi, head trained,
AdamW (lr 5e-4, head 8e-4, weight decay 0.01), clip at 1.0, 6% warmup then linear decay, batch 16.

## Design

Frozen weights dequantized to f32 on the GPU; activations f32. Per-layer checkpointing: the forward pass
keeps only each layer's input, and the backward pass recomputes that layer before differentiating it.
Attention backward in two passes without atomics (per query row: dQ and D = dO·O; per key row: dK, dV),
recomputing probabilities from the saved log-sum-exp. Layer 0's input needs no gradient, so its dX is
skipped.

## Checks and predictions

1. **Gradcheck** (SwiftShader in headless Chromium): one batch of 8 with lengths that exercise the local
   window, random nonzero A and B. P1: loss within 1e-5 relative, every gradient tensor within 1e-4
   relative L2 error, parameters after one AdamW step within 1e-5.
2. **Trajectory** (GPU Bridge, Oskar's GPU): a seeded CPU run and the WGSL run from the same initial
   adapter on the same batch order. P2: per-step loss within 1% relative (mean over epoch 1); final test
   accuracy within 0.005 of the CPU run.
3. **Speed**: P3: on a Mac GPU, one batch-16 step under 150 ms and 4 epochs under 5 minutes, against
   7.5 minutes on the 4-CPU container.

Decision rule: the trainer is "working" if P1 and P2 hold. P3 decides whether it beats the CPU path.
