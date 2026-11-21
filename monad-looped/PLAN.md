# monad-looped — plan (registered 2026-11-21, before any arm ran)

## Question

Monad (PleIAs, 56.7M parameters, 64 layers × 256 wide) reasons in text inside a `<think>` block before it
answers. Thinking Machines' Inkling 2 Mini (2026-11-20) runs its middle layers three times with shared weights
and reports reaching answers with fewer reasoning tokens. Can Monad be made deeper by looping a band of its
own layers, and can that extra latent depth replace enough text reasoning to make answers faster end to end?

Speed accounting. On Oskar's Mac (onnxruntime-web WebGPU, batch 1) a Monad layer costs about 0.17 ms in fp32
and decode is 11.8 ms per token (`monad-layer-parallel`). A looped pass costs as much as a new layer, so
looping n layers for K passes adds (K−1)·n·0.17 ms per token: 8 layers × 2 passes +1.4 ms (+12%), 16 × 2 or
8 × 3 +2.7 ms (+23%). The end-to-end number is milliseconds per correct answer.

## Loop mechanics

Inkling-style: prelude `[0, s)`, band `[s, e)`, coda `[e, 64)`. Pass 1 reads the prelude output x_p. Pass k
reads `x_p + g_k ⊙ (h_{k−1} − x_p)`, h the previous pass's band output, g a per-channel gate. g = 0 is the
base model exactly (checked: `check_parity.py`), g = 1 replays the band on its own output. Each pass keeps
its own KV cache. Inkling mixes through attention residuals; the gated interpolation is a simplification.

## Stages

1. **Baseline** — reasoning length and accuracy of base and loop-edited Monad, from `monad-weight-edit`'s
   existing Mac GPU runs (no new compute).
2. **Training-free screen** (Kaggle T4 x2) — bands B1 = 12–19, B2 = 36–43 (8 layers each; low block
   influence and deletion KL in `monad-weight-edit/results/layers.json`, clear of the sink/loop layers
   22–34), B3 = 6–21 (16 layers); K ∈ {2, 3}; constant gate α ∈ {0.1, 0.25, 0.5, 1}. Metrics:
   teacher-forced answer NLL with 0, 50% and 100% of the reference trace kept, trace NLL and KL to base on
   384 held-out SYNTH rows (`data/eval_synth.jsonl`, shard 500); greedy generation on the 192-prompt dev
   fixture (accuracy, loops, think length).
3. **Retrofit** (Colab A100) — continued training on SYNTH from base Monad with a learned gate (init 0) and
   pass counts sampled from {1, 2, 3}; **control arm: the same tokens without the loop**, so a gain from
   training alone is not credited to looping. Gate: looped arm at K = 2 matches or beats the control on
   teacher-forced answer NLL and dev accuracy.
4. **Shorten the trace** — only if stage 3 passes: fine-tune on shortened traces or RL with a length
   penalty, then milliseconds per correct answer on the Mac through the GPU Bridge.

## Predictions

- P1 (stage 2): no training-free arm beats base on full-trace answer NLL; the best arm is within 0.02 nats
  at small α, and α = 1 is clearly worse (> 0.1 nats) for every band.
- P2 (stage 2): no training-free arm improves dev accuracy by more than noise, and α ≥ 0.5 raises loops.
- P3 (stage 3): the looped arm at K = 2 recovers the control's answer NLL; at K = 3 it beats the control
  by a small margin (0.01–0.05 nats), larger on math and mcq rows than on memorization rows.
- P4 (stage 3): the looped arm's advantage is larger with a truncated trace (f = 0, 0.5) than with the
  full trace, the signature of reasoning moved into the latent.
- P5 (stage 4): trace length falls 30–50% at equal accuracy; net speed per correct answer between 0.9×
  and 1.3×.
