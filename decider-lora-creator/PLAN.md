# PLAN: a decision spike creator and training harness in the page

Registered 2026-11-05, before the first real-GPU run.

## Question

Oskar asked for a page that creates Ettin decision-model LoRA spikes: define a decision (questions with
fixed options), label cases, train in the browser, evaluate, export. Training the LoRA itself needs a
backward pass through the encoder in WGSL, which the bsky-account-decider session started building on
2026-11-05 as its own experiment. This experiment builds the page and the harness against an interface
(`step / score / exportSpike / loadSpike`, sent to that session) and ships the one backend that needs no
backward pass: the decision head trained on the frozen encoder. When the WGSL trainer lands it plugs in
as a second backend.

Two things to measure now:

1. Does the page's path reproduce the Python stack? Same prompt ids as the training code, same logits as
   adapter-library's NumPy reference, and an exported spike that the Spiky Ball engine scores the same.
2. How much of a LoRA spike's gain does head-only training get? On adapter-library's growth protocol
   (invoice workflow, library start from the other three workflow spikes, n revealed cases, no
   validation, `growth_epochs(n)`, temperature 1, 100 test cases), against that experiment's Python arms.

## Prior art

adapter-library and browser-decider are the base; the comparison numbers are theirs. In-browser training
with hand-written kernels exists (image-to-wgsl's splat trainer here; WebGPU training libraries in the
wild). What this adds is a page that authors the decision and its data and exports an adapter in a
format a deployed page loads. The sibling session ran the search for WGSL transformer backward passes.

## Predictions (bridge run on the Mac, head-only from the library start)

Python references, invoice workflow, mean of seeds 0 and 1 (adapter-library `results/G-*`):
library-start LoRA 0.600 / 0.748 / 0.813 at n = 16 / 64 / 300; zero-start LoRA 0.628 / 0.621 / 0.790;
the merge itself at n = 0 scores 0.258.

- **P1.** The page's n = 0 start (the merge cut to rank 8, temperature 1) scores within 0.06 of 0.258. Loose,
  because the page merges the shipped v0.1.2 spikes (fine-tuned on the quantized base), not round 1's.
- **P2.** At n = 64, head-only lands between the zero-start and library-start LoRA: 0.62 to 0.74.
- **P3.** At n = 300, head-only trails the library-start LoRA by at least 0.04 (below 0.773): the encoder
  adapter matters once labels are plentiful.
- **P4.** At n = 16, head-only is within 0.04 of the library-start LoRA (0.600): with few labels the
  head carries most of what can be learned.
- **P5.** On the Mac, encoding costs under 15 ms per sequence, and a head-only epoch at n = 300 (about
  1,500 sequences) takes under 3 s.

Decision rule: if P2 and P4 hold, head-only is the page's default for small label budgets and the LoRA
backend is the upgrade for large ones; if head-only falls below the zero-start LoRA at n = 64, the page
should say so and steer users to the LoRA backend once it exists.

## Round 4 (registered 2026-11-06, before the run): seeds at n = 64, and a learned library merge

Oskar asked for both: more seeds, to tell whether in-browser LoRA's 0.693 at n = 64 (Python 0.748) is
seed noise, and the P²-MOPD-style learned merge (arXiv 2611.03953) as a start. The trainer exposes no merge
coefficients, so the adapter puts the K = 3 library spikes into extra LoRA slots of a rank 8 + 8K trainer:
B_slot = α_{k,m}·B_k and A_slot = A_k, beside a fresh rank-8 residual (A kaiming-uniform, B zero). After each
AdamW step the adapter writes the library slots back from α, and updates α (84 scalars: 3 spikes × 4 modules ×
7 layers) with its own Adam, using dL/dα_{k,m} = Σ G_B[slot k] ⊙ B_k / s, which is exact. Arms, invoices,
adapter-library's growth protocol:

- `warm`: the current page start (SVD-cut merge, all of it trained), n = 16 and 64, seeds 0–7.
- `uniform+res`: α frozen at 1/3, residual and head trained, n = 16 and 64, seeds 0–3.
- `alpha+res`: α learned with the residual and head (the paper's method), n = 16 and 64, seeds 0–3.
- `alpha`: α and head only, no residual, n = 16 and 64, seeds 0–1.

Before training, the job checks dα against central differences of the loss on one batch.

- **P6.** `warm` over 8 seeds at n = 64 lands within 0.03 of Python's 0.748: the round-3 gap was noise.
- **P7.** `alpha+res` beats `uniform+res` by at least 0.02 at n = 16 and is within ±0.02 of it at n = 64.
- **P8.** `alpha` alone trails `alpha+res` by at least 0.03 at n = 64 (the paper's ablation).
- **P9.** `alpha+res` is within 0.02 of `warm` at n = 64.
