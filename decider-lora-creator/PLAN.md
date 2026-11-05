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
