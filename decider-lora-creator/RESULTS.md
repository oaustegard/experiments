# RESULTS: a decision spike creator and training harness in the page

A page where you define a decision (questions with fixed options), label cases, train a spike for the
frozen Ettin-17m decision model on your GPU, evaluate it against its start and the prior, and export it in
the format adapter-library's Spiky Ball page loads. Page: `web/index.html` (published at
austegard.com/experiments/decider-lora-creator/web/). Plan and predictions: `PLAN.md`.

## Answer

The page and its harness work. Training the decision head alone, though, is too weak to be the method.
On adapter-library's growth protocol (invoice workflow, start from the other three workflow spikes, 100
test cases), head-only training on the Mac's GPU scores 0.560 / 0.590 / 0.628 at 16 / 64 / 300 labelled
cases. <!-- @evidence[[results/bridge_creator-head-1.json#outputs.arms]] -->
The prior scores 0.538. Python's LoRA from the same start scores 0.600 / 0.748 / 0.813, and LoRA from zero
0.628 / 0.621 / 0.790. Head-only trails both at every size. On three of the five questions it never moves
off the prior's answer. The encoder has to adapt, so the page's real training path is the LoRA backend,
which waits on the WGSL backward pass. The page's engine path matches the reference: the merged start
scores 0.240 at temperature 1 against Python's 0.258 for its own merge. Encoding costs 1.3 ms a sequence
on the Mac.

A LoRA backend that reaches the Python LoRA numbers in the page would change this answer from "harness
ready" to "method ready".

## Findings

1. **Head-only training barely clears the prior on a new workflow.** Mean of 2 seeds: 0.560 (n = 16),
   0.590 (64), 0.628 (300); prior 0.538; library-start LoRA 0.600 / 0.748 / 0.813; zero-start LoRA
   0.628 / 0.621 / 0.790 (adapter-library `results/G-*-invoice_processing-*`). (bridge `creator-head-1`)
2. **What the head learns is mostly urgency.** At n = 64, `disposition` (0.45), `duplicate` (0.89) and
   `matches_order` (0.54) sit exactly at the prior's accuracy, and urgency rises from 0.34 to 0.54–0.64. At
   n = 300 `disposition` reaches 0.55. The frozen [MASK] features of the merged start do not separate
   the answers of the new workflow's questions linearly enough for a 66k-parameter head.
3. **The browser path reproduces the Python stack.** The prompt builder is a verbatim copy of
   adapter-library's `system.js` layout; `web/check.html` compares its ids with `Enc.build` (the
   training code) and the head-only logits with the NumPy reference (`node run/check.mjs check`). In
   pure JS (`run/check_js.mjs`), the head's gradient matches central differences (worst relative error
   2.5e-5), re-packing a shipped spike is byte-identical, the stacked-rank merge is exact (1e-10), and the
   rank-8 SVD cut matches numpy to 5e-9 (`results/check_truncate.json`). On the Mac, the merge at n = 0
   scores 0.240 at temperature 1, close to Python's 0.258. The two do not merge the same spikes, so they
   are not expected to match exactly.
4. **Encoding is cheap and head epochs are not free.** On the Mac (Chrome 156, Apple Metal), the 2,000
   sequences (708k tokens) encoded in 2.7 s, 1.3 ms each. A head-only arm at n = 300 (1,500 sequences,
   10 epochs) took 37 s in JS, about 3.7 s an epoch. (bridge `creator-head-1`)

## Method

- **Fixture.** Typed Decisions `invoice_processing` (browser-decider `data/td`): 300 train and 100 test
  cases, 5 questions each, soft gold distributions (`run/make_demo.py` -> `web/demo/`).
- **Start spike.** The shipped workflows-set spikes for the other three workflows (v0.1.2), merged as
  the exact mean of their B·A deltas (stacked to rank 24), then SVD-cut to rank 8 with A = sqrt(S)·Vᵀ and
  B = U·sqrt(S), with heads averaged. This is adapter-library's library start, rebuilt from the packed
  fp16 spikes.
- **Head-only backend.** Base + start LoRA frozen on adapter-library's SpikeEngine. Each sequence's
  [MASK] rows are computed once and cached. The head (Linear 256→256, GELU, LayerNorm, Linear→1) trains in
  JS: soft cross-entropy over each question's options, AdamW (lr 8e-4, weight decay 0.01 on the
  matrices), warmup over 6% of steps then linear decay, gradient norm clipped at 1.0, batch 16 sequences.
  These match adapter-library `run/lora.py` for the head.
- **Growth protocol** (`run/bridge_inputs.py`, `web/bridge_job.js`). The n revealed cases come from
  adapter-library's own `revealed(workflow, seed, n)` and the epochs from its `growth_epochs(n)`. There is
  no validation and the last epoch is kept. Scoring is at temperature 1 on all 100 test cases, with the
  dataset's scorer (top option matches gold's top option). These are the conditions of the Python G arms.
- **The page's own runs** differ: a validation split chooses the best epoch, with the untrained start as
  a candidate, and per-type temperatures are fitted on validation.
- **Backend interface** shared with the WGSL trainer: `create`, `step(seqs, targets, {lr, weightDecay})`,
  `score`, `exportSpike`, `loadSpike`. `LoraBackend` imports any module that exports `LoraTrainer` with
  that interface.

## Log

### Round 1 (2026-11-05): the page, its checks, and head-only on the Mac

Oskar asked for an Ettin decision-model LoRA fine-tuning creator and training harness on a page, using
the WGSL backward kernel the bsky-account-decider session started that evening. That kernel did not
exist yet. I sent that session the interface above and built the page against it, with head-only training
as the backend that needs no backward pass. Then Oskar chose to test a P²-MOPD-style learned merge
(arXiv 2611.03953) here and on the GPU Bridge rather than on Colab. That needs the backward pass, so it
is a future start option, and I sent the sibling session an optional `taskVectors` / `learnAlpha`
extension.

GPU Bridge job `creator-head-1` (Mac, Chrome 156, Apple Metal, page v10), commit f3cb8464, 117 s:

| n | seed 0 | seed 1 | mean | library-start LoRA (Python) | zero-start LoRA (Python) |
|---|---|---|---|---|---|
| 0 (start) | 0.240 | | | 0.258 (its merge) | |
| 16 | 0.550 | 0.570 | 0.560 | 0.600 | 0.628 |
| 64 | 0.598 | 0.582 | 0.590 | 0.748 | 0.621 |
| 300 | 0.634 | 0.622 | 0.628 | 0.813 | 0.790 |

Prior (each question's mean train gold distribution): 0.538.

Predictions:
- **P1 held:** 0.240 is within 0.06 of 0.258.
- **P2 failed:** 0.590 is below the predicted 0.62–0.74 band and below the zero-start LoRA.
- **P3 held:** 0.628 is more than 0.04 under 0.813.
- **P4 held at the boundary:** 0.560 against 0.600, exactly 0.04.
- **P5 split:** encoding 1.3 ms a sequence held; a head epoch at n = 300 took about 3.7 s, not under 3 s.

The decision rule's second branch applies. The page now says that head-only training is a weak
baseline and that the LoRA backend is the method once it exists.

The headless SwiftShader run of `web/check.html` was still running when this was written. Its result
goes in `results/check_page.json`.
