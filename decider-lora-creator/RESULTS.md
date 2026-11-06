# RESULTS: a decision spike creator and training harness in the page

A page where you define a decision (questions with fixed options), label cases, train a spike for the
frozen Ettin-17m decision model on your GPU, evaluate it against its start and the prior, and export it in
the format adapter-library's Spiky Ball page loads. Page: `web/index.html` (published at
austegard.com/experiments/decider-lora-creator/web/). Plan and predictions: `PLAN.md`.

## Answer

The page trains real LoRA spikes on the GPU, and they score as well as the Python ones. It uses the
WebGPU backward pass from `ettin-wgsl-train`. On adapter-library's growth protocol on your Mac (invoice
workflow, start from the other three workflow spikes, 100 test cases), in-browser LoRA scores 0.600 /
0.693 / 0.833 at 16 / 64 / 300 labelled cases. <!-- @evidence[[results/bridge_creator-lora-2.json#outputs.arms]] -->
Python's LoRA from the same start scores 0.600 / 0.748 / 0.813. Training only the decision head scores
0.560 / 0.590 / 0.628, barely above the prior's 0.538. All six of the page's checks pass on the Mac: prompt
ids, logits against the NumPy reference, the trainer's logits against the inference path (3.1e-6), the
weight round trip, and train-and-export for both backends. The 300-case spike trains in about 97 s.

Over 8 seeds the page's start reaches 0.629 at 16 cases and 0.732 at 64, within 0.03 of Python. <!-- @evidence[[results/bridge_creator-merge-2.json#outputs.arms]] -->
The round-3 gap at 64 was seed noise. A learned library merge after P²-MOPD (one weight per spike per
module, learned with a fresh residual) reaches 0.616 / 0.720. That is no better than freezing the weights at
an even split (0.611 / 0.706) or the page's plain start, so the page keeps the plain start.

## Findings

1. **In-browser LoRA matches the Python LoRA.** Over 8 seeds the page's start reaches 0.629 (n = 16, sd 0.032)
   and 0.732 (64, sd 0.028), against Python's 0.600 and 0.748 (round 4). With 2 seeds and all four LoRA
   targets it scored 0.600 (n = 16), 0.693 (64), 0.833 (300), against Python's library-start LoRA at 0.600 / 0.748 / 0.813
   (adapter-library `results/G-warm-invoice_processing-*`). Without the Wo_mlp target (round 2) it was
   0.572 / 0.704 / 0.849: adding it closed the gap at n = 16; at 64 and 300 the two runs differ by about
   the spread between seeds. (bridges `creator-lora-1`, `creator-lora-2`)
2. **Training the head alone is not enough; the encoder has to adapt.** Head-only scores 0.560 / 0.590 /
   0.628, prior 0.538. On three of five questions it stays at the prior's answer up to n = 64. LoRA moves
   them: at n = 300 `matches_order` reaches 0.93–0.96 and `disposition` 0.81–0.82, against 0.50–0.60 and
   0.55 head-only. (bridges `creator-head-1`, `creator-lora-1`)
3. **The browser path reproduces the Python stack at every joint** (`web/checks.js`, Mac):
   - The prompt ids match the training code's on 20/20 sequences.
   - Head-only logits are within 1.8e-5 of the NumPy reference.
   - The WGSL trainer's logits are within 3.1e-6 of the inference engine's on the same start (100/100 argmax).
   - Weights survive the trainer's load and export round trip within 4.8e-4 relative (fp16).
   - An exported spike reloads into the Spiky Ball engine with the same answers.
   - In pure JS, the head gradient matches central differences, re-packing a shipped spike is
     byte-identical, and the rank-8 SVD cut matches numpy to 5e-9.
4. **Training is quick enough for the page.** On the Mac (Chrome 156, Apple Metal), LoRA arms take 15 s
   at n = 16, 20 s at 64 and 97 s at 300 (940 steps of 16 sequences, about 0.1 s a step including the
   scheduler and loss read-back). Encoding for head-only takes 1.3 ms a sequence.
5. **Learning the library merge's weights did not help** (P²-MOPD-style, 84 weights: 3 spikes × 4 modules
   × 7 layers). With a rank-8 residual it scored 0.616 (n = 16, 4 seeds) and 0.720 (64). With the
   weights frozen at 1/3 it scored 0.611 and 0.706, and the page's start, trained whole, 0.629 and 0.732
   (8 seeds). Seed spread was 0.02–0.05. The learned weights stayed near the even split (0.35–0.41 per
   spike on average, 0.16–0.56 per module). Without the residual, weights alone reached 0.573 / 0.613.
   The weight gradients are exact: within 7.6e-4 of central differences. (bridge `creator-merge-2`)
6. **The library start is a good initialization and a bad model.** The merged start scores 0.222 to 0.240
   untrained, against the prior's 0.538. Python's merge scored 0.258, which is adapter-library's finding.

## Method

- **Fixture.** Typed Decisions `invoice_processing` (browser-decider `data/td`): 300 train and 100 test
  cases, 5 questions each, soft gold distributions (`run/make_demo.py` -> `web/demo/`).
- **Start spike.** The shipped workflows-set spikes for the other three workflows (v0.1.2), merged as
  the exact mean of their B·A deltas (stacked to rank 24), then SVD-cut to rank 8 with A = sqrt(S)·Vᵀ and
  B = U·sqrt(S), with heads averaged. This is adapter-library's library start, rebuilt from the packed
  fp16 spikes.
- **LoRA backend.** ettin-wgsl-train's `Trainer` (rank 8, alpha 16, LoRA on Wqkv, Wo, Wi and Wo_mlp from round 3, head trained too),
  behind `WgslTrainerBackend`: lr 5e-4 for the LoRA and 8e-4 for the head, AdamW with weight decay 0.01, gradient norm clipped at 1.0, the same warmup and
  decay, batch 16. These are run/lora.py's settings. Packed B carries alpha/r; the trainer's B does not, so the adapter rescales both ways.
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

The checks moved to the bridge in round 2.

### Round 2 (2026-11-05): LoRA in the page on ettin-wgsl-train's trainer

The sibling session pushed its WGSL trainer (`ettin-wgsl-train/web/train.js`, gradients checked against
PyTorch on the Mac). `WgslTrainerBackend` adapts it to the creator's interface. Its API is padded
batches and one flat parameter buffer, its B is unscaled, and it has no Wo_mlp LoRA. `web/checks.js`
now holds every check, shared by `check.html` and the bridge job `web/bridge_lora.js`. Bridge job
`creator-lora-1` (Mac, commit 6a0f1330, 283 s): all six checks ok, then the growth arms:

| n | seed 0 | seed 1 | mean | Python library-start LoRA | head-only (round 1) |
|---|---|---|---|---|---|
| 0 (start, no Wo_mlp) | 0.222 | | | 0.258 | 0.240 |
| 16 | 0.564 | 0.580 | 0.572 | 0.600 | 0.560 |
| 64 | 0.674 | 0.734 | 0.704 | 0.748 | 0.590 |
| 300 | 0.842 | 0.856 | 0.849 | 0.813 | 0.628 |

No predictions were registered for this round. I sent its numbers on to the trainer's session.

Reading: the page's LoRA path is the method; head-only stays as the cheap baseline. The gap at small n is
the right size for the start's missing Wo_mlp delta. A Wo_mlp target in the trainer would test that
directly, and the adapter picks it up from the trainer's `layout` with no other change.

The headless SwiftShader run of the round-1 `check.html` hit its 28-minute timeout before reporting, so
the checks moved to the Mac through the bridge. `check.html` now shows each check as it finishes.

### Round 3 (2026-11-06): the Wo_mlp target

The trainer's session added Wo_mlp as a fourth LoRA target (experiments 04e79a02). Its three-target
version came from matching only attn.Wo; adapter-library's lora.py matches the child name "Wo", which
catches both. The adapter reads the trainer's targets from its `layout`, so the library start now loads
whole. Bridge job `creator-lora-2` (Mac, commit 04e79a02, 286 s): all six checks ok; start 0.240 (as the
head-only path, which always had Wo_mlp).

| n | seed 0 | seed 1 | mean | round 2 (no Wo_mlp) | Python library-start LoRA |
|---|---|---|---|---|---|
| 16 | 0.620 | 0.580 | 0.600 | 0.572 | 0.600 |
| 64 | 0.686 | 0.700 | 0.693 | 0.704 | 0.748 |
| 300 | 0.858 | 0.808 | 0.833 | 0.849 | 0.813 |

Reading: Wo_mlp accounts for the n = 16 gap. The n = 64 deficit (0.05) and the n = 300 lead (0.02) are
the size of the seed spread (0.04–0.06), so two seeds cannot settle them.

### Round 4 (2026-11-06): 8 seeds, and a learned library merge

Oskar asked for both: more seeds at n = 64, and the P²-MOPD-style learned merge. `LearnedMergeBackend`
runs it on the unchanged trainer at rank 32: the three library spikes sit in slots 8–31 as α·B_k with A_k
fixed, and a fresh rank-8 residual sits in slots 0–7. dL/dα comes from the trainer's B gradient. The
library slots' gradients are zeroed before AdamW and the slots are rewritten from α after each step.
α uses its own Adam at 20× the LoRA lr. Bridge job `creator-merge-2` (Mac, commit 85315e3a, 796 s). The
first try, `creator-merge-1`, failed on a bug in the job script: it read the LoRA scale `alpha` of the
plain backend as merge weights.

α gradient check on one batch, against central differences (h = 0.05), relative error: Wqkv 9.8e-5,
Wi 1.6e-4, Wo_mlp 7.6e-4, Wo 4.5e-4. The uniform merge itself, untrained: 0.248.

| arm | n = 16 | n = 64 | seeds |
|---|---|---|---|
| `warm` (the page's start, SVD-cut, trained whole) | 0.629 ± 0.032 | 0.732 ± 0.028 | 8 |
| `uniform+res` (α = 1/3 fixed, residual + head) | 0.611 ± 0.023 | 0.706 ± 0.020 | 4 |
| `alpha+res` (α learned, residual + head) | 0.616 ± 0.048 | 0.720 ± 0.033 | 4 |
| `alpha` (α + head, no residual) | 0.573 | 0.613 | 2 |
| Python library-start LoRA | 0.600 | 0.748 | 2 |

On the same seeds 0–3, `alpha+res` minus `uniform+res` is +0.004 at n = 16 (per seed +0.024, −0.024,
+0.044, −0.028) and +0.015 at 64. Learned α per spike averaged 0.35–0.41 (module range 0.16–0.56). The
merge arms take about 20 s at n = 16 and 26 s at 64, against 16 s and 22 s for `warm` at rank 8.

Predictions:
- **P6 held:** `warm` at 64 is 0.732, 0.016 from 0.748.
- **P7 failed:** +0.005 at n = 16, short of the 0.02 predicted; at 64 it was within ±0.02 (+0.014).
- **P8 held:** `alpha` trails `alpha+res` by 0.107 at 64.
- **P9 held:** `alpha+res` is 0.012 under `warm` at 64.

Reading: on this fixture, learning how much of each library spike to keep does not beat an even mean. The
residual adapter carries the learning. The page keeps its rank-8 SVD-cut start, which is as good and
cheaper. `LearnedMergeBackend` stays in `backends.js` for a fixture where the library spikes disagree
more, and the page does not offer it.
