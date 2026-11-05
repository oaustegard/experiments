# RESULTS: a spiky-ball decider from LoRA spikes and a kNN datastore

A decision model answers typed questions about a piece of state with a probability for each option.
browser-decider built one from Ettin-17m that fits in a 10 MB page. It does well on the workflows it
was trained on and badly on any other. This experiment tests a different kind of generality: a "spiky
ball" of many narrow specialists ("spikes"), a router that sends each input to the right spike, a way
to notice an input that matches no spike, and a cheap way to grow a new spike there. Each spike is a
LoRA adapter on one frozen Ettin-17m. A kNN datastore of labelled cases does the routing and the gap
detection, and predicts while a new spike has too few labels to train.

Terms used throughout: the benchmark (Typed Decisions) has four **workflows**, such as customer service
or security incidents. A **case** is one state (a JSON document) with five typed questions about it.
The **library** is the set of spikes already trained, three when one workflow is held out as unseen.
A **library start** initializes a new spike from the library; a **zero start** trains it from a blank
adapter. A **blend** averages the spike's and the kNN datastore's answer distributions. Predictions
P1–P7 and the decision rule were registered in `PLAN.md` before any run and are tabled in the Log.

## Answer

Spikes are cheap, and the library speeds up new spikes. Numbers are accuracy against the
benchmark's gold answers; always picking the most frequent answer scores 0.470. A LoRA spike (246k
trainable parameters, about 0.5 MB, on a 10 MB base) scores 0.714 on its own workflow, against 0.709 for a full fine-tune. <!-- @evidence[[results/summary.json#S_mean.lora]] -->
On an unseen workflow, a new spike can start from the library (the existing spikes' averaged weight
changes, compressed to one adapter). From 64 labelled cases it reaches 0.671. <!-- @evidence[[results/summary.json#G_mean.n64.warm]] -->
From the same cases, a spike started from zero reaches 0.576 and a full fine-tune 0.610. <!-- @evidence[[results/summary.json#G_mean.n64.scratch]] -->
The library start led on all four workflows.

kNN alone is weak. At 16 cases it ties a spike started from zero, 0.523 against 0.525, and it stops improving at about 0.555. <!-- @evidence[[results/summary.json#G_mean.n16.knn]] -->
A learned per-decision blend of kNN and spike improved calibration, not accuracy:
0.555 at 16 cases, against 0.554 for the library-start spike alone and 0.569 for a fixed 50/50 blend. <!-- @evidence[[results/blend.json#mean.n16.warm.gate]] -->

By the rules registered before each round, the verdict stays "a cheap way to store specialists".
A benchmark with similar spikes, or a held-out estimate of each new spike's accuracy, could change it.

## Findings

1. **A LoRA spike matches a full fine-tune on its own workflow.** Mean over 4 workflows and 3 seeds:
   LoRA rank 8 plus head 0.714, full fine-tune 0.709 (P1 held). Seed spread is wider for LoRA: agent
   traces 0.598–0.704. (`results/S-*.json`)
2. **Starting a new spike from the library cuts the labels it needs.** At 64 cases on an unseen
   workflow: library start 0.671, zero start 0.576, full fine-tune 0.610. <!-- @evidence[[results/summary.json#G_mean.n64.full]] -->
   It was ahead of the zero start on every workflow: agent traces 0.666 vs 0.476, invoices 0.748 vs 0.621,
   customer service 0.604 vs 0.550, security 0.666 vs 0.658. Agent traces at 64 cases matched
   its own fully trained spike (0.666 at 2 seeds vs a 0.666 mean over 3). At 16 cases the edge was +0.029, missing P5's 0.03;
   at 300 cases the two were within 0.010. (`results/summary.json` G)
3. **The kNN datastore plateaus early.** Mean over workflows and 5 seeds: 0.522 at 16 cases, 0.551 at 64, 0.555
   at 300, against 0.725 for a new spike at 300. <!-- @evidence[[results/summary.json#G_mean.n300.scratch]] -->
   On agent traces it never passed 0.40, which suggests (untested) that the embedding of that workflow's state does not track its labels.
   (`results/knn.json`)
4. **A fixed equal-weight blend of spike and kNN helps only where kNN is good.** Library start plus kNN at
   16 cases: 0.569 against 0.554 for the spike alone. <!-- @evidence[[results/summary.json#G_mean.n16.mix_warm]] -->
   At 64 cases the blend fell below the spike alone (0.660 vs 0.671), dragged down on agent traces (0.631 vs 0.666) and helped on
   customer service (0.630 vs 0.604). (round 1, `results/summary.json` G)
5. **A learned blend weight improves calibration but not accuracy.** A logistic gate over 7 features of
   the neighbors and the two models, fit on the other three workflows, stayed within 0.002 of the spike
   alone or above it at every number of revealed (labelled) cases n, on average over workflows (16 cases,
   library start: 0.555 vs 0.554; 64: 0.669 vs 0.671; 300: 0.735 vs 0.734). It lowered KL from gold, the
   calibration measure, against the spike alone (0.303 vs 0.363 at 16; 0.143 vs 0.150 at 300). The fixed
   blend was better calibrated at 16 (KL 0.282), and the gate was the best calibrated at 300 (0.143). <!-- @evidence[[results/blend.json#mean.n300.warm.gate_kl]] -->
   It fell below the fixed 50/50 blend at 16 cases (0.555 vs 0.569) and never beat a single learned constant weight
   by 0.01 (P8a, P9, P10, P11 failed; P8b held). A constant weight tuned per workflow on its test set, an
   upper bound, would have reached 0.586 at 16 cases. <!-- @evidence[[results/blend.json#mean.n16.warm.oracle_const]] -->
   (round 2, `results/blend.json`)
6. **The gate has no estimate of how accurate the new spike is.** Spike confidence is an input, but
   confidence is not held-out accuracy. kNN's leave-one-out
   accuracy on the revealed cases predicts kNN's own test accuracy per question (correlation 0.69 at 16
   cases, 0.92 at 300). It barely predicts whether kNN beats the spike (0.19 at 16), although kNN does on
   40% of questions there. An estimate of the spike's accuracy is the likely missing input; that is
   untested. The gate also transferred worst to the workflow least like the others: on agent
   traces, where kNN is weakest, it leaned toward kNN (spike weight 0.41 at 16 cases, library start; accuracy
   0.505 vs 0.571 for the spike alone). (round 2, `out/blend.log` and the diagnostic in the Log)
7. **With no labels, nothing in the library transfers.** Each library spike applied to an unseen workflow,
   their uniform merge, and the spike the router picks all scored 0.254–0.440, below the 0.470 frequency
   baseline (P3 held). The uniform merge was never the best of them. (`results/summary.json` fallback)
8. **Routing and gap detection are trivially perfect on this fixture.** Routing accuracy 1.0; gap AUROC 1.0
   for every held-out workflow (P2 held). <!-- @evidence[[results/routing.json#accuracy]] -->
   Each workflow's state JSON has its own fields, so this is not evidence that routing would work among
   similar spikes.

## Method

- **Fixture:** Typed Decisions (`browser-decider/data/td`): four workflows, 300 train and 100 test cases each,
  five typed questions per case, scored with the dataset's own argmax-against-gold accuracy.
- **Base:** frozen `jhu-clsp/ettin-encoder-17m`, browser-decider's cross-encoder layout (`train_enc.py`).
- **Spike:** LoRA rank 8, alpha 16, on every `Wqkv`, `Wo` and `Wi` in 7 layers, plus the MLP head:
  245,761 trainable parameters. LoRA learning rate 5e-4; full fine-tune 8e-5 with the head at 10×.
- **Datastore:** the 10 MB build's embedder (`e17c`, 768-dim) over each case's state JSON only. kNN takes
  the top min(8, n) revealed cases per question, weights them by softmax(cos / 0.05), and averages their
  gold distributions, with 1% uniform mixed in.
- **Arms:** S: one spike per workflow, 10 epochs, 30 cases for validation, 3 seeds, LoRA and full.
  R: routing (8-NN majority) and gap detection (1 − max cosine to the library) with each workflow held
  out. G: for each held-out workflow, n ∈ {0, 16, 64, 300} revealed cases (same nested sample for every
  method). Methods: fallback (n = 0), knn (5 seeds), and scratch (zero start) / warm (library start) / full
  (full fine-tune), 2 seeds each, with fixed epochs, no validation split and softmax temperature 1.
  "Nested" means the 16 revealed cases are inside the 64, and the 64 inside the 300. warm = the mean of the three library spikes' B·A deltas, reduced to rank 8 by SVD,
  plus their averaged heads. Mixes average the spike's and kNN's distributions.
- **Blend gate (round 2):** λ = sigmoid(w·f + b) weights the spike, 1 − λ the kNN distribution. f =
  top-1 cosine to a revealed case; kNN leave-one-out accuracy for the question on the revealed cases;
  kNN and spike confidence (max probability); their total-variation distance; log n; a library-start flag.
  For each held-out workflow the 8 weights are fit on the other three workflows' new-spike runs (arm G; soft
  cross-entropy, L2 1e-3, L-BFGS) and applied unchanged. Comparisons: a constant λ fit the same way, the
  fixed 0.5, and the best constant per workflow chosen on its test set (upper bound, post hoc).
- **Compute:** one Colab L4 GPU session ran the 100 GPU runs (24 spikes, 4 fallback sets, 72 growth runs) in
  about 25 minutes of training. The CPU arms took about 5 minutes on a 4-core container.

## Log

### Round 1 (2026-11-05)

Oskar asked for the experiment after discussing kNN-plus-frozen-model and LoRA libraries as a spiky
ball. PLAN.md registered P1–P7 before any GPU run. Prior art (3 searches): Arrow (arXiv 2405.11157), LoRA
Retriever, LoRAuter (arXiv 2601.21795), kNN-CM (EMNLP Findings 2023), Mireshghallah et al. (arXiv
2209.05706), MixLoRA-DSI (EMNLP 2025). The composition tested here was not found.

| n cases | scratch | warm | full | knn | scratch+knn | warm+knn |
|---|---|---|---|---|---|---|
| 16 | 0.525 | 0.554 | 0.520 | 0.523 | 0.539 | 0.569 |
| 64 | 0.576 | 0.671 | 0.610 | 0.554 | 0.597 | 0.660 |
| 300 | 0.725 | 0.734 | 0.723 | 0.555 | 0.715 | 0.719 |

Mean over the four held-out workflows; GPU methods 2 seeds, kNN shown for the same 2 seeds.

| prediction | measured | held |
|---|---|---|
| P1 LoRA within 0.02 of full | 0.714 vs 0.709 | yes |
| P2 routing ≥ 0.98, gap AUROC ≥ 0.95 | 1.0, 1.0 | yes |
| P3 every n = 0 fallback ≤ 0.50 | max 0.440 | yes |
| P4 knn > scratch at 16; scratch ≥ knn + 0.03 at 300 | 0.523 vs 0.525; 0.725 vs 0.555 | no (first half) |
| P5 warm ≥ scratch + 0.03 at 16; within 0.02 at 300 | +0.029; +0.010 | no (first half) |
| P6 mix ≥ better component + 0.01 at n ≤ 64 | warm+knn +0.015 at 16, −0.011 at 64; scratch+knn −0.011 at 16 | no |
| P7 scratch at 300 within 0.03 of S's LoRA | 0.725 vs 0.714 | yes |

What broke: the first Colab launch failed because stock Ettin ships `pytorch_model.bin` and the
download pattern allowed only `*.safetensors`. A local cache hid this in the CPU smoke test. The e17c
embedder's `tokenizer_config.json` names transformers 5's `TokenizersBackend`, so under 4.57.1 the
tokenizer loads from `tokenizer.json` directly. Every S run picked its last epoch as best, so 10 epochs
may undertrain both spike kinds equally.

Reading at the time: the library is useful as an initialization, not as a merged model, and kNN is a
weak predictor on this fixture. Next candidates: a blend weight learned from neighbor similarity; a
merge with learned per-module coefficients (P²-MOPD, arXiv 2611.03953, passed on by a sibling session),
which could improve the warm start; and a fixture with confusable spikes, to test routing at all.

### Round 2 (2026-11-05): a learned blend weight

Oskar: "try the learned blend weight next". PLAN.md registered P8–P11 first. Prior art: adaptive kNN-MT
(Zheng et al., ACL 2021, arXiv 2105.13022) learns the interpolation of kNN-augmented machine translation from neighbor distances with a
small meta-network that transferred across domains. CPU only, about 25 seconds, from round 1's saved
distributions.

| n | start | spike | kNN | fixed 0.5 | constant λ | gate | best constant on test (bound) |
|---|---|---|---|---|---|---|---|
| 16 | zero | 0.525 | 0.523 | 0.538 | 0.529 | 0.532 | 0.556 |
| 16 | library | 0.554 | 0.523 | 0.569 | 0.560 | 0.555 | 0.586 |
| 64 | zero | 0.576 | 0.554 | 0.597 | 0.589 | 0.595 | 0.604 |
| 64 | library | 0.671 | 0.554 | 0.660 | 0.668 | 0.669 | 0.683 |
| 300 | zero | 0.725 | 0.555 | 0.715 | 0.724 | 0.725 | 0.734 |
| 300 | library | 0.734 | 0.555 | 0.719 | 0.728 | 0.735 | 0.746 |

KL from gold (lower is better), library start: 16 cases spike 0.363, fixed 0.282, gate 0.303; 300 cases
spike 0.150, fixed 0.164, gate 0.143.

| prediction | measured | held |
|---|---|---|
| P8a gate > better component at 16 and 64, both starts | library start at 64: 0.669 vs 0.671 | no |
| P8b gate ≥ spike − 0.005 at 300 | 0.725 vs 0.725; 0.735 vs 0.734 | yes |
| P9 gate > fixed 0.5 at every n | below at 16 (both) and 64 (zero start) | no |
| P10 gate ≥ constant λ + 0.01 at 16 or 64 | best +0.006 (zero start, 64) | no |
| P11 zero-start gain over spike largest at 16 | +0.007 at 16, +0.019 at 64 | no |

Diagnostic, per (workflow, question, n, start, seed): correlation of kNN's leave-one-out accuracy on the
revealed cases with kNN's test accuracy was 0.69 / 0.88 / 0.92 at n = 16 / 64 / 300. With kNN's test
accuracy minus the spike's it was 0.19 / 0.44 / 0.72. kNN beat the spike on 40% / 35% / 14% of questions.

Reading at the time: the decision rule says the gated blend is not adopted (P8a failed). Its use is no
harm on average and the best calibration at 300 cases. On agent traces it did harm: 0.505 vs 0.571 at
16 cases with the library start. The likely missing input is the new spike's own held-out accuracy. That
needs cross-fitting: train a few spikes, each on all but one fold of the revealed cases, and score each
on the fold it did not see. Round 1's 16- and 64-case spikes took 6–8 s each on the L4. Without it, a gate fit on other workflows guesses wrong on the
workflow least like them.
