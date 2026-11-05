# RESULTS: a spiky-ball decider from LoRA spikes and a kNN datastore

A decision model answers typed questions about a piece of state with a probability for each option.
browser-decider built one from Ettin-17m that fits in a 10 MB page. It does well on the workflows it
was trained on and badly on any other. This experiment tests a different kind of generality: a "spiky
ball" of many narrow specialists ("spikes"), a router that sends each input to the right spike, a way
to notice an input that matches no spike, and a cheap way to grow a new spike there. Each spike is a
LoRA adapter on one frozen Ettin-17m. A kNN datastore of labelled cases does the routing and the gap
detection, and predicts while a new spike has too few labels to train.

## Answer

Spikes are cheap, and the library helps a new spike learn faster. Numbers are accuracy against the
benchmark's gold answers; always picking the most frequent answer scores 0.470. A LoRA spike (246k
trainable parameters, about 0.5 MB, on a 10 MB base) scores 0.714 on its own workflow, against 0.709 for a full fine-tune. <!-- @evidence[[results/summary.json#S_mean.lora]] -->
On a workflow the library has never seen, a new spike can start from the library: the average of the
existing spikes' weight changes, compressed to one adapter. From 64 labelled cases it reaches 0.671. <!-- @evidence[[results/summary.json#G_mean.n64.warm]] -->
From the same cases, a spike started from zero reaches 0.576 and a full fine-tune 0.610. <!-- @evidence[[results/summary.json#G_mean.n64.scratch]] -->
The library start led on all four workflows, though by only 0.008 on security incidents.

The kNN datastore was a weak predictor. At 16 cases it ties a spike started from zero, 0.523 against 0.525, and it stops improving at about 0.555. <!-- @evidence[[results/summary.json#G_mean.n16.knn]] -->
A fixed equal blend of spike and kNN helped only where kNN was good.

By the decision rule registered before the runs, the verdict is "a cheap way to store specialists".
The spikes-match-fine-tunes prediction held, while the three growth predictions each failed a part.
A fixture with similar spikes, or a learned blend weight, could change it.

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
3. **The kNN datastore plateaus early.** Mean over workflows and 5 seeds (the Log table uses the 2 GPU seeds, hence 0.523): 0.522 at 16 cases, 0.551 at 64, 0.555
   at 300, against 0.725 for a new spike at 300. <!-- @evidence[[results/summary.json#G_mean.n300.scratch]] -->
   On agent traces it never passed 0.40: the embedding of that workflow's state does not track its labels.
   (`results/knn.json`)
4. **A fixed equal-weight blend of spike and kNN helps only where kNN is good.** Library start plus kNN at
   16 cases: 0.569 against 0.554 for the spike alone. <!-- @evidence[[results/summary.json#G_mean.n16.mix_warm]] -->
   At 64 cases the blend fell below the spike alone (0.660 vs 0.671), dragged down on agent traces (0.631 vs 0.666) and helped on
   customer service (0.630 vs 0.604). The blend weight has to depend on how good the neighbors are.
5. **With no labels, nothing in the library transfers.** Each library spike applied to an unseen workflow,
   their uniform merge, and the spike the router picks all scored 0.254–0.440, below the 0.470 frequency
   baseline (P3 held). The uniform merge was never the best of them. (`results/summary.json` fallback)
6. **Routing and gap detection are trivially perfect on this fixture.** Routing accuracy 1.0; gap AUROC 1.0
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
  method). Methods: fallback (n = 0), knn (5 seeds), and scratch / warm / full (2 seeds; fixed epochs, no
  validation, T = 1). warm = the mean of the three library spikes' B·A deltas, reduced to rank 8 by SVD,
  plus their averaged heads. Mixes average the spike's and kNN's distributions.
- **Compute:** one Colab L4 session, 100 arms in about 25 minutes of training; the CPU arms took about 5 minutes here.

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
