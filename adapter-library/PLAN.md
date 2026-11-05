# PLAN: a spiky-ball decider — LoRA spikes, a kNN datastore, and growing a new spike

Registered 2026-11-05, before any training run.

## Question

Oskar asked whether a small frozen decision model, a library of LoRA adapters and a growing kNN
datastore could together cover more than any one model does. Here, "general" means a spiky ball: many
narrow spikes that are each good, plus a router, plus a way to notice an input that falls between spikes
and grow a spike there. It does not mean smooth competence everywhere. browser-decider measured
smoothness (Ettin-17m scores 0.740 in-domain and 0.353 leave-one-workflow-out) and found none. This plan
measures the spiky version instead:

1. Is a spike cheap? Does a LoRA adapter on frozen Ettin-17m match a full fine-tune on its own workflow?
2. Can the system tell which spike an input belongs to, and when it belongs to none?
3. Growing a spike: on a workflow the library has never seen, how fast do the kNN datastore, a new
   adapter and the two together reach in-domain accuracy as labels arrive? Does the existing library
   help a new spike start faster?

## Prior art (3 web searches, 2026-11-05)

The components exist; I did not find this composition.

- **Routing a LoRA library:** Arrow (Ostapenko et al., arXiv 2405.11157) routes per token from each
  adapter's SVD direction. LoRA Retriever (Zhao et al., 2024) and LoRAuter (arXiv 2601.21795) route
  from task representations. LoRAuter reports +5.2 points over LoRA Retriever on unseen tasks.
- **Classifier plus kNN interpolation:** kNN-CM (Findings of EMNLP 2023) takes a convex combination of a
  fine-tuned classifier and a kNN vote over its training set. Mireshghallah et al. (arXiv 2209.05706)
  adapt to temporal drift by changing only a kNN datastore over a frozen encoder.
- **Growing on out-of-distribution data:** MixLoRA-DSI (EMNLP 2025) adds LoRA experts when enough
  out-of-distribution inputs accumulate. SEMA uses per-adapter descriptors as a shift detector.

This plan differs in four ways. The datastore does three jobs: it routes, it detects gaps, and it
predicts while a new spike has too few labels to train. The model is a 17M-parameter decider meant for
a browser. The question is the label curve of a new spike, not static multi-task accuracy.

## Fixture

`LocalLLaMA/typed-decisions`, as in browser-decider (`browser-decider/data/td`): four workflows (agent
traces, customer service, invoices, security incidents), each with 300 train and 100 test cases and five
typed questions per case. Each question has a fixed option set within its workflow. The scorer is the
dataset's own `score_one`: accuracy is the argmax matching gold, plus KL from gold.

Four workflows give four spikes. The fixture's limits are stated up front:

- **Routing is expected to be easy.** Each workflow's state JSON has its own keys, and its question
  wording is unique, so separating spikes on this fixture is not a hard test. Routing between
  *confusable* spikes needs a second fixture with near spikes, for example customer service split by
  topic. That fixture is stage 2 and is not run here.
- **Leave-one-workflow-out has only four folds.**

## Models and representations

- **Base:** `jhu-clsp/ettin-encoder-17m` (MLM), frozen. The cross-encoder layout and MLP head are
  browser-decider's (`run/train_enc.py`): one sequence per question, scored at each option's `[MASK]`.
- **Spike:** LoRA rank 8 (alpha 16) on every `Wqkv`, `Wo` and `Wi` linear layer in all 7 layers, plus the
  MLP head. That is about 180k LoRA parameters plus 66k head parameters, roughly 0.5 MB at fp16 per
  spike, against a 10 MB base.
- **Datastore key:** the 10 MB build's embedder (`e17c`: retrieval-tuned Ettin-17m, mean pool plus a dense
  layer to 768 dims), applied to each case's state JSON only. The question text is left out of the key
  because it would make routing trivially perfect.

## Arms

- **S. Spikes (GPU).** For each workflow, train on its 300 train cases (10% held for validation, best
  epoch by validation KL, per-type temperature), 10 epochs, seeds 0–2. Train both a LoRA spike and a
  full fine-tune. Score each on the workflow's own test set.
- **R. Routing and gap detection (CPU).** *Routing:* the datastore holds all four workflows' train cases.
  Each test case goes to the majority workflow among its 8 nearest neighbors; report accuracy. *Gap
  detection:* with workflow W left out of the datastore, the gap score is 1 minus the maximum cosine
  similarity to any stored case. Report the AUROC for separating W's test cases from the other three
  workflows' test cases, for each W.
- **G. Growing a new spike on W.** The library holds the other three workflows. The first n of W's train
  cases (seeded shuffle, the same sample for every method) are revealed, for n in {0, 16, 64, 300}. All
  methods are scored on W's 100 test cases.
  - `fallback` (n = 0): each of the three library spikes applied to W, their uniform merge, and the spike
    the router picks for each case.
  - `knn` (CPU): for each question, a similarity-weighted average of the gold distributions of the
    top k = min(8, n) revealed cases. Weights are a softmax of cosine / 0.05, then mixed with 1% uniform.
  - `scratch`: a new LoRA spike from zero init, trained on the n cases.
  - `warm`: a new LoRA spike initialized from the library. The mean of the three spikes' weight deltas
    (B·A, per layer) is reduced to rank 8 by SVD, and the three heads are averaged.
  - `full`: a full fine-tune on the n cases (the browser-decider recipe).
  - `mix`: (scratch + knn) / 2 and (warm + knn) / 2, fixed equal weights. A λ sweep is reported
    separately and labelled post hoc.

  The growth runs have no validation split, since n can be 16. They train for a fixed number of epochs,
  max(10, enough for 150 steps) capped at 40, at T = 1. Seeds 0–1 for the GPU methods; seeds 0–4 for
  kNN, which is free.

## Predictions

- **P1.** A LoRA spike's in-domain accuracy is within 0.02 of the per-workflow full fine-tune (mean over
  the 4 workflows and 3 seeds).
- **P2.** On this fixture, routing accuracy is at least 0.98 and gap AUROC at least 0.95 for every W.
  This is expected; see the fixture limits.
- **P3.** At n = 0, every fallback (best single spike, merge, routed spike) scores at most 0.50, near the
  0.470 prior.
- **P4.** At n = 16, knn beats scratch. By n = 300, scratch beats knn by at least 0.03.
- **P5.** At n = 16, warm beats scratch by at least 0.03; by n = 300 the two are within 0.02.
- **P6.** The mix beats both of its components at n ≤ 64, by at least 0.01 over the better component.
- **P7.** At n = 300, scratch is within 0.03 of S's LoRA spike. This is a sanity check: they differ only
  in validation and temperature fitting.

## Decision rule

The spiky-ball design earns stage 2 (confusable spikes, then a browser build with spikes as
downloadable files) if P1 holds and at least one of P4, P5 or P6 holds. P1 means spikes are cheap.
P4–P6 mean the datastore or the library measurably shortens the time a new spike needs. If P1 holds
but P4–P6 all fail, the result is "a LoRA library is a cheap way to store specialists, and nothing
more."

## Round 2: a learned blend weight (registered 2026-11-05, before running)

Round 1's equal-weight blend of spike and kNN helped only where kNN was good. Round 2 learns the weight
per decision: p = λ·p_spike + (1 − λ)·p_knn, with λ = sigmoid(w·f + b). It runs on CPU from round 1's
saved distributions; no new training of spikes.

**Prior art:** adaptive kNN-MT (Zheng et al., ACL 2021, arXiv 2105.13022). A meta-k network of a few
thousand parameters, fed neighbor distances, replaces the fixed λ in kNN-MT. It was trained with few
samples, and a network trained on one domain transferred to others. The gate here is the same idea at
8 weights.

**Features f, all available at deployment** (no test labels):
1. top-1 cosine between the case and the nearest revealed case (is there a close neighbor);
2. kNN leave-one-out accuracy for this question, measured on the revealed cases themselves (is kNN
   good on this question of this workflow);
3. kNN confidence (max probability); 4. spike confidence (max probability);
5. total variation between the two distributions (do they disagree);
6. log n; 7. warm-start indicator.

**Fitting without W's test labels:** for held-out workflow W, fit the gate on the growth runs of the
other three workflows (their test decisions, every n in {16, 64, 300}, seeds 0–1, scratch and warm), by
soft cross-entropy against gold with a small L2 penalty. Apply it unchanged to W. This is
leave-one-workflow-out for the gate, as a library that has already grown three spikes would do for its
fourth. Comparisons: the spike alone, kNN alone, round 1's fixed 0.5, a single constant λ fit the same
cross-workflow way, and, post hoc and labelled as an upper bound, the best constant λ per (W, n,
method) chosen on W's test.

**Predictions:**
- **P8a.** At n ∈ {16, 64}, the gated blend beats the better of spike alone and kNN alone (mean over
  the 4 workflows), for both scratch and warm.
- **P8b.** At n = 300, the gated blend is within 0.005 of the spike alone or above it (no harm).
- **P9.** The gated blend beats the fixed 0.5 blend at every n, for both methods.
- **P10.** The features matter: the gated blend beats the cross-workflow constant λ by at least 0.01
  at n = 16 or n = 64, for at least one method.
- **P11.** For scratch, the gain over the spike alone is largest at n = 16.

**Decision rule:** the gated blend becomes the system's predictor if P8a and P8b hold. If P10 fails,
the constant λ is enough and the features are dropped.

## Round 3: cross-fitted blend weight (registered 2026-11-05, before running)

Round 2's gate had no estimate of the new spike's held-out accuracy. Round 3 measures it inside the new
workflow W, from its revealed cases only. The n revealed cases are split into 4 folds (case i goes to
fold i mod 4 in the seeded reveal order). For each fold, a spike is trained on the other three folds
with round 1's growth recipe (same start: zero or library) and predicts the held-out fold. kNN predicts
each held-out fold from the other three folds' cases. Every revealed decision then has an out-of-fold
spike prediction and an out-of-fold kNN prediction, made under the same handicap: each saw 75% of the
cases. This needs 4 extra spikes per (W, n, seed, start): GPU, n ∈ {16, 64}, seeds 0–1. n = 300 is
skipped; the spike dominates there.

**Arms (CPU, after the GPU folds):**
- `cv_lambda`: for each (W, question), choose λ from {0, 0.1, …, 1} to minimize the out-of-fold
  blend's soft cross-entropy against gold on the revealed cases. Apply it to round 1's final spike and
  kNN (both on all n cases) on W's test.
- `cv_select`: per (W, question), use the spike or kNN alone, whichever had higher out-of-fold accuracy
  (ties go to the spike).
- `gate2`: round 2's gate plus two per-question features, spike out-of-fold accuracy and kNN
  out-of-fold accuracy, fit leave-one-workflow-out as in round 2.

**Predictions (mean over the 4 workflows, accuracy):**
- **P12.** cv_lambda beats the spike alone at n = 16 and n = 64, for both starts.
- **P13.** cv_lambda beats round 2's gate by at least 0.01 at n = 16, for at least one start.
- **P14.** cv_lambda avoids round 2's harm on agent traces: at n = 16, library start, it is within
  0.01 of the spike alone or above it.
- **P15.** gate2 beats round 2's gate by at least 0.01 at n = 16, for at least one start.
- **P16.** Per (W, question, n, seed, start) at n = 16, the out-of-fold difference (kNN accuracy minus
  spike accuracy) correlates with the test difference at r ≥ 0.4. Round 2's kNN-only proxy reached 0.19.

**Known bias:** the out-of-fold spike trains on 75% of the cases, and the final spike trains on all of
them. Growth curves are steep at small n, so the out-of-fold numbers understate the final spike more
than they understate kNN, which pushes λ toward kNN.

**Decision rule:** cv_lambda becomes the blend if P12 and P14 hold.

## Round 4 (registered 2026-11-05, before running)

Two parts: fix round 3's loss at 64 cases, and test a router spike on spikes that are hard to tell apart.

### 4a: repairing the cross-fitted blend at 64 cases

Round 3's fold spikes saw 75% of the revealed cases and understated the final spike.
- `cv8_lambda` (GPU): round 3's method with 8 folds, so each fold spike sees 87.5% of the cases.
- `cv_stop` (CPU, round 3's 4-fold data): per question, λ = 1 (spike alone) when the spike's
  out-of-fold accuracy exceeds kNN's by at least 0.10; otherwise round 3's λ.

**P17.** cv8_lambda is at least the spike alone at 64 cases with the library start (round 3: −0.006)
and at least 0.570 at 16 cases with the library start.
**P18.** cv_stop is no more than 0.002 below the spike alone in any of the four cells (16 and 64 cases,
both starts) and above it on average over the four.

**Decision rule:** a method that meets P18's condition (whether it is cv_stop or cv8_lambda) and keeps
P14 becomes the blend. If both qualify, cv_stop wins because it needs no extra training.

### 4b: a router spike on confusable spikes

**Fixture.** Each workflow is split into 3 families by a factor the dataset records, giving 12
families whose spikes share state format and questions:
- customer service by topic: billing / physical orders / account and product;
- invoices by issue: quantity and timing / pricing (price, tax, currency, freight) / process (duplicate,
  missing PO, clean);
- security by rule: sign-ins and credentials / privileges and configuration / data and hosts;
- agent traces by task: data tasks / infrastructure / housekeeping.

How visible the family is varies by design. The alert rule and the agent task appear verbatim in their
states, so a router can match strings. The customer-service topic appears verbatim in 30 of 300
states; otherwise it must be read from the customer's message. The invoice issue never appears and must
be inferred from numbers such as quantity billed against quantity received. Routing is reported per
workflow for this reason.

**Router spike.** A LoRA decision model on the frozen base (zero start, the spike recipe). One choice
question per case, "Which specialist should handle this case?", with one option per library family
(its text description) plus `none`. The input is the state only.

**Protocol, 12 folds.** For held-out family F (each of the 12 in turn), G is the family three places
later in the list, which is always in another workflow. The router trains on the other 10 families'
train cases, labelled with their family, plus G's train cases labelled `none`. F appears nowhere in
training, neither as an option nor as data. Test on all 400 test cases with two option sets: A, the
library plus `none`; and B, A with F's description added.

**kNN router baseline.** The datastore holds the 10 library families' train cases (e17c state
embeddings). It routes by 8-NN majority and scores gaps as 1 minus the maximum cosine. It can route to F
only after seeing examples of F; it is given 4.

**Downstream.** One sub-spike per family (zero start, growth recipe, the family's train cases), scored
on its workflow's test set. Decision accuracy for library-family test cases is compared under four
routings: the router's choice, kNN's choice, the oracle family, and no routing (round 1's whole-workflow
spike). A case routed to `none` or to another workflow falls back to the whole-workflow spike.

**Predictions (mean over the 12 folds):**
- **P19.** Router-spike routing accuracy on library-family test cases (option set A, `none` counted
  wrong) is at least 0.85 and at least kNN's.
- **P20.** Gap detection for F: AUROC of the router's P(none) on F's test cases against library-family
  test cases is at least 0.75, and above kNN's 1 − max-cosine AUROC.
- **P21.** Zero-shot new spike: with F's description added (option set B), at least 50% of F's test
  cases are routed to F, and library routing accuracy falls by at most 0.05.
- **P22.** kNN given 4 labelled cases of F routes a larger fraction of F's test cases to F than the
  zero-shot router does.
- **P23.** Splitting a workflow into sub-spikes does not pay at this data size. Router-routed
  sub-spikes score at most 0.01 above the whole-workflow spike, and so do oracle-routed sub-spikes.

**Decision rule:** the router spike replaces kNN routing if P19 and P20 hold. P21 decides whether a new
spike can be added by description alone, with no examples.
