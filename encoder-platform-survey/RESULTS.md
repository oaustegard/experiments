# encoder-platform-survey

**Started / finished:** 2026-09-20 · **Status:** done — survey plus one measured
arm. **No open Jev-class model is a base to build on. Build the boring thing
instead: a fixed-schema classifier fine-tuned on frontier-labelled data, on the
Ettin encoder family, served as int8 ONNX on CPU.**

**Question (Oskar).** TypeSafe's Jev put non-autoregressive decision models back
in the conversation. For routing, compliance gates and first-pass paper filters:
(a) is there a solid open version to base work on, (b) if not, should we build
one, and (c) on what — ModernBERT, something smaller, something quantized for
CPU?

## Jev's architecture and accuracy

TypeSafe has published no architecture, size or training data. Two sources
constrain it:

- Archer Hume's black-box probe (~10,000 API calls,
  [archerhume.com](https://archerhume.com/posts/jevs-architecture-unmasked/))
  concludes **a causal decoder, prefill only, no decode loop**: token accounting
  is additive (state + question suffixes), a value hidden in a sibling question
  is unreachable (p=0.00) while the same value in shared state is used (p>0.90),
  100 questions cost about the same as one, ~30k tokens clear in ~160 ms. The
  84.6% MMLU-Pro that TypeSafe reports needs frontier pretraining. His estimate
  is a sparse mixture of experts around 10B active parameters.
- TypeSafe's own evals site ([evals.typesafe.ai](https://evals.typesafe.ai/))
  scores Jev at **67.8% averaged over its four workflows**, against Sol 74.1%,
  Opus 5 73.1%, Sonnet 5 67.8%, Terra 67.9%, DeepSeek v4 Pro 65.5%, Haiku 4.5
  53.6%. Reference labels are the average of GPT-6 Astra and Fable 5.1 at high
  thinking. Jev is Sonnet-accurate at 40–200x the speed; it is not more accurate
  than the frontier.

So "non-autoregressive encoder" is the wrong label for Jev itself. It is a
decoder read out in one forward pass. The encoders are what the clones built,
and the two are not interchangeable at the sizes the clones use.

## Open clones on 2026-09-20

Jev launched 2026-09-15. Five days later: 6 clones in Latent.Space's roundup,
19 projects in `cobanov/awesome-jev`, 11 `typed-decisions` checkpoints and 6
`jev`-named models on the Hub. Everything below is single-author, under a week
old, and untested by anyone but its author unless stated.

| project | backbone | what it does | numbers | verdict |
|---|---|---|---|---|
| `convaiinnovations/laya` (1.4k stars in 2 days) | ModernBERT-large 421M / mmBERT-base | cross-encoder: `[MASK]` per option, softmax over mask positions | base checkpoint **0.36 on typed-decisions, below the 0.46 majority class**; the 0.766 headline is fine-tuned on that benchmark's own train split; ECE 0.081 only after temperature fitting (0.213 raw); Khmer 0.000 acc at 0.952 conf | reviewed 2026-09-19 (memory 767b36a7). RLCD training scripts not released (issue #4). Vocabulary copied from Jev, mechanism is a 2019 cross-encoder |
| `wfzyx/von` | NLI zero-shot cross-encoder | | | reviewed 2026-09-19 (memory e5e8801e): calibration temperature computed and never applied, Phase 2 benchmark-contaminated, a fabricated RLCD citation |
| `intikhab49/open-jev-typed-decision-engine` | ModernBERT-base 150M | fine-tune on typed-decisions train (1,016 cases), soft CE + Brier, per-type temperature, 60/40 ensemble with a frozen probe | **0.697 vs Jev 0.727 zero-shot**, ECE 0.057 vs 0.144; ~60 ms/case | the honest one: states that a fine-tune on the test distribution still lost to Jev zero-shot, that 400 cases cannot resolve <0.02, that the ensemble's probe breaks on unseen questions (drops to 0.624). 4 commits, 7 stars. A recipe, not a base |
| `Heman10x-NGU/Verdict-open-jev` (`heman10x/rlcd-modernbert-151m`) | ModernBERT-base + GLiClass head | zero-shot label scoring, CE + Brier, L-BFGS temperature, WebGPU demo | **48.07% on TypeSafe's 337-case public set vs 90.80% Jev and 88.43% DiffusionGemma 26B**; Banking77 in-domain 95.2% | the one apples-to-apples external number in the field: a 150M encoder zero-shot trails by 42 points. Abstention recall drops from 75.5% → 18% with hard-negative siblings; 3–4.5% of answers flip on option reorder |
| `com-kotobalabs/open-jev-deberta-v3-large` | DeBERTa-v3-large 0.4B | typed questions, CE + Brier, trained on banking77 + sst5 + boolq | in-domain 0.854 / ECE 0.022; **out-of-domain 0.690** / ECE 0.035; 1.8 s for 4 questions on M1 CPU fp32 | clean training report, no Jev comparison, the in-domain/OOD gap is the finding |
| `pngwn/nanodiff-350m-typed-decisions` | 350M masked-diffusion LM (LLaDA recipe) | Bayes-optimal synthetic generator, calibration loss | ECE 0.036 vs 0.333 for the accuracy-only baseline at equal 0.67 accuracy | the most careful calibration study here; synthetic data, no external benchmark |
| `mmastrac` DiffusionGemma / vLLM PR #57250 | DiffusionGemma 26B | one denoise step over seeded answer slots | "roughly tied" with Jev on Mastracci's live evals; 162 decisions/s on a DGX Spark | works because it is 26B. Not a CPU story (memory fe2accf7) |
| `TheoLeeCJ/openjev`, `madiator` Nimble, `jaredpalmer` Kev, `Meanblock/JEV-CPU`, `vagmi/jev-lite`, `Mapika/decider` | Qwen 0.5B–9B, Gemma 4 E4B, LoRA or frozen | read logits at the answer position, prefill only | Nimble 90.12% vs Jev 93.21% on its own set; openjev 0.845 vs 0.883 modal agreement on 102 aligned rows | decoder-readout clones. Closest to what Jev actually is; each needs a GPU for the latency the pitch promises |
| `AbdelStark/jev-benchmarks` | eval harness | accuracy, Brier, NLL, top-label ECE, coverage at a fixed error budget, latency | Jev vs GLiNER2.5-multi: AG News 0.91 vs 0.70, Banking77 0.87 vs 0.61, DAIR Emotion 0.48 vs 0.44 with Jev worse calibrated (Brier 0.846 vs 0.668) | a 300-example pilot, and the right metric set (coverage at error budget is the automation question) |

Three things repeat across the table. Every accuracy comparison that favours a
clone pits its fine-tuned checkpoint against Jev zero-shot. Nobody has released
the training data for a general (schema-agnostic) checkpoint, so nothing is
reproducible past its own README. And the only external, same-benchmark test of a
150M encoder against Jev is 48% vs 91%.

That gap is the same mechanism `hypothetical-classification` measured on 2026-08-31:
zero-shot over an arbitrary schema is a knowledge task, and world knowledge is the
first thing a model loses when it shrinks. The clones did not close it with a
loss function.

## Answer

**(a) No.** There is no open Jev-class model to base work on. The field is five
days old, single-author, self-benchmarked, and the one project with a released
base checkpoint (laya) scores below majority class on its own benchmark before
fine-tuning.

**(b) Yes, build — but not a Jev.** The three use cases named are fixed-schema:

| use case | schema | what a wrong answer costs |
|---|---|---|
| model routing | 3–5 tiers | a cheap-tier miss is caught by the verifier (`agent-routing` 2.1.0 rule: the verifier's holder escalates) |
| compliance gate | pass / flag / block | asymmetric; want abstention and coverage-at-error-budget, not argmax accuracy |
| first-round paper filter | relevant / not, maybe 5–10 topics | a false negative is invisible; measure recall at fixed precision |

None of them needs a dynamic schema at inference. All three can be labelled by a
frontier model at a few cents per hundred rows, which is exactly the regime where
a 150M encoder works: Verdict's 95.2% on Banking77 in-domain, kotobalabs' 0.854
in-domain against 0.690 OOD, Ettin-150m's 88.9 GLUE. A fixed-schema classifier
head plus temperature scaling on a fine-tuned encoder is the 2019 recipe, and the
2019 recipe is what the clones fall back to when they report a good number.

The model is a known recipe. The eval is where the care goes:

- Frontier-labelled train/dev/test with a human-checked slice
  (`judge-eval-discipline`: TPR/TNR above 90% against human labels before the
  teacher labels are trusted; Rogan-Gladen plus bootstrap CI on every rate).
- Report **coverage at a fixed error budget** (AbdelStark's framing), not
  accuracy alone; that is the number that says how much traffic the gate can
  take unsupervised.
- Keep a zero-shot baseline in the table to be beaten before any training:
  `knowledgator/gliclass-modern-base-v2.0` (151M, Apache-2.0, IMDB 0.92 / AG News
  0.71 / Emotions 0.43 F1 zero-shot) and an NLI cross-encoder. BTZSC
  ([arXiv:2603.11991](https://arxiv.org/abs/2603.11991), 38 checkpoints, 22
  datasets) puts rerankers on top for zero-shot (Qwen3-Reranker-8B macro-F1
  0.72) and GTE-large as the accuracy/latency knee; both are candidate
  labellers, neither is the deployable artifact.
- For the routing case specifically, RouteLLM (LMSYS, 2024) shipped a BERT
  router and a matrix-factorisation router with a public preference dataset.
  Not checked this session; check before writing a router from scratch.

**(c) Platform: Ettin encoders, int8 ONNX, CPU.** Reasons, then the measurement.

- **Ettin** ([jhu-clsp](https://github.com/jhu-clsp/ettin-encoder-vs-decoder),
  MIT, 2T open tokens, ModernBERT recipe and tokenizer) beats ModernBERT at
  every matched size and ships six sizes, so the same code path scales from 17M
  to 1B. GLUE averages: 17M 79.2, 32M 83.5, 68M 87.2, 150M 88.9 (ModernBERT-base
  88.4), 400M 90.8 (ModernBERT-large 90.4), 1B 91.6. MNLI 79.5 → 91.8. The paper
  also measures that a 400M encoder beats a 1B decoder on classification, which
  is the whole argument for encoders here.
- **ModernBERT-base** stays the reference: 4.4M downloads, official int8 ONNX in
  the repo, and every clone above already runs on it. If Ettin-150m needs an
  export we do it with Optimum; only the 17M and 32M sizes have community ONNX.
- **NeoBERT** (250M, 89.2 GLUE) is a single size and slower per token than
  ModernBERT-base in the ladder. **mmBERT-small** (140M total, 42M
  non-embedding, 1,800 languages) is the pick if the paper filter must read
  non-English abstracts; two-thirds of its bytes are a 256k-vocab embedding table.
- **DeBERTa-v3** loses to ModernBERT-base on GLUE and runs 2–4x slower; skip.
- Quantization: every candidate ships or has a community int8 export, and
  onnxruntime's dynamic int8 on this CPU (AVX-512 VNNI) is the deployment path
  `jina-int8-remax_kb` already uses. Nothing to build.

### Measured: CPU latency ladder

Batch 1, median of 15, onnxruntime 1.29 CPU EP, 4-vCPU Xeon @ 2.80 GHz with
AVX-512 VNNI (this CCotw container), int8 dynamic-quantized exports, **masked-LM
head cut** so the graph ends at the final norm (`cut_heads.py`). `ladder.py`
reproduces it; raw rows in `ladder_t4.json` / `ladder_t1.json`.

| model | params | int8 size | 128 tok, 4 thr | 512 tok, 4 thr | 128 tok, 1 thr | 512 tok, 1 thr |
|---|---|---|---|---|---|---|
| ettin-encoder-17m | 17M | 21 MB | **6.0 ms** | 26.6 ms | 10.0 ms | 76.7 ms |
| ettin-encoder-32m | 32M | 36 MB | **12.2 ms** | 63.2 ms | 24.1 ms | 167.9 ms |
| mmBERT-small | 140M (42M non-emb) | 142 MB | 29.4 ms | 147.1 ms | 59.2 ms | 441.5 ms |
| ModernBERT-base | 149M | 150 MB | 45.6 ms | 311.8 ms | 130.0 ms | 938.4 ms |
| NeoBERT | 250M | 223 MB | 111.3 ms | 406.9 ms | 204.6 ms | 1105.8 ms |
| ModernBERT-large | 395M | 397 MB | 134.7 ms | 661.9 ms | 334.0 ms | 2001.1 ms |

Sources: `answerdotai/ModernBERT-{base,large}` `onnx/model_int8.onnx` (official);
`onnx-community/{ettin-encoder-17m,ettin-encoder-32m,mmBERT-small,NeoBERT}-ONNX`.

Read of the ladder for the three use cases:

- A **router** sits in front of every call and sees short prompts: ettin-32m at
  12 ms / 128 tokens on four threads is a rounding error against any model
  call, and 36 MB fits in a Worker or a browser tab. Start there; go to 68m or
  150m only if the measured accuracy says so.
- A **compliance gate** reads whole messages: ModernBERT-base or Ettin-150m at
  300 ms / 512 tokens on 4 vCPU is fine for a server-side gate and too slow for
  a per-keystroke one. Ettin-68m (87.2 GLUE, no ONNX yet, export ourselves) is
  the likely knee.
- A **paper filter** runs in batch over abstracts, where latency is irrelevant
  and accuracy is everything: Ettin-150m or 400m, and the question is the eval
  set, not the model.

### Recommended first experiment

1. Pick the paper filter (batch, recall-at-precision, we already have the feed
   and the lens: memory df6f5513). Label 1,500 recent arXiv abstracts with
   Fable 5.1 against a written relevance rubric; human-check 150.
2. Arms: gliclass-modern-base zero-shot; NLI cross-encoder zero-shot; fine-tuned
   ettin-32m / 68m / 150m and ModernBERT-base; a frozen-embedding logistic probe
   (bekko or gte-small) as the cheap floor — the `nl2sh-dense` rule: fit the
   linear probe on frozen features before pricing a fine-tune.
3. Report recall at 95% precision, coverage at a 5% error budget, ECE after
   temperature scaling, and CPU ms per abstract. Seen/unseen topic split.
4. Ship the smallest arm within 2 points of the best as int8 ONNX.

## Limits of this survey

- No training run. The accuracy numbers above are the projects' own or the
  Ettin paper's; only latency was measured here.
- No re-run of any clone's benchmark. Verdict's 48% vs 91% is quoted, not
  reproduced; TypeSafe's 337-case set was not fetched.
- Did not confirm how TypeSafe's 90.80% "cases" figure relates to the 67.8%
  workflow average on its own site; they are different scorings of the same
  four workflows and both are TypeSafe's numbers.
- RouteLLM was not re-examined; the routing prior-art line comes from training
  memory and was not fetched this session.

## Sources

Jev: [TypeSafe launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[evals.typesafe.ai](https://evals.typesafe.ai/),
[Archer Hume's probe](https://archerhume.com/posts/jevs-architecture-unmasked/).
Clones: [Latent.Space roundup](https://www.latent.space/p/ainews-here-are-6-clones-of-jev-in),
[lilting.ch comparison](https://lilting.ch/en/articles/jev-clones-architecture-comparison),
[cobanov/awesome-jev](https://github.com/cobanov/awesome-jev),
[intikhab49](https://github.com/intikhab49/open-jev-typed-decision-engine),
[Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev),
[open-jev-deberta-v3-large](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large),
[nanodiff-350m](https://huggingface.co/pngwn/nanodiff-350m-typed-decisions),
[jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks),
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).
Platforms: [Ettin paper](https://arxiv.org/abs/2507.11412),
[ettin-encoder-150m card](https://huggingface.co/jhu-clsp/ettin-encoder-150m),
[mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small),
[GLiClass modern-base](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0),
[BTZSC](https://arxiv.org/abs/2603.11991),
[ModernBERT paper](https://arxiv.org/abs/2412.13663).
