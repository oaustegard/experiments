# dettin-68m: plan (written 2026-11-10, before any training run)

"dEttin-17M-Q" is the name for what the Ettin-17m experiments built: a stock Ettin encoder
fine-tuned as a typed-decision cross-encoder (`browser-decider`), packed with a 4-bit token table and
int8 matmul weights (68 MB fp32 to about 10 MB), with LoRA rank-8 "spikes" that swap per task
(`adapter-library`). Oskar's two asks, 2026-11-10:

1. Does that recipe carry to Ettin-68m, where the GLUE curve bends (17M 79.2, 32M 83.5, 68M 87.2,
   150M 88.9)?
2. Fine-tune on life-sciences text at the length of an article section, as an embedder and as a
   decision model (a rater and a classifier).

The target is Python on a computer, with a GPU where there is one and still usable on CPU. No browser.

## Part A: the recipe at 68M (Typed Decisions, the 17M protocol)

Same data, layout, loss and scorer as `browser-decider` (`run/train_enc.py`, 10 epochs, batch 16),
rerun for 17M, 32M and 68M under one code path with bf16 autocast, so the sizes are compared on equal
terms. The 17M rerun is the control against the published 0.740 (first run) and 0.7315 (rerun).

- A0. Learning rate per size chosen on validation KL, seed 0, from {3e-5, 5e-5, 8e-5} (17M keeps 8e-5).
- A1. Specialist, 3 seeds per size. Accuracy against gold, KL, ECE.
- A2. Leave-one-workflow-out, 68M, seed 0, four runs. 17M scored 0.353 against a 0.470 prior.
- A3. Packed builds per size from seed 0: fp32, t4-w8L63, t4-w8L31. Accuracy, raw and brotli bytes.
- A4. LoRA r8 spike per workflow against a full fine-tune per workflow, 68M, 3 seeds
  (`adapter-library/run/lora.py` S arms). Spike bytes.
- A5. Speed of one decision (about 1,000 tokens) and one 512-token section: 4 vCPU (PyTorch fp32,
  onnxruntime fp32 and dynamic int8) and Colab GPU (bf16, batch 1 and batch 32).

## Part B: life-science sections, labelled by delegation

**Corpus.** Europe PMC open-access research articles with a CC BY or CC0 licence, sampled by random
publication date 2019 to mid-2025 (so not selected on molecular terms, which skewed `lifesci-ettin`).
Body sections of 120 to 700 words; longer ones split at paragraphs. Split by article: 88/4/8.

**Decision suite** (Typed Decisions question shapes; the state is the section text without its heading):

| spike | question | type | labels from |
|---|---|---|---|
| classify | `role`: introduction, methods, results, discussion or conclusion, other | choice | the article's own headings |
| classify | `study_type`: in vitro, animal, human interventional, human observational, computational, review, other | choice | teacher |
| classify | `field`: nine broad fields | choice | teacher |
| classify | `immunoassay`, `human_samples`, `quantitative_result` | noul | teacher |
| rate | `detail`: how reproducible the description is, 0 to 3 | score | teacher |
| rate | `evidence`: strength of support for the section's main claim, 0 to 3 | score | teacher |
| relevance | `relevance` of the section to a query (state holds both), 0 to 3 | score | teacher |

**Who labels what.**

| tier | job | why this tier |
|---|---|---|
| Jev (jev-1.14.0, direct API, $0.042 per million input tokens) | soft labels for every teacher question on the training sections; relevance ratings of query and candidate-section pairs | returns a probability per option, costs about $1 per 25K calls |
| Haiku 5.5, headless `claude -p`, effort low, 12 to 16 sections per child | one search query per training section | generative, high volume; batching because each `claude -p` call carries about 1,370 tokens of its own (haiku-swarm-poc) |
| Sonnet 5.5, headless | test-set queries (a different writer from the training queries); gold judge A | |
| Opus 5.5, headless | gold judge B | |
| Sonnet subagents | corpus script; fresh-eyes read of the writeup | |
| Haiku swarm (`evolving-with-swarms`) | rewording a question's rubric so Jev agrees with gold. Runs only if Jev is under 0.75 on a question after the first gold check; scorer = Jev accuracy on half the gold set, reported on the other half; $3 cap | the one place here with a cheap automatic scorer and a text artefact to improve |

**Gold.** 400 test sections and 800 query and section pairs, labelled independently by Sonnet 5.5 and
Opus 5.5. Headline accuracy is on items where the two agree; agreement rate is reported per question.

Gates:
- G1. A question with Sonnet and Opus agreement under 0.6 is reworded once, then dropped.
- G2. Jev labels the training set unless its accuracy on gold is more than 0.05 below Haiku's
  (Haiku labels 300 gold sections as the comparison). Then Haiku labels it.

## Part C: training on the sections

**Decision model.** `train_enc.py --data`, max length 1,024 tokens, soft cross-entropy on the teacher's
distributions. Arms: one full fine-tune over all questions at 17M, 32M and 68M (3 seeds); at 68M,
three LoRA spikes (classify, rate, relevance) on the frozen stock base. Scored against gold beside the
teacher itself, Haiku, and the most-common-label prior.

**Embedder.** Mean pooling and a linear head to 768-d, from the stock encoder, documents up to 1,024
tokens. One recipe for every size: in-batch contrastive loss on (query, section) pairs with two
same-article sections and retrieved sections as hard negatives, plus a cosine pull of every query and
section vector toward a frozen long-context teacher (`Alibaba-NLP/gte-modernbert-base`, 149M). The
pull is the round 5 anchor from `lifesci-ettin`, with a strong general teacher in e17c's place.
No query is a title: titles as queries are what broke TREC-COVID there.
- C1. Joint recipe at 17M, 32M, 68M.
- C2. At 68M only: teacher pull alone, and contrastive alone.
- Evals: held-out section retrieval (Sonnet-written queries against every dev and test section plus
  20K training-article sections as distractors; nDCG@10, recall@10, and rank within the source
  article); BEIR SciFact, NFCorpus, TREC-COVID. Baselines on the same path: BM25,
  gte-modernbert-base, e17c, MedCPT, `capemox/ettin-encoder-68m-contrastive-2.5m`.
- G3. If the 68M joint model is more than 0.02 nDCG@10 below the teacher on held-out sections, the
  size sweep stops and the recipe is diagnosed first.

**Packed runtime.** `run/dettin.py`: loads a packed build into PyTorch on GPU or CPU, `decide(state,
questions)`, `embed(texts)`, `load_spike(path)`. Accuracy of packed against fp32 for both models.

## Predictions

| | prediction |
|---|---|
| P1 | Typed Decisions specialist: 17M 0.725 to 0.745 (replication), 32M 0.74 to 0.76, 68M 0.755 to 0.775 |
| P2 | Leave-one-workflow-out at 68M stays below the 0.470 prior (0.36 to 0.44) |
| P3 | 68M t4-w8L63 within 0.005 of fp32; weights file 55 to 58 MB raw |
| P4 | 68M LoRA spike within 0.01 of the per-workflow full fine-tune |
| P5 | Life-science decisions, accuracy on gold: 68M within 0.03 of Jev; 68M minus 17M +0.02 to +0.05; the relevance question shows the largest size gap |
| P6 | Section retrieval nDCG@10: 68M joint beats the teacher by at least 0.03; 68M minus 17M at least +0.04; the 32M to 68M gain is at least as large as the 17M to 32M gain (no elbow before 68M) |
| P7 | BEIR: 68M joint within 0.03 of the teacher on SciFact and NFCorpus, and no more than 0.05 below it on TREC-COVID |
| P8 | 512-token section on 4 vCPU: 68M fp32 150 to 300 ms; a Colab L4 embeds more than 300 sections per second |

## Prior art checked (four searches, 2026-11-10)

- `capemox/ettin-encoder-68m-contrastive-2.5m`: a general 68M Ettin embedder (256-token training).
  Used as a baseline.
- BiCA (arXiv 2511.08029, 33M) and MedEmbed: biomedical embedders trained on generated queries, from
  abstracts and clinical notes. BMRetriever starts at 410M. Did not find a small embedder trained on
  generated queries over full-text sections.
- GraLC-RAG (arXiv 2603.22633): 2,033 cross-section questions over 2,359 PMC articles. A candidate
  external eval if its data is released; not checked yet.
- Study-type and evidence labelling exists for abstracts (CIViC labels, SURUS). Did not find
  rubric-labelled full-text sections distilled into an encoder.

## Budget

Colab: up to 60 of 392 compute units. Jev: under $10. Headless Claude children: capped per run with
`--max-usd`, and the five-hour and seven-day usage windows are read from the first wave before the
bulk run.
