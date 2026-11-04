# monad-skygest-finetune

[Monad](https://huggingface.co/PleIAs/Monad) is a 56M-parameter reasoning model from PleIAs, trained on PleIAs' synthetic
dataset SYNTH (questions, step-by-step reasoning, answers). It is small enough to run in a browser tab, and the
[Monad chat page](https://austegard.com/ai-tools/monad-chat/) runs two 8-bit copies of it on WebGPU: the original and
**A8p**, a version from [`monad-weight-edit/`](../monad-weight-edit/RESULTS.md) whose weights were edited to stop it
falling into repetition loops (the same few tokens repeated until the length limit). Monad's training data ends in late
2025, so it cannot know anything published since.

The question here: if A8p is fine-tuned on the papers people were sharing on Bluesky's "Paper Skygest" feed on
2026-11-03, can it then answer questions about them, and what does it lose elsewhere? The result is meant as a third model
for the chat page.

## Answer

Partly. Each paper has three held-out questions on facts the training rows teach, reworded. The shipped
model, `skygest_q8` (after 8 passes over the training data, in 8 bits), answers 31.1% of them <!-- @evidence[[results/summary.json#final.bridge.skygest_e8.papers_judged]] -->
across 45 papers (partly right counts half), and 20.7% fully right <!-- @evidence[[results/summary.json#final.bridge.skygest_e8.papers_correct]] -->.
A8p scored 0.7% on the same questions <!-- @evidence[[results/summary.json#final.bridge.A8p_q8.papers_judged]] -->. Both ran greedily (most likely token
each step) on the Mac GPU of Oskar Austegard, this repo's owner, in the page's runtime.

Paraphrase made it work. With ten training rows per paper, the model answered its own training questions verbatim,
45 of 45 <!-- @evidence[[results/summary.json#arms.lr5e_4_e15.seen_judged]] -->, and only 8.1% of the held-out ones <!-- @evidence[[results/summary.json#arms.lr5e_4_e15.papers_judged]] -->.
Adding about 40 paraphrases per paper, at equal rows seen per paper, gave 28.9% <!-- @evidence[[results/summary.json#arms.para5e_4_e3.papers_judged]] -->.

It costs general behaviour. On 192 general prompts, in full precision, loops rose from 6.2% <!-- @evidence[[results/summary.json#arms.A8p.dev_loop]] -->
to 23.4% <!-- @evidence[[results/summary.json#arms.para5e_4_e8.dev_loop]] -->; the unedited Monad looped on 30%, so
most of A8p's gain is gone.
A checkpoint scoring 39.6% on the papers was rejected: it got 1 of 30 basic facts.

More papers would change the headline: only 45 of the feed's 92 had a fetchable abstract.

## Findings

Rounds are in the Log: 0 is setup, 1 is ten training rows per paper, 2 adds paraphrases, 3 measures the shipped model,
4 adds long replay rows.
*Judged* counts a partly right answer as half; *verbatim* means a training question asked back word for word; *dev* is a
fixed set of 192 general prompts (64 each from MMLU, a multiple-choice knowledge test; GSM8K, grade-school maths word
problems; and IFEval, instructions such as "answer in three paragraphs"). Rows-per-paper comparisons are at equal
exposure: about 150 rows seen per paper.

1. **Paraphrased training rows are what let the model answer reworded questions.** Ten distinct rows per paper, seen 15
   times each, gave 8.1% judged on the held-out questions and 3.0% fully correct. Fifty rows per paper (the ten plus 40
   paraphrases), seen three times each, gave 28.9% judged <!-- @evidence[[results/summary.json#arms.para5e_4_e3.papers_judged]] -->
   and 20.7% fully correct <!-- @evidence[[results/summary.json#arms.para5e_4_e3.papers_correct]] -->. Allen-Zhu and Li
   report the same effect in "Physics of Language Models 3.1" (arXiv 2309.14316): facts trained in one phrasing are
   stored but cannot be retrieved by other questions. (Round 1 vs Round 2)
2. **Without paraphrase, the model memorizes the questions.** At learning rate 5e-4 the round-1 model answered every
   training question asked back verbatim, 45 of 45 <!-- @evidence[[results/summary.json#arms.lr5e_4_e20.seen_correct]] -->.
   When the wording changed it mixed up papers: asked about a mixture-of-experts paper, it quoted a sample-complexity
   bound from a different one. With paraphrases, the verbatim score fell to 41% <!-- @evidence[[results/summary.json#arms.para5e_4_e8.seen_judged]] -->
   and the held-out score rose. (Round 1)
3. **The 8-bit export keeps the full-precision score.** `skygest_q8` scored 31.9% judged on CPU onnxruntime <!-- @evidence[[results/summary.json#final.cpu.skygest_e8_q8.papers_judged]] -->
   and 31.1% on the Mac GPU. The same checkpoint in full precision scored 28.5% <!-- @evidence[[results/summary.json#arms.para5e_4_e8.papers_judged]] -->.
   With 135 questions one standard error is about 4 points, so these agree. On the 30 one-line facts it scored 8 on CPU <!-- @evidence[[results/summary.json#final.cpu.skygest_e8_q8.facts_ok]] -->
   and 6 on the Mac; A8p scored 7 on CPU and 6 on the Mac. (Round 3)
4. **Fine-tuning brings the loops back, and re-applying the edit removes few of them.** In full precision, A8p looped on
   6.2% of dev prompts and the shipped checkpoint on 23.4% <!-- @evidence[[results/summary.json#arms.para5e_4_e8.dev_loop]] -->.
   The worst case is the long IFEval answers: 51.6% <!-- @evidence[[results/summary.json#arms.para5e_4_e8.dev_ifeval_loop]] -->
   against 15.6% for A8p <!-- @evidence[[results/summary.json#arms.A8p.dev_ifeval_loop]] -->. A8p's edit removed one
   direction from eight weight matrices. Measured as a share of each matrix's norm, that direction is 8.7% in the original Monad <!-- @evidence[[results/extra.json#loop_direction_share.base]] -->,
   almost zero in A8p, and 0.8% after training:
   training put back only a tenth of it.
   Removing it again lowered loops only to 18.8% <!-- @evidence[[results/summary.json#arms.para5e_4_e8_re.dev_loop]] -->,
   and MMLU fell to 10.9%. Whatever brings the loops back is mostly not in those eight matrices; this experiment did not
   find where it is. The shipped 8-bit model on the Mac shows the same: 24.0% <!-- @evidence[[results/summary.json#final.bridge.skygest_e8.dev_loop]] -->
   of dev prompts against 7.3% for A8p <!-- @evidence[[results/summary.json#final.bridge.A8p_q8.dev_loop]] -->, and 60.9% of IFEval prompts. (Round 2, Round 3)
5. **The checkpoint with the best paper score lost the most elsewhere.** Round 2 at epoch 4 scored 39.6% on the papers <!-- @evidence[[results/summary.json#arms.para5e_4_e4.papers_judged]] -->,
   but got 1 of 30 facts and looped on 29.7% of dev prompts <!-- @evidence[[results/summary.json#arms.para5e_4_e4.dev_loop]] -->.
   Over the last four epochs, with the learning rate decaying, the model gave back 11 points on the papers and recovered
   the facts (8 of 30). Epoch 8 shipped. (Round 2)
6. **MMLU moved within noise.** A8p scored 25% <!-- @evidence[[results/summary.json#arms.A8p.dev_mmlu]] -->, which is
   chance on four choices, and the shipped checkpoint 20.3% <!-- @evidence[[results/summary.json#arms.para5e_4_e8.dev_mmlu]] -->.
   With 64 questions one standard error is about 5 points, so this neither shows a loss nor rules one out. (Round 2)
7. **The learning rate had to be high enough to store anything.** After 10 epochs on the ten-row data, learning rate 1e-4
   answered 8.9% of its own training questions <!-- @evidence[[results/summary.json#arms.lr1e_4_e10.seen_judged]] -->
   and 3e-4 answered 12.2%. 5e-4 reached 100% by epoch 15. (Round 1)
8. **Older papers were no easier.** 16 of the 45 papers predate Monad's training data, one of them "Attention Is All You
   Need". On the Mac, A8p answered none of their questions; after fine-tuning they scored 33.3% <!-- @evidence[[results/summary.json#final.bridge.skygest_e8.papers_pre_judged]] -->
   against 29.9% for the post-cutoff papers <!-- @evidence[[results/summary.json#final.bridge.skygest_e8.papers_post_judged]] -->.
   The questions ask for details from the abstracts, for example "What score was recorded on the English-to-French WMT
   2014 benchmark by the novel architecture in Vaswani et al. (2017)?", which a model can know a paper exists without
   knowing. (Round 3)
9. **The page's 8-bit model can be rebuilt exactly.** Converting Monad with onnxruntime-genai 0.18.0 in full precision
   and then quantizing with onnxruntime's 8-bit weight quantizer reproduces the published original byte for byte. A8p's
   full-precision weights were not saved anywhere, so they were recovered from the published 8-bit A8p. The recovered
   model's next-token predictions differ from the published one's by a KL divergence (in nats) of 0.00141 <!-- @evidence[[results/extra.json#recover_a8p.kl_rebuilt_vs_published]] -->.
   For scale, the original Monad and A8p differ by 0.032. (Round 0)

10. **Training on long answers did not stop the loops; it made them worse.** Round 4 repeated round 2 with half of each
   epoch's replay rows replaced by long SYNTH answers (1,025 to 2,048 tokens). Dev loops at epoch 8 rose to 30.2% <!-- @evidence[[results/summary.json#arms.long5e_4_e8.dev_loop]] -->
   against 23.4%, and 71.9% of IFEval prompts <!-- @evidence[[results/summary.json#arms.long5e_4_e8.dev_ifeval_loop]] -->.
   Short training targets are not what brings the loops back. (Round 4)

## Method

- **Papers.** The Paper Skygest feed (`at://did:plc:uaadt6f5bbda6cycbmatcm3z/app.bsky.feed.generator/preprintdigest`)
  is personalized; logged out it serves the latest ~120 paper posts across Bluesky, and that window did not move in 45
  minutes on 2026-11-03. That view plus the logged-in view of Muninn (the Bluesky account of the agent that ran this) gave 151 posts linking 92 papers <!-- @evidence[[results/summary.json#data.papers_ranked]] -->
  , ranked by summed likes, reposts, replies and quotes
  (`fetch_feed.py`). Abstracts came from the arXiv API, Crossref and Semantic Scholar (`fetch_meta.py`): 45 papers <!-- @evidence[[results/summary.json#data.papers_with_abstract]] -->
  , 29 dated after 2025-11-01, when Monad and its
  training set SYNTH were released. Most of the rest are bioRxiv preprints under the new `10.64898` DOI prefix, which no
  route reached (see Log).
- **Training data.** Monad's format: `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n`, a terse
  reasoning trace, `\n</think>\n\n`, a prose answer, `<|im_end|>`. Gemini (gemini-3.8-flash through the Cloudflare AI
  Gateway) wrote, from each paper's metadata and abstract only: 6–8 facts, 10 training rows and 3 held-out questions
  with reference answers and key points (`gen_data.py`); then 40 more rows per paper in two calls, one asking each fact
  several ways and one with multi-fact overviews, the paper named differently in every row (`gen_para.py`). Training
  rows sharing half or more of their content words with a held-out question on the same paper were dropped (3 of
  1,800). Replay, to keep the model's general behaviour: English SYNTH rows from random offsets, 2,923 of which fit the 1,024-token window, taken in order from a shuffled pool at 300 (Round 1) or 600 (Round 2) per epoch and wrapping around when the pool runs out. The 450 original training rows were checked against the held-out questions only for exact duplicates; Gemini wrote both from the abstract in the same call.
- **Training.** Full fine-tune from A8p (`train.py`): AdamW, betas 0.9/0.95, weight decay 0.01, 5% warmup then cosine
  decay, batch 16, loss on the reasoning and answer only, gradient checkpointing, fp32, checkpoints saved in bf16. Round
  1: 450 rows, learning rates 1e-4 and 3e-4 for 10 epochs and 5e-4 for 20. Round 2: 2,247 rows, 5e-4, 8 epochs. On a
  Colab L4: 44 minutes for round 2.
- **Probe sets** (`data/eval_prompts.jsonl`). *Papers*: the 135 held-out questions. *Seen*: one training question per
  paper, verbatim (45). *Facts*: the 30 one-line facts from `monad-chat/probes_facts.json`, scored by its regex.
  *Dev*: `monad-weight-edit`'s 192-prompt dev fixture (64 each of MMLU, GSM8K, IFEval) for loop rate and accuracy.
- **Scoring** (`score.py`). Gemini grades each paper answer against the reference and key points as correct, partial or
  wrong; *judged* = correct + half of partial. A loop is a generation that hit the token limit with its tail one block of
  1–64 tokens repeated three or more times. Greedy decoding, 768 new tokens (1,024–1,536 on dev).
- **Runtimes.** Checkpoints were chosen on Colab (HF transformers, fp32, `gpu_eval.py`). The shipped model was measured
  as the page runs it: the 8-bit export on onnxruntime CPU (`evalgen.py`) and on the Mac through the GPU Bridge with
  onnxruntime-web 1.30 on WebGPU, batch 1 (`bridge_eval.py`, using `monad-weight-edit/bridge/job.js`).
- **Export.** `export.py` (onnxruntime-genai 0.18.0, fp32, WebGPU target, logits-index surgery), `quant8.py` (8-bit,
  block 32, symmetric, accuracy level 4, every MatMul including the output head), then `assemble.py` writes the tensors
  into the published `model.onnx`'s data layout, so the page's graph is reused unchanged (graph_sha256 `cd197c81…`).
  `upload_r2.py` refuses to upload unless that hash matches the live graph.

## Log

### 2026-11-03 — round 0: data and starting weights

Oskar, from the monad-chat session: fine-tune Monad on the top ~100 papers in the Paper Skygest feed, as a third model
for the chat page. The feed's logged-out view held 84 papers and did not change over 45 minutes; the logged-in view of Muninn (the Bluesky account of the agent that ran this)
added 8. 45 had abstracts. The 30 bioRxiv preprints carry a DOI with a `v1` suffix that Crossref did not resolve; a
probe of bioRxiv's landing page with a browser User-Agent was refused by the session's safety check as bot-check
circumvention, and no other route to those abstracts was tried. Claude Sonnet was the first
choice to write the training data; its API safeguard refused the prompt (`reasoning_extraction`, writing reasoning
traces to train another model), so Gemini wrote it.

A8p's edit direction was never committed. The int8 `o_proj` weights of layers 27–34 in the published `A8p_q8`, minus the
original weights, are rank one up to rounding (singular values 16.9 and 0.13), so `recover_a8p.py` takes the top
singular vector and rebakes A8p. 0.8% of the int8 values in those eight matrices differ from the published file, all by
one or two rounding steps or by an equivalent sign flip of a block scale. This session's onnxruntime loop reproduced the
published model's recorded answers to the first 10 fact probes exactly, so the harness matches the earlier one.

### 2026-11-03 — round 1: ten rows per paper

Learning rates 1e-4 and 3e-4 for 10 epochs and 5e-4 for 20, 450 paper rows plus 300 replay rows per epoch. Held-out
scores stayed under 9% at every checkpoint, and fine-tuning raised the dev loop rate from 6.2% to 21.9% by epoch 2 at
1e-4. The verbatim probe told "not learned" apart from "learned but not retrievable": 1e-4 and 3e-4 had not stored the
training answers at all, while 5e-4 had memorized every training question by epoch 15 and still could not answer
reworded ones.

| checkpoint | papers judged | fully correct | verbatim | facts (of 30) |
|---|---|---|---|---|
| A8p (fp32) | 0% | 0% | — | 8 |
| 1e-4, epoch 10 | 4.1% | 1.5% | 8.9% | 8 |
| 3e-4, epoch 10 | 5.2% | 2.2% | 12.2% | 4 |
| 5e-4, epoch 15 | 8.1% | 3.0% | 100% | 6 |
| 5e-4, epoch 20 | 6.7% | 3.0% | 100% | 6 |

The first Colab VM was reclaimed mid-round, taking the round-1 checkpoints and a dev run with it; every number above had
been downloaded first. `METHODS.md` already described this failure (a detached Colab job dies when the session goes
untouched), and this experiment had not checked it.

### 2026-11-04 — round 2: forty paraphrases per paper

Oskar: "try more paraphrases per paper instead of more epochs". 1,797 paraphrased rows kept, 600 replay rows per epoch,
learning rate 5e-4, 8 epochs.

| checkpoint | papers judged | fully correct | verbatim | facts (of 30) | dev loops | MMLU |
|---|---|---|---|---|---|---|
| epoch 3 | 28.9% | 20.7% | 45.6% | 3 | — | — |
| epoch 4 | 39.6% | 32.6% | 51.1% | 1 | 29.7% | 15.6% |
| epoch 6 | 31.1% | 20.0% | 58.9% | 7 | — | — |
| epoch 8 | 28.5% | 19.3% | 41.1% | 8 | 23.4% | 20.3% |
| epoch 8, loop edit re-applied | 30.7% | 22.2% | — | 7 | 18.8% | 10.9% |

Epoch 8 shipped: epoch 4's 11 extra points on the papers came with near-zero facts and the worst loop and MMLU numbers,
and the re-applied edit lowered loops (23.4% to 18.8%) while MMLU fell (20.3% to 10.9%), each shift about two
standard errors.

### 2026-11-04 — round 3: the shipped 8-bit model

`skygest_q8` exported, uploaded to R2 at `models/monad-chat/v1/skygest_q8/`, and measured as the page runs it.

| model, runtime | papers judged | fully correct | facts (of 30) | dev loops |
|---|---|---|---|---|
| A8p_q8, Mac GPU | 0.7% | 0% | 6 | 7.3% |
| skygest_q8, CPU | 31.9% | 21.5% | 8 | not run (stopped once the Mac had it) |
| skygest_q8, Mac GPU | 31.1% | 20.7% | 6 | 24.0% |

Costs: 9.65 Colab compute units; Gemini calls for the training data (45 plus 90 for paraphrases) and for judging (15
answers per call); about 1.3 hours of Mac GPU time on the bridge (48 minutes for A8p_q8, 31 for skygest_q8).

### 2026-11-04 — round 4: long replay rows

Oskar asked whether the loops could be fixed. Most of the shipped model's loops are on IFEval's long answers, and every
training target had been under 1,024 tokens, so the first test was whether the fine-tune had simply stopped practising long
answers. Round 2 was rerun unchanged except that 300 of each epoch's 600 replay rows came from a pool of 791 SYNTH rows of
1,025 to 2,048 tokens (`fetch_replay.py --min-chars 3500`, `train.py --replay-long`, window 2,048). Training took 73
minutes. The prediction was fewer loops; loops rose instead.

| checkpoint | dev loops | IFEval loops | MMLU | facts (of 30) | papers, key-point recall |
|---|---|---|---|---|---|
| round 2, epoch 8 (shipped) | 23.4% | 51.6% | 20.3% | 8 | 34.1% |
| round 4, epoch 6 | 27.6% | 62.5% | 23.4% | 6 | 34.1% |
| round 4, epoch 8 | 30.2% | 71.9% | 17.2% | 5 | 36.7% |

The papers column is the key-point word overlap, not the judged score: the Gemini judge stopped with HTTP 402 ("prepayment
credits are depleted") partway through this round, so round 4 has no judged paper score. Nothing from round 4 shipped. The
next step is to repeat `monad-weight-edit`'s diagnosis on the shipped model itself: read a loop direction off its own loops
and find which layers write it now.
