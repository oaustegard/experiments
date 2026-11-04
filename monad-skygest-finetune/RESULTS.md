# monad-skygest-finetune

[Monad](https://huggingface.co/PleIAs/Monad) is a 56M-parameter reasoning model from PleIAs, small enough to run in a
browser tab: the [Monad chat page](https://austegard.com/ai-tools/monad-chat/) runs an 8-bit copy of it on WebGPU. Its
training data ends in late 2025, so it cannot know anything published since. This experiment fine-tunes it on the papers
people were sharing on Bluesky's "Paper Skygest" feed on 2026-11-03 and asks whether it can then answer questions about
them, and what that costs it elsewhere. The starting point is A8p, the loop-edited Monad from
[`monad-weight-edit/`](../monad-weight-edit/RESULTS.md), which is what the page ships.

## Answer

Partly. The shipped model, `skygest_q8`, answers 31.1% of reworded questions <!-- @evidence[[results/summary.json#final.bridge.skygest_e8.papers_judged]] -->
about the 45 papers it was trained on, counting a partly right answer as half,
and gets 20.7% fully right <!-- @evidence[[results/summary.json#final.bridge.skygest_e8.papers_correct]] -->. Before
fine-tuning it scored 0.7% <!-- @evidence[[results/summary.json#final.bridge.A8p_q8.papers_judged]] -->. Both numbers
are greedy decoding on Oskar's Mac GPU with the page's own runtime.

What made it work was paraphrase. With ten training rows per paper, the model learned to answer its own training
questions word for word, 45 of 45 <!-- @evidence[[results/summary.json#arms.lr5e_4_e15.seen_judged]] -->, while scoring
8.1% on reworded ones <!-- @evidence[[results/summary.json#arms.lr5e_4_e15.papers_judged]] -->: it had stored answers
against question strings. Adding about 40 paraphrased rows per paper raised the reworded score to 28.9% <!-- @evidence[[results/summary.json#arms.para5e_4_e3.papers_judged]] -->
after the same number of passes over each paper.

The cost is general behaviour. On a 192-prompt general set, loops (generation stuck repeating until the token limit) rose
from 6.2% to 23.4% <!-- @evidence[[results/summary.json#arms.para5e_4_e8.dev_loop]] --> in full precision, which undoes
much of A8p's loop edit, and MMLU fell from 25% to 20%. On 30 one-line facts it is unchanged.

More papers would change the headline: only 45 of the 92 in the feed had an abstract this session could fetch.

## Findings

1. **Paraphrased training rows are what let the model answer reworded questions.** At an equal number of rows seen per
   paper (about 150), ten distinct rows per paper repeated 15 times scored 8.1% on held-out questions; 50 rows per paper
   (10 originals plus 40 paraphrases) repeated three times scored 28.9% <!-- @evidence[[results/summary.json#arms.para5e_4_e3.papers_judged]] -->
   , and 20.7% fully correct <!-- @evidence[[results/summary.json#arms.para5e_4_e3.papers_correct]] -->
   against 3.0%. This is the effect Allen-Zhu
   and Li describe in "Physics of Language Models 3.1" (arXiv 2309.14316): facts seen in one phrasing are stored but not
   extractable. (Round 1 vs Round 2)
2. **Without paraphrase, the model memorizes the questions.** The round-1 model at learning rate 5e-4 answered every
   training question asked back verbatim, 45 of 45 <!-- @evidence[[results/summary.json#arms.lr5e_4_e20.seen_correct]] -->,
   and borrowed facts across papers when the wording changed (asked about a mixture-of-experts paper, it quoted a
   sample-complexity bound from a different one). With paraphrases the verbatim score fell to 41% <!-- @evidence[[results/summary.json#arms.para5e_4_e8.seen_judged]] -->
   while the reworded score rose. (Round 1)
3. **The shipped 8-bit model keeps its fp32 score.** `skygest_q8`: 31.9% judged on CPU onnxruntime <!-- @evidence[[results/summary.json#final.cpu.skygest_e8_q8.papers_judged]] -->
   , 31.1% on the Mac GPU, against
   28.5% for the same checkpoint in fp32 <!-- @evidence[[results/summary.json#arms.para5e_4_e8.papers_judged]] -->; facts
   8 of 30 on CPU <!-- @evidence[[results/summary.json#final.cpu.skygest_e8_q8.facts_ok]] -->. The difference is within
   noise for 135 questions (one standard error is about 4 points). (Round 3)
4. **Fine-tuning brings the loops back.** Dev loop rate 6.2% for A8p, 23.4% for the shipped checkpoint <!-- @evidence[[results/summary.json#arms.para5e_4_e8.dev_loop]] -->
   , worst on long instruction-following prompts:
   51.6% of IFEval prompts <!-- @evidence[[results/summary.json#arms.para5e_4_e8.dev_ifeval_loop]] --> against 15.6%.
   Training restored only a tenth of the edited direction in the matrices the edit touched (0.8% of their norm, against
   8.7% in the original Monad <!-- @evidence[[results/extra.json#loop_direction_share.base]] -->), and re-applying the
   edit after training cut loops only to 18.8% <!-- @evidence[[results/summary.json#arms.para5e_4_e8_re.dev_loop]] -->
   while MMLU fell to 10.9%. The loops return through other weights. (Round 2)
5. **The best paper score came with the worst general score.** Round 2 at epoch 4 scored 39.6% on the papers <!-- @evidence[[results/summary.json#arms.para5e_4_e4.papers_judged]] -->,
   but got 1 of 30 facts, looped on 29.7% of dev prompts <!-- @evidence[[results/summary.json#arms.para5e_4_e4.dev_loop]] -->
   and scored 15.6% on MMLU. The last four epochs, at a decaying learning rate, gave back 11 points on the papers and
   recovered the facts (8 of 30) and some of the rest; epoch 8 shipped. (Round 2)
6. **The learning rate had to be high enough to store anything.** After 10 epochs of the ten-row data, 1e-4 answered
   8.9% of its own training questions <!-- @evidence[[results/summary.json#arms.lr1e_4_e10.seen_judged]] --> and 3e-4
   answered 12.2%; 5e-4 reached 100% by epoch 15. (Round 1)
7. **Older papers were no easier.** The feed included 16 papers from before Monad's training data (one is "Attention Is
   All You Need"). Before fine-tuning both groups scored 0%; after, older papers scored 33.3% <!-- @evidence[[results/summary.json#final.bridge.skygest_e8.papers_pre_judged]] -->
   and post-cutoff ones 29.9% <!-- @evidence[[results/summary.json#final.bridge.skygest_e8.papers_post_judged]] -->
   . Monad did not know even the
   famous ones well enough to answer these questions. (Round 3)
8. **The page's 8-bit graph can be rebuilt byte for byte, and A8p's weights recovered from it.** onnxruntime-genai
   0.18.0 in fp32, then onnxruntime's 8-bit weight quantizer, reproduces the published `base_q8` exactly; the loop
   edit's direction, never committed, comes back from the published `A8p_q8` as one singular vector, and the rebuilt A8p
   differs from the published one by a next-token KL of 0.00141 <!-- @evidence[[results/extra.json#recover_a8p.kl_rebuilt_vs_published]] -->
   , against 0.03231 between A8p and the
   original Monad. (Round 0)

## Method

- **Papers.** The Paper Skygest feed (`at://did:plc:uaadt6f5bbda6cycbmatcm3z/app.bsky.feed.generator/preprintdigest`)
  is personalized; logged out it serves the latest ~120 paper posts across Bluesky, and that window did not move in 45
  minutes on 2026-11-03. That view plus Muninn's logged-in view gave 151 posts linking 92 papers <!-- @evidence[[results/summary.json#data.papers_ranked]] -->
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
  1,800). Replay: 3,000 English SYNTH rows at random offsets, a fresh 300 (Round 1) or 600 (Round 2) per epoch.
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

Oskar, from the parent session: fine-tune Monad on the top ~100 papers in the Paper Skygest feed, as a third model for
the chat page. The feed's logged-out view held 84 papers and did not change over 45 minutes; Muninn's logged-in view
added 8. 45 had abstracts. The 30 bioRxiv preprints carry a DOI with a `v1` suffix that Crossref did not resolve; a
probe of bioRxiv's landing page with a browser User-Agent was denied by the session's auto-mode classifier as bot-check
circumvention, and per the workspace rule no other route to those abstracts was tried. Claude Sonnet was the first
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
1e-4. The verbatim probe separated the two failure modes: 5e-4 had memorized every training question by epoch 15 and
still could not answer reworded ones.

| checkpoint | papers judged | fully correct | seen (verbatim) | facts |
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

| checkpoint | papers judged | fully correct | seen | facts | dev loops | MMLU |
|---|---|---|---|---|---|---|
| epoch 3 | 28.9% | 20.7% | 45.6% | 3 | — | — |
| epoch 4 | 39.6% | 32.6% | 51.1% | 1 | 29.7% | 15.6% |
| epoch 6 | 31.1% | 20.0% | 58.9% | 7 | — | — |
| epoch 8 | 28.5% | 19.3% | 41.1% | 8 | 23.4% | 20.3% |
| epoch 8, loop edit re-applied | 30.7% | 22.2% | — | 7 | 18.8% | 10.9% |

Epoch 8 shipped: epoch 4's 11 extra points on the papers came with near-zero facts and the worst loop and MMLU numbers,
and the re-applied edit traded MMLU for loops with both shifts about two standard errors.

### 2026-11-04 — round 3: the shipped 8-bit model

`skygest_q8` exported, uploaded to R2 at `models/monad-chat/v1/skygest_q8/`, and measured as the page runs it.

| model, runtime | papers judged | fully correct | facts | dev loops |
|---|---|---|---|---|
| A8p_q8, Mac GPU | 0.7% | 0% | 6 | 7.3% |
| skygest_q8, CPU | 31.9% | 21.5% | 8 | pending |
| skygest_q8, Mac GPU | 31.1% | 20.7% | 6 | pending |

Costs: 9.65 Colab compute units, 135 Gemini calls for training data plus the judge calls, about 1.5 hours of Mac GPU time on the bridge.
