# dettin-68m: does the small decision-model recipe carry to Ettin-68m, and to life-science article sections?

Ettin is a family of open ModernBERT-style encoders from 17M to 1B parameters. Earlier experiments here
turned the 17M one into a small "decision model": it reads a piece of text and a typed question (yes/no, pick
one option, or a level on a rubric) and returns a probability for each answer. Jev, TypeSafe's hosted model, is
the commercial product of this kind; Typed Decisions is a public benchmark for it, four business workflows with
five questions per case. That recipe has three parts. The
encoder is fine-tuned as a cross-encoder (`browser-decider`). Its weights are packed to a 4-bit token table and
int8 matrices, 68 MB down to about 10. And a task can be added as a small swappable LoRA adapter (rank 8 here), called a "spike",
on a frozen base (`adapter-library`).

Two questions, asked by Oskar on 2026-11-10. Does the recipe carry to the 68M encoder, where the published GLUE
curve bends? And can the same encoders be trained on life-science text at the length of an article section, as
a decision model (a classifier and a rater) and as an embedder for search? The target is Python on a GPU
machine that also runs on CPU.

## Answer

The decision recipe carries to 68M. The section embedder works only inside its own domain.

On the public Typed Decisions benchmark a fine-tuned 68M scores 0.770 <!-- @evidence[[results/summary.json#td.68.acc_mean]] -->
against 0.736 for 17M <!-- @evidence[[results/summary.json#td.17.acc_mean]] -->, and the packed 58.3 MB <!-- @evidence[[results/summary.json#td.68.packed_mb]] -->
file scores the same as full precision. Spikes on the frozen stock 68M come within 0.004 of full fine-tunes.

On nine life-science questions, scored on test items where two stronger models agree, spikes on the stock 68M
reach 0.892 <!-- @evidence[[results/summary.json#ls.spikes68]] --> and a full fine-tune 0.875 <!-- @evidence[[results/summary.json#ls.68.macro]] -->;
17M reaches 0.856 <!-- @evidence[[results/summary.json#ls.17.macro]] -->. Neither the 17M nor the 68M transfers to tasks it was
not trained on.

The 68M embedder ties the 149M model it was distilled from on reworded section queries, 0.733 <!-- @evidence[[results/summary.json#emb.e68-joint.para]] -->
against 0.730 <!-- @evidence[[results/summary.json#emb.base-gte.para]] --> nDCG@10, but scores 0.612 <!-- @evidence[[results/summary.json#emb.e68-joint.scifact]] -->
on SciFact where a 33M general embedder scores 0.713 <!-- @evidence[[results/summary.json#emb.base-bge-small.scifact]] -->.
A longer distillation on broader text, which was not run, is what could change that.

## Findings

1. **Typed Decisions accuracy rises with size at every step.** Three seeds each: 17M 0.736 <!-- @evidence[[results/summary.json#td.17.acc_mean]] -->,
   32M 0.751 <!-- @evidence[[results/summary.json#td.32.acc_mean]] -->, 68M 0.770 <!-- @evidence[[results/summary.json#td.68.acc_mean]] -->.
   The second step, +0.019 <!-- @evidence[[results/summary.json#td.step_32_68]] -->, is larger than the first, +0.015 <!-- @evidence[[results/summary.json#td.step_17_32]] -->,
   and the spread across seeds shrinks from 0.009 <!-- @evidence[[results/summary.json#td.17.acc_sd]] --> to 0.001 <!-- @evidence[[results/summary.json#td.68.acc_sd]] -->
   (`results/a/td/`).
2. **Packing costs nothing measurable at 68M.** The 4-bit table and int8 matrices take the 68M from 273.6 MB <!-- @evidence[[results/summary.json#td.68.fp32_mb]] -->
   to 58.3 MB <!-- @evidence[[results/summary.json#td.68.packed_mb]] --> (46.7 MB <!-- @evidence[[results/summary.json#td.68.packed63_brotli_mb]] --> after brotli).
   The packed model scores 0.772 <!-- @evidence[[results/summary.json#td.68.packed63]] --> against 0.772 <!-- @evidence[[results/summary.json#td.68.fp32]] -->
   and picks the same answer on 0.986 <!-- @evidence[[results/summary.json#td.68.packed63_agree]] --> of questions; at 17M that agreement is 0.929 <!-- @evidence[[results/summary.json#td.17.packed63_agree]] -->.
   The ratio is 4.7x, not the 17M's 5.9x, because the token table is a smaller share of a larger model.
3. **Spikes come within 0.004 of full fine-tunes on the 68M.** On Typed Decisions a rank-8 spike with 1,236,993 <!-- @evidence[[results/summary.json#td.spike_trainable]] -->
   trainable weights averages 0.765 <!-- @evidence[[results/summary.json#td.spike_lora68]] --> against 0.769 <!-- @evidence[[results/summary.json#td.spike_full68]] -->
   for a full fine-tune per workflow (12 runs each; the spike leads on one workflow of four). On the life-science questions three spikes, one per question group, average 0.892 <!-- @evidence[[results/summary.json#ls.spikes68]] -->
   against 0.875 <!-- @evidence[[results/summary.json#ls.68.macro]] --> for one full fine-tune over all nine questions.
4. **On life-science decisions most of the size gain arrives by 32M.** Mean accuracy over nine questions, three
   seeds: 17M 0.856 <!-- @evidence[[results/summary.json#ls.17.macro]] -->, 32M 0.871 <!-- @evidence[[results/summary.json#ls.32.macro]] -->,
   68M 0.875 <!-- @evidence[[results/summary.json#ls.68.macro]] -->. The step from 17M to 32M is +0.015 <!-- @evidence[[results/summary.json#ls.steps.32_minus_17.delta]] -->
   with a 95% interval of 0.009 to 0.022; the step from 32M to 68M is +0.004 <!-- @evidence[[results/summary.json#ls.steps.68_minus_32.delta]] -->
   with an interval that includes zero (`results/decisions.json`).
5. **Haiku was the better labeller on seven of eight questions, and the students stay below it.** On eight section questions Haiku 5.5 (Anthropic's small Claude model)
   agrees with the gold labels on 0.940 <!-- @evidence[[results/summary.json#ls.haiku_macro8]] --> and Jev on 0.883 <!-- @evidence[[results/summary.json#ls.jev_macro8]] -->;
   the widest gap is the reproducibility rubric, 0.849 <!-- @evidence[[results/summary.json#ls.haiku.detail]] --> against 0.621 <!-- @evidence[[results/summary.json#ls.jev.detail]] -->.
   The fully fine-tuned 68M trained on Haiku's labels reaches 0.893 <!-- @evidence[[results/summary.json#ls.macro8.68]] --> on those eight.
   The same model trained on Jev's labels scores 0.850 <!-- @evidence[[results/summary.json#ls.jev_teacher68]] --> over all nine.
   The gold judges are Claude models, which may favour Haiku.
6. **Size does not buy transfer to unrelated tasks.** With one Typed Decisions workflow held out the 68M scores
   0.413 <!-- @evidence[[results/summary.json#td.lowo68]] -->, below the 0.470 that always picking the most common
   answer gets (17M: 0.353). On 104 unrelated decisions (card intents, guardrail yes/no questions and rubric scores, from `claude-workspace`) the 68M Typed Decisions model scores 0.298 <!-- @evidence[[results/ood_td68.json#accuracy]] -->
   and the 17M 0.279 <!-- @evidence[[results/ood_td68.json#reference_17m_10_epochs]] -->; Jev scores 0.894.
7. **The section embedder ties its teacher on reworded queries, and each size step is large.** nDCG@10 on 1,186
   reworded queries over 28,073 sections: 17M 0.526 <!-- @evidence[[results/summary.json#emb.e17-joint.para]] -->,
   32M 0.646 <!-- @evidence[[results/summary.json#emb.e32-joint.para]] -->, 68M 0.733 <!-- @evidence[[results/summary.json#emb.e68-joint.para]] -->;
   its teacher (`gte-modernbert-base`, 149M) 0.730 <!-- @evidence[[results/summary.json#emb.base-gte.para]] -->, bge-small 0.664 <!-- @evidence[[results/summary.json#emb.base-bge-small.para]] -->,
   BM25 0.500 <!-- @evidence[[results/summary.json#emb.base-bm25.para]] -->. With its encoder rounded to the packed format it scores 0.734 <!-- @evidence[[results/summary.json#emb.e68-joint-q.para]] -->.
8. **It is not a general biomedical embedder.** On SciFact the 68M scores 0.612 <!-- @evidence[[results/summary.json#emb.e68-joint.scifact]] -->
   against 0.765 <!-- @evidence[[results/summary.json#emb.base-gte.scifact]] --> for its teacher, and on NFCorpus 0.219 <!-- @evidence[[results/summary.json#emb.e68-joint.nfcorpus]] -->
   against 0.344 <!-- @evidence[[results/summary.json#emb.base-gte.nfcorpus]] -->. The teacher stage saw 41K sections for 1,960 steps and
   nothing shaped like an abstract.
9. **Contrastive training alone learns word matching.** Without the pull toward the teacher the 68M scores
   0.900 <!-- @evidence[[results/summary.json#emb.e68-contrast.orig]] --> on the original queries and 0.437 <!-- @evidence[[results/summary.json#emb.e68-contrast.para]] -->
   on the reworded ones, below BM25.
10. **Queries written by a model from the section are a word-matching test.** On the original Sonnet-written
    queries BM25 scores 0.910 <!-- @evidence[[results/summary.json#emb.base-bm25.orig]] -->, above the dense teacher's 0.874 <!-- @evidence[[results/summary.json#emb.base-gte.orig]] -->.
    Rewording them to avoid the section's own terms drops BM25 to 0.500 <!-- @evidence[[results/summary.json#emb.base-bm25.para]] -->.

## Method

**Encoders.** `jhu-clsp/ettin-encoder-17m`, `-32m` and `-68m` as published (hidden sizes 256, 384, 512; 7, 10, 19
layers). Every model here starts from those stock weights. Training used bf16 autocast on a Colab A100.

**Typed Decisions (Part A).** The public `LocalLLaMA/typed-decisions` set: four workflows, 300 training and 100
test cases each, five questions per case. Trainer, layout, loss and scorer are `browser-decider`'s
(`run/train_enc.py`), 10 epochs at batch 16. The learning rate for 32M and 68M was chosen on validation loss from
three values with seed 0; 17M kept 8e-5. Then three seeds per size, a leave-one-workflow-out run at 68M, packed
variants of each seed-0 model (`run/quant_levels.py`), and `adapter-library`'s spike and full fine-tune arms per
workflow on the 68M (`run/colab_a.py`).

**Corpus.** 6,000 Europe PMC research articles with a CC BY licence, sampled by random publication date from 2019
to mid-2025; 5,944 kept. Body sections of 120 to 700 words, longer ones split at paragraphs, headings removed
from the text: 69,006 sections, split by article into 60,933 train, 2,552 dev and 5,521 test (`run/corpus.py`).

**Decision suite** (`run/suite.py`). Eight questions about a section: which part of an article it is, study
type (7 options), field (10 options), three yes/no flags (immunoassay, human samples, a quantitative result),
and two 0 to 3 rubrics (reproducibility of the description, strength of evidence). A ninth question rates how
well a section answers a query, 0 to 3. Three spikes cover them: classify, rate, relevance.

**Labels.** Gold: 395 test sections labelled independently by Sonnet 5.5 and Opus 5.5, and 683 query and
section pairs graded by both. A model is scored on the items where the two agree (71% to 99% of items per
question); the article-part question is scored against the article's own heading. Training labels: Haiku 5.5
labelled 21,310 sections in batches of 8 through headless `claude -p`; Jev labelled the same sections and
43,857 pairs. The plan's gate gave the section labels to Haiku because it led Jev on gold by more than 0.05.
Relevance training labels are Jev's, and Jev's own accuracy on the relevance gold was not measured. Training
used 8,000 sections and 16,000 pairs (`run/build.py`).

**Decision arms** (`run/colab_b.py`). A full fine-tune over all nine questions at each size, three seeds, two
epochs, sequences up to 1,024 tokens; at 68M one run on Jev's labels, and three spikes on the frozen stock
base, two seeds. Scoring: `run/score.py`, with a paired bootstrap over items for the size steps.

**Embedder** (`run/embed.py`). Mean pooling and a bias-free linear head to 768-d. Stage one pulls every section
and training query toward the vector of a frozen teacher, `Alibaba-NLP/gte-modernbert-base` (two epochs over
41K sections and 20,812 queries). Stage two adds an in-batch contrastive loss over 20,812 Haiku-written queries,
each with its source section, another section of the same article and two BM25 hits. The pull stays on in this stage. A
"negative" that Jev rated as answering the query was dropped where a rating existed. Arms: both stages at each
size; at 68M the first stage alone and the contrastive loss alone.

**Embedder evals.** 1,186 queries written by Sonnet for dev and test sections, each scored against 28,073
sections (every dev and test section plus 20,000 others); nDCG@10 with the source section as the one relevant
item. The same queries reworded by Sonnet to avoid the section's own terms. SciFact and NFCorpus from BEIR.
Baselines through the same code: BM25, the teacher, bge-small, MedCPT, and a published Ettin-68m embedder.

**Runtime** (`run/dettin.py`). Loads a packed build into PyTorch on GPU or CPU; `decide`, `embed`,
`load_spike`, `unload_spike`. A packed stock base takes spikes; a packed embedder carries its head, folded
from 768 to the encoder's width.

| | 17M | 32M | 68M |
|---|---|---|---|
| Typed Decisions, 3 seeds | 0.736 | 0.751 | 0.770 |
| packed file, MB | 11.5 | 24.2 | 58.3 |
| life-science decisions, 9 questions | 0.856 | 0.871 | 0.875 |
| sections, reworded queries, nDCG@10 | 0.526 | 0.646 | 0.733 |
| one 512-token text, 4 CPU threads, ms | 24 | 50 | 162 |
| 512-token texts per second, A100, batch 32 | 2,963 | 2,043 | 1,005 |

**Cost.** 70.3 Colab compute units (three A100 sessions, about 10 hours in total). Headless Claude children:
about $30 at list price. Jev: 87.6M input tokens, which exhausted the account's TypeSafe credits partway through
the pair ratings.

## Log

### Round 1 (2026-11-10)

`PLAN.md` was committed before any training run. Predictions against measurements:

| | predicted | measured |
|---|---|---|
| P1 | Typed Decisions 17M 0.725–0.745, 32M 0.74–0.76, 68M 0.755–0.775 | 0.736, 0.751, 0.770: held |
| P2 | held-out workflow at 68M below 0.470 | 0.413: held |
| P3 | packed 68M within 0.005 of fp32; file 55–58 MB | 0.772 and 0.772; 58.3 MB: accuracy held, size 0.3 MB over |
| P4 | 68M spike within 0.01 of the full fine-tune | 0.765 against 0.769: held |
| P5 | life-science: 68M within 0.03 of Jev; 68M minus 17M +0.02 to +0.05; relevance the largest size gap | 0.893 against Jev's 0.883 on eight questions: held; +0.019: just under; relevance +0.041, second to study type's +0.050: missed |
| P6 | sections: 68M beats the teacher by 0.03; 68M minus 17M at least +0.04; the 32M to 68M gain at least the 17M to 32M gain | reworded queries: +0.003, missed; +0.207, held; +0.087 against +0.120, missed |
| P7 | BEIR: within 0.03 of the teacher on SciFact and NFCorpus | 0.153 and 0.125 below: missed |
| P8 | 68M, 512 tokens, 4 CPU threads: 150–300 ms; more than 300 sections per second on a Colab GPU | 162 ms; 1,005 per second on an A100 (no L4 run): held |

**Part A.** Learning rates chosen on validation: 5e-5 at 32M, 8e-5 at 68M. Held-out workflow at 68M: 0.412, 0.414,
0.404, 0.424. The packed variant of the 68M that stores int8 codes in ±31 instead of ±63 scores 0.774 at 41.3 MB after brotli. Spikes by workflow, spike
then full fine-tune: invoices 0.847 and 0.821, customer service 0.755 and 0.777, security 0.747 and 0.755, agent
traces 0.713 and 0.721. On an A100 a 68M spike for one workflow trains in one to two minutes.

**Gold check.** Sonnet and Opus agreement per question: article part 0.970, study type 0.881, field 0.884,
immunoassay 0.992, human samples 0.970, quantitative result 0.967, reproducibility 0.714, evidence 0.830,
relevance 0.811 (within one level on every pair). Items left after the agreement filter, in that order from
study type: 348, 349, 392, 383, 382, 282, 328 of 395 sections and 554 of 683 pairs; the article-part question
uses the 383 sections whose heading names one part. The immunoassay flag is "no" for 98% of sections, so accuracy
says little there; the 68M ranks the positives at AUROC 0.982. Opus graded 98 of 188 same-article sections as
partly or directly answering the query, which is why those are filtered as negatives.

**Life-science decisions, per question** (three seeds; spikes two seeds):

| question | 17M | 32M | 68M | 68M spikes | Haiku | Jev |
|---|---|---|---|---|---|---|
| article part (against the heading) | 0.904 | 0.912 | 0.919 | 0.927 | 0.923 | 0.930 |
| study type | 0.762 | 0.799 | 0.812 | 0.829 | 0.933 | 0.914 |
| field | 0.787 | 0.798 | 0.804 | 0.821 | 0.928 | 0.905 |
| immunoassay | 0.987 | 0.990 | 0.991 | 0.989 | 0.997 | 0.995 |
| human samples | 0.912 | 0.909 | 0.925 | 0.944 | 0.974 | 0.903 |
| quantitative result | 0.908 | 0.938 | 0.943 | 0.952 | 0.971 | 0.919 |
| reproducibility, 0–3 | 0.832 | 0.853 | 0.833 | 0.860 | 0.849 | 0.621 |
| evidence, 0–3 | 0.915 | 0.916 | 0.913 | 0.944 | 0.944 | 0.878 |
| relevance, 0–3 | 0.694 | 0.725 | 0.735 | 0.763 | not run | not run |

Packed against full precision on all 3,818 test decisions: 17M 0.795 and 0.799, 32M 0.809 and 0.814, 68M 0.814 and
0.815. Spike training on the A100: classify 12.5 minutes (42,659 examples), rate and relevance 4.2 minutes each.

**Embedder, all arms** (nDCG@10):

| model | original queries | reworded queries | SciFact | NFCorpus |
|---|---|---|---|---|
| BM25 | 0.910 | 0.500 | not run | not run |
| gte-modernbert-base, the teacher | 0.874 | 0.730 | 0.765 | 0.344 |
| bge-small | 0.833 | 0.664 | 0.713 | 0.343 |
| MedCPT | 0.751 | 0.485 | 0.711 | 0.321 |
| `capemox/ettin-encoder-68m-contrastive-2.5m`, a published general Ettin-68m embedder | 0.551 | 0.265 | 0.448 | 0.199 |
| 17M, both stages | 0.862 | 0.526 | 0.483 | 0.144 |
| 32M, both stages | 0.903 | 0.646 | 0.592 | 0.185 |
| 68M, both stages | 0.920 | 0.733 | 0.612 | 0.219 |
| 68M, both stages, encoder rounded as packed | 0.920 | 0.734 | 0.609 | 0.213 |
| 68M, teacher stage only | 0.855 | 0.646 | 0.600 | 0.188 |
| 68M, contrastive only | 0.900 | 0.437 | 0.510 | 0.157 |

The folded 512-d packed embedder returns the same top 10 as the 768-d build on 150 queries, scores within
1.4e-6 (`results/fold_check.json`).

**Runtime check** on this container's CPU. The packed 68M specialist scores 0.7705 on all 2,000 Typed Decisions
test decisions through `dettin.py` (0.7725 recorded on Colab) at about 393 ms per five-question case. Spikes
trained against the fp32 stock base, loaded onto the packed stock base: invoices 0.854 (0.858 on fp32),
customer service 0.752 (0.750).

**Haiku as an online decision model.** Oskar asked how the Haiku labelling run was configured and whether Haiku
could be the decision model itself. The labelling runs set no effort and put 5 or 8 sections in a call.
A second run put one section in a call at low effort (`results/haiku_single_low.json`, 388 sections scored by
both): 0.910 over the eight section questions against 0.941 for the batched run, a median of 2.3 seconds per
call and $0.0004 per section at list price. The two runs give the same answer on 77% (reproducibility rubric)
to 99.7% (immunoassay) of sections. For comparison Jev scored 0.883 at 0.46 seconds per section, and a 68M
spike answers in about 30 ms on an A100 with no API call.

**What went wrong.**
- TypeSafe credits ran out after 87.6M Jev input tokens, 43,857 of 83,558 pair ratings in. No eval pair was
  rated, so Jev's accuracy on relevance is unknown, and negatives were filtered only where a rating existed.
- The 68M contrastive stage ran out of memory at batch 24 on a 40 GB A100; it was rerun with gradient
  checkpointing.
- The distractor pool depended on the negatives, so it changed (27,148 to 28,073 sections) between two data
  builds; early models were re-scored on the final pool.
- The packed-variant scorer spent about 25 A100 minutes on a NumPy parity check of six questions at 68M, and
  more compressing fp32 files at the highest brotli level.
- The local drivers that keep a Colab VM alive were stopped four times by a two-hour limit on background jobs,
  once by a hung Colab CLI call and once by a container restart. The remote jobs ran on; each was re-attached.
- Safety filters refused 7 of 2,716 Haiku labelling batches and a handful of query and grading batches; 45
  labelling batches came back with garbled item numbers. All were left out and none was reworded.
- TREC-COVID was scored for some arms (teacher 0.633, 68M 0.630, bge-small 0.760) and then dropped at Oskar's
  direction: neither teacher used here or in `lifesci-ettin` is good at it.
- Colab use passed the 60-unit budget in the plan, by 10 units.

**Reading at the time.** Oskar's view after this round: there are few enough decision use cases that a cheap
spike per use case makes generality a luxury, and of the two goals a stronger spike library matters more than
the embedder. The embedder's weak public scores have the same shape as `lifesci-ettin`'s: a small amount of
domain training was expected to supply retrieval ability that the starting weights did not have. A learning
curve for the teacher stage over more and broader text would test that; it was proposed and not run.

### Follow-ups the same evening (2026-11-10)

**Triangulation: three other spikes answering a held-out workflow** (`run/triangulate.py`, `results/triangulate.json`).
Oskar asked whether a spike library could answer a domain it has no spike for by running the three nearest
spikes and weighting their answers by distance. For each Typed Decisions workflow its own 68M spike was set
aside and the other three answered its 100 test cases. Distance was the cosine between the case's state,
mean-pooled through the frozen base, and the centroid of each spike's training states. Mean accuracy over the
four workflows: the workflow's own spike 0.763, a uniform average of the other three 0.347, the
distance-weighted average 0.349, the nearest spike alone 0.363, the best single spike chosen after the fact
0.425, always answering with the most common training label 0.482. No combination reached the most-common-label
baseline. The distance carried almost no signal: one workflow (security incidents) was the nearest centroid
for nearly every case of the other three, and the weight on the nearest spike averaged 0.37 to 0.45 of 1.
Four unrelated workflows are a hard case for this idea; it was not tried on a library with near neighbours.

**Haiku reading a whole article** (`run/paper.py`, `results/paper.json`). Oskar's idea: use Haiku at low effort as
the decision model itself, on the full paper, and stack calls with skewed instructions. 150 test articles
(median 3,850 words), four article-level questions, gold where Sonnet and Opus agree on the full text (0.87 to
0.99 of articles), one article per call.

| arm | study type | field | human samples | immunoassay | mean |
|---|---|---|---|---|---|
| plain instruction, three separate runs | 0.909, 0.879, 0.894 | 0.869, 0.892, 0.877 | 0.993 | 0.966 | 0.933, 0.932, 0.932 |
| "decide from what the methods and results report" | 0.962 | 0.908 | 0.993 | 0.959 | 0.956 |
| "start from the least committal answer" | 0.947 | 0.938 | 1.000 | 0.966 | 0.963 |
| plain plus six worked examples | 0.917 | 0.877 | 0.993 | 0.946 | 0.933 |
| three plain calls, majority vote | 0.909 | 0.892 | 1.000 | 0.966 | 0.942 |
| plain, methods and least-committal, majority vote | 0.962 | 0.931 | 1.000 | 0.966 | 0.965 |
| Jev on the full article | 0.955 | 0.954 | 0.986 | 0.966 | 0.965 |

A better instruction did as much as the stack. The worked examples were six sentences from dev articles where
plain Haiku and Opus disagreed, all on study type and field, so that arm is a weak test of examples. With about
130 agreed items per question, 0.03 on one question is four articles. A Haiku call cost $0.0015 per article at
list price and took a median of 2.1 to 2.5 seconds; Jev cost $0.0003. The gold for this test cost $15.
