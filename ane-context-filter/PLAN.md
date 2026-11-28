# ane-context-filter: plan

## Question

When an orchestrator delegates a task, can a model small enough to run on the Mac's
Neural Engine pick the transcript chunks the subagent needs as well as Jev does?
The point is a local subagent (Bonsai 2.5 27B on the M5 Max) that receives the
orchestrator's task plus filtered context, where the filter runs on otherwise idle
silicon instead of a cloud call.

`subagent-context-filter/` showed that the pattern works with Jev as the filter:
99 of 105 facts against 102 for the full transcript and 82 for a written brief, at a
20K-token budget. For a local subagent the budget matters more. Bonsai's prefill on
the M5 Max is about 800 tokens/s (estimate, to be measured by `bonsai-metal-bench`),
so 20K tokens of context costs about 25 s before the first token. The filter has to
hold recall at 5K and 10K tokens too.

## Prior art (searched 2026-11-27)

- **Provence** (Naver, ICLR 2025, `naver/provence-reranker-debertav3-v1`): one
  DeBERTa-v3 cross-encoder that reranks and prunes RAG context at sentence level.
  It is the closest component, and a zero-shot arm here.
- A paper that distils Qwen3-32B's relevance and utility judgements into
  Qwen3-1.7B as a passage selector, and Context-Picker (arXiv 2512.14465), which
  mines minimal evidence sets with a generator-judge loop and trains a picker on
  them.
- kapa.ai prunes 68% of RAG context with a small LLM at runtime.

None of these I found targets agent transcripts, distils from a typed-decision model,
or runs on the ANE. Did not find that composition in 3 searches. What this adds:
transcripts (tool calls with results, not retrieved passages), real delegation tasks
from our own sessions, and a model sized for the ANE's fixed-shape regime.

## Data

Archived sessions from the `claude-workspace` `transcripts` branch, chunked and
secret-scrubbed by `subagent-context-filter/chunking.py`. All transcript-derived files
(chunks, tasks, labels) live outside this public repo, under `EVAL_DATA`. Only
counts are committed.

- `prep.py`: 870 sessions, 1,818 delegations, 1,448 of them `[no-context]` fan-out
  workers. **301 usable delegations from 128 sessions** after dropping opt-outs and
  capping templated duplicates at 3 per session. Median context 76K tokens, p90 171K
  (`results/prep.json`).
- `teach.py`: Jev scores every chunk of every usable delegation (the shipped flat
  scorer at 4x view). **Done 2026-11-27: 301 delegations, 35,143 chunks scored, no
  errors, $1.32 of Jev** (105 delegations ran twice by accident; the first label of
  each is kept). The labels and task text live on the private `ctxfilter-data` branch
  of `oaustegard/claude-workspace`, not here.
- Two things the run found. The shipped filter treats Jev's HTTP 400
  `max_tokens_exceeded` as retryable and gives up on the whole delegation after
  ~90 s. Its 3.3 chars/token estimate undercounts dense text such as hex, so some
  windows overflow. `teach.py` bisects such a window instead, as the filter already
  does for WAF blocks. The fix belongs in the `delegating-with-context` skill too.
  Separately, TypeSafe's 100K tokens/s limit stalls 8 concurrent 30K-token windows,
  so `--workers 2` (4 calls in flight) is the setting that ran.

## Status: fact recall settles it: Jev well ahead, students barely above recency

Round 3 (2026-11-28): `factrecall.py`. On the 19 test sessions over 12K tokens, the
`subagent-context-filter` labeller (Sonnet, 1M context) wrote 60 tasks with 232 required facts
and the chunks holding them. Each arm's selection went to a fresh tool-less Sonnet executor;
facts scored by regex, misses not adjudicated (`results/factrecall.json`).

| arm | evidence chunks kept, 5K / 10K | facts found at 10K |
|---|---|---|
| Jev | 0.635 / 0.801 | 217 / 232 (93.5%) |
| student17 | 0.229 / 0.292 | 178 / 232 (76.7%) |
| recency | 0.244 / 0.362 | 153 / 232 (65.9%) |
| student17lw | 0.207 / 0.273 | (executor not run) |

Paired per task at 10K (bootstrap 95% CI): Jev finds +1.07 facts per task more than recency
[+0.67, +1.50] and +0.65 more than student17 [+0.42, +0.93], better on 25 tasks and worse on 1.
student17 beats recency by +0.42 [+0.08, +0.78]. The student keeps fewer of the labelled evidence
chunks than recency yet yields more facts, so it finds facts repeated in other chunks; it still
trails Jev by 39 facts. 180 executor runs cost $13.75.

Reading: a 17M student distilled from 233 delegations is not a usable replacement for Jev. The
gap is large, and the overlap metric understated it. Likely causes, in order: the student scores
each chunk alone, while Jev sees the task with the whole session's index and user messages; the
training tasks (real delegations, median 165 characters, many of them templated) differ from these
fact-carrying tasks; and 233 delegations is little data. Worth trying only with a cheap answer to the
first cause, e.g. prepending the session's user messages to each pair; otherwise the ANE filter
is not worth building, and Jev stays the filter.

## Round 2 status: two students trained; neither reliably beats recency on Jev overlap

Round 2 (2026-11-27): the listwise loss (`train.py --loss listwise`: each step one
delegation's chunks, up to 48, KL to the softmax of Jev's logits, plus 0.2x BCE),
2 epochs, 46 minutes. Also blends of each student's rank with recency's (equal
weight). Recall of Jev's chunks at 5K / 10K / 20K tokens
(`results/agreement_val.json`, `results/agreement_test.json`):

| arm | val (23 delegations) | test (45 delegations) |
|---|---|---|
| recency | 0.367 / 0.499 / 0.626 | 0.370 / 0.433 / 0.572 |
| student17 (BCE) | 0.319 / 0.401 / 0.562 | 0.361 / 0.372 / 0.500 |
| student17lw (listwise) | 0.379 / 0.442 / 0.532 | 0.302 / 0.341 / 0.488 |
| student17 + recency | 0.388 / 0.519 / 0.647 | 0.348 / 0.404 / 0.538 |
| student17lw + recency | 0.381 / 0.492 / 0.632 | 0.372 / 0.405 / 0.522 |

The listwise student leads recency on val and trails it by 7 points on test; with
23 and 45 delegations, swings of that size are noise. No arm separates from recency
on overlap. Every student arm keeps chunks Jev rates more relevant per token
(quality 0.77-0.86 against recency's 0.71-0.78), so the students are picking
different relevant chunks, not irrelevant ones.

What would change the reading: fact recall. Overlap with Jev's exact picks punishes
a student that picks equally useful substitutes, and Jev itself varies between
identical calls. The next round should run the `subagent-context-filter`
fact-recall eval (label.py tasks on the test sessions, executor, adjudicator) for
recency, Jev and the best student, before more training. If students hold facts as
well as Jev, overlap was the wrong yardstick; if they do not, the bottleneck is the
233 training delegations, and synthetic tasks are the way to more data.

## Round 1

Round 1 (2026-11-27): `train.py` defaults, Ettin-17m, 2 epochs on 25,349 pairs,
38 minutes on 4 CPUs. Test split, 45 delegations (`results/agreement_test.json`):

| arm | recall of Jev's chunks at 5K / 10K / 20K | quality at 5K / 10K / 20K |
|---|---|---|
| Jev's own scores (the eval's ceiling) | 0.986 / 0.993 / 0.984 | 1.001 / 1.001 / 0.994 |
| recency | 0.370 / 0.433 / 0.572 | 0.708 / 0.740 / 0.779 |
| bm25 | 0.245 / 0.300 / 0.422 | 0.754 / 0.760 / 0.782 |
| student17 | 0.361 / 0.372 / 0.500 | 0.842 / 0.858 / 0.868 |

On the 23 validation delegations the order is the same (student 0.319 at 5K,
recency 0.367). The student picks chunks Jev rates relevant (quality 0.84 against
recency's 0.71) but not the ones Jev ranks highest. It was trained pointwise with
BCE on soft labels, which teaches calibration rather than ordering, and Jev's
labels are flat: mean 0.367, 25% of chunks at 0.5 or more, 3.7% at 0.8 or more.
Overlap with Jev's exact picks may also understate a student that picks
near-equivalent chunks; fact recall would settle that.

Next, in order: a listwise ranking loss per delegation (softmax over a
delegation's chunks against Jev's scores); recency and the student combined;
68m; then the fact-recall eval for whichever arm leads. Weights (fp16) and
scores are on the private `ctxfilter-data` branch.

## Earlier status: baselines

Teacher agreement on the 45 test delegations (20 sessions), non-user chunks
(`results/agreement_test.json`):

| arm | recall of Jev's chunks at 5K / 10K / 20K | quality at 5K / 10K / 20K |
|---|---|---|
| recency | 0.370 / 0.433 / 0.572 | 0.708 / 0.740 / 0.779 |
| bm25 | 0.250 / 0.302 / 0.424 | 0.754 / 0.760 / 0.782 |

Quality is the token-weighted mean Jev score of what an arm keeps, relative to
Jev's own selection. Jev keeps a median 7, 10 and 16 non-user chunks at the three
budgets. Over all 301 delegations the ordering is the same (recency 0.422 at 5K,
BM25 0.302: `results/agreement_all.json`). Recency beating BM25 says Jev leans on
recent chunks, which a student given position as a feature can learn.

## Arms

All arms pack chunks into the same budget with the shipped `select` function, at
5K, 10K and 20K tokens.

| arm | scorer | runs on |
|---|---|---|
| `jev` | Jev, flat, 4x view (the teacher) | cloud |
| `recency` | newest first | — |
| `bm25` | BM25 of the task against each chunk | CPU |
| `dense` | cosine of task and chunk embeddings (mdbr-leaf-ir, as in Projects Search) | ANE |
| `provence` | Provence reranker score, zero-shot | ANE candidate |
| `student17` / `student68` | Ettin-17m / 68m cross-encoder distilled from Jev's scores | ANE |
| `full`, `none` | everything / nothing | — |

## Metrics

1. **Teacher agreement** (cheap, every held-out delegation): recall of the chunks Jev
   would keep at each budget, and rank correlation with Jev's scores.
2. **Evidence recall** (cheap): on held-out sessions labelled by
   `subagent-context-filter/label.py` (tasks with required facts and the chunk ids
   holding them), the share of evidence chunks inside the budget.
3. **Fact recall** (costly, final arms only): the `subagent-context-filter` executor
   and adjudicator on those tasks.
4. **Latency on the ANE**: ms per chunk at a fixed 512-token shape, and per
   delegation, on the M5 Max alongside a Bonsai decode (`bonsai-metal-bench`
   contention phase).

Splits are by session: 80% train, 10% validation, 10% test. The test sessions also
get `label.py` tasks. Metric 2 must use held-out sessions, since Jev's labels on
training sessions are what the student copies.

## Student

Ettin-17m (and 68m if 17m falls short), fine-tuned as a cross-encoder on
`[task] [SEP] [chunk view]` at 512 tokens, regressing Jev's score (BCE on soft
labels). Same recipe as `browser-decider/`. Chunk view = the 1x view the shipped
filter uses (1,500 chars, or 300 input + 900 result for a tool chunk), clipped to fit.
Trained on CPU here or a Kaggle T4. Converted to Core ML in the ANE layout from
`ane-dettin` (1x1 convs, (1, C, 1, S) activations, fixed length), fp16.

## Risks

- **301 delegations is small.** Pairs number about 50K (median 170 chunks each), but
  they are correlated within a delegation. If the student overfits, add synthetic
  tasks: `label.py`-style tasks on sessions without delegations, labelled the same
  way.
- **The teacher's ceiling.** The student can at best match Jev. Jev's scores also
  vary between identical calls (`subagent-context-filter` finding 5), which caps
  measurable agreement.
- **The 512-token window.** Jev sees each chunk with the whole session's index and
  user messages; the student sees one chunk and the task. Chunks whose relevance
  depends on session context will be missed. Prepending the latest user message is
  the cheap fix to try.
