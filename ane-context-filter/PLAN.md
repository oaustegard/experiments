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
  scorer at 4x view): about 165K Jev tokens and 2 s per delegation, about $2 in all.

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
