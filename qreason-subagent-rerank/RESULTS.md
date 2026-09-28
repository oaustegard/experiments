# QReason's decoupling in a subagent fan-out

QReason ([arXiv 2609.30904](https://arxiv.org/abs/2609.30904), EMNLP 2026) trains a 7B rewriter to write a query's reasoning once, then lets a non-reasoning listwise reranker reuse it across sliding windows instead of regenerating chain-of-thought per window. Oskar asked whether the idea carries over to agentic search in Muninn's own environment, where the "windows" are subagents each judging a batch of results. On 16 BRIGHT-biology queries with Haiku 4.5 judges, a relevance brief written once by the orchestrator beat the raw query by +5.2 NDCG@10 pooled over two runs (95% CI +0.1 to +11.2). Per-batch reasoning landed between the two. A brief that stated a guessed answer scored no better than the raw query. Run-to-run noise was large enough that one run's +8.8 gap became +1.7 on the replicate.

The second half of this file checks the paper itself: its ReasonRank baseline is 1.6 points below ReasonRank's published number, 290 of BRIGHT's 1,384 test queries appear verbatim in the training set both papers use, and the 7.9x speedup compares a 3B-active MoE against a 32B dense model.

## Setup

- **Queries and candidates.** BRIGHT biology, ReasonIR top-100 retrieved with BRIGHT's GPT-4 reasoning queries (the paper's first stage). First-stage NDCG@10 over all 103 queries from this candidate file: 43.40; the paper's Table 1 reports 43.49. 16 queries drawn with seed 20260928 from the 85 with at least one gold passage in the top 30, query 0 excluded because its gold ids were visible before the briefs were written. Top 30 candidates per query, passages truncated at 1,500 characters. `prep.py` rebuilds every input byte-identically from public GitHub copies (the Hugging Face CDN is blocked from this container).
- **Batches.** Retrieval ranks 1-10, 11-20, 21-30 go to three separate subagents, each scoring its batch for all 16 queries. That is the sliding-window analog, run in parallel instead of in sequence.
- **Judges.** Claude Code Agent-tool subagents, `general-purpose`, Haiku 4.5, graded 0-3 per passage. Ranking = score descending, ties broken by retrieval rank. Gold labels are BRIGHT's qrels (tevatron copy).
- **Arms.**
  - A: the original query only.
  - B: original query + a criteria brief (what the asker wants, what a relevant passage contains, which near-misses to discount), written once per query by the orchestrator (Opus 5.5) from the query text alone, before any candidate or gold label was seen.
  - C: original query; each batch's subagent first writes its own 80-150 word analysis of the query, then scores (the per-window-CoT analog).
  - E: original query + a brief stating the orchestrator's guessed answer with specifics (HyDE-style).
  - D: the B brief, all 30 candidates in one subagent context.
- **Replicate.** A, B and C were run a second time (A2, B2, C2) with a stricter reading spec (Read tool only, no scripts, half-size input files); see ERRORS.md for why.
- Subagent transcripts were checked for appended parent context (the `[no-context]` issue in METHODS.md): none; each judge's first message is the prompt alone.
- Predictions were written to `PREDICTIONS.md` before the first subagent ran; an addendum before the replicate states what would count as the run-1 gap being noise.

## Results

NDCG@10 x100, mean over 16 queries.

| arm | run 1 | run 2 | pooled |
|---|---|---|---|
| first stage (ReasonIR order) | 39.82 | | |
| A raw query | 45.70 | 47.28 | 46.49 |
| B criteria brief | **54.47** | 48.94 | **51.71** |
| C per-batch reasoning | 48.81 | 49.13 | 48.97 |
| E answer-guess brief | 46.83 | | |
| D single pass + criteria brief | 51.14 | | |

Paired differences, bootstrap 95% CI over queries, wins/losses out of 16:

| comparison | diff | 95% CI | W/L |
|---|---|---|---|
| B - A, run 1 | +8.77 | [+2.5, +18.3] | 11/1 |
| B - A, run 2 | +1.66 | [-4.0, +6.6] | 9/4 |
| B - A, pooled | +5.22 | [+0.1, +11.2] | 9/5 |
| C - A, pooled | +2.48 | [-5.8, +10.9] | 11/4 |
| B - C, pooled | +2.73 | [-2.2, +8.5] | 6/9 |
| B - C, run 2 | -0.19 | [-5.0, +4.2] | 6/6 |
| E - B pooled | -4.88 | [-14.0, +3.5] | 8/7 |
| E - A run 1 | +1.13 | [-13.2, +16.4] | 7/7 |
| D - B pooled | -0.57 | [-5.5, +4.1] | 8/6 |

1. **The shared brief helped the cheap judge, by an amount one run cannot pin down.** B beat A in both runs, by +8.8 and then +1.7. Pooled, the CI clears zero by a hair. The size of the effect is not established at n=16.
2. **Per-batch reasoning did not beat a shared brief.** C sat between A and B pooled, and tied B in run 2. The paper's claim that query-level reasoning can be done once and reused holds here as "no worse"; "better" is not shown.
3. **An answer-guess brief was worse than a criteria brief and no better than the raw query** (one run). The prediction was the opposite: BRIGHT's gold documents are the ones the accepted answer cites, so a correct guess should name them. The paper attributes TongSearch's reranking losses to unsupported specifics in the rewrite creating false relevance signals, and E points the same direction without clearing noise.
4. **One context versus three batches made no difference at 30 candidates** (D - B pooled -0.57).
5. **Identical runs disagree a lot.** Between run 1 and run 2 of the same arm, 59-60% of the 480 passage scores match exactly (Pearson 0.73-0.80), and per-query NDCG@10 differs with SD 13.5 (B) to 21.2 (A) points. The shared brief gave the highest run-to-run correlation (0.80) and the smallest per-query spread of the three arms.
6. **The cross-batch calibration story did not replicate.** Per-batch reasoning was expected to make scores from different batches less comparable. Arm C's gold-vs-non-gold AUC over the merged 30 was below its within-batch AUC in both runs (0.782 vs 0.815; 0.807 vs 0.831), but B showed a +0.027 gap in run 1 and -0.021 in run 2, and A2 -0.015. No arm separates from the others on this measure.

## Cost

Every valid subagent in every arm used 94k-111k tokens (harness-reported `subagent_tokens`, `results/usage.json`), dominated by reading 25-30k tokens of passages plus the ~32k per-subagent floor measured in `hypothetical-classification`. The per-batch analyses in arm C added about 8k characters (~2k tokens) of output per subagent, ~2% of its total. Wall clock ran 49-140 s per subagent with no arm consistently slower. QReason's efficiency argument rests on decode-bound, sequential windows, where regenerated reasoning dominates latency. In a parallel fan-out whose judges mostly read, the saving from reusing reasoning is small. The 16 briefs cost about 1.1k tokens of orchestrator output.

## Per-batch analyses and the 90% similarity figure

The paper motivates QReason with ">90% average semantic similarity" (BGE-m3) between reasoning traces a listwise reranker writes for the same query in different windows. Arm C's analyses reproduce the number and show what it measures (`similarity.py`):

| measure (same query, different batch vs different queries) | same | different |
|---|---|---|
| gemini-embedding-2 cosine, 8 queries x 3 batches | 0.920 (min 0.882) | 0.623 (max 0.744) |
| TF-IDF cosine, 16 queries | 0.474 | 0.024 |
| word-trigram Jaccard, 16 queries | 0.032 | 0.005 |

Three independent 59-word analyses of the same question score 0.92 on a dense embedding while sharing 3% of their word trigrams. They are paraphrases that name the same concepts, which is real semantic redundancy, and the cosine does not say how many tokens could be skipped. They also differ in emphasis (one analysis of the twins query centres on heteropaternal superfecundation, another on superfetation), which is the difference a single shared brief removes.

## Checks on the paper

- **The ReasonRank baseline is under-reproduced.** ReasonRank's own Table 1 ([arXiv 2508.07050](https://arxiv.org/abs/2508.07050)), same first stage (ReasonIR top-100, GPT-4 queries for retrieval, original queries for reranking): 38.03 for 32B, 35.74 for 7B. QReason's reproduction: 36.45 and 35.38. The 32B gap is -1.58 on average and -4.69 on StackOverflow, -3.52 on Biology, -2.91 on Robotics. The headline "QReason + Qwen3.5-35B-A3B 36.93 beats ReasonRank 32B" holds against the reproduction (7 of 12 subsets) and not against the published number (1.10 lower, 3 of 12 subsets).
- **21% of BRIGHT's test queries are in the training set** (`overlap/overlap.py`). QReason builds both training stages from `reasonrank_data_13k`. 290 of 1,384 BRIGHT queries appear verbatim in its query files: biology 74/103, earth science 73/116, sustainable living 67/108, economics 55/103, robotics 20/101, stackoverflow 1/117. ReasonRank's paper sources those training queries from the same six StackExchange sub-domains and mines positives from documents linked in accepted answers, which is how BRIGHT built its gold sets; I found no mention of deduplication against BRIGHT in it or in QReason. The "gold relevant passages" QReason trains on are DeepSeek-R1-mined, per the dataset card. The match test counts a hit only on a 60-character verbatim window, so these counts are a floor.
- **The overlap does not show up in the rewriter's gains.** Across the 14 rerankers in Tables 1, 6 and 7 (Table 7 uses windows of 10), QReason's mean gain over original queries is +1.85 on the four high-overlap subsets, +1.61 on robotics and stackoverflow, and +2.26 on psychology and pony, which are outside the training domains. Psychology, the one StackExchange subset outside them, has the smallest gain (+0.39) and pony the largest (+4.13); one clean StackExchange subset cannot separate domain familiarity from query overlap. Absolute BRIGHT scores of any model trained on this set, ReasonRank included, carry the overlap; QReason's per-domain gains do not rise with it.
- **The 7.9x speedup is mostly the reranker's architecture.** Table 3 compares ReasonRank 32B (dense) with QReason + Qwen3.5-35B-A3B (3B active parameters). With a dense 32B reranker the speedup is 2.6x, and that configuration's accuracy (35.18) matches ReasonRank 7B's reproduced 35.38, whose latency is not reported. Qwen3.5-35B-A3B with original queries and no rewriter already scores 35.49.
- **λ was chosen on the test set.** Appendix C.1 sweeps the reward weight from 0.1 to 1.5 and picks 0.6 by the NDCG@10 gain of the rewritten queries with Qwen2.5-32B as reranker; the paper names no separate dev set, and every reported evaluation is on BRIGHT. The ablations (Table 2) are single training runs with deltas of 1.3-2.2 points and no variance estimate; greedy decoding removes decoding noise and leaves training-run noise unmeasured.

## Predictions scored

- P1 (B > A by +1 to +4): run 1 +8.8, run 2 +1.7, pooled +5.2. Direction held; the run-1 gap fell above the predicted range.
- P2 (C within 3 of B, at 1.5-3x the tokens): quality held (pooled -2.7, run 2 -0.2); the token multiple was about 1.02x.
- P3 (E >= B): wrong, -4.9.
- P4 (D >= B): tie, -0.6.
- P5 (every arm beats the first stage): all means are above 39.82; A's CI includes zero.
- P6 (same-query cosine > 0.85, different-query > 0.7, trigram Jaccard < 0.2): 0.92 yes, 0.62 no, 0.03 yes.

## Limits

16 queries from one BRIGHT subset; two runs for A, B and C and one for D and E; one orchestrator writing every brief; Haiku 4.5 only. Run 2 used a stricter reading spec than run 1, so run-1-versus-run-2 differences mix judge noise with the spec change; within-run comparisons are clean. 12 of the 16 queries are among the BRIGHT queries found in the ReasonRank training set, which does not touch this experiment (no training) but means Haiku may have seen the posts in pretraining, as it may for any BRIGHT query. Passages were truncated at 1,500 characters.

## Files

`prep.py` (inputs, keys), `keys.json` (selection, id mapping, qrels), `queries.json`, `briefs_criteria.json`, `briefs_guess.json`, `prompts.md`, `PREDICTIONS.md`, `outputs/` (all valid judge outputs), `outputs_invalid/` (the two discarded E runs), `evaluate.py` -> `results/summary.json`, `similarity.py`, `results/c_analysis_embedding_cos.json`, `results/usage.json`, `overlap/overlap.py` -> `overlap/overlap_counts.json`, `ERRORS.md`.
