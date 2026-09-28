# Pre-registered predictions — QReason-in-my-environment (2026-09-28 ~06:20 EDT)

Setup: BRIGHT-biology, 16 queries (seed 20260928, q0 excluded, >=1 gold in top-30),
ReasonIR+GPT-4-reasoning top-30 (first-stage NDCG@10 on all 103 = 43.40; paper: 43.49).
Haiku subagents, graded 0-3 scores, merge by score then retrieval rank.
Batches = contiguous ranks 1-10 / 11-20 / 21-30 (sliding-window analog).

Arms:
- A raw: original query only, no analysis written
- B criteria brief: original query + one-time brief (intent, what relevant evidence contains, what to reject), written by me from the query alone
- C per-batch reasoning: original query; subagent writes its own query analysis per batch, then scores
- E answer-guess brief: original query + my guessed answer with specifics (HyDE/TongSearch-style)
- D single pass: criteria brief, all 30 candidates in one context

Predictions (n=16, so direction only; paired per-query SD ~0.2 means ~+/-10 pt CI):
- P1: B > A by +1 to +4 NDCG points. Low confidence in sign.
- P2: C within +/-3 of B, at 1.5-3x the subagent output tokens.
- P3: E >= B on BRIGHT, contrary to the paper's mechanism. Reason: BRIGHT gold = documents the
  accepted answer cites, so a correct answer guess names the concept the gold docs cover.
  Paper's TongSearch failure was wrong specifics; mine should mostly be right. Low confidence.
- P4: D >= B (one context, consistent calibration across the list). Small effect.
- P5: every reranked arm beats the ReasonIR order on these 16 queries.
- P6: C analyses: same-query cross-batch embedding cosine > 0.85, but different-query cosine
  also > 0.7 (floor); word-3-gram Jaccard same-query cross-batch < 0.2 (paraphrase, not repetition).

## Addendum before replicate runs (after seeing run-1 results)
Run 1: ReasonIR 39.82 | A 45.70 | B 54.47 | C 48.81 | E 46.83 | D 51.14. B-A +8.77 [+2.6,+18.3].
E_2/E_3 run-1 outputs were invalid (saw 77/160 and 6/160 passages; E_3 keyword-scored in Python);
fresh reruns with a Read-only spec replaced them. Replicate run 2 of A, B, C uses that Read-only spec
and half-size input files for all three arms.
Prediction for run 2: B-A stays positive (> +3); C stays between A and B. If B-A in run 2 is <= 0,
the run-1 gap was run-to-run noise.
