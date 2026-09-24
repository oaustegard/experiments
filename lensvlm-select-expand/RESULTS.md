# RESULTS: LensVLM's select-then-expand pattern on Opus 5.5 and Sonnet 5

Source: Apple's LensVLM ([arXiv 2605.07019](https://arxiv.org/abs/2605.07019),
[code](https://github.com/apple-aiml-research/ml-lensvlm)). Appendix 9 of the
paper applies the pattern, untrained, to Sonnet 4.6 and reports +4.8 to +9.0 pp.
Oskar, 2026-09-24: *"try the select-then-expand pattern with Sonnet AND opus
5.5 in experiments as Opus has higher image viewing fidelity"*. Muninn memory
`cdda7027` holds the paper review.

## Answer

Does select-then-expand help Claude, and does Opus 5.5 do it better than
Sonnet 5? Opus 5.5 is much better at the *select* half: shown a whole document
as one sheet of thumbnails, it puts the evidence page first 100/90/70% of the
time at 5x/10x/15x compression, against 75/15/5% for Sonnet 5, and its
expansions reach the evidence 100/100/95% of the time against 85/40/30%. The
*expand* half added little to answer accuracy for Opus (image-only 95/90/85%,
after expansion 90/90/80%) because Opus already read the 5x and 10x sheets
directly, and HotpotQA answers come largely from memory: 85% closed-book for
Opus, 75% for Sonnet. For Sonnet, expansion added 20/15/10 pp over its
image-only answers. Sonnet's image-only answers at 10x and 15x also fell
20 pp *below* its own closed-book score: an unreadable document displaced
answers it knew.

The two models have the same 2576 px vision limit and got identical pixels, so
the gap is perceptual, not a resolution cap. n = 20 documents per cell, one run
each: the selection gaps (e.g. 100 vs 40% at 10x) are large enough to stand,
the 5 to 10 pp accuracy differences are not. A benchmark whose answers the
models do not already know would change the accuracy half of this answer; the
paper's post-cutoff PubMed set is that benchmark.

## Findings

1. **Opus 5.5 selects evidence pages from compressed sheets far better than
   Sonnet 5.** First-ranked page correct: Opus 100/90/70%, Sonnet 75/15/5% at
   5x/10x/15x. Evidence reached within its expansions: Opus 100/100/95%,
   Sonnet 85/40/30%. (`score.py`, columns sel@1 and expHit)
2. **HotpotQA answer accuracy mostly measures memory.** Closed-book, with no
   document: Opus 17/20, Sonnet 15/20. Only 3 (Opus) and 5 (Sonnet) documents
   test reading at all. On those, Opus's final answer was right 2/3 at every
   ratio; Sonnet's 4/5 at 5x and 0/5 at 10x and 15x. (closed-book arm)
3. **Expansion adds accuracy for Sonnet and not for Opus.** Final minus
   image-only: Sonnet +20/+15/+10 pp; Opus −5/0/−5 pp. Opus lost h02 at 5x
   (accepted "SSIH" became "Endura" after reading) and h12 and h18 at 15x.
4. **Opus reads 5x text outright.** Its image-only accuracy (95%) exceeds its
   closed-book (85%) at 5x: it answered 2 of its 3 memory misses from the
   sheet alone, 1 at 10x, 0 at 15x. Sonnet did so once, at 5x.
5. **An unreadable sheet hurts Sonnet's answers below knowing nothing.** At 10x
   and 15x, 4 documents Sonnet answered right closed-book went wrong
   image-only; its image-only accuracy 55% against 75% closed-book. Opus
   never lost a closed-book answer to the image.
6. **Sonnet spends more to get less.** Expansions per document 1.90/2.50/2.60
   (Opus 1.25/1.45/1.75), so effective compression 3.3/4.5/5.4x against Opus's
   3.7/5.6/6.3x. Batches of 10 documents took Sonnet 5.0 to 11.5 min and 96k to
   136k subagent tokens; Opus 2.3 to 3.1 min and 78k to 82k.
7. **At 15x, Opus selects from paragraph titles, not body text.** The bracketed
   `[Title]` headers survive as dark blobs at 136 px page width; body text does
   not (see `examples/h00-r15.png`). Opus's 70% first-page accuracy at 15x is
   consistent with title matching; this is an inference from the images, not
   something the ledgers record.

| model | ratio | closed | image-only | expand | exact-match (expand) | sel@1 | expHit | #exp | ECR |
|---|---|---|---|---|---|---|---|---|---|
| Opus 5.5 | 5x | 85 | 95 | 90 | 65 | 100 | 100 | 1.25 | 3.7 |
| Opus 5.5 | 10x | 85 | 90 | 90 | 65 | 90 | 100 | 1.45 | 5.6 |
| Opus 5.5 | 15x | 85 | 85 | 80 | 55 | 70 | 95 | 1.75 | 6.3 |
| Sonnet 5 | 5x | 75 | 70 | 90 | 65 | 75 | 85 | 1.90 | 3.3 |
| Sonnet 5 | 10x | 75 | 55 | 70 | 40 | 15 | 40 | 2.50 | 4.5 |
| Sonnet 5 | 15x | 75 | 55 | 65 | 35 | 5 | 30 | 2.60 | 5.4 |

Percentages over 20 documents. "closed" is the same 20 questions with no
document, repeated on each row for comparison.

## Method

**Fixture.** 20 HotpotQA distractor-validation questions, `level == hard`,
yes/no answers excluded, seed 20260924 (`build_data.py`). Each document is the
question's 10 paragraphs padded to ~8k nominal tokens with paragraphs from
other questions, placed before or after the original block, as in LensVLM's
`hotpotqa.py` provider. Wrapped at a fixed 28 lines per page, so every document
has 17 pages at every compression. Gold pages are those holding a supporting
sentence (1 to 3 per document).

**Compression.** All 17 pages go on one contact-sheet PNG with a legible
`P<n>` label above each thumbnail. The page scale is searched so the whole
sheet, labels and gutters included, costs `text_tokens / ratio` image tokens by
Claude's formula `ceil(W/28) * ceil(H/28)`. Text tokens are chars/4, which
under-counts Claude's tokenizer, so true compression is somewhat above the
label. Achieved: 5.06x (1,596 image tokens, pages 251 x 220 px), 10.07x (801,
173 x 152), 15.49x (521, 136 x 119). Sheets stay under Claude Code's 2000 x
2000 Read-tool cap, so neither model's image is resized.

**Arms.** One `general-purpose` subagent per model x ratio x half (10
documents), 12 in all, models set by the Agent tool's `opus` and `sonnet`
aliases; every agent logged its served model id (`claude-opus-5-5`,
`claude-sonnet-5`). Per document the agent Reads the sheet, runs
`lens.py commit` with an image-only answer and ranked page guess, then
`lens.py expand` up to 3 times, then `lens.py final`. `lens.py` refuses expand
before commit and after final, caps expansions, and ledgers every call. Page
texts sit zlib-compressed in `pages.bin`. Prompts carried `[no-context]` so the
workspace's context hook did not append this session's transcript, which held
the gold answers. Closed-book arm: one agent per model answers all 20
questions with no document.

**Scoring.** `score.py`: HotpotQA normalisation; a match is exact, or
containment with the shorter string at least 4 characters, or a hand-adjudicated
alternate applied identically to every arm (h02 SSIH/ASUAG, h05 "Ghana national
football team", h09 "2012 season" = the Rams' 75th, h10 "director"; reasons in
`score.py`). Strict exact-match is reported alongside. sel@1: first committed
page is a gold page. expHit: any expanded page is a gold page. ECR: document
text tokens over sheet tokens plus expanded text tokens.

**Controls not run.** No full-text arm (the whole document as text), so there
is no text upper bound. No repeat runs, so no variance estimate. One Sonnet
5x agent reports opening `lens.py` itself; the page texts are compressed and
it could not have read answers from it.

**Cost.** 1.34M subagent tokens: Opus arms 479k, Sonnet arms 682k,
closed-book 119k, pilot 60k. Roughly 32k of each agent is the harness floor
(METHODS, "Batch the candidates").

## Log

### Round 1 — 2026-09-24

Asked: try LensVLM's select-then-expand on Sonnet and Opus 5.5.

Checked first: the Claude Code binary's model table gives both Sonnet 5 and
Opus 5.5 `image_limits {maxWidth: 2000, maxHeight: 2000}` in the Read tool, and
the API docs give both 2576 px. The alias table maps `opus` to
`claude-opus-5-5` and `sonnet` to `claude-sonnet-5`. So the premise that Opus
sees more pixels does not hold here; any difference is in what each model makes
of the same pixels.

Pilot (2 documents, Opus, 10x) confirmed the ledger flow and showed both answers
right before any expansion, which is where the closed-book arm came from.

Viewing the sheets while building: 5x body text looked readable, 15x only the
bracketed titles. The 5x reading was right for Opus (finding 4) and wrong for
Sonnet.

No predictions were written down before scoring, so this is the reading at the
time rather than a test of one. Opus beat Sonnet on image-only accuracy at 10x
(90 vs 55%) and 15x (85 vs 55%), though at 15x Opus's image-only score only
equals its closed-book score. Expansion moved Sonnet +10 to +20 pp, about twice
the paper's gain for Sonnet 4.6, and moved Opus nowhere, because its
image-only answers had little left to recover on this benchmark.
