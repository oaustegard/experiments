# RESULTS: LensVLM's select-then-expand on Opus 5.5, Sonnet 5, Gemini 3.8 Flash and Muse Spark 1.3

Source: Apple's LensVLM ([arXiv 2605.07019](https://arxiv.org/abs/2605.07019),
[code](https://github.com/apple-aiml-research/ml-lensvlm)). Appendix 9 of the
paper applies the pattern, untrained, to Sonnet 4.6 and reports +4.8 to +9.0 pp.
Oskar, 2026-09-24: *"try the select-then-expand pattern with Sonnet AND opus
5.5 in experiments as Opus has higher image viewing fidelity"*, then *"could we
try Gemini and Muse?"*. Muninn memory `cdda7027` holds the paper review.

## Answer

Can a model find the page that matters on a sheet of unreadably small page
thumbnails, and does reading that page then help? Three of four can. Opus 5.5, Gemini 3.8 Flash and Muse Spark 1.3 rank the evidence page first
90 to 100% of the time at 5x and 10x compression and 50 to 75% at 15x. Sonnet 5
manages 75/15/5%. All four got identical pixels, so the gap is perception.
Reading beyond memory: Muse answered 5 of its 6 closed-book misses from the
sheet alone at 5x and 10x, Gemini 3 of 4 at every ratio, Opus 2 of 3 at 5x
falling to 0 at 15x, and Sonnet 1 of 5 at 5x, none after.

The *expand* half added little for anyone but Sonnet (+10 to +20 pp), because
the other three already read the sheet, and HotpotQA answers come largely from
memory (closed-book 70 to 85%). Gemini mostly skipped it: it finalized 13 or 14
of 20 documents per ratio without reading a single page.

Gemini bills every sheet at about 1,090 image tokens whatever its pixel size,
so shrinking the image does not compress anything for Gemini: all three of its
conditions are about 7.4x by its own count, differing only in source
resolution. n = 20 per cell, one run: the selection gaps against Sonnet stand, 5 to
10 pp accuracy differences do not. A post-cutoff corpus would make accuracy
measure reading.

## Findings

1. **Selection separates Sonnet 5 from the other three.** First-ranked page
   correct at 5x/10x/15x: Opus 100/90/70%, Gemini 100/90/75%, Muse 90/95/50%,
   Sonnet 75/15/5%. (`score.py` sel@1, Rounds 1 and 2)
2. **HotpotQA accuracy is mostly memory for all four.** Closed-book, no
   document: Opus 17/20, Gemini 16/20, Sonnet 15/20, Muse 14/20. Only 3 to 6
   documents per model test reading. (closed-book arms)
3. **Muse and Gemini read the most from the image alone.** Closed-book misses
   answered from the sheet before any expansion: Muse 5/6, 5/6, 2/6; Gemini
   3/4 at all three ratios; Opus 2/3, 1/3, 0/3; Sonnet 1/5, 0/5, 0/5.
   Image-only accuracy: Muse 95/95/80%, Gemini 90/95/95%, Opus 95/90/85%,
   Sonnet 70/55/55%. (Round 2)
4. **Expansion helps only the model that cannot read the sheet.** Final minus
   image-only: Sonnet +20/+15/+10 pp, Muse 0/0/+5, Gemini 0/0/0, Opus
   −5/0/−5. (Rounds 1 and 2)
5. **An unreadable sheet hurts Sonnet below knowing nothing.** At 10x and 15x,
   4 documents Sonnet answered right closed-book went wrong image-only (55%
   against 75%). Muse and Opus never lost a closed-book answer to the image;
   Gemini once, at 5x. (Round 1)
6. **Gemini's image cost is fixed near 1,090 tokens per sheet.** Its reported
   image tokens were 1,087/1,102/1,073 for sheets Claude bills at 1,596/801/521.
   A pixel-shrinking compression knob therefore saves Gemini nothing below
   ~1,100 tokens and over-compresses above it; by its own count it saw ~7.4x
   at every label. Muse's input tokens tracked the pixels (1,836/999/754 per
   commit turn, prompt text included). (Round 2)
7. **Gemini rarely expands; Muse always does.** Documents finalized with zero
   expansions: Gemini 14/13/14 of 20, Muse 0. Muse spent 268k reasoning tokens
   over 175 calls (17.9 min at concurrency 2); Gemini 24k over 84 calls
   (2.7 min). (Round 2)
8. **At 15x, selection runs on paragraph titles.** The bracketed `[Title]`
   headers survive as dark blobs at 136 px page width and body text does not
   (`examples/h00-r15.png`). An inference from the images; the ledgers do not
   record what a model looked at.

| model | ratio | closed | image-only | expand | exact (expand) | sel@1 | expHit | #exp | ECR* |
|---|---|---|---|---|---|---|---|---|---|
| Opus 5.5 | 5x | 85 | 95 | 90 | 65 | 100 | 100 | 1.25 | 3.7 |
| Opus 5.5 | 10x | 85 | 90 | 90 | 65 | 90 | 100 | 1.45 | 5.6 |
| Opus 5.5 | 15x | 85 | 85 | 80 | 55 | 70 | 95 | 1.75 | 6.3 |
| Sonnet 5 | 5x | 75 | 70 | 90 | 65 | 75 | 85 | 1.90 | 3.3 |
| Sonnet 5 | 10x | 75 | 55 | 70 | 40 | 15 | 40 | 2.50 | 4.5 |
| Sonnet 5 | 15x | 75 | 55 | 65 | 35 | 5 | 30 | 2.60 | 5.4 |
| Gemini 3.8 Flash | 5x | 80 | 90 | 90 | 60 | 100 | 30 | 0.40 | 4.6 |
| Gemini 3.8 Flash | 10x | 80 | 95 | 95 | 60 | 90 | 35 | 0.40 | 8.7 |
| Gemini 3.8 Flash | 15x | 80 | 95 | 95 | 60 | 75 | 30 | 0.35 | 13.1 |
| Muse Spark 1.3 | 5x | 70 | 95 | 95 | 55 | 90 | 100 | 1.70 | 3.4 |
| Muse Spark 1.3 | 10x | 70 | 95 | 95 | 55 | 95 | 100 | 1.65 | 5.3 |
| Muse Spark 1.3 | 15x | 70 | 80 | 85 | 55 | 50 | 80 | 2.35 | 5.2 |

Percentages over 20 documents. "closed" is the same 20 questions with no
document, repeated on each row. expHit counts only documents where the model
expanded, over all 20, so Gemini's low figure is mostly unexpanded documents.
*ECR uses Claude's image-token formula for every model; by Gemini's own count
its sheets cost ~1,090 tokens at every ratio.

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
Claude's formula `ceil(W/28) * ceil(H/28)`. Text tokens are chars/4. Achieved:
5.06x (1,596 image tokens, pages 251 x 220 px), 10.07x (801, 173 x 152),
15.49x (521, 136 x 119). Sheets stay under Claude Code's 2000 x 2000 Read-tool
cap. Every model received the same PNG files.

**Claude arms.** One `general-purpose` subagent per model x ratio x half (10
documents), 12 in all, via the Agent tool's `opus` and `sonnet` aliases; every
agent logged its served model id (`claude-opus-5-5`, `claude-sonnet-5`). Per
document the agent Reads the sheet, runs `lens.py commit` (image-only answer,
ranked pages), `lens.py expand` up to 3 times, then `lens.py final`. `lens.py`
refuses expand before commit and after final, caps expansions, and ledgers
every call; page texts sit zlib-compressed in `pages.bin`. Prompts carried
`[no-context]` so the workspace's context hook could not append this session's
transcript, which held gold answers.

**API arms.** `api_driver.py` plays `lens.py`'s role for models called over an
API, writing the same ledger events. Gemini: `gemini-3.8-flash` through the
Cloudflare AI Gateway, native multi-turn, default thinking and media
resolution, `maxOutputTokens` 8192. Muse: `muse-spark-1.3` through
claude-workspace `scripts/muse.py` `ask()`, the only permitted route
(`docs/muse-contributor.md`); the gate put every call on the contributor tier
for the declared source `public-url:` HotpotQA. `ask()` is single-turn, so each
turn re-sends the sheet and the transcript so far. Default reasoning effort,
`max_tokens` 32768 (8192 ran out on reasoning in the smoke test). The commit
turn may declare a final answer immediately instead of expanding; both prompts
allowed stopping as soon as the model could answer. The API arms did not log a
served model id for Gemini; Muse reported `muse-spark-1.3-contributor`.
Closed-book arms: one prompt listing all 20 questions (API) or one subagent
(Claude).

**Scoring.** `score.py`: HotpotQA normalisation; a match is exact, or
containment with the shorter string at least 4 characters, or a hand-adjudicated
alternate applied identically to every arm and model (h02 SSIH/ASUAG, h05 Ghana
national (football) team, h06 Bernard Law Montgomery, h09 "2012 season" = the
Rams' 75th, h10 "director"; reasons in `score.py`). Strict exact-match is
reported alongside. sel@1: first committed page is a gold page. expHit: any
expanded page is a gold page. ECR: document text tokens over sheet tokens plus
expanded text tokens.

**Controls not run.** No full-text arm, so no text upper bound. No repeat
runs, so no variance estimate. Gemini's default media resolution was not
varied. One Sonnet 5x agent reports opening `lens.py`; page texts are
compressed and it could not have read answers from it.

**Cost.** Claude arms 1.34M subagent tokens (Opus 479k, Sonnet 682k,
closed-book 119k, pilot 60k), about 32k of each agent being the harness floor.
Gemini 127k input, 3k output, 24k thinking tokens. Muse 305k input, 279k output
(268k reasoning) on the contributor tier, about $0.09 at $0.10/$0.20 per
million.

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

### Round 2 — 2026-09-24

Asked: *"could we try Gemini and Muse?"*

Built `api_driver.py`. Smoke test on h01 at 10x: Gemini answered from the
sheet and finalized without expanding, reporting 1,102 image tokens for a sheet
Claude bills at 801. Muse's first call ran out of an 8,192-token cap on
reasoning alone (8,189 tokens, `finish: length`); at 32,768 it committed after
11,041 reasoning tokens in 73 s, expanded one page, and finalized.

Full run: both models over all 60 document-ratio pairs, concurrency 2 each,
plus closed-book, as one resumable background job. 0 errors, 0 unparseable
replies. Scoring surfaced two paraphrases the matcher missed ("Bernard Law
Montgomery", "Ghana national team"), added to the adjudication list for every
model; neither changed a Claude number.

Reading at the time. The expectation that Opus would lead on fidelity held
only against Sonnet: Gemini matched Opus on selection and Muse matched it at
5x and 10x, and both read more beyond memory than Opus did. Gemini's fixed
image budget means its "15x" row is really a ~7.4x view of a low-resolution
source; that it still answered 95% there, and 3 of its 4 closed-book misses, is
the strongest reading result in the experiment, from the model that expanded
least.
