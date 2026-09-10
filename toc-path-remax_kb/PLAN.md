# Heading-path prefixes in remax_kb chunks — pre-registration

**Written 2026-09-09, before any measurement.** Registered so the reading of
the result cannot be chosen after seeing it.

## Question

Does prepending a chunk's heading path to its indexed text improve retrieval in
`remax_kb`?

Origin: assessing arXiv:2609.03874 (STAIR/SearchTome, IBM Research India). That
paper reports Recall@1 82.6 for a system that sees a book's Table of Contents
against 59.5 for BM25 and 68.7 for a dense retriever, and attributes the gap to
structure. Its baselines index section *content*; only the proposed system is
shown section *titles*. No row in any of its tables varies that axis, so the
paper does not separate "structure helps" from "only one system saw the
titles". The cheap version of the same idea needs no finetuning: put the
heading path in the text you already index. That is what this measures.

Second origin, internal: `lexical-kb/RESULTS.md` ("Design decisions") chose
**metadata stays structured, not folded into indexed text**, and recorded
title-term boosting as deferred and unquantified. This closes that.

## Ground truth about the target

`remax_kb/pack_v2.py` embeds `[c.text for c in pending_adds]` (line 347) and
builds BM25 over `[rows[i].chunk.text ...]` (line 592). One field feeds both
retrieval modalities. `remax_kb/pack.py::default_chunker` (line 45) splits on
blank lines, packs sentences to ~500 chars, and writes
`meta={"source_path": ...}` — no heading is captured, and a chunk starting
mid-section carries no indication of which section it is in.

So the intervention is a change to one string, and both modalities inherit it.

## Arms

All four arms share the same corpus, the same embedder, the same BM25
parameters, the same fusion settings, and the same query set. Only the indexed
text differs.

| arm | indexed text | isolates |
|---|---|---|
| **A** | chunk body (current `default_chunker`) | baseline |
| **B0** | `source_path` + body, identical boundaries to A | filename leakage |
| **B1** | heading path + body, identical boundaries to A | heading text |
| **B2** | heading path + body, chunks never span a heading | heading text + boundaries |

A→B1 is the paper's claim reduced to its cheapest form. B1→B2 is what the
boundary change adds on top. B0 exists because the heading path of a SKILL.md
chunk contains the skill's name, and a query about that skill would match the
name for reasons that have nothing to do with document structure; if B0 ≈ B1,
the effect is filename leakage and I will report it as such.

## Corpus

The 95 `SKILL.md` files under `/mnt/skills/user/` — 782 KB, 1,728 markdown
headings across three real nesting levels (468 h1, 779 h2, 469 h3). Chosen
because it is an actual `remax_kb` corpus (cf. `examples/build_claude_docs_kb.py`),
because its heading hierarchy is genuine rather than induced, and because it has
heavy topical competition between chunks (charting vs charting-vega-lite,
creating-skill vs skill-creator, exploring-codebases vs orienting-codebases,
remembering vs session-memory). Topical competition is required: per
`lexical-kb/RESULTS.md`, a topically disjoint corpus puts the baseline at rank 1
on nearly every query and leaves no headroom for any intervention to show in.

## Query construction

Queries are written by subagents that receive **only the chunk body**. They
never see the heading path, the source filename, or any statement of what this
experiment is testing. Gold label is the chunk the body came from. The same
query set scores all four arms.

The blindness is load-bearing in two directions. A generator that saw headings
would write queries echoing them and hand B1 a win by construction. A generator
told it was testing structure would write to the inferred grader (ops
`eval-realism`). Neither is available to it.

No LLM-free construction was used: sampling a sentence from the body makes the
query a substring of the gold chunk and puts every arm at ceiling, which fails
the diagnostic check below before it starts.

## Metrics

Recall@1, @3, @5, @10 and nDCG@3, reported three ways: **dense only, BM25 only,
and fused**. The modalities are reported separately because the heading path
could plausibly help one and hurt the other — a heading repeated across every
chunk of a section lowers those terms' IDF for BM25 while pulling every dense
vector in the section toward the section's topic — and `remax_kb` ships the
fusion, so the fused number is the one that decides deployment.

Reporting @5 and @10 alongside @1 follows this repo's own finding that a
lexical ranker is a good shortlister and a poor router, and should not be judged
by top-1 alone (`METHODS.md`, `gh-mcp-regex-fit`).

Confidence intervals: paired bootstrap over queries, 10,000 resamples, on the
per-arm difference. A difference whose 95% interval contains zero is reported as
no effect, not as a trend.

## Pre-registered readings

**If heading paths do nothing**, R@1(B1) − R@1(A) is 0 with an interval
straddling zero, and B1 ≈ B0 ≈ A across every k and every modality.

**The metric can read negative and I expect it might.** Prepending 30–60 chars
of text that is identical across every chunk in a section is not a free
addition: it dilutes BM25 IDF for the heading's terms and shifts every vector in
a section toward a common centroid, which blurs *within-section* discrimination
exactly where retrieval needs it most. A negative delta is a real outcome of
this design, not a bug, and gets reported as the finding if it happens.

**Direction I expect**: small positive on fused R@1, larger on queries whose
body is generic, possibly negative for BM25 alone. Confidence: low. Recorded
now so it cannot be revised into a prediction after the fact.

## Controls, scheduled before the arms rather than after

**Positive control — can this harness see a heading effect at all?** Score a
query set consisting of the heading paths themselves. B1 must beat A by a wide
margin on it. If it does not, the measurement cannot detect the thing it is
looking for and no arm result from it is worth anything; stop and fix the
harness.

**Negative control — is the effect just the filename?** B0 above.

**Saturation check — is there headroom?** If arm A's fused R@1 exceeds 0.90,
the corpus is too easy to show a difference, per the `lexical-kb` finding.
Report that and stop rather than reading noise off a ceiling.

## What this cannot show

One corpus, one embedder (`JinaQ4ONNXEmbedder`, official q4), one chunk size
(~500 chars), English markdown with hand-written headings. It does not touch
STAIR's parametric-index claim, which requires finetuning a model per corpus —
this measures only whether the structure signal that STAIR puts in a prompt is
worth anything when handed to an ordinary retriever instead. A null here is
evidence about heading text in `remax_kb`, not evidence that STAIR's result is
wrong.

## Stopping conditions

1. Positive control fails to separate B1 from A → harness broken, stop.
2. Arm A fused R@1 > 0.90 → no headroom, report and stop.
3. Otherwise run all four arms and report as they come out.
