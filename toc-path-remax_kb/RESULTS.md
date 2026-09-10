# Heading paths in remax_kb chunks — no retrieval gain

Prepending a chunk's heading path to its indexed text does not improve
`remax_kb` retrieval. Fused Recall@1 moves from **0.592** (production chunker)
to **0.586** with the heading path added, a difference of **−0.0063
[−0.0325, +0.0188]** over 799 queries. The interval rules out anything better
than about +1.9 points.

The harness can see heading text when heading text is what the query asks for.
With the heading path itself as the query, the same arm gains **+0.2450
[+0.1950, +0.2950]** Recall@1. So the null is a measurement, not a blind spot. [`PLAN.md`](PLAN.md) carries the
pre-registration, committed before any measurement ran.

## Origin

arXiv:2609.03874 (STAIR/SearchTome, IBM Research India) reports Recall@1 82.6
for a system shown a book's Table of Contents, against 59.5 for BM25 and 68.7
for a dense retriever, and reads the gap as evidence that global structure
helps retrieval. Its baselines index section *content*; only the proposed
system is shown section *titles*, and no row in any of its tables varies that
axis. This runs the cheap version of its idea (put the heading path in the text you
already index, no finetuning) on a retriever that was not built for it.

Internally it also closes a question `lexical-kb/RESULTS.md` left open: that
experiment chose "metadata stays structured, not folded into indexed text" and
recorded title-term boosting as deferred and unquantified.

## Method

`remax_kb` feeds one field to both retrieval modalities: `pack_v2.py:347`
embeds `chunk.text` and `pack_v2.py:592` builds BM25 over the same strings,
while `pack.py:45`'s `default_chunker` never captures a heading. One string
change reaches both.

Four arms over 95 `SKILL.md` files (782 KB, 1,728 headings, 1,871 chunks):

| arm | indexed text | chunks |
|---|---|---|
| A | chunk body, today's `default_chunker` | 1,871 |
| B0 | `source_path` + body | 1,871 |
| B1 | heading path + body | 1,871 |
| B2 | heading path + body, chunks never span a heading | 2,675 |

A, B0 and B1 share byte-identical boundaries: the instrumented chunker
reproduces `default_chunker`'s output exactly on all 95 files, asserted at build
time, so arm A is production rather than a reimplementation of it. B0 is the
leakage control, because a SKILL.md heading path contains the skill's name and a
query about that skill would match the name for reasons unrelated to structure.

799 queries over 400 gold chunks, written by ten subagents shown only chunk
bodies, never the heading path and never the purpose of the experiment. Retrieval
runs through `remax_kb`'s own writer and reader, so the fusion under test is the
one that ships. Embedder: `JinaQ4ONNXEmbedder` (official q4). Confidence
intervals are a paired bootstrap over queries, 10,000 resamples.

## Results

Recall@1 / Recall@10:

| arm | dense | bm25 | fused |
|---|---|---|---|
| A | 0.504 / 0.887 | 0.554 / 0.885 | **0.592** / 0.940 |
| B0 | 0.488 / 0.885 | 0.552 / 0.884 | 0.593 / 0.949 |
| B1 | 0.481 / 0.871 | 0.561 / 0.896 | 0.586 / 0.946 |
| B2 | 0.516 / 0.872 | 0.546 / 0.879 | 0.598 / 0.932 |

Every fused Recall@1 contrast against A straddles zero: B0 +0.0013
[−0.0238, +0.0263], B1 −0.0063 [−0.0325, +0.0188], B2 +0.0063 [−0.0275, +0.0401].
B0 ≈ B1 rules out a filename-leakage explanation for the absent effect.

B2 is worse than level once you look past Recall@1. Its fused Recall@3 is
**−0.0300 [−0.0551, −0.0063]** against A, and its BM25 Recall@3 and Recall@5 are
−0.0375 [−0.0626, −0.0138] and −0.0288 [−0.0513, −0.0075]. Cutting chunks at
heading boundaries produced 43% more, smaller chunks and split text that the
lexical index wanted together. Its flat Recall@1 was the least informative slice
of it, and its gold sets average 1.69 chunks per query — an asymmetry that
favours B2 and still leaves it losing.

Baseline fused Recall@1 of 0.592 sits well below the 0.90 saturation stop, so
there was headroom for an effect to appear in.

## Per-modality results

The two modalities move in opposite directions, each significantly, and cancel
at the fused level:

- **dense** A→B1: MRR **−0.0200 [−0.0378, −0.0019]**, excluding zero. Recall@1
  −0.0238 [−0.0513, +0.0038] and Recall@10 −0.0163 [−0.0350, +0.0025] point the
  same way without individually clearing it.
- **bm25** A→B1: Recall@10 **+0.0113 [+0.0013, +0.0225]**, an interval clear of
  zero in the other direction. Recall@1 +0.0063 [−0.0150, +0.0275], MRR +0.0041
  [−0.0087, +0.0169].

Both fused contrasts for B1 straddle zero at every k.

A heading path repeated across every chunk of a section gives BM25 a few extra
matchable terms, and gives the embedder the same 30–60 characters of text in
every vector of that section, pulling those vectors toward a shared centroid
and blurring exactly the within-section discrimination Recall@1 needs. The
prefix is worth slightly more to a lexical index than it costs, and worth less
to a dense one than it costs. Fusion lands on neither.

`PLAN.md` registered this shape of outcome but guessed the signs backwards. It
predicted the dilution would hurt BM25 and named a dense gain as the likely
positive. The opposite happened.

## Exploratory subgroups

Not pre-registered; reported for the mechanism, not as findings. B1 can only
help where the heading names something the body does not, so the split is by
how many heading tokens are absent from the chunk body (mean 4.77; 73% of chunks
gain ≥3, 12.2% gain none).

No subgroup interval excludes zero. The dense loss deepens monotonically with
the number of foreign tokens prepended — −0.008 (0 new), −0.035 (1–2), −0.029
(≥3), −0.067 (≥6), which is what the centroid-drift account predicts, on n = 120
at the far end.

## Positive control

Heading path as the query, Recall@1, all three modes separating cleanly:

| arm | dense | bm25 | fused |
|---|---|---|---|
| A | 0.255 | 0.258 | 0.320 |
| B1 | 0.485 | 0.477 | 0.565 |

## Limits

One corpus, one embedder, one chunk size, English markdown with hand-written
headings. Nothing here touches STAIR's parametric-index claim, which finetunes a
model per corpus; this measures only whether the structure signal STAIR puts in
a prompt is worth anything handed to an ordinary retriever. A null here is
evidence about heading text in `remax_kb`, not evidence that STAIR's result is
wrong.

The 53.4% of gold chunks whose body already contains one of their own heading
strings (`default_chunker` keeps `## Foo` lines as body text) means B1
measures the incremental value of a *consistent full path*, not of heading text
against none.

## Recommendation

Leave `default_chunker` alone, and do not adopt heading-bounded chunking. The
prefix costs 15% more indexed characters per chunk and returns nothing fused;
the boundary change costs 43% more chunks and loses ground at Recall@3.

The one durable signal is that the heading path is worth something to the
lexical index and costs something to the dense one. That argues for BM25F-style
field boosting rather than string concatenation, if a future corpus with deeper
or more discriminative headings than a SKILL.md tree makes it worth revisiting:
a weighted lexical field collects the Recall@10 gain without splicing the same
30–60 characters into every vector of a section. `lexical-kb/RESULTS.md` already
deferred exactly that experiment, and this result is the argument for running it
that way rather than the way tried here.

## Files

| file | role |
|---|---|
| [`PLAN.md`](PLAN.md) | pre-registration |
| [`ERRORS.md`](ERRORS.md) | three errors, how each was caught |
| `build_corpus.py` | four arms + byte-identity assertions against `default_chunker` |
| `make_prompts.py` | gold sampling, blind prompt emission |
| `consolidate_queries.py` | batch merge, near-verbatim rejection |
| `run_arms.py` | `.kb` build + scoring |
| `rescore_modalities.py` | per-modality rescore (see `ERRORS.md` #2) |
| `analyze.py` | paired bootstrap |
| `subgroup.py` | exploratory splits |
| `recheck.py` | asserts this file's numbers against `results.json` |
| `results.json`, `analysis.json`, `subgroups.json` | data |
