# bts-coordinates — growing coordinate systems for cross-field prior-art search

**Status: negative. The mechanism under test shows no effect, and the headline
number from the first pass was destroyed by its own null control.**
Pre-registration in [`PLAN.md`](PLAN.md), written before any data. The
adversarial pass required by that pre-registration is what killed the result,
which is the outcome it was scheduled for.

## The mechanism under test

Large Discovery Models (arXiv:2608.15669, §4.2) fit their reward surrogate over
a **growing** feature set: a newly discovered mechanism enters the value model
as a new coordinate, and the model is refit. Their ablation freezes the feature
set and converges to a worse plateau.

We have carried an objection since May (memory `a8b97f70`): if novelty appears
as a new *direction* in representation space, no search over a fixed flat
embedding reaches it. We published that objection and never tested it, having
no mechanism to test. LDM supplies one.

Task: recover a known cross-field prior-art target from a pool of paper titles,
given the problem stated in the other field's vocabulary. Ground truth is the
ms13 campaign, which reached Doerr 2004 only at phase 7 after roughly 25 hours.

One encoder (bge-small-en-v1.5) does all the work in every arm. A named axis is
a short phrase; a document's coordinate on it is the cosine between the title
embedding and the axis-phrase embedding. Pool, document text and encoder are
identical across arms, so an arm difference cannot come from a stronger model
or more text. Only the basis changes.

| arm | how it ranks |
|---|---|
| A | cosine to the raw query text |
| N | cosine to the blind-extracted signature paragraph |
| mean-axes | mean coordinate across the named axes |
| max-axes | max coordinate across the named axes |
| best-single-axis | the most favourable single axis, hindsight-assisted |
| C | Bayesian-ridge surrogate over axes plus UCB, batch 5, **frozen** axis set |
| B | as C, but widens by 8 further axes whenever a batch returns all-negative |

B against C is the transplant. Everything else is context for it.

## Results

Rank of the target out of the pool. `reads` is reads-to-hit under a
target-only binary oracle, capped at 200. Random expectation is 550 and 620.

| case | n | A | N | mean-axes | max-axes | best-axis | C reads | B reads |
|---|---|---|---|---|---|---|---|---|
| P1_PRE   | 1101 | 496 | 175 | 122 | 38  | 5  | 24   | 19   |
| P1_PARA1 | 1101 | 113 | 347 | 17  | 36  | 9  | 183  | none |
| P1_PARA2 | 1101 | 190 | 236 | 97  | 277 | 29 | none | 195  |
| P1_PARA3 | 1101 | 164 | 122 | 1   | 91  | 5  | 1    | 1    |
| P2       | 1241 | 367 | 110 | 183 | 66  | 25 | none | none |

`P1_PARA1-3` replace only the target's own title with a paraphrase avoiding
"discrepancy" and "unimodular". Pool, axes, query and encoder are untouched.

### Arm B against arm C

One win each, two uninformative ties, sign test p = 1.0.

| case | C | B | |
|---|---|---|---|
| P1_PRE | 24 | 19 | B by 5 reads |
| P2 | none | none | both censored |
| PARA1 | 183 | none | C |
| PARA2 | none | 195 | B |
| PARA3 | 1 | 1 | tie, surrogate never ran |

The pre-registration set 2x as the threshold for calling a difference; 19
against 24 is 1.26x. Both surviving numbers sit against the 200-read cutoff,
where which arm squeaks under is arbitrary. B also spends extra LLM calls to
generate up to 24 more axes, and there is no "C plus 8 arbitrary axes" control,
so growth is not separated from simply having more axes.

**No comparison here shows growth working.** The pre-registered null
holds. The prior recorded in `PLAN.md` was B best; that prior is wrong.

Both arms are also beaten by doing nothing: on P1_PRE a static ranking puts the
target at 5 with zero reads, while the surrogate costs 19 to 24.

### The null control that removed the headline

The first pass reported best-single-axis = 5 of 1101 as evidence that a named
axis exposes a direction the flat space does not. The adversarial pass demanded
the missing null: min over 12 noisy cosines is an order statistic, so what does
it read when the axes are meaningless?

Twelve titles drawn at random from the pool, used as axis phrases, 20 draws:

| | P1 (n=1101) | P2 (n=1241) |
|---|---|---|
| real blind axes, best-of-k | 5 | 25 |
| **random axes, best-of-k, median** | **10** | **41** |
| random axes, best-of-k, best draw | 2 | 6 |
| real blind axes, mean | 122 | 183 |
| random axes, mean, median | 303 | 305 |
| other case's axes, best-of-k | 97 | 20 |

Random axes land the target in the top 10 about half the time on P1. The real
axes' 5 sits inside that distribution, near its 20th percentile. **The
best-of-k result is indistinguishable from chance and is withdrawn.** On P2 the other
problem's axes (20) beat the matched blind ones (25), which is the same
conclusion from a second direction.

What survives is narrower: **mean-over-axes beats both the raw query and the
median random-axis draw in all five configurations** (122 against 496 and 303;
183 against 367 and 305). A spread of short phrases covers a concept's lexical
realisations better than one long query does. That is multi-probe query
expansion with rank fusion, which is prior art, and which `METHODS.md` already
records as losing three times in this repo. Here it wins. The tension is
recorded rather than resolved.

### Rank spread across the four target wordings

Across four wordings of one document, everything else fixed:

| statistic | min | max | spread |
|---|---|---|---|
| A | 113 | 496 | 383 |
| N | 122 | 347 | 225 |
| mean-axes | 1 | 122 | 121 |
| max-axes | 36 | 277 | 241 |

Reads-to-hit moves from 1 to never on the same paper. Rewording one document
out of 1101 moves it by hundreds of ranks, so the wording of the target
dominates every effect this experiment set out to measure.

Note also that arm A improves on all three paraphrases (496 to 113, 190, 164).
The paraphrases are easier instances, not neutral ones, so the spread is not a
clean noise estimate either.

Z-scoring each axis across the pool before taking the max was tried as a fix,
so that a document extreme on one axis wins regardless of that axis's scale. It
made things worse: spread 324 against 241. Recorded as a failed fix.

### Query length against document length

Arm A embeds a 1300-character problem statement and compares it to eight-word
titles. Averaging twelve *random* pool titles as probes (median 303) beats it
(496). A probe made of random titles should not beat the actual query if the
query were a fair probe, so arm A's poor showing is at least partly a
long-query-against-short-document mismatch.

This matters for what can be claimed: **the experiment cannot separate "bridges
are not near their endpoints in a flat embedding" from "a long problem
statement is a poor probe against short titles."** Our published claim is not
confirmed here, and it is not refuted either.

### Score levels on the negative control

N3, a fabricated pseudo-problem with no correct answer, produces a top
candidate at cosine 0.7719. The real P1 target scores 0.7295 on its best axis.
The fabricated problem's best match outscores the real one, so absolute
similarity does not indicate whether an answer exists. Any acquisition function
built on these cosines is fitting a surrogate to a space where the true
positive is not the maximum, which predicts the B-against-C null independently.

### The leak in the inherited test case

Issue #179's P1 case is the lemma as written *after* the reduction was found,
so it contains "totally unimodular" and "lindisc", which are the target's own
words. Handed that text, arm A improves from 496 to 111 with nothing else
changed. The inherited test case measures whether we can find a paper whose
title we have already been given, and a variant with that vocabulary stripped
was written for this run.

## The blind extraction step

A subagent was given the P1 problem with every discrepancy-side term removed,
told to use no tools, and asked to name candidate fields and search terms. It
made **zero tool calls** and returned, as its highest-confidence query:

> linear discrepancy totally unimodular matrices

That is the verbatim title of Doerr 2004. Its structural description also
derived "network matrix, hence totally unimodular" unprompted, in under two
minutes. The July P2 extraction, blind under the same conditions, converged on
"series-parallel graph theory", a literal substring of that target's title.

This is the only part of the run that clearly did something, and it is upstream
of every mechanism the experiment was built to test. It is a single LLM call
with no corpus, no surrogate, no acquisition function and no growth.

Its own caveat, from the adversarial pass: zero tool calls rules out leakage
through the transcript, not through the shared prior. An LLM asked to name axes
for rounding a fractional flow emits discrepancy vocabulary whether or not it
has seen Doerr. Blindness to the title is not independence from the target. The
available claim is the weak one: **the bridge was already in the model's prior,
and this pipeline had not asked it to name the field.** That matches the
operating envelope memory `3d35c1e0` recorded in May, reached by another route.

## Limitations

- Title-only ranking, forced by the target's abstract being elided by its
  publisher. Ranking 1101 documents on eight-word strings is harsh and is
  plausibly the source of the paraphrase instability.
- n=1 per case, three cases, no seeds. Arms B and C are deterministic given the
  pool, so 19 against 24 has no run-to-run noise, but it is still one draw of
  one case.
- The oracle is target-only and binary, so the surrogate sees no positive label
  before the hit and can only learn to move away from what it rejected. LDM's
  surrogate learns from graded rewards. This is a weak instantiation of their
  mechanism, and a fairer one might do better.
- P1's pool contains the target by injection, since Doerr is not on arXiv. P1
  measures ranking rather than retrieval.
- The 12 axes are not orthogonalised and all derive from one paragraph, so
  their effective dimensionality is likely 2 or 3, not 12.

## Infrastructure findings that contradict the July record

`claude-workspace` PR #180 killed the prior-art-probe because arXiv keyword
search failed 6 queries out of 6. Probed again 2026-09-01:

| endpoint | July 2026 | now |
|---|---|---|
| arXiv keyword search | failed 6/6 | **works**; a blind July query returns the P2 target at rank 77 |
| S2 `/paper/search` | dead, 429 | still dead, 429 on first call |
| S2 `/paper/{id}`, `/references` | works, 429 under load | unchanged |
| Doerr `/references` | elided by publisher | unchanged, `data: null` |
| HuggingFace weight download | recorded blocked (memory `6b190772`) | **works** here; `us.aws.cdn.hf.co` returns 206 |

Two of five inherited constraints are no longer true. The July kill was correct
on its evidence and is now out of date on two counts. Re-probe before
inheriting an infrastructure conclusion.

Depth mattered more than query quality: the first P2 pool missed its target
because each query was capped at 40 results, and the target sits at rank 77 of
a query that was already in the blind set. Fixed to 100 uniformly before any
arm ran.

## Measurements that would settle the open parts

- Graded rewards instead of the binary target-only oracle, which is the version
  of LDM's surrogate that the paper actually validates.
- Full-text or abstract-level documents instead of titles, which would test
  whether the paraphrase instability is a title-length artifact.
- More cases. Two targets cannot separate a method from a coincidence.

## Reproduction

```
python3 fetch_corpus.py P2 cache/queries_P2.json   # and P1, N3
python3 run_AN.py            # arms A, N
python3 run_C.py             # arm C
python3 run_B_and_control.py # arm B + paraphrase control
python3 null_control.py      # random-axis and swapped-axis nulls
```

`embed.py` needs `cache/bge-small/` from HuggingFace, about 133 MB. Encoder
settings, which `METHODS.md` requires be recorded or the cache is unusable:
**bge-small-en-v1.5 ONNX, CLS pooling, L2-normalised, max_length 512, query
prefix "Represent this sentence for searching relevant passages: " on queries
and axis phrases only, documents encoded bare.**
