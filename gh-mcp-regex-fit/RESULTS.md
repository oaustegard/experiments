# gh-mcp-regex-fit — fitting the regex router instead of writing it

[`monad-bsky`](../monad-bsky/RESULTS.md) ended with twenty hand-written regex
rules routing an 18-tool Bluesky catalogue at **0.833**, beating a 45M
tool-calling model's best configuration (0.722) and its five-tool oracle ceiling
(0.778), at 0.022 ms against 324 ms. The rules were written after reading that
eval's failures, so the writeup could only say the number was fitted "to an
unknown degree", and the post-mortem found most residual errors were rule bugs
rather than genuine ambiguity — `reposters` missing from an alternation, `\bup\b`
in the status rule swallowing "dig **up** posts".

The obvious next move is to stop writing the rules. This builds the harness that
searches for them instead, against a real and much larger catalogue: the **58
GitHub MCP tools** exposed to a Claude Code on the Web session, of which 50 have
upstream schemas, expanding to **79 routing targets** once the seven
`method`-enum dispatchers are counted properly.

Headline: **the fitted decision list loses to hand-written rules on every
held-out split, and the harness's most useful output is the diagnosis of why.**
Best fitted arm reaches 0.239 on a held-out phrasing family and 0.351 on
hand-authored queries, against 0.546 and 0.486 for rules written by hand from
the schemas. No regularisation setting closes the gap. Meanwhile the structural
cue layer that carried the Bluesky catalogue routes **0.013–0.049** here on its
own, and only **13.5%** of hand-authored queries even contain the `owner/repo`
their tool requires, against 61–77% of generated ones.

## Why this catalogue is not the Bluesky one

`regex_only.py` worked because 9 of its 20 rules carried a structural
precondition and three fired on structure *alone* — a bare post URI is a thread
read, a bare handle is a timeline read. The catalogue was signposted by argument
shape.

This one is not. `owner` and `repo` appear in **40 of the 50** tools, so the
dominant structural cue discriminates nothing. The genuinely discriminating
parameters are sparse: `pullNumber` in 8 tools, `query` in 6, `issue_number` and
`path` in 5, `branch` in 4, and `sha`, `ref`, `run_id`, `tag`, `threadID` in one
or two each.

It also adds a level the Bluesky catalogue had no analogue for. Seven tools take
a required `method` enum and dispatch inside themselves:

| dispatcher | methods |
|---|---|
| `pull_request_read` | 9 — `get`, `get_diff`, `get_status`, `get_files`, `get_commits`, `get_review_comments`, `get_reviews`, `get_comments`, `get_check_runs` |
| `actions_get` | 6 | 
| `actions_run_trigger` / `issue_read` / `pull_request_review_write` | 5 each |
| `actions_list` | 4 |
| `issue_write` | 2 |

Naming `pull_request_read` without naming the method is not a route, so the
label space is 43 plain tools + 36 methods = **79**.

The catalogue is built by `catalogue.py` from `github/github-mcp-server`'s own
committed schema snapshots (`pkg/github/__toolsnaps__/`, 117 of them, at
`8ec62491`), so it is the bytes an MCP client actually receives. Eight of the 58
session tools — `subscribe_pr_activity`, the auto-merge pair, the Copilot job
tools, `get_check_run`, `run_secret_scanning` — have no upstream snapshot and are
excluded.

## What "fitting" means here

`fit.py` induces an ordered decision list by greedy precision-constrained
covering, the CN2/RIPPER shape:

```
while some candidate clears (min_precision, min_coverage):
    pick the highest-scoring candidate, ties broken by coverage
    append (literals -> majority label)
    delete every row it covered, right or wrong
rows never covered are ABSTAINS, not a fallback label
```

Deleting *wrongly* covered rows is what makes it a decision list rather than a
rule set, and it reproduces on purpose the property `needle-bsky/two_stage.py`
found the hard way: an earlier stage's error is unrecoverable.

Candidates are conjunctions of at most two literals with at most one negated,
enumerated from the surviving rows so coverage is never zero. Literals come from
three feature families:

| family | example | source |
|---|---|---|
| `cue:` | `cue:pr_ref`, `cue:sha40`, `cue:path` | 23 structural detectors in `cues.py`, each derived from a parameter that exists in the catalogue |
| `tok:` / `bi:` | `tok:diff`, `bi:sub_issues` | query tokens, optionally restricted to the 613 words the catalogue itself uses |
| `ov:` | `ov:pull_request_read::get_diff` | the top-3 labels by IDF-weighted overlap between the query and the label's schema text |

Rows are bitsets and label purity is computed by iterating set bits, which keeps
a fit to ~100 s over 988 rows and ~650 features on one core.

## Three splits, and why there are three

| split | what it is | n routable | n off-topic |
|---|---|---|---|
| **family A** | the training templates | 948 | 40 |
| **family B** | same 79 intents, deliberately different verbs and sentence shapes, disjoint entity pool | 948 | 40 |
| **wild** | hand-authored, unconstrained phrasing, no template | 74 | 15 |

Family B is the check `monad-bsky` ran for its hand-written rules (0.833 fitted →
0.824 unseen). `wild.jsonl` exists because families A and B share an author with
the rules that would be compared against them; it was written and **committed
before `fit.py` existed**, so the git history carries the provenance rather than
a claim about intent.

## Results

`acc` is exact label accuracy including the method; `tool` ignores the method;
`meth` is method accuracy given the tool was right; `abst` is the share of
off-topic rows correctly abstained on.

| arm | split | cov | prec | acc | tool | meth | abst | args | ms |
|---|---|---|---|---|---|---|---|---|---|
| fitted, schema vocab (112 rules) | A (fitted) | 0.984 | 1.000 | **0.984** | 0.984 | 1.000 | 1.000 | 0.845 | 0.055 |
| | B (held-out) | 0.577 | 0.415 | **0.239** | 0.304 | 0.665 | 0.975 | 0.249 | 0.060 |
| | wild | 0.527 | 0.615 | **0.324** | 0.365 | 0.750 | 1.000 | — | 0.047 |
| fitted, open vocab (113) | B | 0.590 | 0.361 | 0.213 | 0.285 | 0.561 | 1.000 | 0.243 | 0.064 |
| | wild | 0.392 | 0.690 | 0.270 | 0.297 | 0.778 | 1.000 | — | 0.056 |
| fitted, **cues only** (6) | A | 0.049 | 1.000 | **0.049** | 0.049 | — | 0.975 | 0.027 | 0.038 |
| | B | 0.068 | 0.438 | **0.029** | 0.029 | — | 1.000 | 0.017 | 0.039 |
| | wild | 0.027 | 0.500 | **0.013** | 0.013 | — | 1.000 | — | 0.028 |
| fitted, Laplace + min-cov 8 (63) | B | 0.477 | 0.400 | 0.191 | 0.267 | 0.577 | 1.000 | 0.237 | 0.053 |
| | wild | 0.432 | 0.656 | 0.284 | 0.324 | 0.700 | 1.000 | — | 0.043 |
| fitted, + schema overlap (106) | B | 0.663 | 0.342 | 0.227 | 0.303 | 0.581 | 0.850 | 0.257 | 0.082 |
| | wild | 0.622 | 0.565 | **0.351** | 0.432 | 0.571 | 0.933 | — | 0.070 |
| fitted, overlap + Laplace (74) | wild | 0.500 | 0.676 | 0.338 | 0.378 | 0.727 | 0.933 | — | 0.065 |
| fitted, unigrams + Laplace (60) | wild | 0.378 | 0.714 | 0.270 | 0.297 | 0.778 | 1.000 | — | 0.042 |
| **hand-written** (73) | A | 0.950 | 0.733 | 0.696 | 0.706 | 0.972 | 0.925 | 0.636 | 0.044 |
| | B | 0.870 | 0.628 | **0.546** | 0.568 | 0.927 | 0.950 | 0.467 | 0.046 |
| | wild | 0.730 | 0.667 | **0.486** | 0.500 | 0.938 | 0.867 | — | 0.036 |
| hand-written **+ catch-all** (73) | A | 1.000 | 0.696 | 0.696 | 0.706 | 0.972 | **0.000** | 0.636 | 0.044 |
| | B | 1.000 | 0.546 | 0.546 | 0.568 | 0.927 | **0.000** | 0.467 | 0.047 |
| | wild | 1.000 | 0.500 | 0.500 | 0.513 | 0.938 | **0.000** | — | 0.036 |

Paired McNemar on the same queries:

| split | contrast | hand only | fitted only | p |
|---|---|---|---|---|
| family B | hand vs schema | 377 | 86 | 1.7e-44 |
| family B | hand vs overlap | 385 | 82 | 5.6e-48 |
| family B | hand vs Laplace8 | 402 | 65 | 2.3e-60 |
| wild | hand vs schema | 23 | 11 | 0.058 |
| wild | hand vs overlap | 23 | 13 | 0.133 |
| wild | hand vs Laplace8 | 26 | 11 | **0.020** |
| family B | schema vs overlap | 61 | 49 | 0.294 |

On 948 held-out rows the hand-written arm wins decisively. On 74 wild rows it is
directional against the two better fitted arms and significant against the third.

## Why the fitted list loses

Not memorisation of entities, which was the failure the disjoint pools were
built to prevent. The learned rules read like rules a person would endorse:

```
n=12 p=1.00  [tok:diff]                        -> pull_request_read::get_diff
n=12 p=1.00  [tok:branches]                    -> list_branches
n=12 p=1.00  [tok:jobs, NOT tok:failed]        -> actions_list::list_workflow_jobs
n=12 p=1.00  [tok:cancel, cue:run_ref]         -> actions_run_trigger::cancel_workflow_run
```

`tok:diff → get_diff` is correct. It simply never fires on family B's *"what code
does this PR actually change?"*, which contains no learned token. The
hand-written rule for the same target is `\b(diff|patch|changeset)\b`, and the
extra alternates were written for words no training query contained.

**That is most of the gap — but not all of it, and the second pass below
corrects this paragraph.** A fitter can only learn the surface forms it was
shown; a human writing a rule enumerates synonyms from knowledge of the
language. What that account cannot explain is `bm25-schema`, a ranker with
**zero fitted parameters**, which drops 0.611 → 0.200 across the same two
families. Family B is not merely phrasing the fitter had not seen; it is
phrasing that avoids *schema vocabulary*, and it penalises anything reading the
catalogue just as hard. See **Family B is the odd split** below.
The fitted arm's family-A precision of 1.000 at 0.984 coverage against 0.415
precision at 0.577 coverage on family B is the textbook signature, and every
knob aimed at it — Laplace-corrected scoring, min-coverage 8, dropping bigrams —
moves coverage and precision around without moving accuracy: 0.239 → 0.191,
0.178, 0.209, 0.227. The best fitted variant on the wild set (0.351) uses the
`ov:` features, which are computed against the schema at inference time rather
than learned, and that is the one intervention that helps — because it is the
only one that can carry a word the training data lacked.

The schema is a weak synonym source even so. `has anyone approved it` overlaps no
label text at all: "approve" appears in no description among the 79.

## Structure alone routes almost nothing here

The cues-only arm is the direct test of `monad-bsky`'s stated boundary —
*"catalogues whose tools are distinguished by argument shape hand a regex most of
the task; catalogues distinguished by intent do not."*

Six rules. **4.9%** of training rows covered, 0.029 accuracy on family B, 0.013
on wild. On the Bluesky catalogue the same layer did most of the work. The
boundary is real, it is not a matter of degree, and it can be measured in about
one minute before committing to a routing design: fit on cues alone and read the
coverage.

## The catch-all costs everything and buys nothing

`monad-bsky` watched refusal fall from 0.500 on its fitted eval to 0.183 on
unseen templates, and named the catch-all `search_posts` rule as the cause. The
same rule set here, with and without a catch-all `search_code`, isolates it
exactly:

| | abstention | accuracy |
|---|---|---|
| hand-written, abstains | 0.925 / 0.950 / **0.867** | 0.696 / 0.546 / 0.486 |
| hand-written + catch-all | 0.000 / 0.000 / **0.000** | 0.696 / 0.546 / 0.500 |

Abstention goes to zero on all three splits and accuracy moves by **+0.000,
+0.000 and +0.014**. The fallback in the parent experiment was not a trade — it
was a giveaway. A structural router should abstain and hand off; it should never
own the decision that no tool applies.

## What the generator hides

The single most consequential number here is not in the results table.
`context_probe.py` asks how often a query contains the thing it refers to:

| split | any structural cue | `owner/repo` present | all required args extractable |
|---|---|---|---|
| family A (generated) | 0.925 | 0.767 | 0.661 |
| family B (generated) | 0.927 | 0.608 | 0.562 |
| **wild (hand-authored)** | **0.514** | **0.135** | **0.149** |

Real requests say *"go ahead and merge it"*, *"has anyone approved it"*, *"cancel
that run, it's stuck"*. The referent was established several turns earlier and
the sentence is a pronoun. A generated query almost always carries its
identifier, because a template that renders `{pr}` renders it every time.

Every regex-router number in this repo — including `monad-bsky`'s 0.833, whose
62 eval queries all carried their handle, post URI or DID — is therefore measured
on requests that are structurally richer than the ones a deployed router meets.
The routing task is not the harder half of the job either: it is the *argument
binding* that collapses, from 0.66 of required arguments extractable to 0.15.

This does not make a deterministic prefilter useless. It relocates it: the cues
must be read from the conversation state, not from the current sentence.

## Method dispatch is not the hard part

Given the right tool, the hand-written arm picks the right method 0.972 / 0.927 /
0.938 of the time across the three splits, and the fitted arms manage 0.57–0.78.
The nine-way `pull_request_read` enum looked like the interesting difficulty when
the catalogue was surveyed; it is not. Choosing among 43 tools is where the
errors are. Worth remembering before designing a catalogue around dispatcher
enums to keep the tool count down — it does not appear to cost accuracy.

## Two routers agreeing is a nearly free gate

`monad-bsky/synergy.py` found two small models naming the same tool were right
0.880 of the time at 0.455 coverage, beating a calibrated confidence head at
matched coverage — at 11x the latency, because it meant running both models. The
same gate between two *deterministic* routers costs microseconds:

| split | gate | coverage | precision |
|---|---|---|---|
| family B | hand alone | 0.870 | 0.628 |
| family B | fitted alone | 0.577 | 0.415 |
| family B | **both agree** | 0.192 | **0.775** |
| wild | hand alone | 0.730 | 0.667 |
| wild | fitted alone | 0.527 | 0.615 |
| wild | **both agree** | 0.203 | **0.867** |

The shape replicates: agreement buys ~15 and ~20 points of precision for about a
quarter of the coverage. The two arms are less independent than two models are —
they share the cue layer and the catalogue — so this is an optimistic reading,
and where they disagree the hand arm is right 0.518 / 0.533 of the time against
the fitted arm's 0.202 / 0.333, which is another way of stating the main result.

## What this says about where a fitted regex router belongs

The parent experiment's conclusion survives, narrowed. Deterministic routing
earns its place when the catalogue is signposted by argument shape *and* the
requests carry their arguments. The GitHub MCP catalogue fails the first
condition — measured, in a minute, by the cues-only arm — and real requests fail
the second.

What transfers regardless is the part that never needed a model: **argument
binding by extraction**. `cues.extract` is the only thing in this harness that
produces an argument, and where the value is in the sentence it is copied rather
than retyped. That was worth 83% of the reachable argument gap in `monad-bsky`
and it costs nothing here.

And the harness itself transfers, which was the point of building it. Four of its
outputs are cheap pre-commitment tests for any catalogue: fit on cues alone to
see whether structure carries the decision; run the catch-all ablation to price
the fallback; run the context probe to find out whether the queries you are
evaluating on are the ones you will receive; and check the fitted-versus-held-out
gap before believing any number a hand-written rule set reports on its own eval.

## Caveats

- **Families A and B share an author with the hand-written rules.** The hand arm
  is flattered on those splits. `wild.jsonl` was committed before `fit.py`
  existed, which fixes the ordering but not the authorship; it is 74 routable
  rows, and the wild contrasts are directional rather than decisive.
- **No model arm.** Nothing here measures what a frontier or small model scores
  on this catalogue, so "the model layer earns its place" is not tested — only
  "the deterministic layer does not carry it alone".
- **The hand-written arm is deliberately incomplete**, covering the tools and
  methods a person reaches for first and abstaining elsewhere. Writing all 79 by
  hand is the labour the fitter was meant to remove.
- **One fitter.** Greedy covering with ≤2-literal conjunctions is one point in a
  large space; a linear model over the same features, or gradient-boosted rules,
  might generalise better. The vocabulary limitation diagnosed above would
  constrain any of them.
- **Eight session tools have no upstream schema** and are excluded, including
  both PR-activity subscription tools, which are CCotw additions rather than
  upstream MCP tools.

## Reproduce

```bash
python3 catalogue.py --clone          # rebuild from github/github-mcp-server snapshots
python3 gen_data.py --n 12            # families A and B, disjoint pools
python3 fit.py --vocab schema         # ~100 s on one core
sh sweep.sh                           # the regularisation arms
python3 eval.py rules_*.json hand hand+fallback
python3 context_probe.py              # the referent-presence table
python3 agreement.py                  # the two-router gate
python3 mcnemar.py                    # paired significance
```

`data/` is gitignored and regenerable from the committed generator and seed.

---

# Second pass — the rest of the lexical stack, and one rung up

Oskar's follow-up: *"we have a number of lexical tools at our disposal: regex,
grep, bm25. Do the others improve anything? And up the stack there's good old
spaCy."* Three arms were built against the same catalogue, the same three splits
and the same scoring code, behind a common interface (`arms.py`) so every router
in this experiment is directly comparable. Everything below was measured by
three subagents working in parallel and re-verified through this writeup's own
eval path before being believed.

## The table

`label_acc` on family A (fitted) / family B (held-out phrasings) / wild
(hand-authored). For any arm whose documents *are* the family-A queries, the
family-A column is memorisation and should be ignored.

| arm | A | B | wild | abstention (wild) | median ms |
|---|---|---|---|---|---|
| fitted decision list, best | 0.984 | 0.239 | 0.351 | 1.000 | 0.047 |
| hand-written regex | 0.696 | 0.546 | 0.486 | 0.867 | 0.036 |
| hand-written + catch-all | 0.696 | 0.546 | 0.500 | **0.000** | 0.036 |
| `tok-overlap` (IDF schema overlap, no fitting) | 0.539 | 0.193 | 0.351 | 0.667 | **0.030** |
| `bm25-schema` (zero parameters) | 0.611 | 0.200 | 0.405 | 0.200 | **0.014** |
| `bm25-train` (documents = family-A queries) | 0.993\* | 0.318 | **0.635** | 0.533 | 0.025 |
| `bm25-both` (weighted fusion, w=0.2) | 0.997\* | 0.341 | 0.622 | 0.200 | 0.076 |
| `bm25-gated+off` (threshold + negative doc) | 0.981\* | 0.293 | 0.540 | **0.867** | 0.073 |
| `spacy-lemma` | 0.492 | 0.165 | 0.311 | — | 8.2 |
| `spacy-vec-idf` | 0.416 | 0.092 | 0.230 | — | 8.6 |
| `spacy-fusion` | 0.621 | 0.156 | 0.351 | — | 8.2 |
| `enc-schema` (encoder, zero-shot) | 0.567 | 0.260 | 0.378 | 0.467 | 2.0–3.7 |
| `enc-centroid` (encoder, family-A centroids) | 0.916\* | 0.247 | 0.460 | 1.000 | 2.0–3.7 |
| `enc-fusion` | 0.903\* | 0.397 | 0.595 | 0.800 | 2.0–3.7 |
| **`cascade-enc-fusion`** (hand → encoder @0.416) | **0.738** | **0.585** | **0.622** | **0.867** | **0.088** |

\* documents or centroids built from family A.

## The cascade is the result

The parent experiment measured a catch-all fallback at abstention **0.000** for
**+0.014** accuracy and called it a giveaway. The fix is not to drop the
fallback but to give it a score it can decline on. Hand-written rules first;
where they abstain, a thresholded encoder; where that also declines, abstain.

| | A | B | wild | abstention |
|---|---|---|---|---|
| hand, abstains | 0.696 | 0.546 | 0.486 | 0.925 / 0.950 / 0.867 |
| hand + catch-all | 0.696 | 0.546 | 0.500 | **0.000 / 0.000 / 0.000** |
| **hand → enc-fusion @0.416** | **0.738** | **0.585** | **0.622** | 0.925 / 0.925 / **0.867** |

**+0.136 wild accuracy over the abstaining hand arm at zero abstention cost.**
On wild the abstention line is flat at 0.867 for every threshold down to 0.416 —
the encoder is simply never confident about the 15 off-topic rows, which is the
property a catch-all regex cannot have by construction.

It is also cheap where it matters: **82% of requests never reach the encoder**,
so the median is **0.088 ms** against 0.071 for the hand arm alone, with the
2–4 ms encode paid only on the tail (p90 9.3 ms, p99 16.5 ms).

## Family B is the odd split

`bm25-schema` fits nothing — no split contributes a parameter — and still scores
**0.611 / 0.200 / 0.405**, a 3x spread and a −67% A→B drop against the fitted
list's −76%. A zero-parameter ranker cannot overfit, so the A→B collapse cannot
be overfitting alone.

The tell is that **wild outscores family B on every schema-reading arm**: 0.405
vs 0.200 for BM25, 0.378 vs 0.260 for the encoder. Hand-authored requests are
*easier* than the generated "held-out" family. Family B was written to avoid
family A's verbs, and family A's verbs came from the same schemas any
catalogue-reading router consults, so family B is adversarial toward schema
vocabulary specifically.

This does not rescue the fitted arms — they lose to hand-written rules on wild
too (0.351 vs 0.486), which is the split with no such bias. It does mean **the
−0.31 family-B gap in the first pass overstates generalisation loss**, and any
future experiment generating a held-out family this way should expect the same
distortion.

## Was the truncated catalogue to blame? No

`catalogue.py` capped parameter descriptions at 240 characters, which cut the
`method` enum glosses on exactly the three dispatchers that document them —
`pull_request_read` kept 3 of its 9. Every schema-reading arm was handicapped by
the harness rather than by the catalogue. Rebuilding without the cap (2.2x the
method-gloss text, 9 of 9 enums glossed) and re-running:

| arm | B, capped | B, full | wild, capped | wild, full |
|---|---|---|---|---|
| `bm25-schema` | 0.200 | 0.224 | 0.405 | 0.405 |
| `enc-schema` | 0.260 | 0.282 | 0.378 | 0.378 |
| `tok-overlap` | 0.193 | 0.193 | 0.351 | 0.338 |

A real bug, worth +0.02 on family B and nothing on wild. **The conclusion it
threatened survives**: the catalogue does not describe itself in the words people
use to ask for it. "approve" appears in none of the 79 schema texts, and giving
the arms the full text does not change that.

## BM25: a shortlister, not a router

`bm25-train` beats every prior arm on the honest split — **wild 0.635** against
0.486 hand-written and 0.351 best-fitted, McNemar p=0.043 — and loses decisively
on family B (0.318 vs 0.546, p=8e-27). "Soft beats hard" holds on hand-authored
requests and fails on the adversarial family.

Recall@k is where it earns its place:

| arm | wild @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| `bm25-train` | 0.635 | 0.784 | 0.811 | 0.892 |
| `bm25-train-stem` | 0.635 | 0.811 | 0.838 | **0.960** |
| `bm25-both` | 0.622 | 0.784 | 0.851 | 0.919 |

**79 targets narrowed to 5 keeps 85% of gold answers, at 0.03 ms.** That is the
usable product: a candidate generator in front of a more expensive decision, not
a router. It also reframes the tool-catalogue sizing problem — `needle-bsky`
measured a fixed ~750 ms penalty for declaring a sixth tool to Cactus Needle, and
a 0.03 ms shortlister that keeps 85% recall at k=5 is a way to never pay it.

Two further results:

- **A negative document beats a threshold for abstention.** One extra pseudo-label
  whose document is the 40 off-topic training queries buys **+0.400 abstention for
  −0.014 accuracy** — the exact mirror of the catch-all, same 0.014, opposite sign.
  Stacked with a threshold, `bm25-gated+off` reaches abstention 0.925 / 0.867 with
  real accuracy, the first arm here with both.
- **RRF loses to weighted sum at every weight** (wild 0.554 vs 0.622). The two
  component arms are of very unequal quality, and a weighted sum can down-weight
  the weak one to 0.2 where reciprocal-rank fusion votes it as loudly as the
  strong one.

## Stemming: two agents, two mechanisms, one conclusion

Stemming and lemmatisation both *lose* at top-1, and the reason is the same and
worth stating as a rule: **in an API catalogue, grammatical number is semantic.**
Plural names a list endpoint, singular a fetch-one.

The lemmatiser deletes exactly the most discriminative token:

| form | df/79 | idf | → lemma | df/79 | idf |
|---|---|---|---|---|---|
| branches | 1 | 4.37 | branch | 5 | 2.76 |
| files | 2 | 3.68 | file | 5 | 2.76 |
| issues | 4 | 2.98 | issue | 15 | 1.66 |

Method-accuracy-given-tool falls 1.000 → 0.500 on wild under lemmatisation. The
Porter stemmer arrives at the same place from the other side: 96 stems merge more
than one surface form, and the merges hit `tag`/`tags` (`get_tag` vs
`list_tags`), `workflow`/`workflows`, `team`/`teams`, `commit`/`commits`. Family
B verdict changes: **16 fixed, 83 broken**.

The exception: stemming *helps recall@10* (0.892 → 0.960 on wild). It pulls the
right answer into the shortlist while pushing it off the top. **Stem if you are
shortlisting; do not if you are deciding.**

## spaCy: a clean negative, and the sharpest diagnosis in the experiment

`en_core_web_md` installed and downloaded without trouble — the anticipated
proxy 403 on the GitHub-release model never happened. Every spaCy arm then lost
to `tok-overlap`, a 20-line IDF-weighted schema-overlap control with no spaCy in
the path, while running **250x slower** (8.2 ms vs 0.03 ms).

The decisive measurement is the **zero-lexical-overlap slice** — rows sharing no
word with any of the 79 label texts, where lexical routing is structurally blind.
n = 2 / 77 / 7 across the splits, and the vector arm scores **0.000 on all
three**, against 0.437 / 0.108 / 0.254 on rows lexical matching can see. Family
B's n=77 is enough to believe. Vectors add nothing precisely where they were
supposed to.

They do move gold from *absent* to *findable*: "has anyone approved it" puts
`pull_request_read::get_reviews` at **rank 21 of 79**, and "what code does this
PR actually change" puts `get_diff` at rank 30. That is not routing. What a human
supplies writing `\b(diff|patch|changeset)\b` is **domain synonymy** — the fact
that those three name one GitHub concept — and general-English vectors do not
encode it, because it is a fact about this API rather than about English.

Two incidental corrections to the first pass:

- **The referent problem is absence, not pronouns.** The pronoun direct-object
  rate on wild is only **0.108**. Requests do not say "merge it" instead of
  naming the PR nearly as often as they simply never name the repo at all —
  which matches `context_probe.py`'s 0.135 `owner/repo` presence.
- **Latency is no longer part of the regex arms' case.** BM25 runs at
  0.005–0.028 ms against the hand-written regexes' 0.036–0.087. The lexical
  ranker is *faster* than the rule list as well as more accurate on wild.

## Where the semantic ceiling actually sits

Just above the regex floor, not in a different regime. The best single semantic
arm scores 0.622 wild and 0.540 family B against 0.486 / 0.546 for hand-written
regexes — +0.14 on one split, −0.01 on the other, for 50–90x the latency and a
124 MB model.

What carries it is **supervision, not semantics**: zero-shot schema-text
embedding scores 0.298 / 0.392, *below* the regexes. Given the same family-A
supervision the encoder generalises 2.3x better than the fitted decision list on
family B (0.540 vs 0.239) — so what the encoder buys is exactly the synonym
robustness the first pass diagnosed as the fitter's failure, recovering about
half the gap.

The remaining ~0.38 error on wild is not a phrasing problem and no encoder fixes
it. Only 13.5% of hand-authored requests carry the `owner/repo` their tool
requires; *"go ahead and merge it"* has no referent to embed. **Routing above
~0.62 on this catalogue needs conversation state, not a bigger encoder.**

## The agreement gate scales with independence

| split | gate | coverage | precision |
|---|---|---|---|
| B | hand ∧ fitted list | 0.192 | 0.775 |
| B | **hand ∧ encoder centroid** | **0.355** | **0.864** |
| wild | hand ∧ fitted list | 0.203 | 0.867 |
| wild | **hand ∧ encoder centroid** | **0.351** | **0.923** |
| wild | hand ∧ encoder fusion, s≥0.42 | 0.297 | 0.955 |

1.7–1.9x the coverage at higher precision. The fitted list shares this repo's cue
layer and catalogue with the hand arm; the encoder shares neither. That is what
`monad-bsky` meant by the two signals being "close to independent", now with a
gradient: **the more independent the second voter, the more the gate buys.**
Wild is 26 rows at the gate, so read it as directional.

## Second-pass caveats

- **Every threshold was chosen on family A**, the fitted family, and transferred.
  For the BM25 gate the transfer happened to be free (wild's own oracle threshold
  is the same 0.841); for the encoder cascade it was not checked against a wild
  oracle, so its operating point is transferred, not tuned.
- **The wild split is 74 routable rows.** Every wild contrast here is directional.
- **`spacy_arms.py` changed between the cascade agent's two screening runs**
  (wild 0.365 → 0.351), so its fallback-comparison table mixes two versions of
  that one row. The selected cascade does not use spaCy.
- **A local `catalogue.py` shadows the PyPI `catalogue` package** that spaCy
  depends on. The failure is an `AttributeError`, not an `ImportError`, so a
  guard written for missing dependencies does not catch it.
