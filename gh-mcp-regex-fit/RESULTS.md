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

**That is the whole gap.** A fitter can only learn the surface forms it was
shown; a human writing a rule enumerates synonyms from knowledge of the language.
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
