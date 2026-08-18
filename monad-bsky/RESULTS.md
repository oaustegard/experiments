# monad-bsky — Pleias Monad against Cactus Needle on one routing task

[`needle-bsky`](../needle-bsky/RESULTS.md) put Cactus Needle 2 (45M parameters,
purpose-built for tool calling, decode constrained by a grammar compiled from
the declared schemas) in front of 18 Bluesky read tools and measured it on 62
natural-language queries. This runs [PleIAs
Monad](https://huggingface.co/PleIAs/Monad) at the same task: 56M parameters, a
generalist small reasoning model trained on 200B tokens of SYNTH, no grammar,
no confidence head. Same 18 tools, same 62 queries, same scoring code, same 800
training rows from the same generator and seed.

Headline: **fine-tuning takes Monad from nothing to roughly two thirds of
Needle's routing accuracy, and the gap is mostly transcription, not choice.**
Zero-shot Monad produces no parseable call at all: 0 of 62. Fine-tuned it
reaches 0.481 routable top-1 against Needle's 0.611 base and 0.722 best
configuration. But it copies an identifier out of the request correctly only
51% of the time (`austegard.com` → `afethew.com`, `jetstream` → `jetforek`),
against 78–90% for the grammar-constrained arm, and **more training makes the
copying worse**.

One combination of the two is worth having. Where the models independently name
the same tool, that answer is right **88.0%** of the time across 45.5% of
queries, against 74.1% for Needle's own calibrated confidence head at the same
coverage — and agreement needs no confidence head, which is what fine-tuning
takes away. It costs running both models, 11x the latency of Needle alone.

Everything below is CPU-only on the same 4-core container.

## The two models

| | Needle 2 | Monad |
|---|---|---|
| parameters | 45M | 56.7M measured |
| shipped size | 14 MB (CQ2) | 113 MB fp32 safetensors |
| architecture | Simple Attention Network | Llama, 64 layers × 256 hidden, 4 heads |
| vocabulary | 8,192 | 8,192 |
| context | 256-token sliding window, tools pinned | 2,048 absolute |
| decode | byte-level grammar from the schemas | unconstrained |
| confidence | calibrated head | none |
| built for | tool calling, device use, extraction | instruction following with thinking traces |

The asymmetries matter and are not corrected for. Monad ships 8x larger on disk
in fp32 and was not quantized here; Needle was LoRA-tuned because its engine
accepts nothing else, while Monad got a full fine-tune; and Needle sees five
tools per turn through its retrieval head where Monad sees all eighteen.

## Rendering the catalogue

The 18 tool schemas as JSON, the exact bytes Needle consumes, cost **1,972
Monad tokens** against a 2,048 context. There is no room left for the query, the
thinking trace and the answer.

| rendering | Monad tokens |
|---|---|
| full JSON schemas | 1,972 |
| one line per tool, `name(args): description` | **574** |
| name and description only | 493 |

So Monad gets the prose rendering, which is 3.4x cheaper for the same
information and leaves ~1,400 tokens of headroom. A training row comes to 767
tokens at the median, 864 at the maximum.

This is about JSON versus prose, not about one model's tokenizer being worse
than the other's. The same 18 schemas cost Needle 1,642 tokens, within 20% of
Monad's 1,972; Needle avoids the problem with a 256-token sliding window that
pins the tool block as KV sinks rather than by spending less on it.

The rendering also explains why Monad sees the whole catalogue while Needle
sees five of it:
at 574 tokens there is no reason to retrieve, and the model has no retrieval
head to do it with.

## Zero-shot

Monad routes nothing. Across 62 queries the parse rate is **0.000** — not one
completion contained a JSON object. It produces a thinking trace and runs out of
tokens inside it:

```
User query breakdown: "call one tool to answer the request" → direct request.
"reply with one JSON object" → structured data format. "arguments" → specific
type of request. "use only arguments whose values appear in the …
```

It is parsing the instruction as the thing to analyse. That is a reasonable
behaviour for a model whose card advertises "instruction following with thinking
traces" and has never seen a tool schema.

One artefact to name: this arm scores `refusal 1.000` and `tool_acc 0.129`.
Both are vacuous. The eight off-topic queries count as correct refusals because
the model emitted nothing, not because it declined. Read the routable column,
which is 0.000.

## Fine-tuned

Full fine-tune, 800 rows, 3 epochs, batch 4, lr 1e-4 cosine, loss on completion
tokens only, 6,466 seconds on 4 cores. Training loss reaches 0.0001; validation
0.0128 → 0.0040 → 0.0017.

| arm | routable | args | refusal | invented | parse-ok | hallucinated tool | median |
|---|---|---|---|---|---|---|---|
| Monad zero-shot | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 4970 ms |
| Monad 1 epoch | 0.389 | 0.241 | 0.125 | 0.481 | 0.968 | 0.065 | 3068 ms |
| Monad 2 epochs | **0.481** | 0.296 | 0.500 | 0.333 | 0.952 | 0.113 | 3316 ms |
| Monad 3 epochs | 0.444 | 0.296 | 0.500 | 0.315 | 1.000 | 0.145 | 2879 ms |
| Needle base | 0.611 | 0.537 | 0.625 | 0.222 | n/a | 0 by construction | 958 ms |
| Needle LoRA | 0.667 | 0.648 | 0.375 | 0.222 | n/a | 0 | 953 ms |
| Needle 2-stage | **0.722** | 0.685 | 0.625 | 0.111 | n/a | 0 | 324 ms |
| Needle oracle k=5 | 0.778 | 0.667 | 0.625 | 0.185 | n/a | 0 | 199 ms |

The refusal and tool-accuracy figures for the zero-shot row are vacuous, as
above.

Paired McNemar on the same 62 queries:

| contrast | Needle-only wins | Monad-only wins | p |
|---|---|---|---|
| Needle base vs Monad 3 epochs | 18 | 8 | 0.076 |
| Needle LoRA vs Monad 3 epochs | 18 | 7 | **0.043** |
| Needle 2-stage vs Monad 3 epochs | 22 | 6 | **0.0037** |
| Monad 1 epoch vs Monad 3 epochs | 4 | 10 | 0.18 |

Against the best Needle configuration the difference is solid. Against Needle's
plain base weights it is directional at this n. The third epoch does not
separate from the first.

**Validation loss is not tracking anything useful here.** It falls by 7.5x
across the three epochs while routable accuracy goes 0.389 → 0.481 → 0.444. The
80-row validation split is drawn from the same templated pool as the training
rows, so it measures how well the model has fitted those templates, and the
model fits them essentially perfectly (train loss 0.0001) while remaining a
0.45-accuracy router on differently-phrased queries. A held-out split from the
same generator is not a held-out task.

## Routing errors versus transcription errors

Sorting the tuned model's errors splits them cleanly. Some are routing:
`get_followers` where `get_following` was wanted, which Needle gets wrong too.
The rest are the model failing to reproduce a string that was sitting in the
request.

`copy_probe.py` scores exactly that, over the 41 eval arguments whose expected
value appears verbatim in the query: was the value emitted exactly, wherever it
landed?

| arm | verbatim copies | copy accuracy |
|---|---|---|
| Needle LoRA | 37/41 | **0.902** |
| Needle 2-stage | 33/41 | 0.805 |
| Needle base | 32/41 | 0.780 |
| Monad 1 epoch | 23/41 | 0.561 |
| Monad 2 epochs | 21/41 | 0.512 |
| Monad 3 epochs | 21/41 | **0.512** |

The misses are not near-misses of meaning:

| wanted | Monad emitted |
|---|---|
| `simonwillison.net` | `simonwillon.net` |
| `austegard.com` | `afethew.com` |
| `austegard` | `afethewess` |
| `jetstream` | `jetforek` |
| `cactus compute` | `cactus computation` |
| `at://did:plc:s3cqfxbcwnvvyrsttl3wivgp/app.bsky.feed.post/3lxk2mnop4c2v` | `at://did:plc:s3cqfxbcwnvvirior` |
| `why.bsky.team` | `Why.bsky.team's recent` |

The obvious explanation is tokenization, and it is wrong. Both models carry
**8,192-piece vocabularies**, and they segment these strings identically:
`austegard.com` is `['a','ust','eg','ard','.','com']` in each, and across ten
identifiers, handles, DIDs and URLs the totals are 111 Needle pieces against 109
Monad pieces, a ratio of 0.98. Neither model has an easier string to copy.

What differs is what each was trained to do with it. Needle's documented
contract is that arguments carry only values evidenced by the input; span
copying is the behaviour it was built around, and its **base** weights — never
exposed to any of this experiment's data — already copy at 0.780. Monad was
trained to reason in prose and generate, and 800 examples did not install the
copying operation: accuracy *falls* from 0.561 at one epoch to 0.512 at three
while training loss keeps dropping, because fitting the training set's
identifiers is not the same as learning to transcribe an unseen one.

The same failure appears at the tool-name level. Tuned Monad invents names that
were never declared (`get_posts`, `get_replies`, `search_followers`, and once
`get_spammer.bsky.social`, assembled out of the query's own handle) on 6.5% of
queries after one epoch, 11.3% after two and 14.5% after three. Needle cannot do this: its grammar admits only the declared names.
A constrained decoder buys exactly this, and measurably it is worth
more here than the 11M parameter difference.

## Don't ask it to copy

If transcription is the problem, take the strings away from the model.
`repair.py` keeps the tool name Monad chose and refills every structurally
extractable argument from the query — handle, post URI, feed URI, DID, a bare
integer for `limit` — with the same kind of regex the `needle-bsky` two-stage
router uses for its group pick.

| | routable | args |
|---|---|---|
| Monad 3 epochs | 0.444 | 0.296 |
| + regex argument fill | 0.444 (unchanged by construction) | **0.370** |

That closes most of the reachable gap: argument accuracy cannot exceed tool
accuracy, so 0.370 of a 0.444 ceiling is 83% of what was available.

What it cannot fix is the four remaining failures, and they are informative —
three are free-text search queries (`jetforek`, `cactus computation`,
`afethewess`) with no structure to extract, so the model's transcription is all
there is. The fourth put a DID in `feed_uri` instead of
`actor`, a schema error the filler never fired on.

**Deterministic extraction rescues arguments that have structure. Free-text
arguments have none, and there a small prose-vocabulary model is the weakest
link.** As with the `needle-bsky` stage-1 regex, this extractor was written
after reading the failures, so its numbers on this eval are fitted to this
distribution to an unknown degree; the shape is what transfers.

## Per-category

| category | n | Monad 3 epochs | Needle base |
|---|---|---|---|
| analysis | 3 | 0.00 | 1.00 |
| feed | 3 | **1.00** | 0.00 |
| find-users | 3 | 0.33 | 0.33 |
| firehose | 3 | 0.67 | 0.67 |
| graph | 5 | 0.20 | 0.60 |
| identity | 3 | 0.33 | 0.33 |
| interactions | 6 | 0.83 | 0.83 |
| keywords | 2 | 1.00 | 1.00 |
| off-topic | 8 | 0.50 | 0.62 |
| person-posts | 5 | 0.20 | 0.60 |
| profile | 4 | 0.25 | 0.25 |
| search | 7 | 0.71 | 0.71 |
| status | 3 | 0.00 | 0.67 |
| thread | 3 | 0.33 | 0.67 |
| trending | 4 | 0.25 | 0.75 |

Monad wins `feed` outright, 3/3 against 0/3, which is the category where Needle's
retrieval head kept failing to surface `get_feed_posts` — Monad has all 18 in
context and no retrieval stage to fail. It ties on six categories and loses the
rest. `profile` is 0.25 for both models, unchanged by any amount of training on
either, which makes it a property of the eval's phrasing rather than of either
model.

## Latency

Monad's median turn is 2,879–4,970 ms against Needle's 958 ms flat and 324 ms
two-stage: **3 to 9 times slower**. 64 sequential layers at hidden size 256 is
close to the worst possible shape for a CPU: almost no width to parallelise
across, sixty-four dependent matmuls per token. Needle runs on a hand-written
C++ engine with a quantized weight format; this is transformers on fp32 tensors,
so the comparison is implementation as much as architecture.

## Combining the two

Both models answered the same 62 queries and every per-query row is committed,
so `synergy.py` evaluates eight combinations as post-processing. One works, one
fails cleanly, and the rest do not pay for a second model.

### Agreement beats Needle's own confidence head

When Needle's two-stage router and 2-epoch Monad independently name the same
tool, that answer is right **88.0%** of the time, across 45.5% of the queries
where Needle emitted a call. Needle's calibrated confidence head, on the same
55 calls, reaches that precision only by cutting coverage roughly in half:

| gate | coverage | precision |
|---|---|---|
| none | 1.000 | 0.709 |
| Needle confidence ≥ 0.4 | 0.636 | 0.743 |
| Needle confidence ≥ 0.6 | 0.491 | 0.741 |
| Needle confidence ≥ 0.8 | 0.236 | 0.846 |
| Needle confidence ≥ 0.9 | 0.127 | 1.000 |
| **the two models agree** | **0.455** | **0.880** |
| agree *and* confidence ≥ 0.4 | 0.255 | **0.929** |

At matched coverage (~0.46) the confidence head delivers 0.741 against
agreement's 0.880, about 14 points. To reach agreement's precision on confidence
alone Needle has to drop to 0.236 coverage, roughly half. The two signals also
compose: requiring both lifts precision to 0.929.

Two reasons this is worth something beyond the number. Agreement needs **no
confidence head at all**, and fine-tuning Needle destroys its head — so a tuned
deployment, which currently has no gate, could get one back this way. And the
signals are close to independent: one is a calibrated post-hoc score over
Needle's own logits, the other is a second model trained from different data
under a different objective.

The price is honest and steep: you run both models. 324 ms for Needle's
two-stage plus ~3,300 ms for Monad is **11x the latency** of Needle alone, and
it buys about double the coverage at equal precision. On a phone that is a bad
trade; in a batch pipeline where a wrong tool call costs a network round trip
and a wrong answer, it may not be.

### Needle's confidence says nothing about Monad

The head separates Needle's own correctness — mean 0.584 when right, 0.392 when
wrong. Pointed at Monad's answers it is flat and slightly inverted: 0.486 when Monad
is right, 0.532 when Monad is wrong. Sliced by
threshold it gets worse, not better:

| Needle confidence | n | Needle accuracy | Monad accuracy |
|---|---|---|---|
| ≥ 0.4 | 39 | 0.692 | 0.487 |
| ≥ 0.6 | 25 | 0.760 | 0.440 |
| ≥ 0.8 | 15 | 0.867 | 0.400 |

Calibration is a property of the model that produced it, not of the query. A
"hard query" score that transfers between models would have been useful; this
one does not.

### The other five combinations

**Complementarity is thin.** Needle two-stage and Monad are both right on 24
queries, Needle alone on 20, Monad alone on 6, neither on 12. The union ceiling
is 0.806 against Needle's 0.710: six queries of headroom, and reaching it needs
an oracle to know which model to believe.

**Name snapping is smaller than the hallucination rate suggests.** Snapping
Monad's undeclared names to the nearest declared one by edit distance fixes two
queries at epoch 3 (0.444 → 0.481) and none at epoch 2. A decode grammar would
eliminate the 6.5–14.5% hallucinated-name rate, but most of those names sat on
queries that were misrouted anyway, so the accuracy it buys is ~3.7 points, not
14. This corrects the impression the earlier section leaves.

**Splitting the roles does not work.** Monad choosing the tool and Needle
supplying the arguments reaches 0.407 argument accuracy, against 0.370 for
Monad's own regex repair and 0.685 for Needle two-stage doing both jobs. Needle's
arguments are only available where it picked the same tool (25 of 62 queries),
so the split is starved exactly where it would help.

**Fallback rescues almost nothing.** Needle two-stage refuses only 2 routable
queries and Monad rescues 1. Among the 9 queries where Needle is both wrong and
below 0.6 confidence, Monad is right on 2. The flat Needle arm has more room —
16 wrong-and-unsure, Monad right on 7 — but the flat arm is the configuration
you would not deploy.

**Per-category dispatch tops out at 0.758** against 0.710, and it is fitted to
this eval by construction: the categories where Monad wins (`feed` 3/3 vs 2/3,
`search` 6/7 vs 5/7) are three and seven queries. Reported as a bound, not a
design.

## Reading this against the small-reasoner thesis

Monad is a generalist that was never shown a tool schema, and 800 examples plus
two hours of CPU took it from producing nothing to routing about half the
queries in an 18-tool catalogue. That is a real capability transfer at 56M
parameters, and it arrived with no inference engine and no decode
grammar.

It also lost to the purpose-built model at every configuration, and the reason
is neither reasoning nor tokenization — the two vocabularies chop these strings
the same way. It picked defensible tools; it could not transcribe
`austegard.com`. The parts a tool-calling stack needs — a constrained decoder
that cannot name an undeclared tool, argument values that are spans rather than
regenerations, a calibrated score to gate on — are engineering around the model,
not capacity inside it. Two of the three are cheap to add to Monad: a grammar
over the declared names would eliminate the 11–15% hallucinated-tool rate
outright, and the regex fill above recovers most of the argument gap. The
third, calibration, is the one Needle has and neither fine-tune keeps.

## Reproduce

```bash
python3 make_data.py -n 800              # 800 rows from needle-bsky's generator, k=18
python3 train.py --epochs 3              # full fine-tune, ~108 min on 4 cores
sh run_evals.sh                          # base + three checkpoints, then the table
python3 copy_probe.py                    # verbatim-copy accuracy across both experiments
python3 repair.py                        # regex argument fill over the tuned results
python3 recheck.py                       # every number here against the artifacts
```

`model/`, `tuned*/` and `data/` are gitignored: the weights come from Hugging
Face and the data is regenerable from the committed script and seed.

## Caveats

- **n=62**, 54 routable. One query is 1.9 points of routable accuracy. Only two
  of the four contrasts reach significance.
- **Not size-matched.** Monad ran fp32 at 113 MB against Needle's 14 MB CQ2
  export. A quantized Monad was not measured, so nothing here is a claim about
  quality per byte.
- **Not method-matched.** Full fine-tune against LoRA, because Needle's engine
  accepts only LoRA and Monad has no such constraint. Each model got the better
  of what it supports.
- **Not context-matched.** Monad sees 18 tools, Needle sees the 5 its retrieval
  head returns. That advantages Monad on retrieval (it cannot miss a tool that
  is in context — see `feed`) and disadvantages it on selection (17 distractors
  rather than 4).
- **The eval set and the training templates share an author** with everything in
  `needle-bsky`, which the parent writeup already flags. Both models are exposed
  to it equally, so the comparison is less affected than the absolute numbers.
- **Monolingual English model, and one eval query is Norwegian**
  (`kw-02`), which both models happen to route correctly.
