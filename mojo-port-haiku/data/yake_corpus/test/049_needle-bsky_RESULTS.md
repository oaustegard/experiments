# needle-bsky — Cactus Needle 2 as a routing layer over the Bluesky read tools

[Needle 2](https://cactuscompute.com/needle) is a 45M-parameter tool-calling
model from Cactus Compute, 14 MB compressed at CQ2, meant for phones, wearables
and microcontrollers. `cactus-needle` 2.0.0 went to PyPI on 2026-08-10 and
2.0.6 on 2026-08-17; this run used 2.0.6 with engine 2.0.2. The model answers
only in function calls, with a byte-level grammar compiled from the declared
schemas constraining every token, and it carries a calibrated confidence head so
a product can act above a threshold and escalate below it.

This wires it in front of an 18-tool Bluesky read surface (`browsing-bluesky`
and `atprotoing`, both already on disk as skills) and measures what the routing
layer is worth: 62 natural-language queries, four schema arms, an
oracle-retrieval probe, a two-stage router, a LoRA fine-tune, and an extraction
pass.

Three findings, in the order they change what you would build.

**The base model routes 61–70% of queries to the right tool, and the confidence
head is what makes that usable, but only if you declare nothing but required
arguments.** With optional arguments declared, a correct call can score 0.0004,
so the gate cannot separate correct routing from wrong. With them dropped, the
same gate trades coverage for precision monotonically: 38% coverage at 76%
precision, 13% at 100%.

**Keep every agent at five tools, and pick the five with something that is not
a model.** A five-tool catalogue containing the right answer beats the flat 18
by 11–17 points, and the sixth declared tool costs 3.6x the per-turn latency.
Splitting into groups and letting Needle choose the group scores 24 points
*worse* than the flat 18; ~20 lines of regex doing the same job scores 11 points
better and runs 3.8x faster.

**The fine-tune is a wash and costs the gate.** 800 templated examples and two
hours of CPU moved routing from 0.611 to 0.667 (paired p=1.0), left the two
worst categories exactly where they were, and replaced every confidence score
with `None`, because fine-tuning does not update the confidence head.

Everything below is CPU-only, on a 4-core CCotw container. No GPU, no
`ANTHROPIC_API_KEY`, no OpenRouter.

## The declared catalogue

18 tools, all reads:

| From | Tools |
|---|---|
| `browsing-bluesky/scripts/bsky.py` | `get_profile`, `get_user_posts`, `search_posts`, `get_feed_posts`, `get_trending`, `get_trending_topics`, `sample_firehose`, `get_thread`, `get_quotes`, `get_likes`, `get_reposts`, `get_followers`, `get_following`, `search_users`, `analyze_account`, `extract_keywords` |
| `atprotoing/scripts/atproto.py` | `resolve_identity`, `atproto_status` |

`get_all_followers` and `get_all_following` are not declared: they differ from
their non-`all` siblings only in pagination, which is a caller's decision and
not recoverable from natural language.

18 matters because Needle renders at most **five** tools per turn. Above five, a
built-in retrieval head embeds the query and admits only the top five into the
context, with the grammar rebuilt over that subset. Cactus's own documentation
puts it as "an unselected tool is unreachable, not merely unlikely".

## The four arms

A 2×2 over the two things a developer controls when wiring an existing library
to a router:

|  | full arity | required args only |
|---|---|---|
| **existing docstrings** (`needle.tool` introspection, zero editing) | `auto` | `auto-min` |
| **schemas written for the router** | `tuned` | `tuned-min` |

`auto` is what wiring costs nothing: `bsky.py` is already written with
Google-style `Args:` blocks, so one decorator per function produces a schema.
That schema keeps `transcribe` (a model-selection knob), keeps summary lines
written for a human reader, and truncates multi-line argument descriptions at
the first line. `tuned` rewrites the descriptions verb-first around the object
that distinguishes each tool from its neighbours, drops the arguments a router
cannot fill, and puts numeric bounds in the schema. The `-min` variants drop
every optional argument from whichever wording they start with.

## Eval set

`evalset.jsonl`, 62 queries: 54 routable, 8 off-topic. Each carries the accepted
tool name(s), the arguments the query licenses, and which optional arguments are
*evidenced* by the query text ("get 50 posts from…" licenses `limit`, "what has
X been posting" does not). Three things are scored per query:

- **tool**: predicted name is in the accepted set; an off-topic query must
  produce Needle's empty call.
- **args**: every expected key present and equal after normalisation.
- **invented**: the call carries an argument neither expected nor evidenced,
  i.e. a value the query never licensed.

Routing is one `complete()` turn and touches no network. Every arm is
deterministic: re-running the full set produced byte-identical calls
(`repeat 2`, all four arms).

## Results

| arm | tool acc | routable | refusal | args | invented | median turn |
|---|---|---|---|---|---|---|
| `auto` | 0.500 | 0.444 | 0.875 | 0.407 | 0.722 | 2389 ms |
| `auto-min` | 0.565 | 0.518 | 0.875 | 0.444 | 0.352 | 1004 ms |
| `tuned` | **0.710** | **0.704** | 0.750 | **0.593** | 0.518 | 1496 ms |
| `tuned-min` | 0.613 | 0.611 | 0.625 | 0.537 | **0.222** | 958 ms |

Paired McNemar (same 62 queries per arm, exact two-sided):

| contrast | wins A | wins B | p |
|---|---|---|---|
| `auto` vs `tuned` — description authorship, full arity | 4 | 17 | **0.0072** |
| `auto-min` vs `tuned-min` — description authorship, minimal arity | 8 | 11 | 0.65 |
| `auto` vs `auto-min` — arity reduction, auto wording | 3 | 7 | 0.34 |
| `tuned` vs `tuned-min` — arity reduction, tuned wording | 8 | 2 | 0.11 |

Only one contrast is significant at n=62: rewriting the descriptions is worth
+26 points of routable accuracy when the full signature is declared. The arity
contrasts point in opposite directions depending on the wording, and neither
reaches significance on accuracy. Arity's effect lands on the confidence gate
instead.

### Argument arity and the confidence gate

The gate is the reason to run a 45M model at all: act above a threshold,
escalate below it. Sweeping the threshold over the calls each arm emits
(refusals excluded, since a refusal executes nothing at any threshold):

| threshold | `tuned` coverage / precision | `tuned-min` coverage / precision |
|---|---|---|
| 0.0 | 1.00 / 0.679 | 1.00 / 0.600 |
| 0.4 | 0.32 / 0.722 | 0.62 / 0.676 |
| 0.6 | 0.11 / 0.833 | 0.38 / 0.762 |
| 0.8 | 0.05 / 1.000 | 0.20 / 0.909 |
| 0.9 | 0.02 / 1.000 | 0.13 / 1.000 |

`tuned` reaches perfect precision by keeping one call in fifty. `tuned-min`
reaches it keeping one in eight, and every rung between is usable. The reason is
visible in the mean confidences: on correct calls `tuned` averages 0.309 and
`tuned-min` averages 0.584.

Declaring an optional argument the query does not license makes the model fill
it anyway — `get_user_posts(handle='pfrazee.com', limit=10)` for "what has
pfrazee.com been posting lately", where nothing in the query says 10 — and the
confidence head scores the whole call, so one invented argument drags the score
of an otherwise correct routing decision to 0.0004. Dropping optional arguments
cut the invented rate from 0.722 to 0.352 (`auto`) and from 0.518 to 0.222
(`tuned`), and raised mean confidence on correct calls in both wordings.

The practical rule: **declare only the arguments you need the model to fill.**
Not because it routes better, since that contrast is a wash, but because the
confidence score is otherwise uninformative, and the confidence score is the
product.

### The arity effect on a single-tool catalogue

The gate result above is measured across 18 tools, four wordings and a
retrieval stage, any of which could be carrying it. `arity_probe.py` removes all
three: **one** declared tool, so the routing decision is correct by
construction, 30 queries that name an account and license nothing else, and the
only thing that varies is how many optional arguments that one schema declares.

| declared | mean confidence | median | filled an unlicensed argument |
|---|---|---|---|
| `handle` | 0.199 | 0.073 | 0.000 |
| `handle`, `limit` | 0.111 | 0.028 | 0.733 |
| `handle`, `limit`, `transcribe` | 0.068 | 0.016 | 0.233 |

Paired two-sided sign tests over the same 30 queries:

| step | confidence down | up | p |
|---|---|---|---|
| `handle` → `+limit` | 21 | 9 | 0.04277 |
| `+limit` → `+transcribe` | 21 | 9 | 0.04277 |
| `handle` → both | 25 | 5 | **0.00032** |

Monotone and significant at every step, on a call that cannot be misrouted.
The 18-tool gate result is the same mechanism operating through a selection
stage.

Two things this also shows. The head is conservative in absolute terms — mean
0.199 even in the required-only condition on an unambiguous call — so a
threshold has to be fitted to a deployment rather than read as a probability.
And the model fills the unlicensed `limit` on 73% of queries when it is
declared, which is what the confidence head is reacting to.

### Retrieval versus selection

Needle renders five tools; 18 are declared. A wrong answer can mean the right
tool never entered the context, or that it did and the model picked a neighbour.
The Python surface does not expose which five were rendered, so `oracle.py`
measures the other regime: each query gets its own five-tool catalogue
containing the correct tool plus four seeded distractors.

| arm | routable, 18 declared | routable, oracle k=5 | retrieval cost |
|---|---|---|---|
| `auto` | 0.444 | 0.611 | +0.167 |
| `auto-min` | 0.518 | 0.500 | −0.018 |
| `tuned` | 0.704 | **0.815** | +0.111 |
| `tuned-min` | 0.611 | 0.778 | +0.167 |

With perfect retrieval and hand-written schemas the model reaches 81.5% top-1
over 54 queries. Between a third and two fifths of the errors at 18 declared
tools disappear when retrieval is replaced by an oracle: `tuned` goes from 16
errors to 10, `tuned-min` from 21 to 12. Two queries are suggestive of the
mechanism without proving it — `feed-02` ("read the feed at
at://…/app.bsky.feed.generator/…") and `trend-03` ("which topics are hot right
now, and how many posts each") both came back as refusals, at confidence 0.0006
and 0.9464. A refusal is what a rendered-but-unmatched catalogue and an
unrendered tool look like alike; the Python surface cannot distinguish them.

`auto-min`'s −0.018 is within noise (one query on n=54).

### Latency versus catalogue size

Holding the five tools the probe queries need in every subset and padding the
catalogue with the rest:

| tools declared | median turn | p90 |
|---|---|---|
| 5 | **290 ms** | 419 ms |
| 6 | 1051 ms | 1478 ms |
| 8 | 1100 ms | 1234 ms |
| 12 | 1123 ms | 1313 ms |
| 18 | 1187 ms | 1285 ms |

The sixth tool costs 3.63x the fifth. An earlier run of the same probe, hours
apart and before anything else was on the box, gave 284 / 1034 / 1054 / 1122 /
1109 ms — the same shape to within a few percent. Retrieval is a fixed ~750 ms per turn on
this CPU, charged the moment a sixth tool is declared and then almost flat out
to 18. `tool_index_path` does not touch it: with the index persisted, cold and
warm, the median turn stayed
1090–1124 ms against 1150 ms without. That is consistent with the documented
design: the index caches the *tool* embeddings, computed once at init, while
the *query* is embedded every turn.

Five or fewer tools per agent is one performance regime. Six and eighteen are
the other, and cost the same as each other.

One measurement artifact worth naming: `needle.Needle(...)` binds lazily, so
construction returns in ~0 ms and the engine load lands on the first
`complete()`. An `init_seconds` measured around the constructor measures
nothing, and the first query of any session carries ~2 s that belongs to setup.

### Two-stage routing over five-tool groups

Two results above point the same way. A five-tool catalogue containing the right
answer beats the flat 18 by 11–17 points, and the sixth declared tool costs 3.6x
the per-turn latency. Both say: keep every agent at five tools and pick the
five first. `two_stage.py` splits the 18 into five groups of ≤5 (`account`,
`follow_graph`, `one_post`, `find_content`, `plumbing`) and routes in two steps.

| stage 1 | stage-1 group accuracy | routable top-1 | args | invented | median end-to-end |
|---|---|---|---|---|---|
| a Needle turn over 5 group descriptions | 0.481 | 0.370 | 0.333 | 0.333 | 1702 ms |
| ~20 lines of regex over the query text | **0.870** | **0.722** | **0.685** | 0.111 | **316 ms** |
| — flat 18-tool `tuned-min`, for comparison | n/a | 0.611 | 0.537 | 0.222 | 1187 ms |
| — oracle five-tool ceiling, for comparison | n/a | 0.778 | 0.667 | 0.185 | 199 ms |

Asking Needle to classify a request into an abstract category is 24 points
**worse** than just handing it all 18 tools. Its errors are systematic rather
than noisy: of the 15 `account` queries it put 9 in `find_content` and got 3
right, and of the 8 `plumbing` queries it got 1 right. The model is trained to map
a concrete request onto a concrete callable, and a group description
("the request is about one named account…") is not one. Needle's own retrieval
head, a contrastive embedding rather than a decode, does the same 5-of-18 job far
better. Replacing it with a Needle turn moves backwards.

Replacing it with something deterministic is the right one. The regex stage
looks for structural cues only — does the text contain a post URI, a feed or
list URI, a `did:plc:`, a follow word, a handle-shaped token — and costs
microseconds. That combination lands at 0.722 routable, above the flat 18-tool
arm and most of the way to the five-tool ceiling, while cutting invented
arguments by half and running **3.8x faster end-to-end** than the flat arm —
316 ms against 1187 ms, because no agent it touches is ever above the retrieval
threshold. Accuracy was identical across two runs hours apart, one of them
against a saturated box, which is the determinism holding.

**The 0.870 is not a held-out number.** Those rules were written after reading
which groups the Needle classifier confused, so they are fitted to this eval's
distribution to an unknown degree. What generalises is the shape: the cheap
deterministic pre-filter beat the model-based one by 35 points, and a stage-1
error is unrecoverable because the right tool is then absent from stage 2's
catalogue entirely.

One implementation constraint found on the way. The engine holds **one global
session per process**: `_bind()` re-runs `needle_init` whenever a different
`needle.Needle` object is used, so a two-stage router in one process pays an
init on every turn, and a loaded `.cact` can never be unloaded (constructing a
base-weights agent after a tuned one raises rather than silently answering with
the tuned weights). Deploy the stages as separate processes.

### Where the base model fails

Per-category top-1, all four arms:

| category | n | `auto` | `auto-min` | `tuned` | `tuned-min` |
|---|---|---|---|---|---|
| analysis | 3 | 0.000 | 0.000 | 1.000 | 1.000 |
| feed | 3 | 0.333 | 1.000 | 0.667 | 0.000 |
| find-users | 3 | 0.667 | 0.667 | 1.000 | 0.333 |
| firehose | 3 | 0.333 | 0.000 | 0.667 | 0.667 |
| graph | 5 | 0.600 | 0.600 | 0.600 | 0.600 |
| identity | 3 | 0.000 | 0.000 | 0.333 | 0.333 |
| interactions | 6 | 1.000 | 0.833 | 0.667 | 0.833 |
| keywords | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| off-topic | 8 | 0.875 | 0.875 | 0.750 | 0.625 |
| person-posts | 5 | 0.200 | 1.000 | 0.800 | 0.600 |
| profile | 4 | 0.250 | 0.250 | 0.250 | 0.250 |
| search | 7 | 0.429 | 0.429 | 0.714 | 0.714 |
| status | 3 | 0.000 | 0.000 | 0.667 | 0.667 |
| thread | 3 | 0.333 | 0.333 | 1.000 | 0.667 |
| trending | 4 | 0.750 | 0.750 | 0.750 | 0.750 |

`profile` sits at 0.25 in every arm, which is the clearest single weakness. The
errors are all semantically adjacent: "how many followers does pfrazee.com have"
routes to `get_followers`, "look up the account jay.bsky.team" routes to
`get_user_posts`. `identity` fails because the model has no notion that "did"
and "pds" are things `resolve_identity` serves. `graph` sits at 0.600 in all
four arms on the `get_followers` / `get_following` pair, a distinction of one
preposition.

These are the failures a fine-tune should reach, and the ones a schema rewrite
already tried and could not.

## Fine-tune

800 synthetic rows (`gen_data.py`, 15% off-topic refusals, entity pools disjoint
from the eval set), LoRA rank 16 / alpha 32 on the frozen base, 3 epochs, batch
8, `max_len` 640, merged and exported to a 13.74 MB `.cact`. Roughly two hours
of 4-core CPU. Evaluated on the same 62 queries with the same `tuned-min`
schemas.

| | routable | args | refusal | invented | confidence |
|---|---|---|---|---|---|
| base | 0.611 | 0.537 | 0.625 | 0.222 | available |
| fine-tuned | 0.667 | 0.648 | **0.375** | 0.222 | **None** |

Paired McNemar over the same queries: tool selection 4 base-only wins against 5
fine-tuned-only, **p=1.0**; arguments 3 against 7, p=0.34. At this data scale
the fine-tune is a wash on routing and directionally positive on arguments,
with a real regression on refusing off-topic input — it answered "what is the
capital of Norway" with `get_trending_topics` and "block the account
spammer.bsky.social" with `get_followers`.

The per-category table is what makes the run worth reporting:

| category | n | base | fine-tuned |
|---|---|---|---|
| thread | 3 | 0.67 | 1.00 |
| interactions | 6 | 0.83 | 1.00 |
| graph | 5 | 0.60 | 0.80 |
| person-posts | 5 | 0.60 | 0.80 |
| **profile** | 4 | **0.25** | **0.25** |
| **identity** | 3 | **0.33** | **0.33** |
| analysis | 3 | 1.00 | 0.67 |
| search | 7 | 0.71 | 0.57 |
| off-topic | 8 | 0.62 | 0.38 |

`profile` and `identity` are the two categories the base model is worst at, and
both were covered by the training templates — "open the account page for X" for
`get_profile`, "where is X's repo hosted" and "turn <did> into a handle" for
`resolve_identity`. Neither moved at all. The confusions that survive are the
ones where two tools serve genuinely adjacent intents ("how many followers does
X have" is a profile field and a graph query), and 800 templated examples did
not separate them.

And the fine-tune costs the gate. Cactus documents that fine-tuning does not
update the confidence head, so a tuned agent reports `confidence` as `None` and
warns once at construction. Everything in the gate section above stops being
available the moment you load tuned weights. Against a wash on accuracy, that
is the whole trade: the tuning removed the one signal that made a 61% router
safe to act on.

Anyone wanting real gains here should spend the effort on catalogue size
instead. The regex stage-1 arm is +11 points for about 20 lines of code and
keeps the confidence head; this fine-tune is +5.6 points, not significant, and
gives it up.

Training loss reached 0.965 with validation 1.096 at the end of epoch 1. The
epoch 2 and 3 curves were not captured — the trainer's stdout was writing to an
inode that a concurrent run had unlinked, and only the first epoch survived in
a readable file (ERRORS.md #5). The exported model is unaffected; the loss
curve past epoch 1 simply is not in hand.


## Extraction

Needle treats extraction as tool calling with one declared tool, so the same
engine that routes a query can read a post and return a typed record.
`extract_demo.py` runs it over 8 live posts from `@austegard.com` in two schema
shapes, plus 6 constructed posts that carry their field values literally.

| declared | mean confidence | refused |
|---|---|---|
| `subject`, `mentions_handle`, `links_to`, `language` | 0.0105 | 3/8 |
| `subject` only | 0.0165 | 5/8 |
| `what`, `when`, `where`, `amount` on constructed posts | 0.0015 | 1/6 |

**Every extraction in all three conditions scored below 0.05.** The highest
single score across 22 attempts was 0.0434. At any threshold that made the
routing gate useful, all of it escalates.

The outputs say why. A field like `subject` ("the main thing the post is about,
in a few words") asks for a summary, and Needle's contract is that arguments
carry only values evidenced by the input — so it fills `subject` with a copied
fragment and `mentions_handle` with whatever span is nearby:
`{"subject": "Love this", "mentions_handle": "live"}` for the post "I love
this!", and `"language": "q4"` for a post that mentions a q4 quantization.

Span-shaped fields do better on content and no better on confidence. "Standup
moved to 09:30 CET starting Monday, room B2." returned
`{"what": "Standup", "when": "Monday", "where": "room B2"}`, which is right, at
confidence 0.0. The same schema put `"when": "Oslo"` on a for-sale post and
invented `"when": "2025-08-20"` from the Norwegian "20. august" — a year the
post does not contain, and the wrong one.

So for this domain the split is clean: route with it, do not extract with it.
The routing gate has an operating point; extraction has none. That is a
statement about free-form social text against these schemas, not about the
invoice-shaped extraction the model card demonstrates — but it is the shape a
Bluesky toolset would actually ask for.

## Running it

```bash
pip install cactus-needle                      # engine + weights fetched once from HF, then offline

python3 -m needle_bsky route "who liked <post url>"        # decide only, no network
python3 -m needle_bsky ask   "open the account page for austegard.com"
python3 -m needle_bsky ask   "..." --router flat           # declare all 18 to one agent
python3 -m needle_bsky repl                                # keeps the agent warm
python3 -m needle_bsky tools --arm tuned-min               # dump the declared schemas
```

The default router is the two-stage one measured above: a deterministic group
pick, then a ≤5-tool agent. `--router flat` declares all 18 to a single agent,
which is the arm most of this writeup measures.

`ask` applies the gate and exits 3 when it escalates, so a shell caller can
branch on "the small model was not sure" without parsing anything.
`demo.txt` is a live capture; the profile read there is a real AppView call.

Reproduce the measurements:

```bash
python3 eval.py --repeat 2       # four arms -> results_<arm>.json
python3 oracle.py --arm tuned    # oracle retrieval, per arm
python3 analyze.py               # paired McNemar, per-category, gate sweeps
python3 recheck.py               # every number in this file against the artifacts
```

## Caveats

- **n=62.** One query is 1.6 points of tool accuracy and 1.9 of routable
  accuracy. Only the `auto` vs `tuned` contrast survives a paired test; treat
  every other ordering here as directional.
- **One author wrote the eval set and the tuned schemas**, so the tuned arm is
  partly measuring agreement between two artifacts by the same writer. The
  `auto` arm is not exposed to this, since its wording predates the experiment by
  months, which is why the `auto` vs `tuned` gap is the contrast worth
  believing.
- **The eval set is not sampled from real traffic.** Queries were written to
  cover the catalogue and to include the confusions the surface actually has
  (`get_trending` vs `get_trending_topics`, followers vs following). A real
  query distribution would be far more concentrated.
- **CPU, 4 cores, shared.** Absolute latencies are the wrong number to carry
  anywhere; the ratios are the finding. Cactus reports 300–700 tok/s decode on
  sub-$200 phones, and this container measured 280–630 tok/s.
- **Reads only, and the router is not what makes it so.** Nothing in the
  catalogue posts, likes, follows or deletes. Four eval queries ask for a write
  ("post this to my bluesky account", "send a dm to…", "delete my last post",
  "block…"), and the router refuses only 3 of 4 under the `auto` wordings and
  2 of 4 under the tuned ones — "post this to my bluesky account: good morning"
  routes to `get_user_posts` or `get_likes` rather than returning the empty
  call. A read-only catalogue is the safety boundary here; the model's refusal
  behaviour is not one, and should not be treated as one for a catalogue that
  does contain write tools.
