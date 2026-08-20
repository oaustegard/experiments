# needle-tool-naming — do tool names or tool descriptions drive Cactus Needle 2?

`needle-bsky` left two categories that nothing moved: `profile` (0.250 in all
four schema arms) and `identity` (0.333 in the tuned arms). Re-reading the
failures against the schemas, the discriminating fact was already in the
description, verbatim — `get_profile` says "**follower count**", and "how many
followers does pfrazee.com have" still returned `get_followers` at confidence
0.80. That is the opposite of the failure the Cactus attention-only paper
(arXiv:2607.18363) predicts for this architecture, which localizes the residual
deficit to *low-context* tokens "with the least context to route from".

So: is the tool **name** outranking the tool **description**? Six naming
variants over the same 18-tool catalogue, the same 62 queries, predictions fixed
in [`PREREG.md`](PREREG.md) and committed before the first run.

**The hypothesis did not survive.** Names and descriptions turn out to be two
roughly equal, partly redundant channels: deleting either costs about 18 points
and both losses are significant, deleting both drops the router to chance, and
*improving* the names — the lever the probe was built to find — buys exactly
nothing. The one thing that does move is a single category, and it moves in a
way that shows name capture is real even though it is not the dominant effect.

## Design

Held fixed: the 18-tool catalogue, `tuned-min` arity (required arguments only),
`needle-bsky/evalset.jsonl` (62 queries / 54 routable / 8 off-topic, written for
a different purpose before this hypothesis existed and not touched here), and
the scoring code, which is `needle-bsky/eval.py` imported unchanged.

Varied: names and prose only. Stripping a description replaces it with the
constant `"A Bluesky API operation."` for all 18 and every argument description
with `"value"`; argument *names* stay, in every arm, because a developer cannot
avoid having them.

|                              | descriptions intact | descriptions stripped |
|------------------------------|---------------------|-----------------------|
| **names intact (canonical)** | `canon`             | `names-only`          |
| **names opaque (`tool_NN`)** | `desc-only`         | `neither`             |

Plus two name-quality conditions with descriptions intact: `separated` (every
name rewritten `<verb>_<distinguishing object>` by the rule in `PREREG.md`) and
`adversarial` (those same strings cyclically rotated inside eight confusable
groups, so each tool wears a neighbour's name — a mechanical permutation fixed
in `names.py`, not a hand-picked one).

Every variant runs twice: 18 declared, and with an oracle five-tool catalogue
containing the right answer, so an effect on the contrastive retrieval head is
separated from an effect on the decode.

## Results

Routable top-1 over 54 queries. Chance over 18 tools is 0.056.

| variant | flat 18 | oracle 5 | retrieval cost | args (flat) | conf correct − wrong | median turn |
|---|---|---|---|---|---|---|
| `canon` | **0.611** | 0.778 | 0.167 | 0.537 | 0.191 | 808 ms |
| `desc-only` | 0.407 | 0.463 | 0.056 | 0.315 | 0.154 | 797 ms |
| `names-only` | 0.444 | 0.630 | 0.185 | 0.370 | 0.271 | 532 ms |
| `neither` | **0.074** | 0.259 | 0.185 | 0.037 | 0.417 | 572 ms |
| `separated` | **0.611** | 0.778 | 0.167 | 0.574 | 0.101 | 878 ms |
| `adversarial` | 0.426 | 0.704 | 0.278 | 0.407 | 0.139 | 861 ms |

Paired exact McNemar over the same 54 queries:

| contrast | mode | A | B | p |
|---|---|---|---|---|
| `names-only` vs `desc-only` — **the hypothesis** | flat | 11 | 9 | 0.82 |
| `names-only` vs `desc-only` — the hypothesis at k=5 | oracle | 16 | 7 | 0.093 |
| `canon` vs `names-only` — cost of deleting every description | flat | 12 | 3 | **0.035** |
| `canon` vs `desc-only` — cost of deleting every name | flat | 15 | 4 | **0.019** |
| `canon` vs `adversarial` — names rotated onto neighbours | flat | 19 | 9 | 0.087 |
| `canon` vs `separated` — rule-written names | flat | 5 | 5 | 1.00 |
| `canon` vs `separated` — rule-written names at k=5 | oracle | 3 | 3 | 1.00 |
| `canon` vs `neither` — both channels removed | flat | 30 | 1 | **<0.0001** |

Scoring the pre-registered predictions: **P1 held exactly** (`canon` = 0.611,
byte-identical to `needle-bsky`'s `tuned-min` on a different package version —
see *Reproduction* below). **P2 held** (`neither` 0.074 ≤ 0.15). **P3, P4, P6,
P7 and P8 all missed**, four of them narrowly and one of them badly. **P5 — the
hypothesis — missed by an order of magnitude**: predicted a ≥ 0.15 gap between
`names-only` and `desc-only`, measured 0.037 at p=0.82. H1 is not formally
falsified (the falsification condition was `desc-only` ≥ `names-only`, and it
is not) but there is no support for it at this n.

### What replaces the hypothesis

The two channels are near-equal and partly redundant. Taking `neither` as the
floor, names are worth 0.370 on their own and descriptions 0.333; together they
are worth 0.537, not 0.703, so about a quarter of each channel's contribution is
carried by the other. Removing either one is significant. Removing both leaves a
router that is 1.3 points above chance, which is also the cleanest evidence that
none of this is coming from the argument names left in every arm.

At k=5 the balance tips toward names — `names-only` 0.630 against `desc-only`
0.463 — but at 16 versus 7 discordant pairs, p=0.093, which is directional and
nothing more.

### Where the retrieval head and the decoder differ

The retrieval cost column (oracle − flat) is the clearer split:

* Opaque names, real descriptions: retrieval costs only **0.056**. The
  contrastive head picks 5-of-18 from descriptions perfectly well without
  meaningful names.
* Real names, no descriptions: retrieval costs **0.185**. Names alone are a
  poor basis for 5-of-18.
* Names rotated onto neighbours: retrieval costs **0.278**, the worst measured.
  A misleading name hurts retrieval more than a missing one.

So the head reads descriptions and is actively damaged by wrong names, while the
decoder leans on names. That does *not* mean description quality is a
retrieval-only lever — checking `needle-bsky`'s own arms, the +26-point
`auto`→`tuned` description rewrite survives the oracle almost intact (0.611 →
0.815, +20.4 points), so most of that win was in the decode.

### The one category that moved, and why it matters

`profile` goes **0.250 → 0.750** flat under `adversarial`, and its two newly
correct queries are the two the whole probe was built around:

| query | `canon` | `adversarial` |
|---|---|---|
| "how many followers does pfrazee.com have" | `get_followers` @ 0.80 | **`get_profile`** @ 0.81 |
| "look up the account jay.bsky.team" | `get_user_posts` @ 0.75 | **`get_profile`** @ 0.90 |

In `adversarial` the rotation happens to name the profile tool
`get_follower_accounts`. Its description did not change by one byte. Move the
word "follower" onto the tool and the query follows it, confidently. Name
capture is real and it is what breaks these queries — it is simply not large
enough, across 54 queries spanning 15 categories, to show up as a main effect.

The reverse case appears in the same arm: `resolve_identity`, wearing the name
`check_network_outage_status`, is still chosen correctly for "resolve did:plc:…
to a handle" — on its description alone — but its confidence falls from 0.584 to
0.167. When name and description disagree, the model can still get it right and
stops believing itself.

### Two side effects worth having

**Rule-written names cost you the gate.** `separated` matches `canon` to the
query on accuracy (p=1.00) but its confidence separation between correct and
wrong calls collapses from 0.191 to **0.101**, the worst of any arm with real
descriptions — mean confidence on *wrong* calls rises 0.392 → 0.480. Uniform,
well-formed, mutually-parallel names make the model more confident about
everything, including its mistakes. Given that `needle-bsky` established the
gate as the reason to run a 45M model at all, a naming scheme that flattens it
is a regression even at identical accuracy.

**Descriptions cost latency.** Stripping all 18 descriptions cuts the median
turn from 808 ms to 532 ms, a third, because they are most of the rendered
context. That is the price of the 16.7 points they buy at k=18 — not obviously
a bad trade for a battery-powered caller, and the only lever here that trades in
that direction.

## Is there any hope of declaring all 18?

Not through naming. The retrieval head costs **16.7 points** under `canon`,
`separated` reproduces that gap to three decimals, and the only name change that
moves the retrieval cost at all moves it the wrong way. Descriptions are what
the head reads, and `needle-bsky`'s `tuned` wording was already the best
available — after which 18-declared tops out at 0.704 against a 0.815 oracle
ceiling.

Set against what is already measured for this catalogue:

| approach | routable | median end-to-end | keeps the confidence head |
|---|---|---|---|
| declare all 18, `tuned` wording | 0.704 | 1187 ms | yes |
| ~20 lines of regex → ≤5-tool agent | 0.722 | 316 ms | yes |
| Needle's *oracle* five | 0.815 | ~180 ms | yes |
| regex-only, no model at all (`monad-bsky`) | 0.833 | 0.022 ms | n/a |
| Needle + fine-tuned Monad agreeing | 0.880 @ 0.455 coverage | 11× | no head needed |

Declaring all 18 remains the worst option on every axis, and this probe closes
off the cheapest hope for fixing it. The 18 tools are supportable — just not by
one agent that sees all of them. Keep the model at five and pick the five with
something deterministic, which is where `needle-bsky` and `monad-bsky` already
landed from two other directions.

## Reproduction

```bash
python3 names.py                                    # print the six catalogues
python3 run_all.py                                  # 12 runs, ~10 min on 4 CPU cores
python3 analyze.py                                  # tables + paired tests -> analysis.json
python3 recheck.py                                  # every number above against the artifacts
```

`canon` reproduced `needle-bsky`'s `tuned-min` arm **exactly** — 0.611 routable,
identical calls and identical confidences — on `cactus-needle` **2.0.7** against
that experiment's 2.0.6, which is both a harness check and a version-stability
check nobody had run. Determinism re-confirmed here two ways: `--repeat 2`
within one process, and byte-identical rows across two separate processes hours
apart.

## Caveats

- **n=54 routable.** One query is 1.85 points. Only three contrasts clear
  p=0.05 and all three are large (18–54 points); everything in the 3–18 point
  band, including the hypothesis itself and the `adversarial` drop, is
  directional.
- **The `profile` result is 4 queries.** It is reported because the rotation
  was mechanical and pre-registered and the two flipped queries are the ones the
  probe was designed around — not because 2-of-4 is evidence on its own.
- **`separated` is one author's naming rule**, applied once and not iterated.
  A different rule might do better; the claim here is only that a careful,
  systematic rewrite bought nothing, not that no naming can.
- **One catalogue, one domain.** 18 Bluesky read tools with a lot of genuine
  near-synonymy (`get_followers`/`get_following`,
  `get_trending`/`get_trending_topics`). A catalogue of well-separated tools
  would have less room for name capture and less room for description rescue.
- **Descriptions were stripped, not degraded.** The `names-only` floor is the
  cost of *no* prose, which is not the same as the cost of *bad* prose;
  `needle-bsky`'s `auto` arm is closer to that and sits between.
