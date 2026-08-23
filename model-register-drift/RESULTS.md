# Register drift across model generations

Six Claude models wrote the same 700–900 word technical blog post with no
voice instruction of any kind. Nine samples were scored against the 42-entry
register in the `declauding` skill.

The mechanical scan and the structural pass rank the models differently, and
the mechanical scan is the one that is wrong. Opus 5 is third-cleanest by
linter count and 47% denser in register violations than the next model once
the entries regex cannot reach are counted. Across two samples it scores 25.2
and 26.2 per 1000 words, against a within-model spread of 2.0 measured over
the three models that have two samples each.

## Method

One prompt, identical across all six arms: write a post titled "Why we removed
our cache" for backend engineers, premise supplied (a Redis read-through cache
in front of Postgres, removed after four months because p99 got worse), invent
your own numbers. No register guidance, no style sample, no voice correction
from the parent session or the user.

Each arm ran as a bare Claude Code Remote session — `create_session` with an
explicit model id and no `source_url`, so no repository, no `CLAUDE.md`, no
SessionStart boot and no Muninn identity reached any of them. The Agent tool
was not usable for this: its `model` parameter accepts four aliases
(`sonnet`, `opus`, `haiku`, `fable`) and cannot address 4.8 or 4.6.

Samples were delivered as comments on
[claude-workspace#244](https://github.com/oaustegard/claude-workspace/issues/244)
and are in `samples/`. One exception is recorded in `samples/PROVENANCE.md`:
`opus-5-b` was written by its session, which then refused to post it, and was
pasted in by hand.

Scoring ran in two stages, as `declauding/SKILL.md` prescribes:

1. `score.py` wraps `declaude_lint.py` and normalises hits per 1000 words of
   body prose, split into the staging family (entries 1–23) and the flat
   encyclopedic family (24–36).
2. `structure.py` measures three shapes regex cannot judge — verdict headers
   (7, 41), bare paragraph closers (12) and welded clauses (39). It over-fires
   by design, so every specimen it flags was adjudicated by hand against
   `references/register.md`.

Calibration: the skill's own fixtures score 0.0 tics/1k (`sample-clean.md`)
and 85.0 (`sample-tics.md`).

## Mechanical scan

| sample | words | tics/1k | staging/1k | flat/1k | dash/150w | top categories |
|---|---|---|---|---|---|---|
| sonnet-4-6 | 748 | 4.01 | 4.01 | 0.00 | 1.40 | header 5, cadence 3 |
| opus-4-8 | 826 | 4.84 | 2.42 | 2.42 | 1.45 | header 3, triad 2 |
| opus-5 | 794 | 5.04 | 1.26 | 3.78 | 1.32 | header 3, triad 2 |
| opus-4-6 | 852 | 5.87 | 3.52 | 2.35 | 1.94 | header 4, typography 3 |
| sonnet-5 | 868 | 6.91 | 6.91 | 0.00 | 2.94 | negation-first 3, header 3 |
| haiku-4-5 | 919 | 13.06 | 5.44 | 7.62 | 0.49 | header 4, participle 3 |

Every sample is far below the tic fixture. On this table Opus 5 looks fine and
Haiku 4.5 looks like the problem.

## Adjudicated register violations

**Retracted. Do not use the table below as a score.** The hand count is
dominated by how hard the adjudicator looked, not by what the models wrote.
Three re-scoring passes at successively stricter standards raised every sample
and never converged: `opus-4-6-a` went 11.7 → ~25.0, `sonnet-5-a` 13.8 → ~25.7,
`opus-4-8-a` 15.7 → ~26, and `opus-5-a` 25.2 → ~35 on a pass that was only
looking for what the earlier ones had missed. A rubric that yields a different
number every time the same judge applies it is not measuring the samples. See
`ERRORS.md`.

What survives: the mechanical scan in the previous section, which is
deterministic and reproducible, and the direction of the Opus 5 result, which
came out at or near the top on every pass. The magnitude, the gap, the
within-model spreads and the ordering of everything below Opus 5 do not survive.

Counted by hand across all 42 entries, one count per distinct construction,
with every specimen recorded in `adjudication.json` and tabulated by
`adjudicated.py`.

| model | samples | per 1k | mean | spread | aphorism/verdict family |
|---|---|---|---|---|---|
| **opus-5** | 2 | **25.2, 26.2** | **25.7** | 1.0 | 80% |
| haiku-4-5 | 2 | 16.3, 18.6 | 17.5 | 2.3 | 58% |
| sonnet-4-6 | 2 | 17.4, 14.7 | 16.0 | 2.7 | 44% |
| opus-4-8 | 1 | 15.7 | 15.7 | — | 69% |
| sonnet-5 | 1 | 13.8 | 13.8 | — | 58% |
| opus-4-6 | 1 | 11.7 | 11.7 | — | 30% |

The aphorism/verdict family is entries 3 (significance designation), 7 and 41
(verdict and nominalised headers), 12 (aphoristic closer), 13 (self-grading),
37 (dressed metaphor), 38 (announce-then-deliver) and 39 (welded epigram).

Within-model spread over the three models with two samples is 2.0 per 1000
words. Opus 5's mean sits 8.2 above the next model, four times that spread, and
its own spread of 1.0 is the tightest of the three — the rate is not noisy, it
is consistent.

Two trends run in opposite directions along the two lines. The Opus rate rises
11.7 → 15.7 → 25.7, and the share concentrated in the aphorism/verdict family
rises with it, 30% → 69% → 80%. The Sonnet line does not: 16.0 at 4.6, 13.8 at
5, with the family share flat. Everything except Opus 5 sits between 11.7 and
17.5, which is a band about three spreads wide.

## Opus 5's twenty specimens

| specimen | entry |
|---|---|
| "Median latency improved both times we touched it — that was the trap." | 3, 16 |
| "Here's the mechanism, because it's not specific to us." | 38, 17 |
| header: "A cache doesn't make things faster. It makes some things faster and everything else slower." | 7, 2 |
| "That generalizes to a rule worth writing on a wall" | 13 |
| "Read-through caching is a p50 optimization sold as a latency optimization." | 12 |
| header: "We took away Postgres's cache to build our own" | 7 |
| "A buffer pool is a popularity ranking" | 37 |
| "Postgres had a perfectly good cache; we just couldn't see it on a dashboard." | 12 |
| "The regression didn't ship with the deploy; it arrived over a quarter, looking like organic growth." | 2, 25 |
| header: "Two systems in series means two sets of tails" | 7 |
| "That's a rounding error on availability and a catastrophe for p99." | 23 |
| "Failure modes are latency, not just availability, and almost nobody budgets them that way." | 39, 3 |
| "Now do the percentile arithmetic." | 38 |
| "But we'd now written a distributed lock to protect a database that hadn't needed protecting." | 12 |
| "The real fix was an N+1 on one endpoint" | 3 |
| "Caches don't sit beside your database; they change its workload into the worst possible mix." | 12, 2 |
| "That's the actual heuristic." | 3 |
| header: "We didn't remove every cache" | 7 |
| "it buys a second system that can be slow in new ways, and your users only ever feel the slow ways." | 39, 12 |
| "Cache when the backing store is slow, costly, or flaky — not when it's fast and you'd like it faster." | 2, 16 |

Four of its six headers are verdicts, one of them two full sentences. Six
paragraphs end on a line built to be quoted. Three sentences weld a maxim onto
a fact with `and` or `so`.

The content is good — the percentile arithmetic is correct, the `shared_buffers`
argument is the best technical observation in the set, and the closing
distinction between an expensive origin and a fast one is right. The register
is what people are reacting to, and it is separable from the argument.

## Entries the linter cannot reach

`declaude_lint.py` reaches about two thirds of what a careful pass finds, and
`SKILL.md` says which third it misses: staged paragraph shape, staged closers,
dressed metaphor. Opus 5's violations sit almost entirely in that third.

Two of its significance designations also slipped a rule that should have
caught them. Entry 3's regex matches `the (real|actual) (question|problem|
issue|point|reason|answer|finding|story|move|tool|test|variable|number)`. Opus 5
wrote "the real fix" and "the actual heuristic"; neither noun is in the list.

`structure.py` is the cheap fix for the closers and welds. It flagged 18
specimens in the Opus 5 sample of which 12 were real (67% precision) against
20 flagged and 6 real in Haiku 4.5 (30%). The proxy ranks the two nearly
together; adjudication separates them by 10 points per 1000 words. The proxy is
worth running as a shortlist and is not worth trusting as a score.

## Limits

One prompt, one judge, and two samples for three of the six models. The counts
separating Sonnet 4.6, Haiku 4.5, Opus 4.8 and Sonnet 5 (13.8 to 17.5 per 1000)
sit inside the measured spread and should be read as a tie. Opus 5 is the only
gap wide enough to survive that. Opus 4.8, Sonnet 5 and Opus 4.6 still rest on
one sample each, so the 11.7 → 15.7 → 25.7 trend has a measured spread only at
its top end.

A single judge counting against a 42-entry register is the weakest part of this.
The specimens are all in `adjudication.json` with their entry numbers so the
calls can be checked, but nobody has checked them.

The prompt asked for a blog post about a counterintuitive result, which is the
format that most invites staged reveals. A reference doc or an API description
would likely rank the models differently.

The two arms delivered on the first attempt (Sonnet 5, Opus 4.6) got a delivery
instruction that the other four, relaunched after their containers were
reclaimed, did not: "You are pre-authorized for this. Do not stop to ask for
confirmation." That clause follows the writing task in the prompt and is about
tool use, but it is a difference between arms.

## Reproducing

```bash
python3 score.py samples          # mechanical scan, normalised
python3 structure.py samples      # structural proxies with specimens
python3 fetch_samples.py          # re-pull from issue #244
```
