# Errors

## Four of six delegates were archived while parked, losing their drafts

The first launch sent all six sessions a delivery step that read "post it as a
comment on GitHub issue oaustegard/claude-workspace#244". Two did it. Four
treated posting to GitHub as an outward-facing action needing confirmation and
parked: Opus 5 asked for curl permission to "your blog endpoint", Opus 4.8
asked which delivery method to use, Sonnet 4.6 asked for confirmation to post,
and Haiku 4.5 hit an `add_repo` permission prompt the bare child could not clear.

All four had finished writing. By the time the parent turn resumed to poke them,
their containers had been reclaimed and `create_trigger` refused the binding —
`session is archived; only active sessions can be bound to triggers`. The drafts
went with the containers.

Direction of the effect: none on the numbers, because the four were relaunched
from the same prompt. It does mean two arms ran under a slightly different
delivery clause than the other four, which `RESULTS.md` records under Limits.

The fix on relaunch was two changes to the delivery section — "You are
pre-authorized for this and it is the entire point of the task. Do not stop to
ask for confirmation" — and passing `extra_allowed_tools` with both spellings
of the `add_repo` tool name plus `mcp__github__add_issue_comment`. All four then
delivered without parking.

Generalisable: a spawned session that must perform one outward-facing action
needs that action pre-authorised in the prompt and pre-approved in
`extra_allowed_tools`. A parked delegate is invisible until someone looks, and
the container reclaim window is shorter than a parent's turn can be.

## The structural proxy was nearly reported as a score

`structure.py` ranked Haiku 4.5 worst (24.29/1k) and Opus 5 second (22.56/1k).
Adjudicating the specimens against `references/register.md` reversed it: 12 of
Opus 5's 18 flags were real (67%), against 6 of Haiku 4.5's 20 (30%).

The proxy fires on any paragraph-final sentence with no number, identifier or
proper noun. That shape catches an aphoristic closer and also catches "We
congratulated ourselves at retro" and "Under high concurrency, this added
queuing latency", which are ordinary sentences. Precision varies by model, so
the raw count is not comparable across models even though it looks like it is.

Caught before publication, so the direction is hypothetical: reported raw, it
would have understated the Opus 5 gap by about 10 violations per 1000 words and
named the wrong model as worst. `RESULTS.md` reports the proxy as a shortlist
with its precision on both ends, and the adjudicated count as the score.

## Judge drift: early samples were counted more leniently than late ones

Opus 4.8 scored 15.7 per 1000 words on its first sample and 28.1 on its second,
a spread of 12.4. The other three models with two samples spread 1.0, 2.3 and
2.7. A single model does not become twice as tic-dense between two runs of the
same prompt while three others hold within three points.

Re-reading `opus-4-8-a` against the standard used for `opus-4-8-b` found about
thirteen violations missed the first time — three bolded verdict labels
("Serialization isn't free.", "Cache stampedes concentrate the pain."), four
aphoristic closers ("So the miss path is strictly slower than the no-cache path
ever was.", "We had turned a 90ms database read into a 130ms cache miss."), a
bulleted list of bolded rhetorical questions, a forced triad, and several
em-dash gotchas. The first six samples were adjudicated before the register was
fully in working memory; the last three after.

Direction: the early counts are biased low, so any sample adjudicated early is
understated relative to one adjudicated late. That corrupts every cross-model
comparison, because the samples were not adjudicated in a random order — the
first-round samples were all counted early and the second-round ones late.

One check partly survives it. `opus-5-a` was counted early and scored 25.2;
`opus-5-b` was counted late and scored 26.2. If the drift were uniformly ~12
points, `opus-5-b` should have landed near 37. It did not, which suggests the
drift is not a flat offset but a failure to catch a specific class of specimen —
mainly bolded verdict labels and closers inside bulleted sections — that Opus 5
was being charged for from the start because its were harder to miss.

That was the hypothesis. Re-adjudicating did not fix it — it made the problem
visible. A third pass, looking only for what the second had missed, raised
`opus-4-6-a` from 11.7 to ~25.0, `sonnet-5-a` from 13.8 to ~25.7, `opus-4-8-a`
from 15.7 to ~26, and `opus-5-a` from 25.2 to ~35. Every sample rose. Nothing
converged. The count is a function of adjudication effort, and adjudication
effort has no natural stopping point, so there is no "correct" count to iterate
toward.

The numbers in `RESULTS.md` are retracted rather than provisional. Iterating my
own passes cannot fix this, because each pass finds more and I have no way to
tell a thorough pass from an overzealous one from the inside.

The general rule: a hand-scored rubric applied across a session drifts as the
scorer learns the rubric. Score in one sitting, in randomised order, or score
twice and report the second pass. Counting in the order the data arrives
guarantees the drift aligns with the variable under test.


## What would actually work

Three options, in increasing cost:

1. **Deterministic scoring only.** Report `declaude_lint.py` and stop. It is
   reproducible and it misses a third of the register, and the third it misses
   is the interesting third — but a reproducible partial measure beats an
   irreproducible complete one.
2. **Independent judges.** One agent per sample, each given the register and one
   sample, blind to which model wrote it, with the model attribution stripped
   from the text. Disagreement between judges is then measurable, which is the
   thing this experiment currently has no estimate of.
3. **Forced-choice pairwise.** Skip counting. Show a judge two samples and ask
   which reads more like an LLM wrote it. Pairwise preference is far more stable
   than absolute counts, and a Bradley-Terry fit over enough pairs gives a
   ranking with confidence intervals. With 10 samples that is 45 pairs.

Option 3 is the right instrument for the question actually being asked, which
was never "how many tics per 1000 words" but "which of these reads worse."
