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
