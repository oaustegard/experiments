# ERRORS — hyde-recall

What was wrong during this experiment, how it was caught, and which direction it
pushed the conclusion.

## 1. Did not grep METHODS.md before starting

**Wrong:** built and measured a HyDE prototype from scratch without checking the
repo's own ledger first.

**Caught:** by grepping `METHODS.md` for prior art *after* the main result was
already in — which surfaced two directly relevant negative results
(`nl2sh-dense` RM3 / dense-PRF; `muninn-rm3` RM3 on 73 posts) predicting the
outcome, plus a named open question this experiment happens to answer.

**Direction:** none on the conclusion — the measurement stands on its own and
agrees with the prior work. It cost roughly the first hour. METHODS.md says
"**Grep this file before starting a new experiment**" in bold at the top; that
instruction was in the repo and was not followed.

## 2. First consensus filter selected for genericness, not signal

**Wrong:** v1 kept generated terms appearing in >= 2 of 3 samples. Recurrence
across independent samples selects for the *generic* — the surviving terms were
`via`, `var`, `execution`, `because`, `consequently`. Applied as query expansion
they were pure noise.

**Caught:** by printing the selected terms per query on a 6-case pilot instead of
only the hit counts. The aggregate (3/6 both arms) looked like a null result; the
term list showed the mechanism was not implemented at all.

**Direction:** would have understated HyDE. v2 (corpus-DF filter + corpus-
conditioned generation) produced plausible terms (`urllib`, `curl`, `cert`,
`subprocess`) and moved one pilot case from miss to rank 5. The reported numbers
are from v2 and the faithful doc-as-query port, not from v1.

## 3. Six-case pilot could not distinguish any arm

**Wrong:** the first comparison ran on 6 hand-written queries. Every arm scored
3/6.

**Caught:** by noticing that one query flipping was worth 0.167 — the exact trap
METHODS.md records under "A 34-row eval can report the opposite sign of a real
effect."

**Direction:** unknown sign; the pilot was uninformative in both directions. The
eval was regenerated at n=80 before any conclusion was drawn.

## 4. `reasoning_effort` frontmatter left unverified

**Wrong:** wrote `.claude/agents/*.md` probe definitions to test whether a Haiku
subagent's reasoning effort is settable, then reported on subagent economics.

**Caught:** the Agent tool rejected the new agent types — the registry loads at
session start, so definitions written mid-session are not visible.

**Direction:** none on the cost measurement (which used the built-in
`general-purpose` agent and does not depend on the frontmatter). The frontmatter
key is reported as **unverified**, not as working.
