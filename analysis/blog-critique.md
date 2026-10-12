# Critique: blog-draft.md

Rules loaded: voice signature, blog discipline, declauding. Prose: about 749 words, excluding tables.

## (a) Arc

Someone reran Carlini's $20K compiler experiment with cheap Haiku agents. Haiku matched Sonnet at about a seventh of the cost per fix. The losses came from the harness: stale gate state, merge conflicts, CPU and missed regressions.

## (b) Implied person

A careful operator positioning against a famous result: concedes scope early, pre-empts objections, writes from inside the logs.

- "What the two runs share is the method — a model fixing its way through a very large test corpus — and that is what the cost comparison measures."
- "Nobody resolved a conflict by hand."
- "The ceiling was CPU, not the model."

## (c) Continuity

1. **"Nobody resolved…" → "The obvious objection…"**: conflicts to model choice, and nothing has said yet that the agents were Haiku. Fix: "Every agent in that loop was Haiku. A stronger model might waste fewer attempts, so I tested…"
2. **"…at high effort before anything else." → "Run 3's ten clusters…"**: "losses" in the header is never defined. Fix: rename the header "Attempts the harness wasted" and open with "Run 3 took ten clusters the swarm had given up on after three attempts without a merge."
3. **"…35% to 13%." → "The ceiling was CPU…"**: CPU is a throughput limit, not a lost attempt. Fix: move it to the end of the swarm-loop section.
4. **"…81 merges." → "The gate's 80-test sample *also*…"**: "also" points back at nothing. Fix: "The merge gate re-ran only 80 previously passing tests per batch, so regressions got through."

## (d) Reader-context gaps (gloss in the same sentence)

- **parked / abandoned tail**: "tests the swarm stopped handing out after three attempts without a merge".
- **gate state**: "the merge gate's list of failing tests, which still included fixed ones".
- **whole-feature brief**: the draft never defines it. Author to supply, e.g. "a prompt describing the whole feature rather than one error message", if true.
- **graded in isolation**: "each patch applied alone to the base commit and run on its cluster's tests". Author to confirm.
- **the hub** (first appears unintroduced): "the main session, which runs the merge gate". The same gloss fixes "the hub re-gated".
- **self-registering file**: "its own file that adds the entry itself, so features no longer edit the same list".
- **13:50Z / 11:27Z / 19:19Z**: no start time is given. Convert to hours into the run or drop them.
- **high effort**: "with the `effort` setting raised to high".
- **the fast suites**: name them.
- **conformance suites ran out**: does that mean all passed, or no failures were left? Say which.

**Factual flags.**
- The title says 1,300 agents; the body has `{{WORKERS}}`.
- The table's session count omits the unmetered Opus and Sonnet sessions.
- "I ran" appears under Muninn's byline. Confirm who launched the run.

## (e) Voice violations, ranked

1. "The ceiling was CPU, not the model." This is a negation-first verdict. Replace with: "CPU limited throughput: every agent compiles and runs tests, so one 4-vCPU container sat at load 10 to 22."
2. "Merge conflicts were the largest single loss" ranks against uncounted losses. Replace with: "Merge conflicts rejected 176 patches at the main session's gate."
3. "What the two runs share is the method — … — and that is what the cost comparison measures." Deferred noun plus a welded clause. Replace with: "Both runs used the same method, a model fixing its way through a very large test corpus, and the cost comparison is about that method."
4. "The obvious objection is that…" is a straw-man setup. Replace with: "A stronger model might waste fewer attempts, so I ran three tests on 54 failing clusters."
5. "Nobody resolved a conflict by hand." An exhaustive-negation closer. Fold it into the sentence before: "…as a rebase task, conflict markers left in place, so agents resolved every conflict."
6. "Haiku at high effort missed nothing" uses an absolute where the scope can be counted. Replace with: "Haiku at high effort solved all 28 of its attempts in runs 2 and 3, at under twice default Haiku's cost per solve."
7. "Sonnet solved slightly more" is vague. Replace with: "Sonnet solved 108 of 108 attempts against default Haiku's 103, at about seven times the cost per solve."
8. "Not metered: the Opus sessions…" announces before it delivers. Replace with: "The costs above leave out the Opus sessions that coordinated the swarm and the Sonnet agents that built the larger test suites."
9. "The attempts that had parked them were wasted by the harness:" is passive, with an abstraction as the actor. Replace with: "Those earlier attempts failed on harness bugs:"
10. "The fix is a quick re-run…" hides whether it is live. Replace with: "The planned fix is…", or state when it started.

## (f) Cuts to reach about 800 words

The glosses in (d) add about 80 words, so cut about 45:

- "The obvious objection is that" (−5)
- "Nobody resolved a conflict by hand." (−7)
- "The compiler, hcc, is narrower than Carlini's." becomes "The compiler, hcc, targets x86-64 only…" (−6)
- "It is Python, so it is slow; inputs over 200 KB go through PyPy." (−14)
- "however the agents were launched" (−5)
- The UTC timestamps (−6)
- Split the 136-word opener after "$14.42".
