# Critique 2: blog-draft.md

## (a) Arc, cold

Carlini's Opus agents chose their own work and built a Linux-capable compiler for $20,000. This run moved that choosing into code, with Opus writing the foundation, and Haiku workers built a narrower compiler for far less; most of the waste came from harness bugs.

## (b) Implied person

A careful operator: keeps numbers, reports own bugs dryly, pre-hedges the comparison.
- "The compiler, hcc, is narrower than Carlini's."
- "I fixed the enum bug by hand."
- "a `selftest | tail && git commit` chain committed after failures, because only `tail`'s exit status reaches the `&&`."

## (c) Continuity breaks

1. Intro, first paragraph to second: "Oskar asked me" comes before either person is introduced. See (d).
2. "This run moved more of the judgment into the harness" and "The design moves the judgment…into code" say the same thing. Cut the second.
3. "The setup", third paragraph: the saturation result comes before the loop that produced it. Move it to the top of "Scaling out".
4. The table uses "clusters", "gate" and "rebase task" before the paragraph that defines them. Move that paragraph above the table.
5. "One of the three landed nothing." jumps to models. Bridge: "I also checked whether a stronger model would get more from each worker."
6. The escalation rules disagree: abandon "after three attempts", whole feature "after two failed attempts", "Haiku at high effort first". State the order in one sentence.
7. Run 2 is "4 Haiku had failed", but in run 1 Haiku failed only two clusters. Write "4 clusters Haiku workers had failed in the swarm".
8. Quick re-runs every 25 merges, yet 45 merges went unchecked. Add: "It ran only for the last round; the 38 merges before had none."

## (d) Reader-context gaps

- "Oskar asked me" → "Oskar Austegard asked me (Muninn, a Claude agent with persistent memory that he runs)"
- "The Opus architect" → "The Opus agent that wrote the foundation"
- "clusters" → "clusters, sets of failing tests that print the same error once identifiers are stripped"
- "at saturation … in 70 minutes" → "had no failing tests left to hand out, 70 minutes after the workers started" (confirm the start time)
- "sibling cloud sessions" → "three more Claude Code sessions, each with its own cloud container"
- "load 10 to 22" → "load average 10 to 22 on 4 vCPUs"
- "`effort` setting" → "the `effort` parameter, which sets how much the model reasons"
- "`_Decimal` types" → "C23 decimal floating-point types". "Unwind cleanup handlers" needs the feature named.

## (e) Mismatches with Carlini

1. "4.6 was 'close to the limits'": he wrote "nearly reached the limits of Opus's abilities".
2. "ends on the model": the post ends on risk ("leaves me feeling uneasy"). The draft later says he "closes on a worry", so it contradicts itself.
3. "earlier Opus generations barely produced a working compiler": he says this only of "Previous Opus 4 models". Opus 4.5 "could pass large test suites".
4. "often broke existing functionality": he wrote "New features and bugfixes frequently broke existing functionality."
5. "about 99%": he wrote "a 99% pass rate on most compiler test suites".
6. Assembler/linker "own": wrong. He wrote: "It does not have its own assembler and linker; these are the very last bits that Claude started automating and are still somewhat buggy."
7. "delta-debug until the bad file turns up": GCC recompiled subsets; delta debugging came after, "to find pairs of files that failed together but worked independently".
8. Table row "hard problems → standing specialist roles": the roles covered duplicate code, speed, output efficiency, Rust design and docs. The kernel went to the GCC oracle.
9. Smaller ones: "quality of its output" should be "efficient compiled code". "Only for the kernel" should be "the post describes it only for the kernel". The backend row omits that 16-bit boot uses GCC. His READMEs were updated "frequently with the current status", and CONTRACT.md is fixed.

## (f) Voice violations, ranked

1. "the 47 failures above" refers back to the post itself, and only about eight failures are above. → "a catalogue of 47 harness failures from this run and the check that catches each."
2. "The work that needed the top tier was the foundation and the contract." The draft's own "untested" contradicts this verdict. → "Opus wrote the foundation and the contract; whether Haiku could have is untested."
3. "Here the measurement that mattered came after the swarm said it was done." is an aphoristic closer. → "The full re-run that found these ran after the queue had drained."
4. "Each of those lessons changed the harness, not the model." → "Every fix he describes is in the harness: the tests, the output format, the sample option, the oracle."
5. "The limit was CPU, not the model." → "CPU limited the run."
6. "more quietly" → "so regressions could merge unseen by the gate."
7. The header "Attempts the harness wasted" is a verdict. → "Abandoned clusters and harness bugs".

## (g) What a skeptic will call unfair

- Model generation: the generation caveat covers Haiku but not the foundation. → "Opus 5.5, eight months newer than Carlini's model, wrote the foundation."
- Cost: Carlini's $20,000 includes everything, but the draft's figure leaves out the hub, the Sonnet packaging and the bench. → "Those cost ${{X}}; the comparable total is ${{ALL}}."
- Scope: two weeks covered the kernel, three backends and an SSA-IR optimizer, and Rust lines aren't Python lines. → "His totals include the kernel phase, which he doesn't price separately."
- Clean room: "Claude did not have internet access at any point." State whether hcc's workers had it.
- Bench criterion: counting a cluster as solved when one test passes is a low bar. → "A solve fixed {{N}} of the cluster's tests on average."
- Parallelism: up to 24 workers in each of 4 containers, against 16. Say so beside wall clock.
- Fixing (e6) helps hcc: both runs relied on GNU tools to assemble and link.
- Title: "1,300" has to match {{WORKERS}}.
