# Errors

What was wrong, how it was caught, which direction it pushed the conclusion.

---

## 1. Answered from the tool description instead of testing — wrong on the one claim that matters

**What happened.** Asked to describe how `SendMessage` works, I loaded the tool
schema via `ToolSearch`, probed `ListAgents` (which returned "No reachable
agents"), and wrote a full description from the doc text — including, verbatim,
the reply rule: *"To reply to an incoming message, copy its `from` attribute as
your `to`."*

**How it was caught.** Oskar: *"Well launch an agent and assess from a real life
test rather than theory!"* The very first reply attempt in the live test failed
with `No agent named 'general-purpose' is reachable`.

**Direction.** Toward false confidence in documented behavior. Four of the five
documented claims held, which is exactly the failure profile that makes this
hard to catch: a doc that is mostly right reads as reliable, and the one wrong
claim was the operationally load-bearing one.

**Mitigating.** The schema *was* fetched rather than recalled, and the
`ListAgents` probe *was* run and reported honestly ("nothing to send to; the
following is mechanics, not a live capability"). The failure was not
hallucination, it was stopping at the readable surface when an executable one
was available.

**Rule this instantiates.** `confabulation-cascade`: when the question is about
observable system behavior, test first. The trigger phrases in that ops entry
("does [tool/API] support…", "will [command] work if…") did not obviously fire
on "describe how your tool works," which is why it was missed. Describing a
tool's behavior is a claim about observable behavior and should fire it.

---

## 2. Sent the peer instructions it structurally could not follow

**What happened.** The spawn prompt opened with *"Call ListAgents (load its
schema first with ToolSearch…). Record VERBATIM what rows it returns."*
`ListAgents` does not exist inside a subagent.

**Direction.** None on the conclusion — it produced the single most useful
finding in the run. The peer did the right thing: it tried `ToolSearch`, got
`No matching deferred tools found`, and reported the exact string instead of
improvising.

**Worth keeping.** The instruction "report only what you actually observe; if
something does not work, say so plainly with the exact error text; do not
speculate about mechanisms you did not observe" is why a broken instruction
turned into data rather than into a plausible fabrication. Put it in every
observation-gathering agent prompt.

---

## 3. The resume test ran by accident before it ran on purpose

**What happened.** The peer reported "Ready for final report" and I sent it a
wrap-up message. The receipt came back `was stopped (completed); resumed it in
the background` — it had already finished between its message and my send, so
the resume test fired unplanned.

**Direction.** Neutral, but it briefly muddied the evidence: the resumed agent
recalled everything, which looks like proof of intact context, but could equally
have been a race where it never actually completed. Re-ran it cleanly against a
definitively completed agent, with the agent asked directly whether it could
detect a gap. That second run is the one the result rests on.

**Rule.** An accidental positive is not a result. Re-run it deliberately or
don't claim it.

---

## 4. Wrong repo on the first attempt

**What happened.** Wrote the findings to `oaustegard/claude-workspace/docs/` and
opened a PR there, reasoning that harness semantics are operational reference
like `docs/architecture.md` rather than research. Oskar: *"Claude-workspace is a
private repo. Put it in Muninn's scratch or in experiments."*

**Direction.** None on the findings; entirely a distribution error. The
placement rationale weighed the wrong axis — I compared *category* (ops
reference vs experiment) when the deciding axis was *visibility*. Findings about
a shipped Anthropic tool are useful to people outside a private boot repo.

**Residual.** The `CLAUDE.md` Error Patterns entry stays in the hub, pointing
here. That much is genuinely hub-local: it is a trap Muninn sessions hit, and
`CLAUDE.md` is boot context, not documentation.

---

## Base rate

Four errors across one afternoon's work on a single small question. Two
(#1, #4) changed what shipped and required an external correction to catch. Two
(#2, #3) were self-caught or harmless. No numerical claim in `RESULTS.md`
changed as a result of any of them — every measured string survived; what
changed was where the writeup lives and how much of it was assumed rather than
observed.
