# Rubric: what a turn of work in a coding-agent session deserves

A "turn" is one user message to an AI coding agent (Claude Code) plus the work the agent does in reply.

## effort: the least reasoning effort at which a capable model does this turn well
- low: acknowledgement, go-ahead, a named command to run, a simple lookup, a short factual or conversational answer, a mechanical edit the message spells out.
- medium: routine focused work with a clear deliverable: implement or fix a described change, summarise or review one thing, a standard multi-step task with no hard judgment.
- high: work that needs design judgment, debugging an unknown cause, careful analysis or critique, weighing trade-offs, or a plan others will build on.
- xhigh: hard reasoning where a subtle mistake is costly: research or experiment design, novel algorithms, subtle correctness or security, long autonomous projects with many dependent decisions.

## tier: the least capable model that would reliably do it well
- haiku: mechanical or fully specified; no judgment, design or debugging.
- sonnet: a bounded task whose deliverable the message names; nothing has to be found out first.
- opus: a goal whose work has to be discovered: find what needs changing then change it, audit, design, debug an unknown cause, synthesise across sources; needs initiative.
- fable: frontier-hard: novel research, subtle correctness, long autonomous projects spanning hours.

## scope
- quick: one answer or one small change, minutes.
- task: a focused task of several steps, up to about an hour.
- project: extended research, experiments or many changes over hours.

## delegate (foresight): should the agent hand part of this to subagents?
- none: do it inline.
- subagent: there is a separable chunk (a broad search, reading many files, an independent check) worth handing to one subagent.
- fanout: several independent pieces worth running in parallel subagents.
