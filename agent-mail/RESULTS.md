# agent-mail: how Haiku subagents should talk to each other

Question (Oskar, 2026-10-14): earlier Haiku swarms gave each agent one task and
coordinated them only through a shared tree and a merge gate. What is the most
effective and efficient way for peer agents to introduce themselves and
exchange messages?

## The mechanism built here

`mail.py` is a file mailbox: an append-only `log.jsonl`, a `roster.jsonl` of one
card per agent, and a per-reader cursor. `hook.py` is a `PostToolUse` hook that
binds a subagent's `agent_id` to its name the first time it runs
`mail join|send|recv NAME`. After that, every tool call the agent makes returns
with its unread mail and any new or changed teammate cards as
`additionalContext`. Nobody polls. `mail recv NAME --wait N` blocks until mail
arrives, so an agent with nothing to do can wait for a question rather than
end its turn.

A per-team wrapper `<tree>/mail` pins the board to `<tree>/.board`, so several
teams can run at once. The hook is wired in `.claude/settings.local.json`,
which Claude Code picks up mid-session.

## Probes (2026-10-14, Claude Code 2.1.296, CCotw)

- **Native peer-to-peer works now.** Two Haiku subagents exchanged
  `SendMessage` pings by agentId. The envelope's `from` is now the sender's
  agentId, so a reply to `from` reaches the sender. This changed since
  `subagent-messaging/` (2026-08-12), when `from` was the agent type and replies
  failed.
- **`ListAgents` is still absent inside a subagent**, so the parent has to hand
  out the roster after spawning everyone.
- A send to a finished subagent resumes it from its transcript.
- The hook's `additionalContext` reaches the subagent as
  `PostToolUse:Bash hook additional context`.
- A foreground `sleep N` is blocked by a hub guard, so probes cannot simply
  wait. `mail recv --wait` is the wait primitive.

## Round 1: a 4-agent build with split knowledge

**Task** (`task/`): build a 4-module `orders` package (parser, inventory,
pricing, report/entry point). Each agent owns one file and sees only its own
section of the spec. None of the interfaces between modules are specified.
Grading uses 48 hidden end-to-end cases generated from `harness/ref.py`.

There were five arms, each one team of 4 Haiku 5.5 `general-purpose`
subagents. All 20 agents ran concurrently.

| arm | channel | intro |
|---|---|---|
| none | none; agents can read each other's files | — |
| native-plain | `SendMessage`; the parent sends the agentId roster after spawn | free-form |
| native-contract | `SendMessage` | each agent's first message is its exact interface contract |
| mail-plain | mailbox + push hook | join card: name and file |
| mail-contract | mailbox + push hook | join card IS the interface contract |

**Results** (`runs/r1/summary.txt`; `usd` is the input+output cost at the
Haiku 5.5 rate card):

| arm | pass | messages | tool calls | turns | output tokens | usd | wall |
|---|---|---|---|---|---|---|---|
| none | 48/48 | 0 | 30 | 27 | 859 | 0.030 | 48 s |
| mail-contract | 48/48 | 7 | 37 | 36 | 643 | 0.036 | 65 s |
| mail-plain | 48/48 | 17 | 54 | 53 | 610 | 0.049 | 72 s |
| native-plain | 48/48 | 19 | 62 | 55 | 1,094 | 0.053 | 63 s |
| native-contract | 48/48 | 37 | 82 | 59 | 2,045 | 0.059 | 75 s |

An audit of all 20 transcripts found no agent reading another agent's brief,
another arm's tree, or the hidden cases.

## What round 1 shows

**Every arm hit the ceiling, including no channel.** The only coupling in this
task was interfaces. dave owned the entry point, wrote it last, read his
teammates' files, and adapted to them. A shared tree that agents can read is
already a sufficient channel for interface agreement. This round therefore
measures what each channel costs, not what it buys.

**The contract-first card cut mailbox traffic from 17 messages to 7.** In
mail-plain, all four agents broadcast a proposal at the same moment, then
spent six messages converging on key names (`line` vs `line_no`, `order` vs
`order_id`). One proposal tried to drop the merge rule, and alice caught it
because only she held that rule. In mail-contract, every card was a full
signature from the start. The remaining messages were `done` notices and
`SHIP`.

**Native messaging costs the most and the contract intro made it worse.**
Agents started before the roster arrived, so several first sends to bare
names failed. Each contract broadcast or `done` notice woke peers that had
already finished. native-contract alice handed back six times, each time
replaying her transcript. With no blocking receive, a native agent can only
end its turn and be resumed.

**Mailbox + push hook + contract card is the efficient design**, at 1.2x the
no-channel cost. It had no failed sends, no resumes, and `recv --wait` kept
agents available until `SHIP`.

**Caveats:** one replicate per arm, one task, Haiku only. The task did not
require communication, so this round says nothing about whether any channel
improves correctness. Round 2 needs a task where the shared tree cannot carry
the needed information: a rule known to one agent that changes another
agent's code, which cannot be read off any file.

## Files

- `mail.py`, `hook.py`: the mechanism
- `task/`: the shared brief (`COMMON.md`) and each agent's private section
- `harness/`: reference implementation, case generator, grader, brief builder
  (`mkbriefs.py`), cost analysis
- `runs/r1/<arm>/`: the code each team produced, plus mailbox logs
