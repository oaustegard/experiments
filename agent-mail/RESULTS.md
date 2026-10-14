# agent-mail: how Haiku subagents should talk to each other

## Answer

Question (Oskar, 2026-10-14): earlier Haiku swarms gave each agent one task and
coordinated it only through a shared tree and a merge gate. What is the most
effective and efficient way for peer agents to introduce themselves and
exchange messages?

Three things have to work: delivery, acceptance and verification. A file
mailbox with a `PostToolUse` push hook delivers at about 1.2x the cost of no
channel. Acceptance needs a team charter in the agent's system prompt. In
round 2, 6 messaging teams relayed 19 of the 24 rules a teammate needed
and implemented **none**. The charter in a custom agent's system prompt
raised that to **10 of 12**. Verification needs a gate. With `mail owe`
(send a rule plus a test), `mail ack`, and `mail ship` (SHIP only when every
rule is relayed, acknowledged and passing), **all 3 teams scored 60/60 and
implemented 12 of 12 rules**, at about **$0.026 per team**. The gate caught
one rule that had been acknowledged but only half implemented.

What would change this: more teams, a second task, or a model other than
Haiku 5.5.

## Findings

1. **Charter in the system prompt plus a handoff gate is the working design:
   12 of 12 rules, 60/60 in all 3 teams (Round 4).** Without the gate, the
   charter alone reached 10 of 12 (Round 3).
2. **A team charter in the system prompt is what makes peers act on each
   other's messages.** In round 2, holders in 6 messaging teams relayed 19 of
   24 rules and owners implemented none, all scoring 14/60. In round 3, `team-member` agents with
   the charter in their system prompt implemented 10 of 12 rules, scoring
   60, 43 and 28 of 60.
3. **Agents read a peer's message as an unverified claim, not as spec.** Round-2
   holders sent 19 of their 24 rules to the right owner (16 of 20 in the
   mailbox teams, 3 of 4 in the native team). Owners refused every one as "not
   in my brief", called the quote suspect, or decided "the briefs disagree",
   and dave held `SHIP` waiting for a lead who never came (Round 2 mailbox logs).
4. **An authority clause inside a brief file trips the auto-mode classifier.**
   "A teammate's rule overrides your section", in a file the agent reads, got
   the agent's next Bash call denied as "Instruction Poisoning" (Round 2,
   `mail-auth`).
5. **When interfaces are the only coupling, a shared tree is enough.** Every
   round-1 arm, including no channel, scored 48/48, because the integrator
   wrote last and read his teammates' files (Round 1).
6. **A contract-first join card cuts chatter.** Mailbox traffic fell from 17
   messages to 7, and cost from $0.049 to $0.036 per team, against $0.030 with
   no channel (Round 1).
7. **Native `SendMessage` costs the most.** Each send to a finished peer
   restarts it from its transcript (one agent handed back 6 times), early sends
   fail before the roster arrives, and there is no blocking receive. That was
   $0.053–0.059 per team (Round 1).
8. **Native subagent-to-subagent `SendMessage` works on CC 2.1.296.** `from`
   is now the sender's agentId, where it was the agent type on 2026-08-12, but
   `ListAgents` is still absent inside subagents (Log: Probes).
9. **A relay + ack + test gate before SHIP closes the gap.** In round 3 the
   two misses were a rule never relayed and a rule never acknowledged. With
   the gate, all 3 teams scored 60/60 and implemented 12 of 12 rules, at
   $0.026 per team against $0.021 without it (Round 4).
10. **An acknowledgement is not an implementation; the attached test is what
   catches it.** In round-4 team 1, the GIFT- rule was acknowledged while the
   invoice still printed the catalog unit price. The holder's test blocked
   SHIP, the agents routed the fix to the right file, and the second ship
   attempt passed (Round 4, `runs/r4/mail-gate-1/mail-log.txt`).

## Method

**Mechanism.** `mail.py` keeps an append-only `log.jsonl`, a `roster.jsonl`
with one card per agent, and a per-reader cursor. `hook.py` (`PostToolUse`)
binds an agent to its name on its first `mail join|send|recv NAME` call, keyed
by `agent_id`, or by `session_id` for a headless child. After that, every tool
call returns the agent's unread mail and any new or changed teammate cards as
`additionalContext`. `mail recv NAME --wait N` is the blocking receive. A
per-team `<tree>/mail` wrapper pins each team to its own `.board`.

**Task.** Four agents (alice, bob, carol, dave) each own one file of an
`orders` package: parser, inventory, pricing, and the report entry point. Each
sees only its own section of the spec, and no interfaces between modules are
given. Round 1 is graded on 48 hidden end-to-end cases. Rounds 2 and 3 add four
rules, each held by one agent but belonging in a teammate's file: `GIFT-` skus
are free, 2 units of each sku stay in reserve, a status line, and
case-insensitive order_id. Those rounds use 60 hidden cases. The grader matches
each team's output against reference implementations with every subset of the
four rules switched on, to name the rules a team implemented. No rules scores
14/60.

**Arms.**

| arm | channel | intro |
|---|---|---|
| none | none; agents can read each other's files | none |
| native-plain / native-contract | `SendMessage`; the parent sends the roster after spawn | free-form / interface contract as the first message |
| mail-plain / mail-contract | mailbox + push hook | join card: name and file / interface contract |
| mail-team (Round 3) | mailbox + push hook, `team-member` agent | contract card, plus the team charter in the system prompt |

All agents were Haiku 5.5. Rounds 1 and 2 ran as Agent-tool `general-purpose`
subagents; the Agent tool runs at most 20 at once, so round 2 ran in waves.
Round 3 ran as headless `claude -p --agents <file> --agent team-member`
children with `--allowedTools` scoped to file and shell tools, because agent
definitions only load when a session starts. A transcript audit found no agent
reading another agent's brief, another arm's tree, or the hidden cases.

## Log

### Probes (2026-10-14, Claude Code 2.1.296, CCotw)

- Two Haiku subagents exchanged `SendMessage` pings by agentId. The envelope's
  `from` is the sender's agentId, so a reply to `from` reaches the sender. This
  changed since `subagent-messaging/`.
- `ListAgents` is absent inside a subagent. A send to a finished subagent
  resumes it from its transcript.
- The hook's `additionalContext` reached the subagent as
  `PostToolUse:Bash hook additional context`. Hooks added to
  `.claude/settings.local.json` load mid-session.
- A hub guard blocks a foreground `sleep N`, so `mail recv --wait` is the wait
  primitive.

### Round 1: interfaces only, one team per arm

| arm | pass | messages | tool calls | turns | output tokens | usd | wall |
|---|---|---|---|---|---|---|---|
| none | 48/48 | 0 | 30 | 27 | 859 | 0.030 | 48 s |
| mail-contract | 48/48 | 7 | 37 | 36 | 643 | 0.036 | 65 s |
| mail-plain | 48/48 | 17 | 54 | 53 | 610 | 0.049 | 72 s |
| native-plain | 48/48 | 19 | 62 | 55 | 1,094 | 0.053 | 63 s |
| native-contract | 48/48 | 37 | 82 | 59 | 2,045 | 0.059 | 75 s |

`usd` is input plus output at the Haiku 5.5 rate card (`runs/r1/summary.txt`).
In mail-plain, all four agents proposed interfaces at the same moment, then
spent six messages converging on key names. One proposal tried to drop the
merge rule, and alice, the only holder of that rule, caught it. In
native-contract, contract broadcasts and `done` notices kept waking finished
peers. Reading at the time: the task did not need communication, so this round
measured what each channel costs, not what it buys.

### Round 2: rules only a teammate can implement

| arm | teams | score (/60) | rules implemented |
|---|---|---|---|
| no channel | 3 | 0, 0, 14 | none; 2 teams never integrated |
| native-plain | 1 | 14 | none |
| mail-plain | 3 | 14, 14, 14 | none |
| mail-contract | 2 | 14, 14 | none |

Holders relayed 19 of the 24 rules (counted from the mailbox logs and the
native team's `SendMessage` calls); owners implemented none of them. A `mail-auth` arm then added an
authority clause to the brief, and the auto-mode classifier denied the first
agent's next Bash call as "Instruction Poisoning". That arm stopped after one
agent, per the hub's first-denial rule. Three queued teams (native-plain-2/3,
mail-contract-3) were dropped once all graded teams had failed the same way. Oskar
then approved two changes: a custom agent, and scoped permission rules for the
experiment directories.

### Round 3: team charter in the system prompt

| team | score (/60) | rules implemented | messages | cost |
|---|---|---|---|---|
| mail-team-1 | 60 | A B C D | 17 | $0.019 |
| mail-team-2 | 28 | A C D | 17 | $0.025 |
| mail-team-3 | 43 | A B C | 16 | $0.020 |

In team 2, bob never relayed the reserve rule. In team 3, alice received dave's
order_id rule but did not implement it. A frontmatter `permissionMode` could
not have helped: the CLI ignores one that would widen the session's inherited
mode. Cost comes from the headless result JSON, so it is not directly
comparable with round 1.

### Round 4: relay, ack and test gate before SHIP

The coordinator seeds the four handoffs per team (`mail seed FROM TO`). A
holder sends its rule with `mail owe`, attaching an input and the invoice
lines the rule must produce. `mail ack` records the acknowledgement and
notifies the team. dave can only ship through `mail ship dave`, which runs
every test against the team's current tree and messages the responsible agent
about anything open.

| team | score (/60) | rules implemented | messages | ship attempts | cost |
|---|---|---|---|---|---|
| mail-gate-1 | 60 | A B C D | 20 | 2 (1 blocked) | $0.026 |
| mail-gate-2 | 60 | A B C D | 17 | 1 | $0.027 |
| mail-gate-3 | 60 | A B C D | 16 | 1 | $0.026 |

The one block came in team 1. bob had acknowledged the GIFT- rule, but the
invoice still printed `@ 7.00` for a free item. bob traced it to dave's unit
price display, dave fixed it, and the next `ship` passed. All 12 handoffs were
relayed and acknowledged without the gate having to prompt for them. That
suggests the seeded obligations and the `owe` instructions also prevent the
"never relayed" miss, not just detect it.

### Files

- `mail.py`, `hook.py`: the mechanism
- `task/`, `harness/`, `runs/r1/`: round 1 task, reference, grader, brief
  builder and team outputs
- `team-member.md`: the custom agent definition (copy to `.claude/agents/`);
  `harness2/team-member-r3.md` is the round-3 version, which acknowledged with
  a plain `ok <rule>` reply instead of the brief's ack command
- `task2/`, `harness2/`: round 2/3 task, rule-toggling reference and grader,
  headless launchers
- `gate.py`: the owe/ack/ship gate
- `runs/r2/`, `runs/r3/`, `runs/r4/`, `runs/r2-r4-grades.txt`: team code,
  mailbox logs, headless result JSON, round-4 obligations, grades
