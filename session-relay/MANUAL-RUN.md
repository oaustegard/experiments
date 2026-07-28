# Manual run: two Claude.ai sessions over the relay

The CCotw run used two subagents; nothing in the transport depends on that —
`relay.py` speaks HTTPS to Turso, which the Claude.ai code-execution container
can reach, and Muninn-project sessions already have the credentials in
`/mnt/project/*.env`. What changes on Claude.ai is the **choreography**: a
session only executes while a turn is in flight, so each side must run its
whole protocol inside one agentic turn, blocking on `wait` between messages.

## Choreography

1. Open two Claude.ai chats in the Muninn project (call them **B** and **A**).
2. Paste the **responder prompt into B first** and send. B bootstraps, then
   blocks waiting for the opener.
3. Immediately paste the **opener prompt into A** and send.
4. Watch. Each session's turn runs until CONSENSUS/ACK or bailout.
5. If either turn ends prematurely (harness cut a long wait short), reply
   `continue from your last cursor` — the seq cursor makes resume safe, which
   is why the prompts require reporting the cursor after every relay call.

## Claude.ai-specific adjustments (vs. the CCotw prompts)

- **Bootstrap block** fetches `relay.py` from the repo via the Contents API
  and sources the env files — CCotw agents had both already on disk/in env.
- **Short waits, retried**: one blocking `wait --timeout 240` risks the
  per-call execution limit; use `--timeout 90` and re-run up to 5 times.
- **Post via stdin** (`post CHANNEL SENDER -` with a heredoc) — the live run
  showed inline message args break on shell metacharacters.
- **Fresh channel name per run** (e.g. `manual-20260702`) so stale traffic
  doesn't confuse a human reading `history` (the cursor logic doesn't care).

## Prompt for session B (paste FIRST — responder)

```text
You are "muninn-b", coordinating with another Claude session ("muninn-a") that you share NO context with. Your only channel is a message relay. Set up:

set -a; . /mnt/project/GitHub.env 2>/dev/null; . /mnt/project/turso.env 2>/dev/null; . /mnt/project/muninn.env 2>/dev/null; set +a
curl -s -H "User-Agent: muninn-raven" -H "Authorization: token $GH_TOKEN" -H "Accept: application/vnd.github.v3.raw" "https://api.github.com/repos/oaustegard/claude-workspace/contents/experiments/session-relay/relay.py?ref=main" -o relay.py
(If TURSO_URL/TURSO_TOKEN aren't set after sourcing, inspect /mnt/project/*.env and export the Turso URL and token under those names.)

Relay usage (channel: manual-20260702, your sender name: muninn-b):
  post:  printf '%s' "message text" | python3 relay.py post manual-20260702 muninn-b -
  wait:  python3 relay.py wait manual-20260702 --after <cursor> --not-sender muninn-b --timeout 90 --interval 5
Track a cursor = highest seq you've seen from ANY message including your own posts (post prints its seq; wait prints JSON with seq). If wait exits 2 (timeout), re-run it — up to 5 times before treating it as a real timeout. State your current cursor in chat after every relay call so a human can resume you.

Your role: RESPONDER with the *practical* perspective. Joint task: negotiate with muninn-a a checklist of exactly 5 items titled "Ground rules for two AI sessions coordinating through a shared message relay" — muninn-a brings protocol-design ideas; you stress-test them against practical failure modes (timeouts, lost turns, ambiguous cursors, runaway chattiness) from your own reasoning.

Protocol: (1) run the setup; (2) immediately wait with --after 0; (3) reply substantively to muninn-a's opener — critique, refine, supply what it asked for; (4) loop wait→respond, at most 4 messages from you total; (5) when you agree, whoever holds the pen posts a message starting "CONSENSUS:" with the 5 items and the other replies starting "ACK". If muninn-a's CONSENSUS is acceptable, post "ACK: agreed" and stop. Two consecutive real timeouts → post that you're ending, and stop.

When done, summarize in chat: the agreed checklist, message counts per side, and anything awkward about coordinating this way.
```

## Prompt for session A (paste SECOND — opener)

```text
You are "muninn-a", coordinating with another Claude session ("muninn-b") that you share NO context with. Your only channel is a message relay. Set up:

set -a; . /mnt/project/GitHub.env 2>/dev/null; . /mnt/project/turso.env 2>/dev/null; . /mnt/project/muninn.env 2>/dev/null; set +a
curl -s -H "User-Agent: muninn-raven" -H "Authorization: token $GH_TOKEN" -H "Accept: application/vnd.github.v3.raw" "https://api.github.com/repos/oaustegard/claude-workspace/contents/experiments/session-relay/relay.py?ref=main" -o relay.py
(If TURSO_URL/TURSO_TOKEN aren't set after sourcing, inspect /mnt/project/*.env and export the Turso URL and token under those names.)

Relay usage (channel: manual-20260702, your sender name: muninn-a):
  post:  printf '%s' "message text" | python3 relay.py post manual-20260702 muninn-a -
  wait:  python3 relay.py wait manual-20260702 --after <cursor> --not-sender muninn-a --timeout 90 --interval 5
Track a cursor = highest seq you've seen from ANY message including your own posts (post prints its seq; wait prints JSON with seq). If wait exits 2 (timeout), re-run it — up to 5 times before treating it as a real timeout. State your current cursor in chat after every relay call so a human can resume you.

Your role: OPENER with the *protocol-design* perspective. Joint task: negotiate with muninn-b a checklist of exactly 5 items titled "Ground rules for two AI sessions coordinating through a shared message relay" — you propose the design (identity, cursor discipline, turn-taking, termination handshake, message budgets); muninn-b stress-tests it against practical failure modes. Genuinely incorporate its critiques.

Protocol: (1) run the setup; (2) post an opening message: introduce yourself, propose a draft structure with your key design principles, and ask muninn-b to attack it; (3) loop wait→respond, at most 4 messages from you total; (4) when you agree, post a message starting "CONSENSUS:" with the 5 items (or ACK muninn-b's). Two consecutive real timeouts → post that you're ending, and stop.

When done, summarize in chat: the agreed checklist, message counts per side, and anything awkward about coordinating this way.
```

## Afterward

From any session with the credentials (either of the two, a third, or CCotw):

```bash
python3 relay.py history manual-20260702
```

Note the prompts fetch `relay.py` with `?ref=main` — valid once the
session-relay PR merges. Before merge, substitute
`?ref=claude/bluesky-post-discussion-hipdm9`.
