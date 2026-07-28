# session-relay — inter-session Claude coordination over Turso

Reproduces the pattern from [Joshua Shew's Bluesky post](https://bsky.app/profile/joshuashew.bsky.social/post/3mnx2urp3f22n)
(itself inspired by [lathrys.at](https://bsky.app/profile/lathrys.at) — two
agents passing messages under `/tmp`): give two Claude sessions a shared
message channel and a turn-taking protocol, and let them reconcile their
working knowledge live.

Joshua's version required one Claude session to deploy a relay HTTP service.
This version needs **zero new infrastructure**: the rendezvous is the Turso DB
every Muninn session already has credentials for. The relay is one table
(`relay_messages`) plus a ~180-line CLI.

## Files

- `relay.py` — the relay CLI: `init` / `post` / `poll` / `wait` / `history`.
  Talks to Turso via the HTTP pipeline API (`TURSO_URL` / `TURSO_TOKEN` env),
  with 5xx backoff for cold starts.
- `RESULTS.md` — transcript and observations from the live two-agent run.

## Design

```
relay_messages(seq AUTOINCREMENT, channel, sender, body, created_at)
```

- **Append**: `relay.py post CHANNEL SENDER BODY` → prints assigned `seq`.
- **Poll with cursor**: `relay.py poll CHANNEL --after N [--not-sender ME]`
  → JSON lines. AUTOINCREMENT `seq` is the cursor; callers track the max seq
  they've seen (including their own posts) and pass it back.
- **Blocking wait**: `relay.py wait CHANNEL --after N --not-sender ME
  --timeout 240 --interval 5` → returns on first new peer message, exit 2 on
  timeout.
- **Transcript**: `relay.py history CHANNEL` — the whole conversation persists
  in the DB, readable after the fact. Half the value.

The behavioral protocol lives in the session prompts, not the code: name each
participant, seed each with knowledge the other lacks, designate an opener and
a responder, cap message counts, and define a terminal handshake
(`CONSENSUS:` → `ACK`).

## Running a coordination session

1. `python3 relay.py init` (idempotent).
2. Give each session a prompt containing: its sender name, the channel, the
   CLI invocations verbatim, its knowledge seed, opener/responder role, a
   message cap, and the CONSENSUS/ACK stop condition.
3. Afterward, `python3 relay.py history CHANNEL` to read the exchange.

Works identically whether the sessions are subagents in one container or
separate Claude.ai / CCotw sessions on different machines — the transport is
the remote DB either way.
