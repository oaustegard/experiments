# avo-supervisor-assessment

Empirical assessment for [claude-workspace#233](https://github.com/oaustegard/claude-workspace/issues/233):
can an AVO-style long-horizon supervisor loop be built on CCotw? Four questions,
answered by running things, 2026-08-22, in CCotw session `03c849ea` (claude-fable-5,
CLI 2.1.239). Verdict posted on the issue; this directory holds the raw numbers
and artifacts.

## Q1 — Stop hook mechanics: confirmed live

Method: instrumented the production Stop hook (`scripts/check-store-on-stop.py`
in claude-workspace) with stdin logging plus a one-shot armed exit-2 block
(`artifacts/stop-hook-instrumentation.diff`), then ended the response.

- Exit 2 from a Stop hook **blocks the stop and injects stderr verbatim** as the
  next user turn. The session was woken immediately; the directive text arrived
  intact and was acted on (`artifacts/stop-hook-log.jsonl`).
- The settings.json relay form `python3 script; rc=$?; [ $rc -eq 2 ] && exit 2;
  exit 0` relays the block correctly.
- Stop stdin schema observed: `session_id`, `transcript_path`, `cwd`,
  `prompt_id`, `permission_mode`, `effort`, `hook_event_name`,
  `stop_hook_active`, `last_assistant_message`, `background_tasks`,
  `session_crons`. `stop_hook_active` was `false` on the first stop.
- Infinite-block protection: the `stop_hook_active` guard returns exit 0 when
  the flag is true (verified synthetically against the armed block), and
  one-shot state (flag file consumed on first fire) worked as designed.
- The hook **script is re-read on every Stop event** — edits take effect
  mid-session with no re-registration. The supervised agent can therefore edit
  its own supervisor; the loop design should treat the hook script as
  agent-writable.
- CCotw also installs its own harness-level Stop hook
  (`~/.claude/stop-hook-git-check.sh`) that blocks on uncommitted changes.
  Multiple Stop hooks run and each one's stderr is delivered. A
  commit-per-candidate ledger gets this pressure for free.
- Untested: registering a *new* hook in settings.json mid-session. The tested
  path edits an already-registered script.

## Q2 — Plateau detection: arithmetic in the hook, forced and correct

- Hook wall clock: **38 ms** over this session's 794 KB transcript. Ledger
  parse + best-of-last-N comparison is noise on that.
- An LLM call from inside the hook is not possible on CCotw: no
  `ANTHROPIC_API_KEY` (by design), and neither the harness Agent tool nor MCP
  is reachable from a hook shell. The division is forced: hook = arithmetic +
  directive injection; strategy reasoning happens in the main loop after
  injection, done by the frontier model with full context — better than any
  side-channel LLM call would be anyway.

## Q3 — Ledger: JSON in-repo is the state; Turso is the postmortem

| operation | time |
|---|---|
| local JSON read (200-candidate ledger) | 0.2 ms |
| local JSON write | 1.0 ms |
| Turso `config_get`, warm | 0.49–0.69 s |
| Turso cold start | documented 503s, backoff required |

The Stop hook reads the ledger on every stop, so the ledger must be a file —
Turso from hook context adds credential sourcing (`/etc/profile.d` is
login-shell only) and 503 risk for no benefit. Commit per candidate; the boot
clone then delivers the ledger to any resumed or fresh session for free.
Context is disposable, the file is the state. Turso keeps the cross-session
narrative (run summaries, verdicts), not per-candidate state.

## Q4 — Throughput: evaluation is not the bottleneck

remex (shallow clone, 0.9 s), synthetic clustered corpus per its own bench:

| run | wall clock |
|---|---|
| full `bench/benchmark.py` suite (2 corpus sizes × 5 configs) | 43 s |
| single-candidate fitness (`artifacts/fitness_probe.py`): cold python, one config, 10k×384, 200 queries, recall@10 | 1.4–1.7 s |
| encode+search alone | ~0.5 s |

Fitness range at 10k×384: 4-bit recall@10 = 0.849, 3-bit = 0.736 — enough
dynamic range to optimize against. Per-candidate wall clock is agent-turn
dominated (minutes, not seconds), so realistic throughput is ~15–30
candidates/hour and the search stays a search. AVO's ~20 min/direction was
also agent-bound; same shape. The real-embedding SPECTER2 cache (~90 MB in
claude-container-layers releases) is reachable after `add_repo` (verified
in-session).

## Fable arm — runnable; the assessment session is the proof

- CCotw exposes model selection: this session ran **claude-fable-5** (both
  `session_context.model` and `last_served_model`), created from iOS.
  Container CLI 2.1.239 ≥ the 2.1.170 floor.
- `create_session` (claude-code-remote MCP) takes a `model` parameter — the
  Opus 5 arm is spawnable programmatically from a Fable session.
- A nested `claude -p` cannot authenticate (no API key), so each arm is a
  CCotw session — which forces the cross-session ledger design Q3 already
  picked.
- No routing/downgrade observed during this session.

## Side findings

- `send_later` was denied by the auto-mode permission classifier: self-scheduled
  wake-ups cannot be assumed as loop glue. The Stop hook is the reliable
  continuation surface.
- `persist-transcript.sh` (Stop-hook transcript archival) has not landed an
  artifact since 2026-04-05: `transcripts-latest` does not exist and the API
  calls target an out-of-scope repo, so the agent proxy 403s them and
  `curl -sf … 2>/dev/null` swallows the evidence. Separate fix needed.

## Artifacts

- `artifacts/stop-hook-instrumentation.diff` — the temporary hook modification
- `artifacts/stop-hook-log.jsonl` — captured Stop event + directive compliance
- `artifacts/fitness_probe.py` — single-candidate fitness timing probe
