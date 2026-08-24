# HALO on Claude Code on the Web

[context-labs/halo](https://github.com/context-labs/halo) installs and runs in a
CCotw container. Every component works except the one that needs an API key:
`pip install halo-engine` succeeds, the bundled `deno` and `ripgrep` wheels
execute, the Deno+Pyodide WASM sandbox boots numpy/pandas/pydantic in 7.5s, and
the trace index, store, and tool surface all accept trace data converted from a
live CCotw session. The engine's RLM driver needs an OpenAI-compatible
`OPENAI_API_KEY`, which CCotw does not provide.

Measured 2026-08-24 in a CCotw container, `halo-engine` 0.3.5.

## Engine requirements against the container

| Requirement | State in CCotw | Evidence |
|---|---|---|
| `pip install halo-engine` | works | clean install, `halo --help` runs |
| bundled `deno` 2.7.14 wheel | works | `deno --version` from the venv bin |
| bundled `ripgrep` 15.1.0 wheel | works | linux x86_64 wheel is in the marker set |
| Pyodide fetch from npm + jsdelivr | works | `registry.npmjs.org` 200, `cdn.jsdelivr.net` 200 |
| WASM sandbox executes Python | works | pyodide 3.13.2, numpy 2.2.5, pandas 2.3.3 |
| egress to an LLM provider | works | POST to OpenAI and OpenRouter `/v1/chat/completions` returns the provider's own 401, not a proxy 403 |
| `OPENAI_API_KEY` | **absent** | no key in env, `/mnt/project` does not exist, `ANTHROPIC_API_KEY` unset by design |
| OTel trace input | **needs conversion** | Claude Code writes its own transcript format |

The proxy reports `selective: false`, so no host allowlist stands between the
container and a provider. Supply a key through the CCotw environment's
variables and the engine runs; nothing else about the network needs changing.

There is no way to route HALO's model calls through the harness instead. The
Agent tool is a model tool call available to the session's main loop, not an
HTTP endpoint a subprocess can reach, so the usual CCotw workaround —
`--emit-prompts` and dispatch through the Agent tool — has nothing to attach
to. HALO's `openai-agents` runtime issues its own requests and expects a
provider to answer them.

## The trace format gap

HALO reads one OTel/OpenInference span per JSONL line. Claude Code writes one
conversation event per JSONL line to
`~/.claude/projects/<slug>/<session_id>.jsonl`. `cc_to_halo.py` converts one to
the other.

The transcript already carries most of what a span needs: `uuid` and
`parentUuid` give the span tree, `sessionId` gives the trace id, `timestamp`
gives an end time, and assistant records carry `message.model` plus a full
`usage` block. The mapping:

| Transcript | Span |
|---|---|
| `sessionId` | `trace_id`, plus a root AGENT span `session-<id>` |
| `uuid` / `parentUuid` | `span_id` / `parent_span_id` |
| assistant record | LLM span, `SPAN_KIND_CLIENT` |
| `tool_use` block | TOOL span spanning the call to its `tool_result` |
| user record | CHAIN span (skipped when it holds only tool results) |
| `isSidechain` | `inference.agent_name = claude-code-subagent` |
| `version`, `cwd`, `gitBranch`, `entrypoint` | `resource.attributes` |

Two places where the conversion is lossy or easy to get wrong:

**Durations are derived.** Each transcript record carries a single `timestamp`
rather than a start/end pair. Tool spans get genuine wall-clock: the gap
between the `tool_use` record and its `tool_result` record. LLM spans get the
gap since the nearest ancestor record, which is model latency plus whatever the
harness spent in between, so it is an upper bound. Latency findings from
converted traces need that caveat attached. The hub's
`scripts/bash-timing-hook.py` already logs true Bash durations and could feed
exact values for that one tool.

**Prompt tokens are three fields, not one.** Anthropic reports `input_tokens`,
`cache_read_input_tokens` and `cache_creation_input_tokens` separately.
`inference.llm.input_tokens` gets the sum. Counting only `input_tokens` reports
106 prompt tokens for a session that processed 13.7 million.

The parent chain also walks through records that never become spans —
`attachment`, hook output, tool-result-only user turns — so a child's parent
resolves to its nearest *emitted* ancestor. Skipping that step left 35 dangling
parents in the index on the first pass.

## Validating against the engine's own code

`validate_dataset.py` runs the three layers a HALO run exercises, with no LLM in
the loop. Over one 11-minute CCotw session (357 transcript records, 1.0 MB →
161 spans, 505 KB):

```
[index]  cc_traces.jsonl.<hash>.engine-index.jsonl (2,458B) in 0.01s
[store]  trace_count=1
[store]  missing_parent_count: 0
         missing_agent_identity_count: 0
         project_id_mismatch_count: 0
         total_input_tokens: 13,718,108
         total_output_tokens: 90,153
[sandbox] run_python in 7.8s
view_trace spans: 0 oversized: True
  161 spans, 391,488B > 150,000B budget
  top span names: [('llm.claude-opus-5', 99), ('tool.Bash', 57), ...]
search_trace tool-span matches: 50
```

The index's three health counters all read zero, which is the check that the
conversion is structurally sound rather than merely parseable.

One CCotw session overflows `view_trace`'s 150 KB per-call budget on its own.
The engine handles it as designed, returning an `OversizedTraceSummary` that
tells the agent to use `search_trace` and `view_spans` instead, but it means
a HALO run over CCotw data never gets to read a whole session at once, even a
single short one.

## Transcript archival

HALO's premise is variance across many executions. Its README is explicit that
high-traffic environments are where it works, and that a general harness
overfits to a single trace.

CCotw gives one transcript per container, discarded at reclamation. The hub has
archival machinery for exactly this. `persist-transcript.sh` is wired as a Stop
hook and pushes each session's transcript to GitHub Releases, and it has
archived nothing since 2026-04-05. Three releases from that one day; the
`transcripts-latest` tag that `CLAUDE.md` documents as the archive was never
created.

Two faults. The `find` bug stops the script early; the 403 would stop it anyway.

**The transcript is never found.** The hook locates it with:

```bash
TRANSCRIPT=$(find "$TRANSCRIPT_DIR" -name "*.jsonl" -newer /tmp/.workstation-booted 2>/dev/null | head -1)
[ -z "$TRANSCRIPT" ] && exit 0
```

`/tmp/.workstation-booted` no longer exists in the container. `find` errors,
stderr goes to `/dev/null`, `$TRANSCRIPT` comes back empty, and the script exits
0. The Stop hook payload names the file outright as `.transcript_path`, so the
scan was never needed.

**Release writes are blocked outright.** Measured from a CCotw container:

```
POST /repos/oaustegard/claude-workspace/releases        -> 403
     "Creating, editing, or deleting releases is not permitted for this session type."
POST /repos/oaustegard/claude-container-layers/releases -> 403
GET  /repos/oaustegard/claude-container-layers/releases -> 200
PUT  /repos/oaustegard/claude-workspace/contents/...    -> 403
     "Write access to this GitHub API path is not permitted through this proxy."
```

The proxy blocks the whole release-write operation class and the Contents API
with it, on any repo, regardless of PAT scope. Reads are fine. So a Stop hook —
which runs in shell context, with no MCP and no `add_repo` — has no HTTP channel
for this at all. Fixing the `find` gets the script to the upload and no further.

Both faults share a shape. Every helper in the script is
`curl -sf ... 2>/dev/null`, so an HTTP error arrives as an empty string, each
`[ -n "$X" ]` guard skips its step, and the script exits 0 having archived
nothing. The hook is wired as `bash ./persist-transcript.sh 2>/dev/null || true`,
which erases what little evidence remained. This is the in-band-failure pattern
the hub's own `CLAUDE.md` names: a probe degrading to a value the caller will act
on, where empty stands for both "no new transcript" and "the write was refused."

[claude-workspace#249](https://github.com/oaustegard/claude-workspace/pull/249)
fixes the discovery and adds a preflight so the 403 is stated once instead of
swallowed. It does not move the transport. `git push` still works from the
container, so a dedicated branch is the obvious channel, but which branch and what
retention are Oskar's calls, and a transcript runs about 1 MB per session.

## Shape of a loop here

With a key in the environment and archival working:

1. The Stop hook archives each session's transcript somewhere durable.
2. A later session pulls the archive, runs `cc_to_halo.py` over N transcripts
   into one JSONL, and runs the engine against it.
3. The engine reports patterns that hold across sessions rather than within one.
4. Claude edits the harness, which here means `CLAUDE.md`, `boot-ccotw.sh`, the
   hooks in `.claude/settings.json`, and the skills.

Step 4 fits CCotw well. HALO's output is meant to be fed to
a coding agent that edits the harness, and CCotw's harness is a git repo the
session is already sitting in. `CLAUDE.md`'s "Error Patterns" section is a
hand-written version of what HALO generates from traces: nine entries, each a
diagnosed failure with a date and a fix. Every one of them was written after
someone noticed the failure by hand.

HALO also ships `skills/claude/SKILL.md`, a Claude Code skill for exactly this
loop, with the engine as diagnostic and Claude as executor. It assumes
`OPENAI_API_KEY` and a clone of the repo.

## Reproducing

```bash
pip install halo-engine
python cc_to_halo.py ~/.claude/projects/*/*.jsonl -o cc_traces.jsonl
python validate_dataset.py cc_traces.jsonl            # no API key needed
halo cc_traces.jsonl -p "..." --model gpt-5.4-mini    # needs OPENAI_API_KEY
```

`validate_dataset.py` needs the engine importable. `pip install halo-engine`
puts `engine` on the path; otherwise point `PYTHONPATH` at a checkout of the repo.
