# Hook mechanics: what was measured on this Mac (CLI 2.1.307), 2026-11-15

The standalone `claude` CLI was logged out all evening, so no model request ran. Run `./run_tests.sh all` after `claude` + `/login`; output lands in `out/summary.txt`.

| # | Question | Answer | Status |
|---|---|---|---|
| 1 | UserPromptSubmit input keys | `session_id, transcript_path, cwd, prompt_id, permission_mode, hook_event_name, prompt`. No model, no effort. `additionalContext` is recorded in the transcript as a `hook_additional_context` attachment | keys measured; model echo unmeasured |
| 2 | Raise effort by telling the model to load a skill with `effort:` frontmatter | Verified by Oskar 2026-11-01 for xhigh in a cloud session (memory 3db7fa22). Ladder skills for low/medium/high written here | not re-measured here |
| 3 | Async advice delivered at a tool boundary | Hook pair works offline (detached writer, atomic deliver-once). Binary schema accepts `additionalContext` on PreToolUse, PostToolUse, PostToolBatch, PostToolUseFailure, Stop, UserPromptSubmit | plumbing measured; model uptake unmeasured |
| 4 | Delegation nudge | scripted, two trials plus control | unmeasured |
| 5 | Haiku call latency | `latency.py` ready. A call that fails at auth takes 0.8 s: the startup floor | unmeasured |
| 6 | Can a hook see model/effort | Not in UserPromptSubmit input. Transcript tail has `effort`, `perTurnEffort`, `message.model` per assistant record (nothing before the first reply). `CLAUDE_EFFORT` in the hook's env can be the parent app's value, not the session's: do not trust it | partly measured |
| 7 | Recursion from `claude -p` inside a hook | `--setting-sources ""` drops settings hooks, `--no-session-persistence` drops the session file (without it even a failed call wrote a transcript). `--bare` disables OAuth, unusable | flags from help; session-file effect measured |

Gotchas: under `-p`, project `permissions.allow` is ignored in an untrusted workspace, so the test script passes `--allowedTools`. The `escalating-effort` skill is not in the claude-workspace checkout on this Mac.
