# compaction-offload

Moves the useful part of Claude Code compaction onto a cheap model.

- `hooks/precompact.sh` reads `transcript_path`, sends the conversation text to `claude -p --model haiku` (hooks disabled for that call, so it uses the session's own auth), and writes `$OFFLOAD_DIR/brief.md`.
- `CLAUDE.md` asks the native compaction for a two-line stub, so the session model's output is small.
- `hooks/sessionstart.sh` injects `brief.md` as `additionalContext` when `source` is `compact`.
- `hooks/postcompact.sh` logs the length of the native `compact_summary`, for comparing.

Brief format follows Belcak et al. 2026 (arXiv 2611.05306, Appendix A): verbatim task, status, next steps, plan, completion criteria, artifact paths.

## Test

`./run.sh` starts a nested headless session with `settings.json`, a low auto-compact threshold, and a multi-step task that needs state from before compaction to finish. It checks that the brief was written, that it was injected, and that the task finished correctly. Results go to `results/`.

The hook scripts are only active through `--settings compaction-offload/settings.json`; nothing here changes the repo's own settings.
