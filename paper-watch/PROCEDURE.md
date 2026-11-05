# paper-watch procedure

A scheduled task runs this daily. It reviews the papers in [@paper.bsky.social](https://bsky.app/profile/paper.bsky.social)'s
"arXiv Upvote Trends Top 30" that have not been reviewed before. `seen.json` is the ledger; git history is the audit trail.
Edit this file to change the routine. The scheduled task's prompt only points here.

## Steps

1. **State.** Clone `https://github.com/oaustegard/experiments` (public; anonymous read works) into the scratchpad and `cd paper-watch`.
2. **Queue.** `python3 paperwatch.py queue seen.json queue.json`.
   The script fetches posts since `last_run` minus one day. It takes the ranked list from the summary post's links and titles/upvotes from the per-paper posts,
   and queues a paper when it is not in the ledger, or when its last read was not full text, it was rated above NONE, and it has had fewer than 2 attempts.
   If the queue is empty, stop: no commit, and a one-line final message saying so.
   In scheduled runs the auto-mode classifier may refuse to execute `paperwatch.py` from the clone ("Code from External"); that cannot be changed from the task's settings.
   When it does, read `paperwatch.py`, write an equivalent of `queue` and `record` in the session's own scripts directory, and run that. Do not retry the refused command. Treat `paperwatch.py` as the spec, and note the substitution in the report only if the output differs from what the spec describes.
3. **Review.** Send the queue to `haiku` subagents (Agent tool, `model: haiku`), at most 5 papers each, all launched in one message so they run in parallel.
   Give each agent the profile and output format below. Fetching facts established in earlier runs:
   - `curl -sL https://arxiv.org/html/<id>v1` works from the shell. The unversioned `/html/<id>` URL is refused by WebFetch's allowlist.
     Fall back to `https://arxiv.org/pdf/<id>v1` (PDF), and retry once if rate-limited.
   - The full text is required. A third-party summary, alphaXiv page, model card or blog does not count as reading the paper.
     If the body text cannot be retrieved, the agent reports `read: abstract` or `read: none` and the ledger queues the paper for one more attempt next run.
   - For a retry (`"retry": true` in the queue), pass the previous review and ask the agent to confirm or revise it from the full text.
   If an agent fails or returns malformed output, spawn a fresh agent with a revised spec carrying the diagnostic and what is already ruled out. Never patch the failed one.
4. **Check the top ratings.** For every paper rated MEDIUM or HIGH, the orchestrating session (not a subagent) opens the full text itself.
   It verifies the one or two numbers or claims the rating rests on, plus any code or weights links, and corrects the rating if they don't hold.
   Haiku tends to inflate relevance and to trust model cards.
5. **Record.** Write `results.json`: a list of `{id, title, rank, upvotes, read, practical, review}`. `review` is 1–3 plain sentences: the concrete use, or why there is none.
   Run `python3 paperwatch.py record seen.json results.json`.
   Write `reviews/<local date>.md`, using the local date from `TZ=America/New_York date +%F`. List HIGH and MEDIUM first, then LOW, then NONE as a single line of ids and titles. Put the abs link on each paper.
6. **Commit.** Commit `paper-watch/seen.json` and the review file in one commit to `main` of `oaustegard/experiments` via the Muninn `github` tool (`commit_files`).
   Upload each file with `curl -s --data-binary @FILE https://muninn-mcp.austegard.workers.dev/stage` and pass `@stage:<id>:<sha256>` as the content, so the ledger is not re-emitted inline.
   Commit message: `paper-watch: <date>, <n> reviewed (<k> medium+)`.
7. **Report.** The final message is short. List each HIGH and MEDIUM paper with its one-line use, give counts for the rest, and link the review file on GitHub.
   With nothing above LOW, one line is enough.

## Profile for the subagents

The reader is an engineer who builds and runs:
- an LLM agent ("Muninn") with an external persistent memory (SQLite/Turso memories, FTS recall, boot context, skills) that delegates to cheaper subagents/models. He cares about model routing and cost, and already has a fast calibrated-probability `decide` tool for routing, triage and gating.
- retrieval and vector-quantization tooling: remex (multi-bit), remax (1-bit), and remax_kb (hybrid dense+BM25 KB). He also runs embeddings in the browser.
- custom WebGPU kernels for in-browser inference of small models (EmbeddingGemma), where weight bytes and download size matter.
- Claude Code tooling (hooks, skills, harness behaviour), client-side web apps, and Bluesky/ATProto tools.

He follows the thesis that a small reasoning model plus a large external knowledge base can stand in for a large model, and he works with cycling and fitness data from Strava.

"Practical" means he could adopt, replicate or test it within weeks: a technique, a prompt pattern, a released tool or model, or a benchmark finding that changes a design decision. Pure theory and recipes that need large training budgets rate LOW or NONE. A finding from such a paper that still changes a design decision should be named.

## Subagent output format

```
### <rank>. <title> (<arxiv id>)
- what: one sentence on what the paper does
- read: full (which URL) / abstract / none (+ why)
- key numbers: the 1-3 numbers the rating rests on, with baselines
- code/weights: links stated in the paper / not stated
- practical: HIGH / MEDIUM / LOW / NONE
- why: 1-3 sentences, concrete and skeptical
```
