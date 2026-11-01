# paper-watch

Daily review of [@paper.bsky.social](https://bsky.app/profile/paper.bsky.social)'s arXiv Upvote Trends Top 30, deduplicated across runs.

- `PROCEDURE.md` is what the scheduled run follows.
- `paperwatch.py queue|record` handles feed parsing and the dedup ledger.
- `seen.json` is the ledger: every reviewed arXiv id, with its rating, read depth and attempt count.
- `reviews/<date>.md` holds one file per run that reviewed anything.
