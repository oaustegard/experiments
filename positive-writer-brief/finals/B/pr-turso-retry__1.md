The Turso client's `execute()` now retries HTTP 503 and 429 responses up to 4 times, with jittered exponential backoff starting at 0.5 s. Any other status fails at once, as it did before. The public API does not change.

## Problem

Turso returns HTTP 503 on a cold start, for the first request after the database has been idle. Boot treated that 503 as a credential failure and printed `auth failed`. The request was never retried, because the retry count had been 0 since a refactor on 2026-09-14.

## Change

- `execute()` retries on 503 and 429, up to 4 times.
- The backoff is exponential with jitter and starts at 0.5 s.
- All other statuses fail immediately, including 401.
- The error message names the status code.

## Measurements

From 40 cold boots on 2026-11-09:

- 31 of the 40 hit at least one 503.
- With the fix, all 40 completed.
- On the boots that retried, the median added latency was 1.1 s.

## Tests

`tests/test_turso_retry.py` adds 6 cases that run against a fake server:

1. 503, then 200: succeeds
2. 429, then 200: succeeds
3. four 503s: fails
4. 401: fails immediately, with no retry
5. backoff timing stays within its bounds
6. the error message names the status code
