Turso client `execute()` now retries HTTP 503 and 429 responses with backoff. Before this change, a cold start made boot print "auth failed".

## Problem

When the database has been idle, Turso answers the first request with HTTP 503 during a cold start. Boot read that 503 as a credential failure and printed "auth failed". The client could not retry it because the retry count has been 0 since a refactor on 2026-09-14.

## Change

- `execute()` retries 503 and 429 up to 4 times with jittered exponential backoff, starting at 0.5 s.
- Every other status still fails immediately.
- The error message names the status code, so a 503 no longer reports as an auth failure.
- The public API is unchanged.

## Measurement

I ran 40 cold boots on 2026-11-09. Without the fix, 31 of them hit at least one 503. With the fix, all 40 completed. On the boots that retried, the median added latency was 1.1 s.

## Tests

`tests/test_turso_retry.py` adds 6 cases that run against a fake server:

- 503, then 200: succeeds
- 429, then 200: succeeds
- four 503s: fails
- 401: fails immediately, with no retry
- backoff timing stays within bounds
- the error message names the status code
