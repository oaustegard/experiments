Turso client `execute()` now retries HTTP 503 and 429 responses up to 4 times. The waits use jittered exponential backoff that starts at 0.5 s. Any other status fails at once, as it did before. The public API does not change.

## Problem

Turso returns HTTP 503 on a cold start, for the first request after the database has been idle. Boot treated that 503 as a credential failure and printed `auth failed`. Nothing retried the request, because the retry count has been 0 since a refactor on 2026-09-14.

## Change

- `execute()` retries on 503 and 429, up to 4 times.
- The backoff is exponential with jitter and starts at 0.5 s.
- All other statuses, including 401, fail immediately.
- The error message names the status code, so a cold-start 503 can no longer show up as an auth failure.

## Measurements

These are from 40 cold boots on 2026-11-09:

- Without the fix, 31 of the 40 hit at least one 503.
- With the fix, all 40 completed.
- The boots that retried took a median of 1.1 s longer.

## Tests

`tests/test_turso_retry.py` adds 6 cases that run against a fake server:

1. 503, then 200: succeeds
2. 429, then 200: succeeds
3. four 503s: fails
4. 401: fails immediately with no retry
5. backoff timing stays within its bounds
6. the error message names the status code
