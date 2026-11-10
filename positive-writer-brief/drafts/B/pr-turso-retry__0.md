Turso's client now retries `execute()` on HTTP 503 and 429. It makes up to 4 retries with jittered exponential backoff starting at 0.5 s. Any other status fails at once, as it did before. The public API is unchanged.

## Problem

Turso returns HTTP 503 on a cold start, for the first request after the database has been idle. Boot treated that 503 as a credential failure and printed `auth failed`. The retry count had been 0 since a refactor on 2026-09-14, so every cold start that hit a 503 failed boot.

## Change

- `execute()` retries on 503 and 429, up to 4 times.
- The backoff is exponential with jitter, starting at 0.5 s.
- All other statuses, including 401, fail immediately.
- The error message names the status code.

## Measurement

I ran 40 cold boots on 2026-11-09. Without the fix, 31 of them hit at least one 503. With the fix, all 40 completed. On the boots that retried, the median added latency was 1.1 s.

## Tests

`tests/test_turso_retry.py` adds 6 cases that run against a fake server:

- 503, then 200: succeeds
- 429, then 200: succeeds
- four 503s: fails
- 401: fails without retrying
- backoff timing stays within bounds
- the error message names the status code
