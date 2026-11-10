The Turso client's `execute()` now retries HTTP 503 and 429 up to 4 times, using jittered exponential backoff that starts at 0.5 s. Any other status fails at once, as it did before.

## Problem

When the database has been idle, Turso answers the first request with a 503 during its cold start. Boot treated that 503 as a credential failure and printed `auth failed`. Nothing retried the request, because the retry count had been 0 since a refactor on 2026-09-14.

## Change

- `execute()` retries on 503 and 429, at most 4 times.
- The backoff is exponential with jitter, starting at 0.5 s.
- Every other status fails immediately, with no change to that behaviour.
- The public API does not change.

## Measurement

Of 40 cold boots run on 2026-11-09, 31 got at least one 503. With the fix, all 40 boots completed. On the boots that retried, the median added latency was 1.1 s.

## Tests

`tests/test_turso_retry.py` adds 6 cases that run against a fake server:

- a 503 followed by a 200 succeeds
- a 429 followed by a 200 succeeds
- four 503s in a row end in failure
- a 401 fails immediately
- the backoff timing stays within bounds
- the error message names the status code
