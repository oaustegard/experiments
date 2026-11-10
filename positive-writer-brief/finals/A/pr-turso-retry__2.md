Adds retry with backoff to the Turso client's `execute()`.

## Problem

Turso returns HTTP 503 on a cold start, for the first request after the database has been idle. Boot treated that 503 as a credential failure and printed `auth failed`. The retry count had been 0 since a refactor on 2026-09-14, so nothing retried the request.

## Change

`execute()` now retries 503 and 429 up to 4 times. The backoff is exponential with jitter and starts at 0.5 s. Any other status fails at once, as before. There is no change to the public API.

## Measurements

Of 40 cold boots on 2026-11-09, 31 hit at least one 503. With the fix, all 40 completed. On the boots that retried, the median added latency was 1.1 s.

## Tests

`tests/test_turso_retry.py` adds 6 cases that run against a fake server:

- 503, then 200
- 429, then 200
- four 503s, then failure
- 401 fails immediately
- backoff timing stays within bounds
- the error message names the status code
