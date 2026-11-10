Adds retry with backoff to `execute()` in the Turso client. During a cold start, Turso returns HTTP 503 to the first request after the database has been idle. Boot treated that 503 as a credential failure and printed `auth failed`. The retry count had been 0 since a refactor on 2026-09-14.

## Change

- `execute()` retries 503 and 429 up to 4 times. Backoff is exponential and jittered and starts at 0.5 s.
- Any other status fails at once, as before.
- The public API does not change.

## Measurement

Over 40 cold boots on 2026-11-09, 31 hit at least one 503. With the fix, all 40 boots completed, and on the boots that retried the median added latency was 1.1 s.

## Tests

`tests/test_turso_retry.py` adds 6 cases that run against a fake server:

- 503, then 200
- 429, then 200
- four 503s, then failure
- 401 fails immediately
- backoff timing stays within bounds
- the error message names the status code
