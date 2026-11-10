The Turso client's `execute()` now retries HTTP 503 and 429. It makes up to 4 retries with jittered exponential backoff that starts at 0.5 s. Any other status fails at once, as before.

## Problem

When the database has been idle, Turso returns HTTP 503 on the first request because the database is cold-starting. Boot treated that 503 as a credential failure and printed `auth failed`. The retry count had been 0 since a refactor on 2026-09-14, so nothing absorbed the 503.

## Change

- `execute()` retries 503 and 429 up to 4 times, with jittered exponential backoff starting at 0.5 s.
- All other statuses, including 401, fail immediately, as before.
- The error message names the status code.
- The public API is unchanged.

## Measurement

On 2026-11-09 I ran 40 cold boots. 31 of them hit at least one 503. With this change all 40 completed. On the boots that retried, the median added latency was 1.1 s.

## Tests

`tests/test_turso_retry.py` adds 6 cases against a fake server:

- 503 then 200 succeeds
- 429 then 200 succeeds
- four 503s, then failure
- 401 fails immediately
- backoff timing stays within bounds
- the error message names the status code
