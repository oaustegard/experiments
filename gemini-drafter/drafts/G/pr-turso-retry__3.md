This pull request adds jittered exponential backoff to the Turso client's `execute()` method so that cold starts and rate limits retry instead of failing as credential errors.

### Problem

Turso returns HTTP 503 on the first request after an idle database wakes up. Boot had assumed any error response meant bad credentials and printed `'auth failed'`. The client's retry count had been 0 since a refactor on 2026-09-14.

### Change

`execute()` now retries HTTP 503 and 429 responses up to 4 times with jittered exponential backoff starting at 0.5 s. Any other error status fails at once as before, and the error message names the status code. The public API does not change.

### Measurements and tests

On 2026-11-09, 31 out of 40 cold boots hit at least one 503. With this fix all 40 completed, with a median added latency of 1.1 s on the boots that retried.

`tests/test_turso_retry.py` adds 6 cases against a fake server: 503 then 200, 429 then 200, four 503s then failure, 401 failing immediately, backoff timing within bounds, and the status code appearing in the error message.
