This pull request adds retry with jittered exponential backoff to the Turso client's `execute()` method so that cold starts complete instead of reporting `'auth failed'`.

## Cause

Turso returns HTTP 503 on the first request after an idle database wakes up. Boot treated a 503 response as a credential failure and printed `'auth failed'`. The client's retry count had been 0 since a refactor on 2026-09-14.

## Change

`execute()` now retries HTTP 503 and HTTP 429 responses up to 4 times with jittered exponential backoff starting at 0.5 s. Any other HTTP status fails immediately, as before. The error message names the status code. The public API is unchanged.

## Verification

Over 40 cold boots on 2026-11-09, 31 hit at least one 503. With this change, all 40 completed, with a median added latency of 1.1 s on the boots that retried.

`tests/test_turso_retry.py` adds 6 cases against a fake server:

- 503 followed by 200
- 429 followed by 200
- Four 503 responses followed by failure
- 401 fails immediately
- Backoff timing stays within bounds
- Error message names the HTTP status code
