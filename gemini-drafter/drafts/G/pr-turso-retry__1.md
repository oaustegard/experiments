This pull request adds jittered exponential backoff to the Turso client's `execute()` method so that cold starts and rate limits retry instead of failing boot.

## Cause

Turso returns HTTP 503 on the first request after the database has been idle. Boot treated that 503 as a credential failure and printed `'auth failed'`. The client's retry count had been 0 since a refactor on 2026-09-14.

## Change

`execute()` now retries HTTP 503 and HTTP 429 responses up to 4 times, with jittered exponential backoff starting at 0.5 s. Any other HTTP status fails immediately as before, and the error message names the status code. The public API is unchanged.

## Verification

On 2026-11-09, 31 of 40 cold boots hit at least one 503. With this branch, all 40 cold boots completed, with a median added latency of 1.1 s on the boots that retried.

`tests/test_turso_retry.py` adds 6 cases against a fake server:

- 503 followed by 200
- 429 followed by 200
- Four 503 responses followed by failure
- 401 fails immediately
- Backoff timing stays within bounds
- Error message names the HTTP status code
