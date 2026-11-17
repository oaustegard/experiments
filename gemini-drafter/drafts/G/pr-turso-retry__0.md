This PR adds jittered exponential backoff to `execute()` in the Turso client, retrying HTTP 503 and 429 responses up to 4 times so cold starts complete instead of reporting an authentication failure.

## Cause

Turso returns HTTP 503 on the first request after a database has been idle. The client's retry count had been 0 since a refactor on 2026-09-14. Boot treated that 503 as a credential failure and printed `auth failed`.

## Change

`execute()` now retries HTTP 503 and 429 responses up to 4 times with jittered exponential backoff starting at 0.5 s. Any other HTTP status fails at once, as before. The public API is unchanged.

## Verification

Over 40 cold boots on 2026-11-09, 31 hit at least one 503. With this branch, all 40 boots completed, with a median added latency of 1.1 s on the boots that retried.

`tests/test_turso_retry.py` adds 6 cases against a fake server:
- 503 followed by 200
- 429 followed by 200
- Four 503s followed by failure
- 401 failing immediately
- Backoff timing staying within bounds
- Error message naming the HTTP status code
