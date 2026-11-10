This branch adds retry with backoff to `execute()` in the Turso client.

## Problem

The first request after the database has been idle hits a Turso cold start, and Turso answers it with HTTP 503. Boot treated that 503 as a credential failure and printed "auth failed". The retry count had been 0 since the refactor on 2026-09-14, so `execute()` never tried the request a second time.

## Change

`execute()` now retries a 503 or a 429 up to 4 times, with jittered exponential backoff that starts at 0.5 s. Any other status fails at once, as it did before. The public API is unchanged.

## Measurement

Over 40 cold boots on 2026-11-09, 31 hit at least one 503. With the fix, all 40 completed. On the boots that retried, the median added latency was 1.1 s.

## Tests

`tests/test_turso_retry.py` adds 6 cases that run against a fake server:

- 503, then 200
- 429, then 200
- four 503s, then failure
- 401 fails immediately
- backoff timing stays within bounds
- the error message names the status code
