Block POST gh api calls to /releases and /git/refs in guard_gh_route.py

`scripts/guard_gh_route.py` denies GitHub release and tag creation calls from a session, since those return 403, and names the Muninn `github` tool call to use instead. The guard matched only `gh release create`, so `gh api -X POST repos/O/R/releases` got past it and received a 403 in a session on 2026-11-09.

The guard now also matches `gh api` calls with method POST to a path ending in `/releases` or `/git/refs`. Adds 4 cases to `tests/test_guard_gh_route.py`; all 31 tests pass.
