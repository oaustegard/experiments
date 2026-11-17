Block POST gh api calls to /releases and /git/refs in guard_gh_route.py

`scripts/guard_gh_route.py` denies GitHub release and tag creation calls from a session, which return 403, and names the Muninn `github` tool call to use instead. The guard previously matched only `gh release create`, so `gh api -X POST repos/O/R/releases` went through and received a 403 in a session on 2026-11-09.

`scripts/guard_gh_route.py` now also matches `gh api` calls with method POST to a path ending in `/releases` or `/git/refs`. `tests/test_guard_gh_route.py` adds 4 test cases, and all 31 tests pass.
