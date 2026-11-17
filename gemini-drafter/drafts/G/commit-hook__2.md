Deny gh api POST calls to /releases and /git/refs in guard_gh_route.py

`scripts/guard_gh_route.py` denies GitHub release and tag creation calls from a session before they fail with 403, and it names the Muninn `github` tool call to use instead. The guard previously matched only `gh release create`. On 2026-11-09, a session ran `gh api -X POST repos/O/R/releases`, which bypassed the guard and received a 403.

`scripts/guard_gh_route.py` now also matches `gh api` calls with method POST to a path ending in `/releases` or `/git/refs`. `tests/test_guard_gh_route.py` adds 4 cases for these paths, and all 31 tests pass.
