Block gh api POST calls to /releases and /git/refs in guard_gh_route

`scripts/guard_gh_route.py` denies GitHub release and tag creation calls from a session and names the Muninn `github` tool call to use instead, because direct calls return 403. It previously matched `gh release create` only. In a session on 2026-11-09, `gh api -X POST repos/O/R/releases` bypassed the guard and received a 403.

The guard now also matches `gh api` calls with method POST to a path ending in `/releases` or `/git/refs`.

`tests/test_guard_gh_route.py` adds 4 cases, and all 31 tests pass.
