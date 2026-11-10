Deny gh api POST to releases and git refs in route guard

scripts/guard_gh_route.py denies GitHub release and tag creation calls
from a session, because they return 403, and names the Muninn `github`
tool call to use instead. It matched only `gh release create`, so
`gh api -X POST repos/O/R/releases` went through and got the 403. This
was seen in a session on 2026-11-09.

The guard now also matches `gh api` calls with method POST to a path
ending in `/releases` or `/git/refs`.

Added 4 cases to tests/test_guard_gh_route.py. All 31 tests pass.
