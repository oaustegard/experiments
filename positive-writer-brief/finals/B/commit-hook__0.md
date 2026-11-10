Guard gh api POSTs to /releases and /git/refs

scripts/guard_gh_route.py denies GitHub release and tag creation calls
from a session, because those calls return 403 there. When it denies
one, it names the Muninn `github` tool call to use instead.

The guard matched only `gh release create`. A call such as
`gh api -X POST repos/O/R/releases` got past it and then failed with
the 403. This was seen in a session on 2026-11-09.

The guard now also matches `gh api` calls that use method POST on a
path ending in `/releases` or `/git/refs`.

Adds 4 cases to tests/test_guard_gh_route.py. All 31 tests pass.
