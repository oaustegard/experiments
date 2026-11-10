Guard gh api POSTs to /releases and /git/refs

scripts/guard_gh_route.py denies GitHub release and tag creation calls
from a session, because those calls return 403, and names the Muninn
`github` tool call to use instead. It matched only `gh release create`,
so `gh api -X POST repos/O/R/releases` got past the guard and hit the
403. This was seen in a session on 2026-11-09.

The guard now also matches `gh api` calls that use method POST on a
path ending in `/releases` or `/git/refs`.

Added 4 cases to tests/test_guard_gh_route.py. All 31 tests pass.
