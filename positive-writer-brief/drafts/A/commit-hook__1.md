guard_gh_route: also block gh api POSTs to releases and git/refs

The guard denies GitHub release and tag creation from a session, since
those calls return 403, and names the Muninn `github` tool call to use
instead. It matched only `gh release create`, so a call such as
`gh api -X POST repos/O/R/releases` got past the guard and then failed
with the 403. This was seen in a session on 2026-11-09.

The guard now also matches `gh api` calls that use method POST on a path
ending in `/releases` or `/git/refs`.

Added 4 cases to tests/test_guard_gh_route.py. All 31 tests pass.
