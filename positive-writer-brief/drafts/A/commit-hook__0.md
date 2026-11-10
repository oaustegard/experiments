guard_gh_route: also deny gh api POST to releases and git/refs

GitHub returns 403 when a session tries to create a release or a tag.
scripts/guard_gh_route.py denies those calls before they run and names
the Muninn `github` tool call to use instead.

The guard matched only `gh release create`. A call such as
`gh api -X POST repos/O/R/releases` went through and got the 403. This
happened in a session on 2026-11-09.

The guard now also matches `gh api` calls with method POST to a path
ending in `/releases` or `/git/refs`.

Added 4 cases to tests/test_guard_gh_route.py. All 31 tests pass.
