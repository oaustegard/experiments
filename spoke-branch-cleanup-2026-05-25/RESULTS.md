# Spoke-repo branch cleanup — 2026-05-25

Swept 28 spoke repos for abandoned non-default branches. 186 candidates → 183
deleted, 3 kept.

## Method

`audit-branches.sh` walks each repo and emits one row per non-default branch:
ahead/behind vs default, associated PR (open + closed history), last commit
date. Categorization:

| Category | Criterion | Action |
|---|---|---|
| Open PR | branch appears in `gh pr list --state open` | KEEP |
| Fully merged | `ahead_by == 0` vs default | DELETE |
| Squash-merged via PR | PR state `MERGED`, ahead may be >0 (squash artifact) | DELETE |
| PR closed, not merged | PR state `CLOSED`, no merge | DELETE (abandoned) |
| No PR, has commits | orphan branch, never PR'd | DELETE (mostly Claude session leftovers) |

Edge cases verified by hand: `auto/update-tools-index-*` daily branches in
`muninn.austegard.com` are leftovers of the
`.github/workflows/update-tools-index.yml` workflow whose `gh pr create` step
isn't firing — branches accumulate without PRs. Two Oskar-authored branches
(`Lottie_Playlist/perceived-color-totality`, `transformer-vm/add-linux-openblas`)
were KEPT because they're not Claude session debris.

## Result

| Repo | Branches before | Deleted | Kept |
|---|---|---|---|
| `Lottie_Playlist` | 1 | 0 | 1 |
| `bookmarklets` | 8 | 8 | 0 |
| `browser-extensions` | 4 | 4 | 0 |
| `claude-container-layers` | 2 | 2 | 0 |
| `claude-github-and-spoke` | 2 | 2 | 0 |
| `claude-jjithub-and-spoke` | 3 | 3 | 0 |
| `claude-skills` | 40 | 40 | 0 |
| `claude-tangled-spoke` | 7 | 7 | 0 |
| `claude-workspace` | 12 | 12 | 0 |
| `claude-workspace-fuse` | 4 | 4 | 0 |
| `eml-sr` | 5 | 5 | 0 |
| `fusemojo` | 2 | 2 | 0 |
| `install-manifest-spec` | 3 | 3 | 0 |
| `jina-v5-nano-mirror` | 1 | 1 | 0 |
| `lemur-numpy` | 1 | 1 | 0 |
| `llm-as-computer` | 6 | 6 | 0 |
| `mojo-bm25s` | 8 | 8 | 0 |
| `muninn-backup` | 1 | 1 | 0 |
| `muninn-utilities` | 7 | 7 | 0 |
| `muninn.austegard.com` | 16 | 16 | 0 |
| `muninns-inbox` | 2 | 2 | 0 |
| `oaustegard.github.io` | 11 | 11 | 0 |
| `remax` | 9 | 9 | 0 |
| `remax_kb` | 1 | 1 | 0 |
| `remex` | 18 | 17 | 1 |
| `thinking-traces-eval` | 1 | 1 | 0 |
| `transformer-vm` | 5 | 4 | 1 |
| `tree-sitter-mojo` | 6 | 6 | 0 |
| **total** | **186** | **183** | **3** |

Zero deletion failures (`delete-log.tsv`).

## Kept branches

- `remex/refactor/mojo-out-of-python-pkg` — open PR #55
- `transformer-vm/add-linux-openblas` — Oskar's commit, USE_OPENBLAS autodetect
- `Lottie_Playlist/perceived-color-totality` — Oskar's commit, addresses #26

The two open PRs on `muninn-utilities` (#32, #33) are from external fork
`dreamer0129` — no oaustegard-side branches to keep.

## Follow-up

`muninn.austegard.com/.github/workflows/update-tools-index.yml` is creating
daily branches without opening PRs. Either the `gh pr create` step is failing
silently or the workflow is being short-circuited before that step. Worth a
look before the next daily run regenerates a fresh stale branch.

## Files

- `audit-branches.sh` — the audit script (reusable for future cleanups)
- `branch-audit.tsv` — raw per-branch data
- `delete-list.txt` — `repo<TAB>branch` pairs that were deleted
- `delete-log.tsv` — per-deletion status (all `ok`)
