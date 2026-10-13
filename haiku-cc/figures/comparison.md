haiku-cc numbers as of 2026-10-13T02:52Z, hub HEAD 5c1d6d81.

| | Carlini (Opus 4.6) | haiku-cc |
|---|---|---|
| model | Opus 4.6 | Haiku 5.5 workers (Opus 5.5 for the 3 foundation agents and coordination) |
| parallel agents | 16 | 6-24 per container; hub + 3 sibling containers from 11:27 |
| sessions | nearly 2,000 | 1,436 Haiku workers (1,070 hub + 366 on 3 siblings) |
| cost | just under $20,000 | $40.50 Haiku workers (hub $26.84, siblings $13.66) + $14.42 Opus foundation, API list price; Opus coordination not metered |
| tokens | 2B input, 140M output | hub only: 613M cache-read, 12.1M output |
| wall clock | two weeks | 21.9 h until the hub swarm stopped |
| language | Rust (std only) | Python 3.11 (stdlib only), PyPy for large inputs |
| size | 100,000 lines | 22,644 lines + 718 selftests |
| assembler/linker | GCC's (own ones 'still somewhat buggy') | system as + gcc as linker driver |
| headline result | builds Linux 6.9 on x86, ARM, RISC-V; QEMU, FFmpeg, SQLite, postgres, redis, Doom | CPython 3.12, SQLite, mbedtls, Redis, QuickJS, Lua, Doom, tcc; no Linux attempt |
| test suites | '99% pass rate on most compiler test suites' incl. GCC torture | 38,451/38,451 tracked tests pass |
| coordination | lock files in current_tasks/, agents pull/merge/push, resolve own conflicts | orchestrator queue by failure signature, per-worker worktree, serial no-regression gate; 518 merges on main |

| node | workers | merged | rejected | failed | spend | local merges | landed on hub main |
|---|---|---|---|---|---|---|---|
| hub | 1070 | 427 | 316 | 301 | $26.84 | 428 | 428 |
| s1 | 271 | 72 | 165 | 34 | $7.98 | 101 | 72 |
| s2 | 40 | 0 | 27 | 5 | $0.34 | 0 | 0 |
| s3 | 55 | 9 | 19 | 27 | $5.34 | 12 | 9 |
| escalation patches | | | | | | | 9 |
| **total** | **1436** | | | | **$40.50** |  | **518** |

| run | design | arm | runs | solved | $ total | $ per solve | median turns | median s |
|---|---|---|---|---|---|---|---|---|
| run 1 | 40 random failing clusters | haiku | 80 | 76 | 0.81 | 0.011 | 14 | 63.5 |
| run 1 | 40 random failing clusters | sonnet | 80 | 80 | 6.01 | 0.075 | 9 | 45 |
| run 2 | 4 clusters Haiku had failed | haiku | 8 | 8 | 0.10 | 0.012 | 14.5 | 125 |
| run 2 | 4 clusters Haiku had failed | haiku-high | 8 | 8 | 0.18 | 0.023 | 23 | 108 |
| run 2 | 4 clusters Haiku had failed | sonnet | 8 | 8 | 0.61 | 0.077 | 12 | 91 |
| run 3 | 10 clusters the swarm abandoned | haiku | 20 | 19 | 0.22 | 0.012 | 14 | 77.5 |
| run 3 | 10 clusters the swarm abandoned | haiku-high | 20 | 20 | 0.35 | 0.018 | 18 | 87 |
| run 3 | 10 clusters the swarm abandoned | sonnet | 20 | 20 | 1.74 | 0.087 | 9 | 47 |
