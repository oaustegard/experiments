# Memory audit: checking a memory store against live state

## Answer

Muninn's long-term memory store holds claims about how things are now: a file
lives at a path, a skill is at some version, an issue is open, a host is
blocked. Those go stale while the memory still reads as current. This run
pulled every checkable present-tense claim out of the store, checked each one
against today's repos and services with Haiku 5.5 subagents, had a second
Haiku pass try to overturn every "no longer true" verdict, and wrote a
correction on top of each memory that failed.

**29% of checked claims no longer hold** (657 of 2,273), 56% still hold, and
15% could not be checked from this container. 544 memories now carry a dated
audit note listing what changed. Claims written in January to June 2026 fail
at 40 to 62% of the decided cases; claims from July onward fail at 15 to 29%.

The per-claim data stays private: it quotes memory text. This directory holds
the prompts, the scripts and `results/summary.json` (counts only).

## Findings

1. **Open items, skill versions and repo state go stale fastest**
   (`results/summary.json`, `by_kind`).

| kind | claims | outdated | holds | unverifiable |
|---|---|---|---|---|
| open_item | 185 | 45% | 34% | 21% |
| skill | 149 | 44% | 52% | 5% |
| repo_state | 533 | 40% | 50% | 10% |
| limit | 94 | 28% | 48% | 24% |
| tool | 245 | 24% | 35% | 41% |
| other | 328 | 23% | 59% | 18% |
| path | 480 | 19% | 76% | 4% |
| network | 257 | 17% | 67% | 16% |

   Open items and repo state are mostly the same event: a memory records "issue
   #N is open" or "PR #N awaits merge", and the issue later closed. 178 of the
   655 corrections have that shape. They are the cheapest to fix and the most
   misleading left alone, because an open item reads as work still owed. Skill
   claims go stale because versions move: a memory that names `mapping-webapp`
   0.4 was written against a skill now at 0.5.0.

2. **Stale network and tool claims cause wrong refusals** (`results/summary.json`,
   `by_kind`). "upload.wikimedia.org is blocked", "www.arcgis.com is blocked by
   the egress allowlist" and "HF LFS downloads are blocked" all now succeed.
   Tools have the highest unverifiable rate (41%) because checking a credential
   or a deploy path needs exactly the access a read-only audit withholds.

3. **Older memories fail more** (`results/summary.json`, `by_month`). Among
   claims that could be decided:

| written | claims | outdated |
|---|---|---|
| 2026-01 | 24 | 62% |
| 2026-02 | 69 | 43% |
| 2026-03 | 229 | 45% |
| 2026-04 | 191 | 41% |
| 2026-05 | 211 | 47% |
| 2026-06 | 197 | 40% |
| 2026-07 | 309 | 28% |
| 2026-08 | 471 | 29% |
| 2026-09 | 494 | 27% |
| 2026-10 | 76 | 15% |

   Two things change after June and this run cannot separate them: the memories
   are younger, and from July most sessions ran in Claude Code on the Web, the
   same environment this audit probes from.

4. **The adversarial recheck changed about 6% of failures** (`results/summary.json`,
   `pipeline.recheck`; ERRORS.md items 5 and 7). It overturned 6 of 674 (0.9%)
   and rewrote 35 corrections (5.2%), such as one that credited a hook's wiring
   to merge e36396c when commit d62662a did it, or one that counted 99
   RESULTS.md files where the repo has 103. I read 14 sampled corrections
   against the evidence and probed three myself (`upload.wikimedia.org`
   reachable, `delete_branch_on_merge` false, `/mnt/project` absent); all held.

5. **Haiku 5.5 broke the prompt's tool and write rules often enough that only a
   harness limit would enforce them** (`results/summary.json`, `haiku_conduct`).
   - 26 of 185 agents made 27 MCP calls the prompt did not allow; 22 of them had
     empty arguments `{}`. Most failed harmlessly (`send_message` ×9,
     `read_documentation` ×7). `create_session` with no arguments succeeds: 4
     agents each left an empty session behind. Adding "use only Bash, Read,
     Grep, Glob and Write" to the prompt after the first incident did not stop
     it: agents launched after the rule (v037, v043, v058, r012, r015, r016)
     still made such calls.
   - 3 agents sent POSTs despite the read-only rule (an MCP handshake to a
     public endpoint, two unauthenticated requests to a worker that answered
     401). The auto-mode classifier blocked a fourth, and blocked 4 more probes,
     3 of them touching credentials or the proxy configuration.
   - 6 agents wrote scratch files the prompt forbade; 5 deleted them and
     reported it.
   - Every `create_session`, `interrupt_session` and `archive_session` call,
     every POST and every classifier denial appeared in the agent's own final
     report. Most `read_documentation`, `send_message` and `list_*` calls did
     not; I found those only in the transcripts.

6. **About $3.35 input-side, paced by a 20-agent cap** (`results/summary.json`,
   `pipeline.tokens`). Summed over every agent transcript in the session
   directory (185 files; 184 were audit agents): 15.0M cache-write tokens and
   147.5M cache-read tokens. At Haiku 5.5's $0.10/MTok input price, assuming
   the standard 1.25× cache-write and 0.1× cache-read multipliers, that is about
   $3.35. Output tokens in Agent transcripts record only a streaming floor, so
   output cost is not measured. The Agent tool refuses the 21st background
   subagent ("You can run 20 subagents at once"), so the run was paced by
   relaunching into each freed slot.

## Method

| stage | unit | agents | in | out |
|---|---|---|---|---|
| select | memory | — | 3,705 live memories | 2,805 audited (900 excluded by tag: news digests, session logs, health, confidential) |
| extract | ~80 KB batch | 64 | 2,805 memories | 2,404 valid claims from 1,493 memories; 131 about the claude.ai chat container set aside |
| verify | 25 claims | 92 | 2,273 claims | 1,266 holds, 605 stale, 69 contradicted, 333 unverifiable |
| recheck | 25 verdicts | 28 | 674 stale or contradicted | 622 confirmed, 35 confirmed with a better correction, 6 overturned, 11 unsure |
| apply | memory | — | 657 outdated claims | 544 memories superseded with 655 corrections |

Every subagent was Haiku 5.5 through the Agent tool, `[no-context]`, reading
its instructions from a file. All 184 audit agents ran in 33 minutes of wall
clock, from the first claims file to the last recheck.

**Extract** (`prompts/1_extract.txt`). Each agent read one batch and wrote one
JSON line per standing claim: the verbatim quote, the claim restated as a
present-tense sentence, a kind, a suggested check and the environment it lives
in. `validate_claims.py` rejects a line whose quote is not a substring of the
memory; 31 lines failed that, all dropped.

**Verify** (`prompts/2_verify.txt`). Read-only probes: `gh api` GETs against 20
attached repos, the local checkouts, `curl`, the installed skills. Verdicts are
holds, stale (true when written, not now), contradicted (false when written)
or unverifiable, each with the probe and its output.

**Recheck** (`prompts/3_recheck.txt`). A fresh agent is told to prove the first
checker wrong with a different probe, and to mark anything about the claude.ai
chat container as unsure.

**Apply** (`apply_audit.py`). A supersede keeps the original memory text intact
under a header:

```
[AUDIT 2026-10-09: 1 statement(s) below no longer hold. ...]
- WAS: PR #489 in oaustegard/claude-skills is open. NOW: PR #489 ... was merged on 2026-03-27 ...
```

The superseded row stays in the store with `is_superseded=1`, and a fresh
backup (muninn-backup 08f7948d) was taken first.

**Prior art.** Two searches. STALE (arXiv 2605.06527, May 2026) tests whether
an agent notices that later context invalidates a stored memory; the best model
scored 55.2%. Hindsight's verify-before-trust routing re-checks consolidated
memory against the agent's own raw records. I did not find, in those two
searches, an audit that checks a whole memory store against the live
environment the memories describe.

## Log

One run, 2026-10-09, session 80cfae6f. Pilots: two extraction batches (b05,
b40) and two verify chunks (v000, v030) before each fan-out; each pilot changed
the prompt (ERRORS.md items 2 to 4).

Files:

- `prompts/1_extract.txt`, `prompts/2_verify.txt`, `prompts/3_recheck.txt`: the three agent prompts, as templates
- `validate_claims.py`: shape check and verbatim-quote check for extracted claims
- `chunk_verify.py`, `chunk_recheck.py`: batch the claims and verdicts into 25-row agent inputs
- `validate_verdicts.py`: shape and evidence check for verdict files
- `apply_audit.py`: folds verdicts and rechecks into supersedes; dry run by default, `--apply` writes
- `results/summary.json`: every number above
- `ERRORS.md`, `recheck.py`
