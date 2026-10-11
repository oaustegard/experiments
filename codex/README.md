# codex

Every live memory in Muninn's store, plus every archived work session, compiled
into a sourced wiki: one page per topic, every factual sentence citing the memory
or session it came from, every citation then checked by a second model pass.
Built 2026-10-11 by 314 Haiku 5.5 subagents for $11.61.

Oskar's ask (2026-10-11): use the near-free Haiku subagents to *build* something
that cost had ruled out. Cost had ruled out reading everything: `recall` returns fragments that match a query, so "what do
I believe about X, and since when" had no answer at any resolution, and the
2026-10-09 audit (`memory-audit/`) corrected 657 claims one at a time without
producing an overview.

## Result

| | |
|---|---|
| memories compiled | 3,699 of 3,745 (46 tagged confidential or health left out) |
| sessions compiled | 102 (1,026 archived stop-commits, deduplicated by session) |
| topics / pages | 149, in 9 sections |
| page text | 144,710 words, 7,774 citations |
| citations to an id not in the topic's sources | 0 after build (5 in drafts: a session id cited with the memory prefix, repaired by kind) |
| statements checked by the verifier | 11,043: 10,533 supported, 481 rewritten to match the source, 29 cut |
| open items surfaced | 1,213 |
| unresolved contradictions surfaced | 393 |
| independent spot check (me, 20 random cited sentences) | 20/20 consistent with the cited source; one carries a derived figure the source window did not state |
| agents | 16 session-card, 149 writer, 149 verifier; all Haiku 5.5, `auditor` agent type |
| tokens | 41.2M cache-write, 298.8M cache-read, 160K output; 484 of 6,184 turns past 100K |
| cost | $11.61 at $0.10/$0.50 per MTok, 5× for turns past 100K (`METHODS.md`) — cards $0.38, writers $6.27, verifiers $4.95 |

The output lives in three places. Each page is a memory in the store (149
compact `analysis` memories, below), so `recall` returns a topic's synthesis next
to its raw memories on every surface: claude.ai, Claude Code, the connector. The
browsable site is a private claude.ai artifact (index.html + codex.json, 8.9 MB
with every memory's text so a citation opens its source). The full pages as
markdown are in the private hub, `claude-workspace/codex/pages/`, read with
`scripts/codex.py find|show`. None of the memory text is in this public repo;
this directory holds the pipeline only.

## Pages in the memory store

Oskar's question after the first build: could the codex be used at all, given
the pages lived in a git repo that `recall` never sees? It could not, outside a
Claude Code session that remembered to call `codex.py`. `store.py` loads every
page into the store as a compact memory: summary, the current-state paragraph,
three open items and one unsettled point, each item clipped to 35 words, with
citations inline and every cited memory id in `refs` (216 to 300 words; `recall`
prints memories in full, so a whole 1,000-word page per hit would flood context).
Type `analysis`, conf 0.7, tags `codex`, `codex-2026-10-11`, `codex-<section>`
plus the topic's own tags.

Measured on 18 queries against a top-10 baseline taken before any page was
stored (`store.py measure`):

| stage | queries returning a relevant codex page | codex pages per top 10 | baseline memories displaced |
|---|---|---|---|
| pilot, 10 pages | 7 of 10 target queries (ranks 2–8); 0 of 8 controls | 0–1 | 0–1 |
| all 149 pages | 17 of 18 | 0–3 | 0–3 |

Every page that surfaced matched the query's subject by title (Ann Arbor →
"Ann Arbor property search" at 2, Strudel → "Strudel FM generative radio" at 1,
polarized training → "Cycling training science" at 1). The three pilot misses
were queries whose words the compact text lacks ("polarized", "benchmark",
"403"): `recall` ranks a page on its words like any other memory, so length
does not let it crowd. The worst case is a news subject spread over several
pages: "Iran Hormuz shipping" returns three (the war page and two Zeitgeist
brief pages), displacing three raw memories.

A rebuild stores each new page, then forgets the previous build's page with the
most citation overlap (Jaccard ≥ 0.3 over refs). `remembering.supersede()` would
keep the chain but cannot carry refs. Two non-page memories already carried the
`codex` tag (this run's own record, and an April analysis of OpenAI's Codex
repo), so both the rebuild match and the monthly audit's exclusion key on the
`CODEX PAGE:` text prefix rather than the tag.

## Pipeline

1. `scripts/codex_export.py` (hub) dumps the store to the gitignored `.codex/`.
2. `corpus.py` reduces each transcript on the hub's `transcripts` branch to what
   was said: user messages and assistant prose, tool calls as one-line names,
   tool output dropped. 3.7 GB of stop-commits became 320 MB of distinct sessions
   and 1.3M characters of conversation: tool output is 99.6% of the bytes.
3. `cluster.py` embeds tags + text with model2vec potion-base-32M (seconds, CPU),
   clusters with Ward linkage at k=160, then splits topics over 110K characters
   and folds those under 12K into their nearest neighbour: 149 topics, median
   34K characters.
4. `tasks.py sessions`: 16 agents write a structured card per session (asked,
   did, outcome, corrections, lessons). Each card joins its nearest 1–3 topics by
   centroid similarity (253 placements).
5. `tasks.py writers`: one agent per topic reads its memories and cards and
   writes a page — summary, body with a citation on every factual sentence,
   timeline, open items, contradictions, terms. Audit-corrected memories carry
   `WAS/NOW` headers; the prompt says the NOW sentence wins.
6. `tasks.py verifiers`: a fresh agent per page reads the page and the same
   sources and marks every sentence supported, fixable (rewrite to match) or
   unsupported (delete), returning the corrected page plus a change log.
7. `build.py` repairs citation-kind mismatches, links related pages (title/term
   mentions, then centroid neighbours), and writes the site and the markdown
   store. `qa.py` is the model-free check: unknown ids, uncited sentences,
   share of a topic's memories cited (mean 77%).

Dispatch ran from the main loop with `claude-workspace/scripts/fanout.py`
prompt files, one Agent call per freed slot, cap 20.

## Findings

- **Transcripts are almost all tool output.** The archive looked like a 100M+
  token reading job when the codex was proposed; the conversation in it is 1.3M
  characters. The memories (6.4M characters) are the bulk of what there is to
  know. Any estimate of transcript-mining cost should be made after stripping
  tool results, not before.
- **The verifier rewrote 4.4% of statements and cut 0.3%.** Most rewrites
  remove inference the writer added on top of a source (an arithmetic
  reconciliation, a "the grader now counts tests" that the source stated only
  as a lesson). In those sentences the citation id is correct and the sentence
  says more than the memory does, so an id check would pass them.
- **A per-agent Read limit bounds topic size.** Topics were capped at 110K
  characters so one agent reads its whole input in two or three `Read` calls;
  the largest writer contexts still crossed 100K and paid the long-context rate
  on 8% of turns.
- **Haiku 5.5 conduct.** 23 of 314 agents stepped outside the `auditor` tool
  allowlist: 50 `Edit` calls, all refused by the harness because the type has
  no Edit; a handful of `sed -i`, `mkdir`, `cp` on `.codex/` paths, including one
  verifier that edited the draft page it was checking before writing its own
  output. No final output was damaged — the build reads verifier outputs only.
- **`fanout.py next` over-reports free slots.** It marks a task done as soon as
  its output validates, which can be before the agent stops, so the next Agent
  call hits the 20-agent cap. Dispatching one call per completion notification
  avoided it.

## Not done

- Section overview pages and a regenerate-monthly routine; the build is
  repeatable (`codex_export.py` then steps 2–7) but not scheduled.
- No answer-quality test: the store measurement shows pages surface for the
  right queries, not that an answer drawn from page plus memories beats one
  drawn from memories alone.
- Duplicate subjects. Clustering split remex across five pages (t011, t013,
  t057, t097, t139) and world news across several; a merge pass before writing
  would cut the three-pages-per-query case.

## Files

`corpus.py`, `cluster.py`, `tasks.py` (prompts and task lists for all three
waves), `build.py`, `qa.py`, `store.py` (load pages into the memory store,
measure recall), `site/template.html` (the viewer).
