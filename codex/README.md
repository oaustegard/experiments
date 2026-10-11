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

Results, findings and the round-by-round log are in [`RESULTS.md`](RESULTS.md):
the build, the pages loaded into the memory store, an answer-quality A/B, and a
pass that worked through the pages' 393 listed contradictions.

The output lives in three places. Each page is a memory in the store (149
compact `analysis` memories prefixed `CODEX PAGE:`), so `recall` returns a
topic's synthesis next to its raw memories on every surface. The browsable site
is a private claude.ai artifact (index.html + codex.json). The full pages as
markdown are in the private hub, `claude-workspace/codex/pages/`, read with
`scripts/codex.py find|show`. None of the memory text is in this public repo;
this directory holds the pipeline only.

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

The build's dispatch ran from the main loop with `claude-workspace/scripts/fanout.py`
prompt files, one Agent call per freed slot, cap 20.

## Evaluation and upkeep

8. `store.py DATA write --all` loads the compact pages into the store; `measure`
   ranks them against a pre-codex baseline.
9. `ab.py DATA questions|answers|judge [ROUND]` then `report`: the blind A/B of
   answers drawn from recall with and without pages.
10. `reconcile.py DATA adjudicate`, `recheck`, `snapshot`, `live`, `live-recheck`,
    `plan`, `apply`: classify each listed contradiction, check the rest against
    read-only repo snapshots and the web, and write confirmed corrections.

Steps 9 and 10 run their task files through headless `claude -p` children with
`dispatching-subagents/scripts/headless_fanout.py` (claude-skills).

## Not done

- A monthly rebuild routine. The build is repeatable (`codex_export.py` then
  steps 2–8) but not scheduled.
- Duplicate subjects. Clustering split remex across five pages and world news
  across several; a merge pass before writing would cut the
  three-pages-per-query case.
- Stored pages still show the contradiction each was built with in their
  `Unsettled:` line, including the ones round 4 found were not contradictions.

## Files

`corpus.py`, `cluster.py`, `tasks.py` (prompts and task lists for all three
waves), `build.py`, `qa.py`, `store.py` (load pages into the memory store,
measure recall), `ab.py` (answer-quality A/B), `reconcile.py` (contradiction
pass), `site/template.html` (the viewer).
