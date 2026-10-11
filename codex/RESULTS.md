# RESULTS: codex, a sourced wiki of Muninn's memory store

Built 2026-10-11 from 3,699 memories and 102 work sessions by Haiku 5.5 subagents,
then loaded into the store, tested for answer quality, and used to correct the
store. The pipeline and how to rerun it are in [`README.md`](README.md).

## Answer

Can near-free Haiku agents compile the whole memory store into something that
improves answers? Yes, for one kind of question. 314 Haiku agents wrote and
citation-checked 149 topic pages for $11.61. Each page is stored as a compact
memory, and it surfaces in recall for its own subject (17 of 18 test queries).
In a blind A/B over 50 questions, an answer drawn from recall that includes the
page beat one drawn from recall without it on synthesis questions that retrieved
their own page: 13 wins, 5 losses, 2 even (sign test p ≈ 0.10). In 5 of those 20,
the page turned an outdated answer into a current one, and none went the other
way. Fact questions gained nothing (0 wins, 2 losses of 6). Identical-context
controls split 4–6, which is the noise floor. The pages' 393 listed
contradictions were mostly not errors. Checked against the cited notes, 16 repo snapshots and the web, 193 were not
errors. 48 wrong statements in 45 memories were corrected after a second
reviewer confirmed each one, and 131 could not be settled from here.

What would change this: more synthesis questions, since the p-value rests on 18
decided ones, and a judge from outside the Claude family.

## Findings

1. **A page helps where the answer is "where does X stand now".** On 20 synthesis
   questions that retrieved their own page, answers had 0.45 → 0.23 errors and
   0.72 → 0.97 currency under the Opus judge. The page arm won 13 questions and lost 5.
   (Round 3, `ab.py`)
2. **It does not help a point fact, and an off-topic page costs a little.** On 6
   fact questions whose context changed, 3 through another topic's page, the
   page arm won none and lost 2. (Round 3)
3. **Same-context answers still split a third of judge votes.** In 21 questions
   where both arms got identical memories, Sonnet's two answers drew 15 non-tie
   votes of 42. A judge comparison without that control reads noise as effect.
   (Round 3)
4. **Most contradictions a summarizer lists are not errors.** Of 393, 193 were two values from different dates or two
   different measures. 63 were a statement that was wrong, and 48 of those were
   corrected in the store. The rest were in session summaries, which are not
   memories, or were already corrected by the 2026-10-09 audit. Haiku sent 59%
   to a live check, and a Sonnet recheck overturned 6 of the 69 errors Haiku
   proposed.
   (Round 4, `reconcile.py`)
5. **Dispatching cost more than the work.** The Agent-tool build made 1,458 Opus
   parent calls, about $160–170, to launch $11.61 of Haiku agents. The A/B and
   the reconciliation ran as headless `claude -p` children: 747
   children from about a dozen parent calls. (Rounds 1, 3, 4)
6. **The build is accurate to its sources.** The verifier rewrote 4.4% of 11,043
   statements and cut 0.3%. A spot check of 20 random cited sentences found
   20/20 consistent. (Round 1)
7. **Transcripts are 99.6% tool output.** 320 MB of distinct sessions held 1.3M
   characters of conversation. (Round 1, `corpus.py`)
8. **Compact pages surface by their words, not their length.** They come back at
   ranks 1–8 for queries naming their subject, and they miss queries whose words
   the compact text lacks. A subject split across pages returns up to 3 of them.
   (Round 2, `store.py`)

## Method

- **Corpus.** Every live memory except those tagged confidential or health, and
  the hub's archived transcripts reduced to user and assistant prose. Clustered
  with model2vec and Ward into 149 topics.
- **Build.** One Haiku writer per topic, one Haiku verifier per page, citations
  repaired and checked by `qa.py`.
- **Store.** Each page goes in as a 216–300-word `analysis` memory prefixed
  `CODEX PAGE:`, with every cited memory in `refs`.
- **A/B** (`ab.py`). 50 topics not used to tune the store. Sonnet 5.5 wrote one
  synthesis and one fact question per topic from a sample of up to 30 of its
  memories, never from the page. Round 1 took 20 topics and alternated kinds;
  round 2 took 30 and kept synthesis only.
  - Arm A is recall's top 10 with codex pages removed. Arm B is recall's top 10.
  - Sonnet 5.5 answered from each arm's memories alone, with no tools.
  - Opus 5.5 judged each pair twice, once per order, against every memory in the
    topic plus the question writer's reference answer.
  - It counted errors, reference points covered, and whether the answer gave
    the current state. It then picked the better answer.
  - Questions with identical contexts are the control.
- **Reconciliation** (`reconcile.py`).
  - Haiku classified each listed contradiction from the full text of the notes
    it cites.
  - Sonnet tried to overturn every "note wrong" verdict.
  - Items needing a live source went to Haiku with web search and read-only repo
    snapshots: blobless clones with the remote removed, plus a commit log dump.
    Sonnet then rechecked each of those, opening the cited sources itself.
  - Confirmed corrections were written as a `[RECONCILED 2026-10-11 ...]` header,
    with WAS/NOW lines, on a superseding memory.
- **Dispatch.** Rounds 1–2 used the Agent tool with the `auditor` type. Rounds
  3–4 used `claude -p` children through `dispatching-subagents/scripts/headless_fanout.py`
  (claude-skills).

## Log

### Round 1: build (2026-10-11)

| | |
|---|---|
| memories compiled | 3,699 of 3,745 (46 tagged confidential or health left out) |
| sessions compiled | 102 (1,026 archived stop-commits, deduplicated by session) |
| topics / pages | 149, in 9 sections |
| page text | 144,710 words, 7,774 citations |
| citations to an id not in the topic's sources | 0 after build (5 in drafts: a session id cited with the memory prefix, repaired by kind) |
| statements checked by the verifier | 11,043: 10,533 supported, 481 rewritten to match the source, 29 cut |
| open items / unresolved contradictions surfaced | 1,213 / 393 |
| independent spot check (20 random cited sentences) | 20/20 consistent with the cited source; one carries a derived figure the source window did not state |
| agents | 16 session-card, 149 writer, 149 verifier; all Haiku 5.5, `auditor` agent type |
| tokens | 41.2M cache-write, 298.8M cache-read, 160K output; 484 of 6,184 turns past 100K |
| cost | $11.61 at $0.10/$0.50 per MTok, 5× for turns past 100K: cards $0.38, writers $6.27, verifiers $4.95 |

Reading at the time:
- The verifier's rewrites mostly removed inference the writer had added on top of
  a source. The citation id was right and the sentence said more than the memory.
- 23 of 314 agents stepped outside the `auditor` allowlist (50 refused `Edit`
  calls, a few `sed -i`/`cp` on `.codex/` paths). No final output was damaged.
- `fanout.py next` marked tasks done before their agents stopped, so refills hit
  the 20-agent cap.
- Topics were capped at 110K characters so one agent reads its input in two or
  three `Read` calls. The largest writer contexts still crossed 100K tokens and
  paid the long-context rate on 8% of turns.

The parent's dispatching cost was measured after the fact from the session
transcript: 1,458 Opus 5.5 calls, 601.3M cache-read, 3.58M cache-write and 0.94M
output tokens. That is about $160–170, not all of it dispatch.

### Round 2: pages in the memory store (2026-10-11)

Oskar asked whether the codex could be used at all, given that the pages lived
in a git repo `recall` never sees. `store.py` stored each page as a compact
memory, measured against a top-10 baseline taken before any page existed:

| stage | queries returning a relevant page | pages per top 10 | baseline memories displaced |
|---|---|---|---|
| pilot, 10 pages | 7 of 10 target queries (ranks 2–8); 0 of 8 controls | 0–1 | 0–1 |
| all 149 pages | 17 of 18 | 0–3 | 0–3 |

The three pilot misses were queries whose words the compact text lacks
("polarized", "benchmark", "403"). A rebuild forgets the previous build's page
with the most citation overlap (Jaccard ≥ 0.3 over refs). The monthly audit
skips pages by their `CODEX PAGE:` prefix, because two non-page memories already
carried the `codex` tag.

### Round 3: answer-quality A/B (2026-10-11)

Oskar's question after round 2: was this worth it? Round 1 asked 20 questions
(10 synthesis, 10 fact). Round 2 added 30 synthesis questions on unused topics,
because round 1's 5 synthesis questions with a page were too few.

Per question, net of the two judge orders:

| set | questions | B better | A better | even | sign p |
|---|---|---|---|---|---|
| synthesis, own page retrieved | 20 | 13 | 5 | 2 | 0.096 |
| fact, context changed | 6 | 0 | 2 | 4 | 0.50 |
| identical context (control) | 21 | 4 | 6 | 11 | 0.75 |

Judge means on the 20 own-page synthesis questions:

| | errors | points covered | current |
|---|---|---|---|
| A (no page) | 0.45 | 6.78 | 0.72 |
| B (page) | 0.23 | 7.72 | 0.97 |

- B retrieved a page on 29 of 50 questions. 20 were its own topic's synthesis
  questions, 3 were synthesis questions where the page changed only the other
  memories, and 6 were fact questions.
- The judge orders agreed on 37 of 50 questions.
- One of A's losses (a18) was an answer saying no retrieved memory touched the
  subject, where B's page supplied it.
- One B loss was a judge artefact. The answer cited the codex page, which is not
  in the judge's reference set, and the judge counted that as an error.

Cost: questions $4.69 (Sonnet), answers $7.39 (Sonnet), judges $20.37 (Opus),
$32.44 in total over 300 headless children.

### Round 4: working through the 393 contradictions (2026-10-11)

Oskar's ask: "work through the 393 contradictions". Each page lists the
points where its sources disagree, with the ids involved.

| stage | model, tools | in | out | cost |
|---|---|---|---|---|
| adjudicate, one task per page | Haiku, none | 393 items on 147 pages | 89 different measures, 32 changed over time, 23 memory wrong, 17 unresolvable, 231 need a live check, 1 missing | $0.62 |
| recheck the memory-wrong verdicts | Sonnet, none | 23 | 18 confirmed (2 with a better NOW), 5 overturned | $0.61 |
| live check | Haiku; Read/Grep/Glob on 16 repo snapshots, WebSearch, WebFetch | 231 | 72 consistent, 46 note wrong, 113 could not check | $2.86 |
| recheck the live note-wrong verdicts, opening every cited source | Sonnet, same minus WebSearch | 46 | 45 confirmed (1 with a better NOW), 1 overturned | $1.97 |

Of the 63 confirmed errors:
- 48 statements in 45 memories were written as a `[RECONCILED 2026-10-11 ...]`
  header with WAS/NOW lines, on a superseding memory tagged `codex-reconciled`;
- 9 were in session summaries, which the store does not hold;
- 4 had already been corrected by the 2026-10-09 audit;
- 1 quoted text that is not in the memory;
- 1 was dropped at review: a cycling-science claim whose only evidence was a
  search tool's paraphrase.

I read every planned correction before applying it. One NOW was rewritten by
hand: an election-result correction that read as if it replaced a later poll
figure.

Corrections by kind:
- the owner's own repos, where a note said a PR was open, a file absent or a gate
  missing and the snapshot showed otherwise;
- world facts, such as a May 2026 note describing a "Labour+Centre" Norwegian
  government that Centre had left in February 2025;
- dates and attributions in paper citations;
- pricing, such as the 1-hour cache write billed at 1.25× instead of 2×.

The pass also found that some write path loses dollar amounts to shell
expansion. One memory read "512 x ~5k = ~8M hardware = 0.25% of the .2B budget",
where the session had "$35k", "$18M" and "$7.2B". Another had "spent 10B" for
"$210B". A scan of the export finds 6 memories with a ".2B"-style remnant, all
news digests and perch notes. Whole-number amounts cannot be found that way.

Not settled: 113 live items needed Strava, the store's own timestamps,
private services or sources the web did not show, plus 17 the notes could not
settle. They stay listed on their pages.

All four stages ran as 447 headless `claude -p` children for $6.06.
