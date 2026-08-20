# nl2sh-retrieval — building the deterministic tier, and gating the small model

**Started / finished:** 2026-08-19 · **Status:** done — **the small model works
after 25 minutes of fine-tuning, retrieval is now the bottleneck, and an
adversarial pass returned OVERSTATED on the first draft of the retrieval numbers.**

[`nl2sh-scoping`](../nl2sh-scoping/README.md) established the shape of the
problem — utility selection over a long tail of utilities, not flag composition
— and argued for a heterogeneous chunked corpus with a small model reading it.
This builds that corpus, the parameter extractor and the retrieval tier, and
runs the one test that decides whether the small-model component exists at all.

Headline, in the order the numbers should be trusted:

- **The zero-shot gate is negative, and fine-tuning fixes it.** Pleias-RAG-350M
  produces **0 usable commands in 40** handed the right source every time; after
  600 rows and **25.5 minutes on 4 CPU cores** it reaches **0.923 on the
  non-`find` slice, where a constant answer scores 0.000**. The failure was
  output shape, not capability.
- **The premise that chose this model did not survive, and the ablation confirms
  it**: verbatim rate is 0.000 before *and* after, and the RAG-less sibling
  fine-tunes to **0.846 non-`find` against the RAG model's 0.923** — one example
  out of 13. The source-quoting the model was chosen for is used by neither; the
  model choice was not the important decision (§7).
- **Retrieval is now the bottleneck**: **0.233 recall@5** on the deployment
  slice, and a query-*independent* frequency prior beats the naive headline.
- **Parameter extraction works**: 0.971 precision held out — but only 37.4% of
  requests contain every operand, and only 53% of correct extractions are a
  whole command token.
- **Rule iteration neither degrades nor transfers**: strictly monotone
  (McNemar 125–0), and wild accuracy sat at 0.540 across all three rounds.

Everything below that was measured by one agent was recomputed by another
instructed to break it. Where the two disagree, the correction is stated inline
rather than the original quietly replaced.

## Provenance of the committed corpus

`data/chunks.jsonl` redistributes worked examples verbatim from
[tldr-pages](https://github.com/tldr-pages/tldr), copyright the tldr-pages team
and contributors, licensed **CC-BY-4.0**. The `man_option` and `man_example`
chunks come from the manuals installed in the build container and carry each
package's own licence. This note was missing until 2026-08-20 and is the
attribution CC-BY requires wherever the content travels.

## 1. The corpus — 31,169 chunks over 4,698 utilities

`build_corpus.py` turns tldr pages and roff man pages into three chunk kinds:

| kind | n | runnable | role |
|---|---|---|---|
| `tldr_example` | 29,437 | yes | quotable; covers 96% of the top-50 utilities |
| `man_option` | 1,601 | no | universal coverage, but needs *composition* |
| `man_example` | 131 | 65% | quotable, rare — see below |

4,688 of 4,698 utilities have at least one runnable chunk. `runnable=true` on
`man_example` is a **kind label, not a guarantee**: after two parser fixes the
share of man examples that actually invoke their own utility went 19% → **65%**,
and the remaining 35% are `psql` SQL session transcripts. 65% is the honest
number. 756 tldr stub/redirect pages and 250 exact duplicates were dropped. Chunks are **median
19 tokens, p90 34** — far smaller than the scoping pass estimated, which matters
for the retrieval design: at that length a dense embedding has very little to
work with and the lexical arm should carry the load.

**A correction to `nl2sh-scoping`.** That directory measured man-page option
chunks by splitting on the roff `.TP` macro and reported a median of 56 tokens
over 1,237 chunks. `.TP` is **not** the universal option idiom: 32 of the 60 man
pages on this container carry zero `.TP`, because DocBook-XSL generated pages
(the PostgreSQL ones here) spell an option as `.PP` / `\fB\-a\fR` / `.br`
instead. The scoping figure was therefore computed over the 28 pages that happen
to use `.TP`, and the parser here handles both idioms. The qualitative
conclusion — man pages chunk into retrieval-sized units — survives; the specific
number did not.

## 2. Parameter extraction — precision 0.97 on held-out, and two structural limits

`extract_params.py` pulls literal values out of the request text: paths, globs,
extensions, sizes, durations, ports, PIDs, permissions, users, hosts, quoted
literals. It never generates a value, which is `monad-bsky`'s one unconditional
transfer and, in a shell, a safety property rather than an accuracy one.

Scored against NL2Bash — does an extracted value actually appear in the gold
command? 300 dev pairs, 100 held out, seed 20260819:

| | dev (n=300) | holdout (n=100) |
|---|---|---|
| precision (substring) | 0.907 [0.875–0.931] | **0.971** [0.927–0.989] |
| recall, all gold values | 0.828 | 0.872 |
| recall, values it does **not** mark | **0.453** | **0.545** |

A marked-value recall of 0.98 appeared in an earlier draft of this table and has
been removed: the agent that built the extractor flagged it as **near-tautological**,
because the "marked" denominator is roughly the set the extractor fires on.
Quote 0.838 and 0.469 (both splits pooled).

Holdout scoring higher than dev is the good direction — nothing was tuned into
the dev split. Per-kind precision runs 0.86–1.00 with `glob`, `identifier`,
`var`, `user` and `ip` at 1.00; `duration` is worst at 0.71.

**The honest weakness is the unmarked half.** NL2Bash annotates some values in
its prompts; on those the extractor is near-perfect (0.95–0.98). On values the
corpus does not mark — the ones phrased naturally rather than quoted — recall is
**0.45–0.55**. That is the real-world case, and it is where this component will
actually be judged.

## 3. The Pleias gate — two engineering findings before the verdict

Two things had to be fixed before the model could be judged at all, and both are
the kind of mistake that produces a false negative.

**The prompt must end with `<|language_start|>\n`.** The first attempt built the
prompt from the special-token list — query block, source blocks — and got a
**0.000 parse rate** that looked exactly like `monad-bsky`'s zero-shot Monad
result. It was not a capability verdict: without that trailing token the model
has no signal that the source list is closed, so it keeps emitting
`<|source_start|>` blocks and degenerates into repetition. Reading
`Pleias-RAG-Library`'s own `_format_prompt` cost two minutes and cost the
first measurement entirely. **Read the reference implementation before
inferring a prompt protocol from a token list.**

**Pre-filling the reasoning scaffold is a 9x latency win.** The model generates
`language` → `query_analysis` → `query_report` → `source_analysis` →
`source_report` → `draft` before it reaches `<|answer_start|>`, about 700 tokens
of preamble. Pre-filling a minimal scaffold and starting generation at the
answer span took a query from **61.2 s to 5.4 s** on 4 CPU cores, without
changing what the model says.

An independent spot-check (not the agent's own eval) on five hand-written
requests reproduced exactly that split: `.log` / `/var/log` / `30 days`,
`port 8080`, `100MB` / `~/Downloads` and `644` all extracted correctly, while
`src/` as a relative directory and `deploy` as a username after "as" were both
missed. The failures are unmarked, naturally-phrased values, which is the case
that matters and the one to fix first.

Two structural findings from the same run bear on the architecture more than the
precision figure does. **Only 37.4% of requests contain every operand their
command needs** — the rest of the time the value simply is not in the sentence,
which is `gh-mcp-regex-fit`'s 13.5% referent-presence result reappearing in a
new domain. And **only 53% of correct extractions are a whole command token**:
the rest compose (`20` → `+20M`, `1080` → `-D1080`), so splicing extracted spans
straight into argv would be wrong about half the time *even with perfect
extraction*. Extraction feeds a template; it does not assemble a command.

### The verdict: 1 of 40, and the one is an accident

Best configuration — 3 sources (the gold utility's tldr example plus two
distractors), scaffold prefilled, greedy decoding, 40 NL2Bash requests where the
gold utility is guaranteed to be among the sources:

| metric | value |
|---|---|
| parse rate | 1.000 |
| **names the gold utility somewhere** | **0.550** |
| **emits anything command-shaped** | **0.025** |
| **utility correct** | **0.025** (1 of 40) |
| **quotes a source command verbatim** | **0.000** |
| median latency | 6.5 s (4 CPU cores) |
| median prompt | 121 tokens |

The single "hit" does not survive inspection: for *"Delete all regular files
named 'FILE-TO-FIND' under current directory tree"* it answered *"The file
system you're looking for is the one that contains the directory path
'*.ext'…"*, which matched only because a token collided. **The true rate is
0 of 40.**

What it produces instead is confident encyclopedic prose about shell:

> The command "kill" is used to terminate a program by terminating its execution.

> The file system is running in a container, and it's running in the current folder.

This is `monad-bsky`'s zero-shot result replicated one model generation later —
0.000 routable there, 0.025 here — and for the same reason. A model trained to
answer questions from sources, handed shell documentation, writes an
encyclopedia entry about shell. It is doing its job; its job is not this.

**The specific bet that failed is worth naming, because it was the good one.**
The architecture's premise was that converting *generation* into *extraction*
would rescue a tiny model: retrieval supplies the command, the model only has to
quote it, and Pleias-RAG is trained precisely to quote. **Verbatim rate is
0.000.** It does quote — the earlier traces show correct `<ref>` spans lifting
source text exactly — but it quotes *descriptions into prose*, never a command
as an answer. Span-copying is real and points the wrong way.

Two things this does **not** show. It does not show 350M is too small: source
comprehension is partial but present (0.55 names the right utility), and it
degrades with source count — 6/8 at 3 sources, 4/8 at 5, worse at 15 — which is
an architectural constraint (**keep the shortlist at k≈3**), not a verdict. And
it does not show fine-tuning would fail: `monad-bsky` took the 56M sibling from
**0.000 to 0.481** with 800 rows and three epochs on CPU. Installing the output
shape is exactly what fine-tuning does, and it is the open question.

What it does show is that **there is no zero-shot path here**, and that any plan
resting on one should stop. That is what the gate was for.

## 4. Does a frontier compiler degrade when fed more failure cases? Not measurably — and the round that would have tested it had no signal

Oskar's pushback on the third pass of `gh-mcp-regex-fit`: the iteration that
regressed was *Gemini* revising its own rules, not a frontier model doing it by
reading failures, and "I would expect YOU to not make more errors when fed
another failure case." That arm had never been run. It has now, under the
identical protocol — start from the clean-room rule set, score on family A,
show at most 120 family-A errors, revise, repeat; never look at family B or wild.

| round | rules | family A | family B | wild |
|---|---|---|---|---|
| clean room | 154 | 0.863 | 0.504 | 0.540 |
| **Claude** iter1 | 173 | 0.995 | 0.545 | 0.540 |
| **Claude** iter2 | 174 | **1.000** | 0.545 | 0.540 |
| **Claude** iter3 | 174 | **1.000** | 0.545 | 0.540 |
| | | | | |
| Gemini iter1 | 80 | 0.892 | 0.219 | 0.405 |
| Gemini iter2 | 82 | 0.912 | 0.210 | 0.419 |
| Gemini iter3 | 83 | **0.789** | 0.176 | 0.419 |

**He was right that Claude does not degrade — but the round-3 test he was
pointing at never actually ran.** `rules_claude-iter2.json` and
`rules_claude-iter3.json` are byte-identical (sha `01ffa759965f`), and an
earlier draft of this section read that as the model declining to change
anything when shown fresh failures. It is not. **Claude saturated family A at
round 2 — 1.000, zero errors — so round 3 had no error signal to show it.**
Round 3 is the identity revision by construction. Gemini still had ~88 family-A
errors at the same point, which is why its round 3 had something to do and did
it badly.

What *is* measured, and is strong: **every Claude revision was strictly
monotone.** Paired McNemar, clean room → iter1: **125–0** on family A and
**39–0** on family B. Not one previously-correct row was lost, on any split, at
any round. Gemini's *improving* round 1 already broke 62 family-A rows and its
round 3 broke 193. So the "fixes shadow correct rules" mechanism is real and is
Gemini's; Claude's edits did not exhibit it.

As a substitute for the missing test the agent ran `iter3-speculative` — ten
broadening edits made with no error signal at all — and it scored identically on
all three splits (1.000 / 0.545 / 0.540). Revision without signal did not
degrade either. That is suggestive, not the experiment.

**One confound to name:** the two arms did not revise the same way. Claude
patched by anchor into an existing list; Gemini regenerated the whole list each
round, which is the mechanism `gh-mcp-regex-fit` blames for the regression. So
this compares *model plus edit procedure*, not model alone. My third-pass
writeup stated the regression as a general law about error-driven iteration;
that was an overreach either way.

**But the second finding matters more for the architecture, and it is not
good news.** Iteration bought perfect in-sample fit and essentially nothing
else:

- family A: 0.863 → **1.000** (all 948 rows)
- family B: 0.504 → 0.545 (+0.041, all of it in round 1)
- wild: 0.540 → **0.540 → 0.540 → 0.540**

Wild did not move by a single query across three rounds. Round 1 changed 5 of 74
wild predictions and broke exactly as many as it fixed. So a frontier model
iterating on its own errors is **safe but non-transferring**: it closes the
in-sample gap and leaves generalisation where it found it.

That is a sharper design rule for the logging component than the one it
replaces. The reason to log real requests is **evaluation, not supervision** —
not because feedback degrades the rules, but because it does not carry past the
rows it was computed on. And the one round that helps is the first: round 2 was
+0.000 everywhere, round 3 was a no-op the model declined to make.

## 5. The retrieval tier: 0.233 recall@5 where it counts, and the reason is corpus breadth

`retrieve.py` builds a BM25 index with code-aware tokenization — a compound
token emits the whole token and its parts, which is what made
`hybrid-code-index` work — over all 31,169 chunks. Scored against NL2Bash: for a
600-query sample, is the gold utility among the utilities of the top-k chunks?

| slice | n | @1 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| all | 600 | 0.098 | 0.280 | 0.382 | 0.473 |
| non-find | 225 | 0.116 | 0.324 | 0.396 | 0.480 |
| tail (stratified) | 369 | 0.071 | **0.203** [0.165–0.247] | 0.252 | 0.290 |
| **non-find AND prompt does not name the utility** | **180** | **0.061** | **0.233** | — | **0.389** |

Median latency **0.184 ms** for the metric actually reported, index build 1.3 s.
Speed was never the question.

**An adversarial verification pass (`verify_retrieval.py`, 8 checks) recomputed
all of this independently and returned OVERSTATED.** The arithmetic reproduces
to four decimals; three framing problems do not survive, and the bottom row of
that table is the one it says to quote.

**(a) The `all` row loses to a constant.** A query-*independent* list of the 20
commonest NL2Bash utilities scores **0.625@1 / 0.787@20**, beating BM25's
0.098 / 0.473 at every k. That is the corpus's 60% `find` skew, not retrieval
skill. The prior collapses to 0.000@1 / 0.431@20 on non-find and to **0.000 at
every k** on the tail, which is why only the non-find and tail rows carry a
claim, and why they need the prior printed beside them. Against a random
retriever BM25 is fine (0.473@20 vs 0.010); random was simply the wrong baseline.

**(b) A third of the recall comes from prompts that name the answer.** 34.7% of
NL2Bash prompts contain the gold utility as a literal token — annotators wrote
the English while looking at the command (*"Convert \*.au files … using `sox`"*).
Those queries score **0.586@5**; the other 65% score **0.117@5**. The premise of
this tier is a user who does *not* know which utility to reach for, so the
second number is the deployment number.

**(c) The quoted latency was for a different computation** — 0.119 ms times
`topk` only, while the reported ranking needs `rank_utilities(pool=300)` at
0.184 ms. Sub-millisecond either way; the number was wrong by 1.5x and is
corrected above.

Two checks came back clean. **No leakage:** zero exact normalised matches
between the 600 prompts and the 29,437 tldr descriptions, median best-Jaccard
0.16, one query in 600 above 0.5 — the corpora do not share a source. **Gold
parse is sound:** 29/30 spot-checks correct, 1.4% junk labels (mostly
`VAR=$(cmd)` collapsing under quote-masking), which *deflates* recall by at most
1.4 points rather than explaining it.

For comparison, `gh-mcp-regex-fit` measured BM25 at **0.851 recall@5** over 79
GitHub tool targets. Here it is 0.280 over 4,698 utilities. **The retrieval tier
as built does not work.**

### Why: 7,232 tldr pages bury the classic utilities

The failing queries say exactly what is wrong:

| query | top-5 utilities returned |
|---|---|
| "Removes all top-level *.pdf files in a current folder" | `bun`, `alr`, `wapm`, `pnpm` |
| "find files bigger than 100MB" | `oneliner`, `rmlint`, `blkdiscard`, `roll`, `tomb` |
| "kill the process listening on port 8080" | `ss`, `fkill`, `rc`, `uvicorn`, `opencode` |

BM25 finds a literal lexical match inside some obscure tool's example — `bun why
--top` matches "top-level" — rather than the common utility the request means.
The corpus covers the entire modern CLI ecosystem, and the ~400 classic Unix
tools are a rounding error inside it.

**This is `xr`'s documented cross-repo confusability, in a new domain.** That
tool's own guidance says account-wide search returns nothing relevant for a
query whose answer exists, because one large repo "contributes 8,088 chunks
thick with API-invocation vocabulary and buries the writeup", and scoping fixes
it. Same mechanism, same fix.

### Scoping to installed utilities nearly doubles recall@1

Restricting the corpus to utilities actually present on `$PATH` — 4,698 → 569
utilities, 31,169 → 6,921 chunks — on an **identical 400-query sample**, so only
the corpus differs and not the sample composition:

| k | full corpus | PATH-scoped | change |
|---|---|---|---|
| 1 | 0.090 | **0.170** | +89% |
| 5 | 0.270 | **0.417** | +54% |
| 10 | 0.378 | 0.490 | +30% |
| 20 | 0.440 | 0.575 | +31% |

A one-line filter, and it is the largest single intervention measured tonight.
It is also free in deployment: a helper knows what is installed.

The verification pass bounds how much more is available this way. **90.9% of the
31,169 chunks belong to utilities NL2Bash never mentions** — tldr is thick with
`git` subcommands, `aws`, `kubectl`, `npm`, `cargo`, while NL2Bash is 2016-era
POSIX shell. Oracle-scoping the corpus to just the 356 gold utilities (2,833
chunks) lifts recall@5 from 0.280 to **0.533** and @20 to 0.765. That oracle is
not deployable, but it brackets the intervention: PATH-scoping's measured 0.417
sits between the unscoped 0.280 and the oracle's 0.533, so **scoping is most of
the available headroom and it has largely been taken.**

**And it is still not enough.** 0.417 at k=5 misses the right utility for most
requests, and the gate established that the model degrades past **k≈3** — so the
tier has to deliver at k=3, where it is worse. The two measurements meet in the
wrong place.

### The tail figures are a floor, and one ablation is untestable here

Only **38.9%** of tail-bucket gold utilities are in the corpus at all, against
98.7% for the head, and on the stratified 369-pair tail run **193 of 246
utilities are never retrieved at any k**. Roughly half the tail loss is coverage
and half is ranking.

The ablation makes this sharper. At k=5 on the tail: tldr-only **0.198**,
both **0.203**, **man-only 0.008** — because the 60 man pages this container
ships are all `java` and `postgres` tooling, covering 0.8% of tail golds. So
`nl2sh-scoping`'s prediction that **man pages carry the tail is untestable here,
not refuted**, and re-running this on a machine with a real man corpus is the
single measurement most likely to change the verdict on the retrieval tier.

## Where this leaves the six-component architecture

| # | component | verdict tonight |
|---|---|---|
| 3 | regex parameter extraction | **works** — 0.971 precision holdout; 0.55 recall on unmarked values is the gap |
| 4a | corpus construction | **works** — 31,169 chunks, 4,698 utilities, 1.25 s to index |
| 1 | LLM-composed regex rules | **works, and does not improve with feedback** — 0.540 wild, flat over 3 rounds |
| 6 | error logging | **reframed** — an eval set, not a supervision signal |
| 2 | small model reading the sources | **fails zero-shot** (0 of 40) — **works fine-tuned**: 0.923 non-`find`, 25.5 min CPU |
| 4b | retrieval over that corpus | **the bottleneck** — 0.233 recall@5 on the deployment slice (0.417 PATH-scoped), needs to serve k=3 |
| 5 | ICL grounded in valid calls | **untested** — now unblocked |

The deterministic half holds up. Both middle tiers failed their first
measurement, and they failed *independently*: even a perfect retriever would not
help, because the gate handed the model the right source and it still produced
prose; and even a working model would be starved, because the shortlist misses
the utility three times in four.

That is a more useful outcome than one failure would have been, because the two
have different fixes and neither is speculative:

- **Retrieval** is a corpus-scoping and query-matching problem. Scoping to
  installed utilities is measured at +54% on recall@5 and is free, and the oracle
  bound says that is most of the headroom scoping can give. The remaining gap is
  a vocabulary mismatch between a full-sentence request and a 19-token chunk,
  which is what a dense arm or a query rewriter is for — neither tried. One arm
  that *was* tried points the same way: **utility-level documents** (one document
  per utility instead of per chunk) beat chunk-level on non-find, 0.369 vs 0.324
  at k=5.
- **The model** is an output-shape problem, and `monad-bsky` is the precedent:
  its 56M sibling went **0.000 → 0.481** on exactly this kind of shape
  installation. `finetune_gate.py` is staged with the training rows built (600
  train / 100 holdout, 93 distinct utilities, identical prompt format to the
  gate so a gain cannot come from a format change) and deliberately not run —
  it is hours of CPU and the recipe should be chosen deliberately.

**What would change my recommendation.** If the fine-tune moves the gate off
zero, the architecture is alive and the work is retrieval quality. If it does
not, the middle tier needs a 7B-class model, where the published NL2SH number is
61% against GPT-4o's 74% — and at that size the scaffold-prefill trick and the
k≈3 constraint both stop mattering, which is a different and much more
conventional system.

## Caveats

- **Nothing here is scored on the metric that matters.** Every number is utility
  selection or retrieval recall. The published NL2SH benchmark scores functional
  equivalence by executing both commands in Docker and judging outputs, and
  reports 74% for GPT-4o. Until that harness exists, none of these figures are
  comparable to the literature.
- **The container has 60 man pages.** Man coverage, the tail retrieval bucket,
  and the `man_option` chunk count are all floors, not estimates.
- **The gate is n=40** on one model at greedy decoding with a fixed source
  construction. It is decisive about zero-shot and says nothing else.
- **NL2Bash is 60% `find`** and its NL side is written by annotators, not users.
  It is a correctness corpus, not a usage distribution.

## 6. Fine-tuning answers the gate's open question — and invalidates its premise

The gate deliberately left one thing open: whether zero-shot failure was a
*shape* problem or a *capability* problem. `monad-bsky` is the precedent —
it took this model's 56M sibling from 0.000 to 0.481 by installing an output
shape. So: 600 rows, one epoch, batch 2, lr 1e-4 cosine, loss masked to the
completion span, **25.5 minutes on 4 CPU cores**. Training prompts are byte-identical
in format to the gate's, so no gain can come from a format change.

**Scored against the baseline that makes the number mean something.** 27 of the
40 gate rows have `find` as their gold, so answering `find` unconditionally
scores 0.675 — a trap the retrieval verification pass caught once already, and
one an earlier draft of this section walked straight into by reporting "8 of 9
correct" without noting that most of those golds were `find`.

| slice | base | **fine-tuned** | always-`find` |
|---|---|---|---|
| all (n=40) | 0.025 | **0.950** | 0.675 |
| **non-`find` (n=13)** | 0.000 | **0.923** | **0.000** |
| uncontaminated (n=38) | 0.026 | 0.947 | — |
| uncontaminated non-`find` (n=13) | 0.000 | 0.923 | 0.000 |
| commands emitted at all | 0.025 | **0.950** | — |

The non-`find` row is the claim: **0.923 where a constant scores 0.000.** It
routes `less`, `cat`, `mount`, `yum` and `set` correctly and misses only `kill`.
The zero-shot failure was **shape, not capability**, and 25 minutes of CPU fixed
it. Two of the 40 rows leaked into training through duplicate NL2Bash English
strings; excluding them changes nothing (0.947 / 0.923).

### The premise that justified this model did not survive

**Verbatim rate is 0.000 before fine-tuning and 0.000 after.** The whole reason
Pleias-RAG-350M was chosen over any other 350M model is that it is trained to
quote sources literally, and the architecture's bet was that converting
generation into extraction would rescue a tiny model — with `monad-bsky`'s 51%
identifier-copying failure as the thing being fixed. The fine-tuned model does
not quote the supplied command. **It generates one.**

So the result is real and the reasoning that produced it was wrong. What was
installed is "emit a shell command", not "copy the right span". That has a
direct consequence: **there is no evidence here that the RAG-specialised model
was the right base**, and the obvious ablation — fine-tune a non-RAG 350M model
of similar size on the identical rows — has not been run. If it matches, the
source-quoting property is decorative for this task and the model choice should
be made on other grounds.

### What this does and does not establish

It establishes that the middle tier can exist at 350M, on a laptop, at ~12 s per
query on four CPU cores unquantised.

It does not establish that the commands are *correct*. `utility_ok` compares the
leading utility, so `find . -type f -print0 | xargs -0 grep -i '.*\.*'` counts as
a success while being nonsense. Functional-equivalence scoring — Docker execution
plus an LLM judge, the metric behind the published 74% for GPT-4o — is the
missing harness, and until it exists **0.923 is a routing number wearing a
generation number's clothes**.

Other limits: n=13 on the slice that carries the claim; one epoch with training
loss still oscillating 1.4–2.1; training and evaluation drawn from the same
corpus and phrasing distribution, so this measures in-distribution shape
installation and says nothing about the wild-request case that `context_probe.py`
showed is much harder.

### Where the bottleneck moved

| component | before tonight | after |
|---|---|---|
| 2 — small model for intent | unknown | **works after 25 min of fine-tuning** |
| 4b — retrieval | untested | **binding constraint at 0.233 recall@5** |

The architecture is alive and the work is now retrieval quality, which is the
better problem to have: it is deterministic, measurable in milliseconds, and has
measured headroom nobody has spent yet — PATH-scoping (+54% at k=5, done),
utility-level documents (0.369 vs 0.324 on non-find, tried once), an oracle
bound at 0.533, and a dense arm and query rewriter neither of which was built.

## 7. The RAG base was worth ~1 example out of 13, and the reason it was chosen was not used at all

The premise that selected Pleias-RAG-350M — literal source-quoting — showed a
verbatim rate of 0.000. So: does the RAG mid-training matter, or would any
350M model fine-tune to the same place? The ablation fine-tunes **the RAG-less
sibling `Pleias-350m-Preview`** (same lab, same llama architecture L26×h1024,
same base data, same tokenizer vocabulary) on the byte-identical 600 rows and
scores it on the same gate.

| arm | utility (all 40) | utility (non-`find`, n=13) | verbatim |
|---|---|---|---|
| base (RAG), zero-shot | 0.025 | 0.000 | 0.000 |
| fine-tuned (RAG) | 0.950 | **0.923** (12/13) | 0.000 |
| **fine-tuned (non-RAG)** | 0.900 | **0.846** (11/13) | 0.000 |
| always-`find` prior | 0.675 | 0.000 | — |

**The RAG base is better by one example out of thirteen.** 12/13 against 11/13
on the slice that carries the claim — a difference of a single query, well
inside noise at n=13. On all 40 it is 0.950 vs 0.900, two queries. Both
fine-tune from a useless zero-shot base to a working router; the RAG head buys a
margin that this eval cannot distinguish from zero.

**And the property it was chosen for is used by neither.** Verbatim rate is
0.000 for both fine-tuned models. Whatever the ~8-point non-`find` gap is, it is
not source-quoting — both generate the command rather than copying it.

**One confound makes even that small gap unattributable.** `Pleias-350m-Preview`
lacks the RAG special tokens, so a `<|source_start|>…<|source_end|>` block costs
it **18 tokens against the RAG model's 5**. It trained in **45.8 minutes against
25.5** for that reason, and its longer per-row encoding is a plausible source of
the one-example difference on its own. So the ablation varies two things — RAG
mid-training *and* native source delimiters — and cannot separate them. A clean
single-variable run would give both models a plain-text source format.

**The practical conclusion holds regardless of which variable it is:** you do
not need the RAG-specialised model. A plain 350M sibling fine-tunes to within
one example of it on this gate, and the source-quoting capability that motivated
the choice does not appear in either model's output. The model selection was not
the important decision; the fine-tuning was.
