# nl2sh-retrieval — building the deterministic tier, and gating the small model

**Started / finished:** 2026-08-19 · **Status:** the gate is the result.

[`nl2sh-scoping`](../nl2sh-scoping/README.md) established the shape of the
problem — utility selection over a ~377-utility long tail, not flag composition
— and argued for a heterogeneous chunked corpus with a small model reading it.
This builds that corpus, the parameter extractor, and the retrieval tier, and
then runs the one test that decides whether the small-model component exists at
all.

Four things were built and measured. Three work. The fourth is the interesting
one.

## 1. The corpus — 31,169 chunks over 4,698 utilities

`build_corpus.py` turns tldr pages and roff man pages into three chunk kinds:

| kind | n | runnable | role |
|---|---|---|---|
| `tldr_example` | 29,437 | yes | quotable; covers 96% of the top-50 utilities |
| `man_option` | 1,601 | no | universal coverage, but needs *composition* |
| `man_example` | 131 | yes | quotable, rare |

4,688 of 4,698 utilities have at least one runnable chunk. Chunks are **median
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

## 2. Parameter extraction — precision 0.97 on held-out

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
| recall, values NL2Bash *marks* | 0.984 | 0.954 |
| recall, values it does **not** mark | **0.453** | **0.545** |

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

## 4. Does a frontier compiler degrade when fed more failure cases? No — it stops.

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

**He was right, and the mechanism is better than "doesn't degrade": it stops.**
`rules_claude-iter2.json` and `rules_claude-iter3.json` are **byte-identical**
(sha `01ffa759965f`). Shown a fresh batch of failures at round 3, the model
changed nothing. Gemini rewrote the entire ordered list every round, which is
why a fix for one error kept shadowing a rule that was already right, and why it
lost 0.123 *in sample* at round 3. That regression is a property of that model's
revision behaviour, not a law about error-driven iteration. My third-pass
writeup stated it as the latter; that was an overreach.

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
