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
