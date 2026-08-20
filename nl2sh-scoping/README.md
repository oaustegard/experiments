# nl2sh-scoping — what shape is the terminal-helper problem?

Scoping measurement for a natural-language-to-shell helper, run before building
anything, to choose between two architectures that want opposite things from the
data:

- **Cascade** — deterministic rules answer, abstain, model handles the rest.
  Needs a **thick head**: a small set of intents covering most requests.
- **Retrieval** — a lexical shortlister narrows candidate utilities and extracts
  parameters, then documentation goes to the model. Needs a **long tail the
  model does not already know**.

The intended input was a real shell history. That was unavailable, and the
substitute turns out to be the better corpus: **a helper is asked about what you
would have to look up, not what you type most.** `ls`, `cd` and `git status`
dominate any history and never reach a helper. NL2Bash was scraped from forums,
tutorials and Stack Exchange, which is approximately the "what do people need
help with" distribution.

## Caveat that governs every number

**60.3% of NL2Bash leads with `find`** — an artifact of how it was collected.
That makes the corpus fine for *correctness* evaluation and useless for
*frequency weighting*. Quote the non-find column.

Parsing is crude on purpose: quoted spans masked, then split on `| || && ;`.
Stage counts are approximate; flag counts are distinct `-x`/`--xyz` tokens
outside quotes.

## Results (n=12,607; non-find n=5,055)

| | all | non-find |
|---|---|---|
| distinct leading utilities | 378 | 377 |
| utilities appearing exactly once | 177 | 176 |
| coverage, top 10 utilities | 71.2% | **29.1%** |
| coverage, top 50 | 88.1% | **70.5%** |
| coverage, top 100 | 95.8% | 89.5% |
| commands with ≤1 flag | 44.4% | **72.6%** |
| single-stage (no pipe) | 63.9% | 57.3% |

## What it says

**The head is thin and the tail is long.** Ten utilities cover 29% of non-find
requests; you need about fifty for 70%, and 176 utilities appear exactly once.
That is a weak case for a rule tier trying to *answer* most requests, and a
strong case for one that *narrows* candidates — which is the role
[`gh-mcp-regex-fit`](../gh-mcp-regex-fit/RESULTS.md) already measured BM25 into:
0.622 top-1 but **0.851 recall@5 / 0.919 recall@10** at 0.03 ms.

**The difficulty is utility selection, not flag composition.** 72.6% of non-find
commands carry at most one flag. So the model does not mainly need to be taught
a utility's option grammar; it needs to be pointed at the right utility out of
~377, most of which it has seen and a long tail of which it has not.

**Which argued, in the first draft of this file, for `tldr` pages over man
pages. That was wrong, and the error is worth naming**: it conflated *RAG over
man pages* with *putting a man page in the context window*. Retrieval chunks.
`doc_corpus.py` measures both corpora properly and the answer is both, tiered.

| | tldr | man |
|---|---|---|
| coverage, top 50 utilities | **96.0%** | whatever is installed |
| coverage, top 200 | 85.0% | " |
| coverage, used-once tail (n=150) | **50.0%** | " |
| whole document, median tokens | ~120 | **2,653** (p90 14,226, max 47,097) |
| natural chunk | one worked example | one `.TP` option entry |
| chunk size, median tokens | ~25 | **56** (p90 271) |

Stuffing a man page is indeed hopeless — a 47,000-token maximum against a
4,096-token context. Chunking one is a non-problem: roff supplies free,
self-delimiting boundaries, `.SH` for sections and `.TP` for one option plus its
description, and **93.4% of `.TP` chunks fit in 350 tokens**, so ten retrieved
option entries sit comfortably in a small model's context. The chunk is also
keyed by exactly the token you retrieve on — the flag.

The coverage numbers are what decide it. **tldr covers 96% of the fifty
most-used utilities and only 50% of the used-once tail** — and the tail is where
the requests are (top 10 utilities are 29.1% of non-find) and where a model most
needs help, since it knows `find` and `grep` cold already. So man pages are the
coverage backbone, not the fallback.

The retrieval corpus is therefore heterogeneous, and the three chunk types are
not equally easy to use:

1. **tldr examples** — runnable, quotable, head coverage.
2. **man EXAMPLES sections** — also runnable and quotable; present in 36 of the
   60 pages sampled here.
3. **man `.TP` option entries** — universal coverage, but a flag plus a
   description is not a command. Using one requires *composition*, not
   quotation.

That ordering matters for model size. A span-copying model answers from (1) and
(2) natively — which is the operation `monad-bsky` measured its 56M sibling
failing at, 51% identifier copying. (3) is where composition starts and where a
350M model would be expected to break, so it is the natural escalation boundary.

One further thing man pages have and tldr does not: **SYNOPSIS is a grammar.**
`needle-bsky` got its best numbers from decoding constrained by a grammar
compiled from tool schemas; a SYNOPSIS line is the same object for a shell
utility, and nothing here has tried compiling it.

**The failure mode this cannot see** is which of the two layers is actually
wrong on a given request. That needs the eval, not the corpus statistics:
NL2Bash's 9,305 curated pairs and the 600-pair verified set with a
functional-equivalence metric from *LLM-Supported Natural Language to Bash
Translation* (arXiv:2502.06858), which reports **74% for GPT-4o** as the number
to beat.

## Reproduce

```bash
git clone --depth 1 https://github.com/TellinaTool/nl2bash.git
git clone --depth 1 --filter=blob:none --sparse https://github.com/tldr-pages/tldr.git
cd tldr && git sparse-checkout set pages && cd ..

python3 utility_distribution.py --data nl2bash/data/bash/all.cm
python3 doc_corpus.py --nl2bash nl2bash/data/bash/all.cm --tldr tldr/pages
```

`man -w` is not a usable coverage probe on a minimised container — Debian's stub
exits 0 for any argument including a nonsense one, which produced a spurious
100% before it was caught.

Writes `results.json`. The clone is not vendored — it is a public corpus with
its own licence.
