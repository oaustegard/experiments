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

**Which argues against feeding whole man pages.** If the answer needs one flag,
a full `man find` is enormous overkill, and `monad-bsky` already priced this
class of mistake: the same 18 tool schemas cost 1,972 tokens as JSON against 574
as prose one-liners, a 3.4x difference for identical information. Man pages are
far worse than that ratio. **`tldr` pages match the measured usage profile** —
a handful of worked examples per utility, ~10-20x smaller — with man pages as
escalation for the rare multi-flag case.

**The failure mode this cannot see** is which of the two layers is actually
wrong on a given request. That needs the eval, not the corpus statistics:
NL2Bash's 9,305 curated pairs and the 600-pair verified set with a
functional-equivalence metric from *LLM-Supported Natural Language to Bash
Translation* (arXiv:2502.06858), which reports **74% for GPT-4o** as the number
to beat.

## Reproduce

```bash
git clone --depth 1 https://github.com/TellinaTool/nl2bash.git
python3 utility_distribution.py --data nl2bash/data/bash/all.cm
```

Writes `results.json`. The clone is not vendored — it is a public corpus with
its own licence.
