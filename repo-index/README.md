# repo-index — semantic lookup for the "I never looked" failure

`CLAUDE.md` says to grep `METHODS.md` before starting an experiment. That works
**if you already know the term**. The repo's documented rediscoveries suggest
the problem is usually that you don't.

Measured on this repo's own three documented rediscovery failures plus two
findings from `bekko-embedding-bench`, over 5 realistic "what I'm about to do"
queries:

| retrieval | hit@5 |
|---|---|
| grep, using words from the query itself | **1/5** |
| **this index** | **5/5** |
| grep, handed the ideal keyword by hand | 5/5 |

The last row is the honest framing: **grep isn't beaten, it's unblocked.** With
`concurrency`, `ITQ`, `codebook`, `matryoshka`, `power analysis` in hand, grep
finds every prior. This index removes the need to already know the word.

## Use

```bash
python3 repo-index/ask.py "about to fan out concurrent LLM calls through a gateway"
#   te-bridges/RESULTS.md:103
#   te-bridges/path_c_cross_domain/RESULTS.md:71
#   METHODS.md:246

python3 repo-index/ask.py --build      # after adding or editing markdown
```

Complement to the grep instruction, not a replacement — run both.

## What it is

| | |
|---|---|
| corpus | 1,266 markdown chunks from 262 files (headings-split) |
| encoder | `bekko-embedding-v1-a8m`, 384-d, mean-pooled |
| codec | **remex 2-bit** — this repo's own measured sweet spot |
| index | **0.18 MB** committed (packed codes + `(path, line)` pointers) |
| rebuild | ~33 s for the whole repo |

It stores **pointers, not text** — the repo is the corpus, so there is no reason
to carry a second copy of it. Both choices come from
[`bekko-embedding-bench`](../bekko-embedding-bench/RESULTS.md): 2-bit at 96 B/vec
is statistically indistinguishable from uncompressed fp32 (n=59, p=1.0), and
1-bit is measurably worse (p=0.008).

The encoder (~124 MB) is fetched to `~/.cache/repo-index` on first use, or set
`$BEKKO_HOME`. Requires `onnxruntime`, `tokenizers`, `numpy`, and `remex` on
`PYTHONPATH`.

## Caveat on the evaluation

n=5, and the queries were written by someone who knew what was in the repo. The
*direction* is credible because the mechanism is not subtle — grep needs a
keyword and these queries deliberately withhold it — but treat 5/5 as a
demonstration, not a measurement. The same experiment found dense retrieval only
**ties** grep on scikit-learn file discovery, where the query *did* contain the
identifiers.
