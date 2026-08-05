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

## Keeping it current

`.github/workflows/repo-index.yml` rebuilds on any markdown change to `main`
and commits the result. A full rebuild is ~33 s, so there is no incremental
path to maintain — and because the build is deterministic on a fixed toolchain,
an unchanged corpus produces a byte-identical artifact and no commit.

**The encoder is pinned by sha256, with Hugging Face as the source and this
repo's own releases as the intended mirror** (`repo-index-model-v1`).

> **The mirror release is not published yet.** Until it is, `ask.py` logs one
> expected `HTTPError` per file and falls back to Hugging Face — which works,
> and the sha256 check still runs, so integrity is enforced either way. What is
> missing is only the rate-limit and availability insurance. To publish it:
>
> ```bash
> gh release create repo-index-model-v1 --title "repo-index encoder (pinned)" \
>   --notes "Mirror of hotchpotch/bekko-embedding-v1-a8m (MIT). sha256 96d8cc61…"
> python3 repo-index/mirror_model.py | cut -d' ' -f1 \
>   | xargs gh release upload repo-index-model-v1
> ```
>
> (Claude Code sessions cannot create releases — the agent proxy returns
> *"Creating, editing, or deleting releases is not permitted for this session
> type"* — so this step is a human one.) bekko-embedding-v1-a8m
is MIT, so mirroring is permitted. Pinning is not only about availability and
rate limits: **a different encoder silently changes the embedding space**, so
`ask.py` verifies the hash and refuses to build against the wrong file.
`mirror_model.py` prints the files and hashes to upload.

### The failure this design is actually guarding against

`remex`'s own rule is that *the rotation is part of the encoding*. Here it is
**regenerated from the seed at query time** rather than stored — and numpy's
LAPACK QR can drift across BLAS builds. A CI-built index queried on a different
machine could therefore land in a **different space with no error at all**, just
quietly worse results.

So `manifest.json` records a **fingerprint of the actual rotation matrix**, and a
query that recomputes a different one prints a loud warning. Versions
(`numpy`, `onnxruntime`, `remex`) are recorded too but are *informational* — a
git checkout of remex reports `0.0.0+unknown` where CI installs `0.6.0`, so the
enforced invariant is the fingerprint, not the version string. The workflow pins
`numpy`, `onnxruntime`, `tokenizers` and `remex` for the same reason: an
unpinned bump would rewrite every code and churn the diff.

## Requirements

`onnxruntime`, `tokenizers`, `numpy`, `remex`. The encoder (~124 MB) lands in
`~/.cache/repo-index`, or set `$BEKKO_HOME`.

## Caveat on the evaluation

n=5, and the queries were written by someone who knew what was in the repo. The
*direction* is credible because the mechanism is not subtle — grep needs a
keyword and these queries deliberately withhold it — but treat 5/5 as a
demonstration, not a measurement. The same experiment found dense retrieval only
**ties** grep on scikit-learn file discovery, where the query *did* contain the
identifiers.
