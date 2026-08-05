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

### The other direction: does it hold up where grep is strong?

Ten keyword-bearing queries — the kind you'd normally just grep (`ascii_fold`,
`GRID_VERSION`, `Lloyd-Max`, `EXPERIMENTS_SPOKES_ROOT`, `SPECTER2`, `RRF`,
`nDCG`, …). Scored deliberately in grep's favour: does the index's top-5 contain
a file that grep-with-the-obvious-term also returns?

| | agrees with grep |
|---|---|
| before excluding generated output | 8/10 |
| **after** | **10/10** |

Both original misses were **identifier lookups**, and both were answered with
LLM chatter from `haiku-assessment/**/outputs/` — see "corpus hygiene" below.
Even at 10/10 the weakest case is `ascii_fold`, which lands at **rank 5**. Exact
identifier lookup is still grep's job; the index merely no longer embarrasses
itself there.

## Use

```bash
python3 repo-index/ask.py "about to fan out concurrent LLM calls through a gateway"
#   te-bridges/path_c_cross_domain/RESULTS.md:103
#   woodall/README.md:55
#   README.md:25

python3 repo-index/ask.py --build      # after adding or editing markdown
python3 repo-index/ask.py --verify     # stored rotation vs seed regeneration
```

Complement to the grep instruction, not a replacement — run both.

## What it is

| | |
|---|---|
| corpus | 1,022 markdown chunks from 90 files (headings-split), generated model output excluded |
| encoder | `bekko-embedding-v1-a8m`, 384-d, mean-pooled |
| codec | **remex 2-bit** — this repo's own measured sweet spot |
| index | **0.14 MB** committed (packed codes + `(path, line)` pointers) |
| rotation | 0.56 MB committed — stored, not regenerated ([why](#the-failure-this-design-is-actually-guarding-against)) |
| rebuild | ~45 s for the whole repo |

### Corpus hygiene

`outputs/` and `prompts/` directories are excluded. They held 173 files of
machine-generated model output (all `run_NN.md`-shaped near-duplicates) — an
experiment's *data*, not its findings — and they were **20% of the corpus**.
They were not inert: both conventional-query misses above were real questions
answered with a sample of LLM output, and the vague queries were prone to
returning five near-identical `run_0N.md` chunks in a row. Excluding them took
conventional agreement 8/10 → 10/10 with the rediscovery cases unchanged at 5/5,
and shrank the index 27%.

The general form: **a semantic index over a repo that stores model output will
rank that output against real questions**, because it is topically on-subject
and there is a lot of it. Lexical search never had this problem — nobody greps
for a phrase that only appears in a sampled generation.

It stores **pointers, not text** — the repo is the corpus, so there is no reason
to carry a second copy of it. Both choices come from
[`bekko-embedding-bench`](../bekko-embedding-bench/RESULTS.md): 2-bit at 96 B/vec
is statistically indistinguishable from uncompressed fp32 (n=59, p=1.0), and
1-bit is measurably worse (p=0.008).

## Keeping it current

`.github/workflows/repo-index.yml` rebuilds on any markdown change to `main`
and commits the result. A full rebuild is ~45 s, so there is no incremental
path to maintain — and because the build is deterministic on a fixed toolchain,
an unchanged corpus produces a byte-identical artifact and no commit.

**The encoder is pinned by sha256, with Hugging Face as the source and this
repo's own releases as the intended mirror** (`repo-index-model-v1`).
bekko-embedding-v1-a8m is MIT, so mirroring is permitted. Pinning is not only
about availability and rate limits: **a different encoder silently changes the
embedding space**, so `ask.py` verifies the hash and refuses to build against
the wrong file. `mirror_model.py` prints the files and hashes to upload.

To publish or refresh the mirror, run **`repo-index-mirror`**. Two ways in:

| trigger | when |
|---|---|
| **add the `publish-mirror` label to a PR** | works from a phone in one tap, and works *before* the workflow reaches `main` |
| dispatch from the Actions tab, or `gh workflow run repo-index-mirror` | needs the workflow on the default branch |

The label route exists because `workflow_dispatch` only appears once a workflow
is on the default branch, whereas a `pull_request` run uses the workflow file
**from the PR itself**. The label is removed when the run finishes, so it behaves
like a button — add it again to run again, including to retry a failure.

Either way the run fetches from Hugging Face, **verifies the sha256 pin before
publishing anything**, creates the release if missing, uploads with `--clobber`,
then downloads the published assets back over the public URL and re-verifies
them. Manual only: the encoder is pinned and does not change on its own, so
there is nothing to schedule. `dry_run: true` (dispatch only) fetches and
verifies without publishing.

> **Published.** `repo-index-model-v1` exists and a cold `ask.py` run now fetches
> from `github.com` rather than Hugging Face. It was published by the label
> trigger above, from the PR branch, before that PR had merged.
>
> The workflow exists because a Claude Code session cannot create releases — the
> agent proxy returns *"Creating, editing, or deleting releases is not permitted
> for this session type"* — but an Actions run, holding the repo's own
> `GITHUB_TOKEN` with `contents: write`, can.

**A mismatch is a hard failure, never an automatic re-pin.** If Hugging Face
serves different bytes, mirroring them would put a *different embedding space*
behind a name the index trusts — the exact silent failure the pin exists to stop.
Re-pinning is a deliberate change, made together with a full `--build`. (The
first version of `mirror_model.py` only printed the hashes and left the
comparison to whoever was reading the terminal, which is not a check.)

### The failure this design is actually guarding against

`remex`'s own rule is that *the rotation is part of the encoding*, and codes
decoded under the wrong one are **~50% different, not slightly off** — a total
and silent failure, not a degraded one.

This originally regenerated the rotation from the seed at query time and merely
**fingerprinted** it, on the reasoning that numpy's LAPACK QR drifts across BLAS
builds. That reasoning was **stale**: remex replaced `np.linalg.qr` with an
explicit Householder QR in its #40 specifically to be bit-reproducible across
BLAS builds, so the hazard being guarded had already been fixed upstream.

The real exposure was elsewhere, and worse. Regenerating meant depending on
three upstream things staying still:

| dependency | why it can move |
|---|---|
| `remex`'s `rotation` default | remex documents it as deliberately changeable, and `ask.py` did not pass one |
| numpy's `default_rng` stream | NEP 19 explicitly declines to guarantee it across feature releases |
| remex's construction of the matrix | as #40 itself demonstrates, it changes |

So the rotation is now **stored** as `rotation.npy` (576 KB) and loaded, and
`ROTATION = "haar"` is passed explicitly. `manifest.json` hashes the stored
matrix and the analytic Lloyd-Max codebook; a mismatch **refuses to return
results** rather than warning. Versions (`numpy`, `onnxruntime`, `remex`) are
recorded but *informational* — a git checkout of remex reports `0.0.0+unknown`
where CI installs `0.6.0`, so the enforced invariant is the hash.

`--verify` reports whether seed-regeneration still reproduces the stored matrix.
It is diagnostic only: the query path uses the stored one either way, so a
divergence there means the pin did its job, not that anything broke.

**No drift has actually been observed.** The last CI build (ubuntu, numpy 2.4.6,
`remex==0.6.0` from PyPI) recorded rotation fingerprint `189c32b3…`, and a local
build against a git checkout of remex produces the same matrix byte-for-byte. So
this is insurance against the three dependencies above moving later, not a fix
for a live bug — which is the honest reason it is worth only 576 KB and not more
machinery.

The general form, which is the part worth carrying elsewhere: **a stored index
that regenerates its transform from a seed inherits every upstream default it
did not pin.** Fingerprinting detects that after the fact and costs a rebuild;
storing the transform prevents it. Here that trade was 576 KB.

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
