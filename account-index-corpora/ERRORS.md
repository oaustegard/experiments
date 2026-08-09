# Errors

What was wrong, how it was caught, and which direction it pushed the conclusion.
The base rate matters more than any single entry.

## 1. The tombstone corpus bypassed every exclusion the live index applies

**Wrong.** `tombstone_chunks` filtered deleted paths by extension and nothing
else. `hcindex.discover` also applies `skip_dirs`, `skip_names`, the repo's own
`exclude` list, and a 1 MiB size cap — but those live on `Path.stat()` and
`rglob`, neither of which a deleted file has.

**Caught by** the first measurement, immediately: 74,822 tombstone chunks
against a 13,257-chunk working tree, 1.7× the entire live account index from
three repos. A number that absurd is its own alarm. Reading the largest
contributors found `phase-a-bridges/data/full_body_embeddings.json` at 767,692
lines — a file the live index had never carried, because it is over the cap.

**Direction.** Inflated the apparent cost of tombstones by 47×. Fixing it moved
tombstones from *obviously unaffordable* to *affordable but not worth it*, so
the conclusion survived the correction while its reason changed completely.
Reported the wrong-but-loud number is why this is worth writing down: had the
bug been subtler — say, only the size cap missing — the corpus would have looked
merely large rather than broken, and the recommendation would have been right
for the wrong reason.

**Generalized** into `METHODS.md`: any corpus assembled from git history, an
API, or a database rather than from the file walker must re-implement the file
walker's exclusions.

## 2. `corpora` first computed each corpus by slicing off a suffix

**Wrong.** The measurement built tree-only, then tree+tombstones, and took
`[len(tree):]` as the tombstone chunks. `corpus_for` interleaves per repo — repo
A's tree, repo A's tombstones, repo B's tree — so the extra chunks are not a
suffix.

**Caught by** review before it ran, not by the data. It would have produced a
plausible wrong number rather than an error, since the slice has the right
length and the wrong contents.

**Direction.** None; fixed before any reported figure came from it. The
measurement now builds once with both corpora on and partitions by label, which
is also cheaper.

## Not errors, but limits on the result

- **3 of 65 repos.** Only the repos this session could clone. It includes the
  account's largest (`experiments`, 11,109 chunks of ~42,500), so the percentages
  are not drawn from a toy — but they are not the account either.
- **Size only.** Nothing here measures whether the extra corpora improve
  answers at account scale. That needs a full encode and a benchmark that does
  not exist. The per-repo wins are assumed, not re-verified.
- **Clone timings come from a proxied network.** Absolute seconds are inflated
  and only the comparison between depths is meaningful.
