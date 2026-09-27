# Errors — mxbai-edge-remex-quant

What was wrong, how it was caught, and which way it pushed the conclusion.

## 1. remex `rotation="rht"` ignores the seed at power-of-two d (a remex bug, not this experiment's)

**What.** The first bench design took its seed floor from rht at seeds 0 and 1. On the
synthetic smoke fixture, both seeds gave bit-identical nDCG at 2 and 1 bit. The cause
is in remex 0.8.0: `rht_plan` uses one round when the block size equals d. In that
round, the seed's permutation and sign flips sit after the Walsh–Hadamard transform
on the encode path. The rotated vector is then a seed-dependent signed permutation of
a fixed WHT of x. A symmetric per-coordinate codebook is equivariant to signed
permutations, so decodes are identical across seeds.

It holds at every power-of-two d checked (64, 128, 256, 1024), not at 384 or 768,
which use two or more rounds. A Walsh row lands on a single rotated coordinate
(max |y| = 1.0) for every seed. remax's stacks are unaffected (asym score correlation
0.64 between stacks, the same as haar).

**Caught by.** The smoke run's seed-1 rows matching seed 0 to four decimals, then a
direct decode comparison.

**Effect here.** The seed floor was moved to haar seeds 0 and 1 before the real run.
The rht rows stay in the tables. For the uncentered arms they fall inside the two
haar seeds' range. For the centered arms they land within 0.011 of the nearer haar
seed, the same size as the haar seeds' own spread. The missing randomisation did not
measurably cost retrieval on these tokens.

**Effect elsewhere.** Any seed-averaged rht result at a power-of-two d measured one
rotation several times. Filed against remex; memory `94957b5c`.

**Update, remex 1.0.0.** remex fixed this (#89) by flooring rht at two rounds,
which changes rht codes at d=64. Re-measured on SciFact under 1.0 (default
rotation, seed 0), plain → centered nDCG@10:
- 1-bit: 0.527 → 0.696
- 2-bit: 0.680 → 0.732
- 4-bit: 0.730 → 0.743

The pre-fix rht plain 1-bit row in RESULTS.md (0.564) sits inside the haar
seed range (0.506–0.613). No haar arm was run at 4 bits, so the 4-bit shift
(0.740 pre-fix, 0.730 under 1.0) has no seed floor to compare against. Finding 2 stands with a larger margin: +0.17
at 1 bit on SciFact under 1.0, against +0.12 before.

## 2. remax on centered tokens was an invalid construction

**What.** `extra.py` added remax k=1/2 on `T − mean`, scoring the float query against
the sign codes of the residual. I argued that q·mean is a per-query-token constant that
MaxSim passes through, and that argument holds. The construction still drops the
residual's magnitude, which after centering varies per token (mean 0.32) and is most
of the signal. Retrieval collapsed to nDCG@10 0.006 (SciFact) and 0.019 (NFCorpus).

**Caught by.** The result itself: worse than random-looking, on both corpora.

**Effect.** None on the conclusions. The rows are reported in the tables as-is and
excluded from the findings. A valid centered 1-bit code needs a per-token magnitude,
and remex's centered 1-bit is that code: it stores m in the norm column.

## 3. Top-k selection guard dropped a hit before sorting (fixed before any real run)

**What.** A guard added so the smoke fixture (60 docs) could run took
`argpartition(-s, 100)[:101][:100]`. That slices an unsorted partition, which can
drop the best document before the sort.

**Caught by.** Re-reading the diff before the real run; `topk()` replaced it and was
checked against a full `argsort` on 500, 60 and 101 items.

**Effect.** None; no real-corpus number was produced with the faulty guard.
