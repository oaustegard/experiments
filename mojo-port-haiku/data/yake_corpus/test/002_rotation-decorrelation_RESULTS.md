# Resolving the rotation pendulum: protocol × k × embedder, + a decorrelation probe

**Question (from the user).** My recall-per-byte run said learned ITQ rotations
*beat* remax's random SimHash; remax#46/PR#47 said they *lose*. "How can I trust
either end? Is it embedder-specific — SPECTER2 is a specialized embedder, Jina is
general?" And: test the decorrelation angle #46 left open.

**Method.** remax/bench's own metric — self-retrieval recall@10 of the float32
top-10 (`exact_knn` ground truth). Pure numpy on precomputed caches, no embedding.
Two embedders, same scientific domain: **SPECTER2** (specialized, 10k) and
**Jina-v5-nano** (general, SciFact 5183). k-stack ladder k∈{1,2,4,8} (codes
concatenated, Hamming). Three methods: **simhash** (k independent random
rotations), **itq** (k independent ITQ rotations), **decorr** (k =
QR(½·R_itq_shared + ½·G_i): shared alignment + per-stack random diversity).
Protocols: **in-corpus** (fit rotation on eval split) vs **transfer** (fit on a
disjoint split). 3 seeds averaged.

## Results (recall@10)

**SPECTER2 (specialized)**

| method | protocol | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| simhash | in-corpus | 0.646 | 0.679 | 0.702 | **0.715** |
| simhash | transfer  | 0.646 | 0.678 | 0.701 | 0.714 |
| itq | in-corpus | 0.661 | 0.683 | 0.691 | 0.701 |
| itq | transfer  | 0.631 | 0.656 | 0.676 | 0.685 |
| decorr | in-corpus | 0.642 | 0.680 | 0.705 | 0.714 |
| decorr | transfer  | 0.642 | 0.681 | 0.702 | 0.713 |

**Jina-v5 (general)**

| method | protocol | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| simhash | in-corpus | 0.730 | 0.785 | 0.822 | **0.847** |
| simhash | transfer  | 0.729 | 0.784 | 0.823 | 0.847 |
| itq | in-corpus | **0.778** | 0.815 | 0.838 | 0.852 |
| itq | transfer  | 0.734 | 0.785 | 0.811 | 0.832 |
| decorr | in-corpus | 0.730 | 0.783 | 0.824 | 0.847 |
| decorr | transfer  | 0.725 | 0.780 | 0.823 | 0.848 |

## The pendulum was never a contradiction — it's three interacting axes

Both ends were right about different corners of the same cube:

1. **k (the ladder).** ITQ's edge is largest at k=1 and decays with k — its
   correlated stacks defeat the ladder's 1/k variance reduction (#46's mechanism,
   reproduced). SPECTER2: itq goes from *winning* k=1 (0.661>0.646) to *losing*
   k=8 (0.701<0.715).
2. **Protocol (overfit).** ITQ in-corpus > ITQ transfer at every cell; gap
   ~0.02–0.03 (SPECTER2), ~0.04–0.05 (Jina). simhash & decorr have **zero** gap
   (in-corpus == transfer) — random rotations can't overfit. This is #46's
   transfer control, now measured on both embedders.
3. **Embedder generality.** ITQ's in-corpus edge is **3× larger on the general
   embedder** (Jina +0.048 at k=1 vs SPECTER2 +0.015). So it *is* partly
   embedder-specific: the general embedder has more learnable rotation structure.
   **But that extra edge is mostly overfit** — transfer erases most of it.

**My NFCorpus "win" was the perfect storm: general × in-corpus × k=1** — the exact
intersection where ITQ looks best. #46 used specialized × transfer × ladder —
where it looks worst. Neither measurement was wrong; they sampled opposite corners.

## The trustworthy verdict (transfer + full ladder = the honest config)

On the shipped stacked ladder, fit honestly (transfer), **ITQ does not beat random
SimHash on either embedder**:

| k=8, transfer | simhash | itq | Δ |
|---|---|---|---|
| SPECTER2 | 0.714 | 0.685 | **−0.029** |
| Jina (general) | 0.847 | 0.832 | **−0.015** |

#46's "keep parameter-free SimHash" decision holds for general embedders too.

## The decorrelation lead: tested, does not pay

The α=0.5 decorrelation (shared ITQ direction + per-stack random diversity) **ties
SimHash** on both embedders, both protocols, at every k (e.g. Jina k=8: 0.848 vs
0.847), and carries no overfit gap. So mixing alignment into random rotations
neither helps nor hurts — **SimHash's independence is already the asset, and it's
near the ceiling.** The open lead, in this form, returns nothing.

Not fully closed: a *more aggressive* decorrelation (explicit cross-stack
orthogonality penalty, or residual/sequential rotations that each capture a
different subspace) is untested — but the α-mix probe suggests thin headroom,
because the thing you'd add (alignment) is exactly what re-introduces correlation
across stacks. Likely a dead end; would need a decorrelation that adds alignment
*without* sharing direction across stacks to have any chance.

## Bottom line

- **Trust the transfer + ladder protocol.** In-corpus and k=1 numbers are real but
  don't generalize; they flatter any learned rotation.
- **ITQ is genuinely embedder-sensitive (bigger effect on general embedders) but
  the effect is mostly overfit;** the honest config gives the same verdict
  (SimHash wins) on both specialized and general.
- **Decorrelation (α-mix) is a wash.** remax should keep parameter-free SimHash.

## Reproduce

```bash
# caches auto-fetched from oaustegard/claude-container-layers releases:
#   specter2-nlp-broad-10k / jina-v5-nano-scifact  (see sweep.py header)
python3 sweep.py
```
