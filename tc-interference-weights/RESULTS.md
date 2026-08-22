# tc-interference-weights

Review of Turner, Wu & Batson, *"Characterizing interference weights in a tiny language
model"*, Transformer Circuits Thread, 21 Aug 2026
([link](https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html)).

Every number below is re-derived from the paper's own figure data files
(`shared/data/figures/*.js`) by `check_claims.py`. Run `mirror.sh` first to fetch them.

```
./mirror.sh mirror            # ~107 MB: page, 21 figures + assets, gallery, feature_vis
python3 check_claims.py mirror
```

## The paper

A one-layer transformer — 2.9M parameters, 4,096-token vocabulary, no normalization and no
bias terms, trained on English and code from Common Corpus — plus a JumpReLU transcoder on
the MLP. The two are re-expressed as 331M explicit virtual weights across six families
(QK, Tokens→Features, Tokens→OV→Features, Tokens→Logits, Features→Logits,
Tokens→OV→Logits), which reproduce the forward pass exactly up to a transcoder error term.

Each weight gets two scores. **Fisher effectiveness** is a second-order estimate of the KL
between the model's output with and without the weight, `½E[aᵀFa]`; it is cheap enough to
compute for all 331M weights over 537M tokens. **Helpfulness** is the mean change in loss
when the weight is ablated, computed for a random sample of 7,765 weights over 1B tokens.

The central demonstration: the largest raw virtual weight from the token `" IN "` to the
logits points at `" utions "`, a token that never once follows `" IN "` in the training
set. Its Fisher effectiveness sits about three orders of magnitude below the top weights
from the same token, and its helpfulness is negative. That is a specific interference
weight identified inside a trained transformer with a loss measurement attached to it,
which had not been done before.

## What holds up

Pruning by Fisher effectiveness is cheap and works: removing the least effective 70% of
weights costs 0.0107 nats and 85% costs 0.0702 nats (verified against `threshold_data.js`).
It beats raw virtual-weight magnitude at essentially every density.

The effectiveness tail really is one-sided. In all six families the most effective helpful
weight exceeds any harmful one by an order of magnitude or more.

## Three problems

### 1. The stated pruning exception names the wrong family

The text says Fisher beats magnitude "for every density and individual weight family
(except for negative Features→Logits weights)". `threshold_data.js` says the reverse for
that family — Fisher wins at all 19 sampled densities, by 3–10× in ΔL. The family where
Fisher actually loses is **Tokens→OV→Logits, negative-only**: worse at 8/19 densities, by
up to 2.22× across the density range 0.05–0.40.

### 2. The main conclusion does not survive the paper's own effect-size analysis

The Discussion concludes that "the model is still dense in this basis" and that "no
saliency scheme will perform much better", resting on 47.6% of weights having positive
mean helpfulness with "tens of percent" surviving significance testing.

The ROPE appendix asks what fraction survives at a *practically meaningful* effect size
rather than mere statistical significance at 1B tokens. Population-weighted over all 331M
weights:

| threshold ε | negative | practically zero | uncertain | **positive** |
|---|---|---|---|---|
| ε → 0 (main-text convention) | 11.3% | 9.4% | 64.3% | **15.1%** |
| ε = budget / N_total = 1.5e-8 nats/tok | 0.9% | 90.1% | 6.1% | **2.9%** |
| ε = budget / N_family | 0.1% | 97.8% | 1.0% | **1.1%** |

At the paper's own naive yardstick, 2.9% of 331M is ~9.6M weights — roughly 3× the original
transformer's parameter count, not the "tens of millions to interpret" the text describes.
It also lands next to the paper's own helpfulness-mass estimate (2.43% density preserves
90% of positive helpfulness mass). Three routes converge on 2–3%; the headline pessimism
comes only from the sign-of-the-mean route, and the ROPE section is never referenced in
the Discussion.

The honest caveat is the paper's own: removing many sub-ε weights at once can compound
nonlinearly, so 2.9% is not a demonstrated achievable density. But that caveat applies
equally to 47.6%, and only the pessimistic reading is stated as a result.

### 3. Helpfulness and Fisher effectiveness are the same quantity on the tail

Expanding the paper's own helpfulness formula to second order in `u = sw`:

```
Δℓ  = log(1 − p_j(1 − e^−u)) + u·1_{j=t}
    = u(1_{j=t} − p_j) + ½u²p_j(1 − p_j) + O(u³)

helpfulness(w) = w·E[s(1_{j=t} − p_j)] + fisher(w) + O(u³)
               = −w·∂L/∂w + fisher(w)
```

which is the Optimal Brain Damage saliency the paper cites in related work but never
connects to its headline. `fisher(w)` is a positive-definite quadratic form, so it is
always ≥ 0 and grows as `w²`, while the gradient term grows as `w`. Weights above a
crossover in `|sw|` are therefore *forced* to measure as helpful.

Measured on the helpful branch of `fisher_panels_data.js`:

| family | log-log slope | r | median h / fisher |
|---|---|---|---|
| QK | 0.96 | 0.956 | 0.99 |
| Tokens→Features | 1.06 | 0.990 | 1.15 |
| Tokens→OV→Features | 0.89 | 0.969 | 1.22 |
| Tokens→Logits | 0.99 | 0.982 | 0.58 |
| Features→Logits | 0.93 | 0.976 | 0.76 |
| Tokens→OV→Logits | 0.93 | 0.977 | 0.73 |

Slope 1 and a ratio near 1 in every family, over six or more orders of magnitude. So
"the model puts its most effective weights in helpful directions" is partly an identity
between the two metrics rather than evidence about training pressure alone. The paper
offers only the training-pressure reading ("a model trained to minimize loss has every
reason to get its most consequential weights pointing in a good direction").

This also inverts the methodological claim that "Fisher already prunes to a similar
density, [so] a sharper metric has little room to improve". Fisher tracks helpfulness on
the tail because the two coincide there. Where they differ is the first-order term — which
is exactly what separates helpful from harmful, and exactly what a better metric would
target.

## Checked, no issue

The 7,765-weight sample is stratified by family at roughly 1,210–1,379 each while the
families range from 16.8M to 104.9M weights. Population-weighting moves h>0 from the
paper's pooled 47.6% to 48.4%, so the headline is not biased by the stratification. (It
does matter for the dead fraction, 12.7% pooled vs 9.4% weighted, and for the ROPE panels,
which the paper only shows per family.)

The `[20.3, 72.7]` interval on the ALL row of the helpfulness table decodes as
[CI excludes zero, positive-or-uncertain] — confirmed against `rope_data.js` at ε → 0.

## Minor

The dead fraction of the same 7,765-weight sample is reported three ways: 12.7% in the
helpfulness table, 13.1% in the histogram figure, and 13.07% implied by the mass figure's
own counts (3,693 + 3,057 + 1,015 = 7,765).

Two assets 403 on the live site — `shared/data/token_splits.js` and
`shared/data/token_vocab_split.js`, both referenced by the Feature 157 figure.
