# ASD-STE100 as a basis for the drafter

Basing the EAGLE drafter on [ASD-STE100](https://www.asd-ste100.org/) does not
help, for three separate reasons. The controlled vocabulary reproduces what
frequency-truncation already gives for free, the vocabulary lever it would pull
is already flat where it would land, and STE-shaped text does not make
Baguettotron more predictable in the measurement I can run.

STE is worth using if the deliverable has to be STE-compliant. Constrained
decoding meets that correctness requirement, and it is a different mechanism
from drafting.

## The specification

Two parts: a set of writing rules, and a controlled general dictionary of
roughly a thousand approved words, most carrying one meaning and one part of
speech. Issue 9 dates from January 2025. The dictionary is copyright ASD and
distributed only on request through the STEMG, so it is not something a training
pipeline can just pull.

The STEMG states the approved words were chosen for "simplicity, flexibility and
frequency of use". A frequency count applies that same criterion, which is what the next section
measures.

## STE as a draft vocabulary

**The dictionary is the closed half of an open language.** STE's technical names
and technical verbs are deliberately not listed. The STEMG is explicit that they
are company- and project-specific, defined only by category, and that an
otherwise-unapproved word "automatically becomes part of that technical name or
technical verb, and as such acceptable." So the bounded half is the function
words and common verbs, and the unbounded half is the content words. A drafter
already predicts the first half well and misses on the second. STE closes the
part that was never the problem.

**A thousand words is not a thousand tokens.** A draft head is indexed by BPE
tokens. Taking the top-K word types of a general corpus as a stand-in for the
approved list and expanding each into its cased and leading-space forms:

| Approved words | BPE token types | Tokens per word | Word-occurrence coverage |
|---|---|---|---|
| 500 | 1,925 | 3.85 | 67.8% |
| 900 | 3,296 | 3.66 | 77.8% |
| 1,500 | 5,061 | 3.37 | 85.8% |
| 3,000 | 8,505 | 2.83 | 94.7% |

A 900-word dictionary spans about 3,300 token types before any technical name is
admitted. Only 54.3% of the BPE tokens in that corpus are plain alphabetic words
at all — the rest is punctuation, digits, and subword fragments, none of which a
word list supplies and all of which a draft head must carry.

**The lever is already flat there.** From
[`RESOURCES.md`](RESOURCES.md), draft cost against draft-vocabulary size:

| Draft vocab | c | Projected at α=0.7 |
|---|---|---|
| 16,384 | 0.019 | 2.78× |
| 8,192 | 0.011 | 2.93× |
| 4,096 | 0.010 | 2.96× |

An STE-sized vocabulary lands between the last two rows, where the decoder layer
has become the floor and the output projection no longer matters. Going from
8,192 to an STE-derived 3,300 is worth roughly 0.03× of projected speedup. The
8,192-token head is derived by counting tokens on the deployment corpus, costs
nothing, carries no license, and covers 98.75% of occurrences.

## Target entropy by corpus

The mechanism that would help is a genuine one: acceptance is bounded by how
peaked the target's next-token distribution is, so a controlled language could
raise it for any drafter. Measured on Baguettotron:

| Corpus | Tokens | Mean entropy (nats) | Mean top-1 prob | Greedy hit rate |
|---|---|---|---|---|
| STE-style procedure | 198 | 3.01 | 0.453 | 0.457 |
| General prose | 149 | 2.79 | 0.452 | 0.392 |
| Python code | 72 | 2.12 | 0.557 | 0.620 |

The STE arm is not lower-entropy — it measures 0.22 nats higher than general
prose, with an identical top-1 probability. Its greedy hit rate is 17% better,
which says the model's argmax is more often the word that actually comes next
even though the distribution is no flatter, and that is what formulaic text
should look like.

These corpora are 72–198 tokens of hand-written proxy, and the STE arm follows
the published rules rather than the copyrighted dictionary. They rule out a
large effect in the expected direction and cannot resolve a small one. Code, which nobody designed as a controlled language, is
far more predictable than either.

**EAGLE does not have a vocabulary prior to constrain.** Its accuracy comes from
conditioning on the target's own hidden states, not from a language model over a
restricted vocabulary. Shrinking its output projection is a compute trick that is
orthogonal to how it earns acceptance.

## Constrained decoding

If output must be STE-compliant, enforce it with a decoding grammar over the
approved dictionary and the project's technical names. That is a compliance
feature. It also shrinks the live token set at each step, which would raise acceptance
for any drafter. The speedup is a side effect; the documentation requirement is
what has to justify the constraint.

Licensing is the practical gate either way: the dictionary is ASD copyright,
supplied on request, and a decoding grammar or a draft head derived from it is a
derived work.

## Files

| File | What |
|---|---|
| `ste_entropy.py` | Target entropy and greedy hit rate by corpus → `ste_entropy.json` |
| `ste_vocab.py` | BPE token span of a controlled word list → `ste_vocab.json` |
