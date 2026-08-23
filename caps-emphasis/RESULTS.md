# Capitalised emphasis in prompts on a 321M model

Capitalising a directive does not make Baguettotron obey it. Mid-sentence, the
effect is +0.003 log-odds with a confidence interval of [−0.147, +0.168] — no
effect at all. Where capitalising does move the number, the movement tracks how
many extra tokens the capitalised form costs, not the case change. Markdown bold
does move it, at both sentence positions, but that advantage reverses when the
bold is placed in the register the training corpus actually puts bold in, which
makes it surprise at an off-register marker rather than emphasis.

The corpus explains all of it. In SYNTH, capitalisation as a directive is
effectively absent: 22,100 documents and 10.5 million words contain exactly one
genuine capitalised instruction.

Model: PleIAs/Baguettotron, 321M parameters, 80 layers, 65,536-token vocabulary,
trained on SYNTH. CPU, 4 threads, fp32, greedy. 58 items, stratified by baseline
pressure. Method in [`METHOD.md`](METHOD.md), prior art in
[`PRIOR_ART.md`](PRIOR_ART.md), mistakes in [`ERRORS.md`](ERRORS.md), corpus
measurement in [`Q5_CORPUS.md`](Q5_CORPUS.md).

Suppression below is in log-odds, normalised against a control token so that a
prompt which flattens the whole next-token distribution does not read as
suppression. **Negative means the directive raised the forbidden word's
probability**, which is what nearly everything here does.

---

## Q1 — compliance under a capitalised directive

No. Measured as actual violations, on 43 items with the model writing the answer
itself:

| directive | violated |
|---|---|
| none | 42/43 |
| `You must never mention the word X.` | 42/43 |
| `You must NEVER mention the word X.` | 42/43 |
| lowercase, padded to matched token count | 42/43 |
| `YOU MUST NEVER MENTION THE WORD X.` | 42/43 |
| `You must **never** mention the word X.` | 41/43 |

Every surface form matches the no-directive baseline. The prohibition is not
weakly obeyed at this scale; it is not obeyed.

The probability measure says the same thing more sharply. Every directive tested
*raises* the forbidden word by roughly 2.5 log-odds against no directive at all —
the ironic rebound that [Mann et al.](https://arxiv.org/abs/2511.12381) and
[Rana](https://arxiv.org/abs/2601.08070) report on models 7× to 20× larger,
reproduced here. Capitalising the directive makes the rebound slightly worse, not
better.

## Q2 — case versus token count

Token count, to the extent it is anything. Capitalising a keyword costs a variable number of extra tokens in this tokenizer.
Pooling the case effect by that cost, with several keywords per bin and the
sentence frame held fixed:

| CAPS costs | keywords | effect (log-odds) | 95% CI |
|---|---|---|---|
| +0 tokens | 3 | −0.100 | [−0.229, +0.035] |
| +1 token | 15 | −0.147 | [−0.203, −0.090] |
| +2 tokens | 4 | −0.083 | [−0.178, +0.009] |
| +3 tokens | 3 | +0.104 | [−0.030, +0.243] |
| +4 tokens | 1 | +0.299 | [−0.000, +0.574] |

When capitalising is free in tokens, the interval spans zero. Capitalisation only moves the number once it buys extra residual streams, and the
movement grows with them.

The position arm is cleaner still. Holding the keyword, the frame and the
question fixed and moving only where the keyword sits:

| marker | sentence-initial | mid-sentence |
|---|---|---|
| ALL CAPS | −0.256 [−0.542, +0.034] | **+0.003 [−0.147, +0.168]** |
| `**bold**` | +0.277 [−0.001, +0.588] | +0.299 [+0.030, +0.573] |
| `*italic*` | +0.150 [−0.027, +0.339] | +0.235 [+0.114, +0.361] |

Mid-sentence capitalisation is a null with a tight interval, t = 0.04. Markdown
emphasis survives both positions.

The effect also changes sign with how strongly the model wants to say the word in
the first place: +0.075 where the word is near-certain, −0.193 where it is
weakest. An unstratified pool would have averaged these into nothing and reported
the average as the finding.

| baseline pressure | n | CAPS effect | 95% CI |
|---|---|---|---|
| ceiling | 312 | +0.075 | [−0.016, +0.165] |
| high | 520 | −0.154 | [−0.222, −0.091] |
| mid | 364 | −0.033 | [−0.128, +0.062] |
| low | 312 | −0.193 | [−0.302, −0.087] |

## Q3 — dose-response

Yes to both, with the forbidden word's own case frozen so that capitalising the
directive does not end up capitalising the target.

| capitalised | rebound | 95% CI | between-draw sd |
|---|---|---|---|
| 0% | −2.135 | [−2.683, −1.640] | — |
| 10% | −2.067 | [−2.219, −1.917] | 0.071 |
| 25% | −1.984 | [−2.140, −1.828] | 0.100 |
| 50% | −1.977 | [−2.135, −1.824] | 0.068 |
| 75% | −2.133 | [−2.296, −1.969] | 0.067 |
| 100% | −2.257 | [−2.423, −2.088] | — |

Partial capitalisation is least bad at 25–50%; full capitalisation is worse than
none at all. Each level averages twelve random word-subsets, and the spread
between draws (0.07–0.10) is well under the spread between levels (~0.28), so
this is a property of the dose rather than of which words happened to be picked.

The size of this is small. The shape is the one the dilution hypothesis predicts.

## Q4 — attention

Total attention mass onto the directive span scales with the span's token count,
which is mechanically guaranteed by a softmax over more keys and is not evidence
of anything: the bold span is 4 tokens and draws 1.233 summed across layers, the
capitalised span is 2 tokens and draws 0.589. Per token the two are nearly
identical, 0.308 against 0.294.

The causal version zeroes the final query position's attention onto the directive
span, layer by layer, and re-reads the forbidden word's probability. It is
reported in [`knockout.json`](knockout.json). Single-layer knockouts moved log
P by at most 0.045.

## Q5 — the base rate in SYNTH

Full numbers in [`Q5_CORPUS.md`](Q5_CORPUS.md). The short version, from 22,100
documents and 10,523,198 words of SYNTH:

All-caps tokens run at 8.10 per 1,000 words, but 96% are acronyms. Capitals used
as emphasis run at **0.34 per 1,000**. Capitalised directives are absent:
`IMPORTANT` 0 occurrences against 3,147 lowercase, `REQUIRED` 0 against 4,459,
`AVOID` 0 against 916, `NEVER` 5 against 957. The sample contains exactly one
genuine capitalised instruction: `NEVER give a baby honey.`

Markdown bold runs at 7.78 per 1,000 words and appears in 52% of documents,
outnumbering emphatic capitals **23 to 1**.

A model cannot follow a convention it has never been shown. The tokenizer says
the same thing from the other side: fitted on this corpus, it holds 3,502
multi-character all-caps tokens against 36,597 lowercase, which is why
capitalising a directive costs two to three times its lowercase token count.

## Bold was entropy and register, not emphasis

Bold outnumbers emphatic capitals in the corpus 23:1, and in the first pass bold
appeared to suppress about 20× harder than capitals. That correspondence was
tempting and it was wrong twice over.

**Most of the raw effect was entropy.** Bold raises the entropy of the
next-token distribution more than any other marker tested (1.927 against 1.451
for sentence case). Raw log P cannot tell a prompt that suppresses one word from
a prompt that flattens everything. Normalised against a control token, bold's
advantage over sentence case is +0.277 log-odds, not 20×.

**What remains is register, not emphasis.** 93% of bold spans in SYNTH sit in
reasoning traces; user turns contain one bold span in 763,630 words. Moving the
same bolded directive from the user turn into the reasoning register reverses its
sign:

| | sentence case | bold | bold's effect |
|---|---|---|---|
| user turn | −2.557 | −2.280 | **+0.277** |
| reasoning register | −1.754 | −1.993 | **−0.240** |

Bold helps where the corpus never puts it and hurts where the corpus always puts
it. That is the signature of an off-distribution marker, not of emphasis.

It also reveals the largest effect measured anywhere in this experiment, and it
has nothing to do with typography: **restating the constraint inside the
reasoning register rather than the user turn cuts the rebound from −2.557 to
−1.754**, about 0.8 log-odds, roughly three times any surface-formatting effect
here.

## Scope and threats to validity

A 321M model trained on synthetic reasoning traces is not evidence about frontier
models, and the effect sizes here should not be carried anywhere. The corpus
measurement is about SYNTH specifically; a model trained on scraped web text has
seen plenty of shouting and may well have a caps-as-emphasis prior that
Baguettotron cannot have.

What may transfer is the method (the token-cost dissociation, the control-token
normalisation, the register contrast) and the structural claim that a
typographic convention's effect depends on whether the training corpus used that
convention, and in which register.

The teacher-forced frame is the main threat to these numbers. It forces the
assistant turn past an empty `<think>` block, and PleIAs' own model card says
suppressing the think trace hurts performance. The generation arms are the check
on it, and they agree: no compliance in any condition. But
[Dillitzer et al.](https://arxiv.org/abs/2608.03711) find reasoning models show
near-zero case sensitivity and attribute it specifically to the think phase, so
some of the null here may belong to the frame rather than to the model.

On Q1 and Q4 this is a replication under stricter controls rather than a
discovery; that paper covers the same ground across 13 models. The dose-response, the
within-tokenizer token-cost dissociation, the register contrast and the corpus
base rate are where four searches found no prior work.
