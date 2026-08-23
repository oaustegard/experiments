# Q5 — what capitalisation means in the training corpus

Measured first, because it sets the prior for everything else. Sample: 22,100
documents, 10,523,198 word tokens from `PleIAs/SYNTH` (config `default`, split
`train`), drawn at random offsets through the HF datasets-server rows API with
seed 20260823. That API exposes a partial index of 1,089,584 of the corpus's
79,648,272 rows, so the sample comes from the first shards rather than the whole
corpus. Text measured is `query` + `synthetic_reasoning` + `synthetic_answer`.

## Capitals are acronyms

All-caps tokens of two or more characters run at 8.10 per 1,000 words, and 61.8%
of documents contain at least one. Almost all of them are acronyms. Excluding a
curated acronym and roman-numeral list drops the rate to 5.36 per 1,000;
excluding acronyms identified from the data itself drops it to 0.56. Requiring
the token to be non-acronym, on a mixed-case line of at least five words, in a
document that is not shouting throughout, leaves **0.34 per 1,000 words — 4.2% of
all capitalised occurrences**.

Sampling 200 random caps occurrences and bucketing them: acronym or initialism
176, heading or label 8, emphasis inside a sentence 8, structured marker 8.

## Capitalised directives are absent

Per 10.5 million words:

| word | ALL CAPS | lowercase |
|---|---|---|
| IMPORTANT | 0 | 3,147 |
| REQUIRED | 0 | 4,459 |
| AVOID | 0 | 916 |
| MUST NOT | 0 | 108 |
| REMEMBER | 0 | 396 |
| CAUTION / ATTENTION | 0 | — |
| WARNING | 1 | 241 |
| NOTE | 2 | 808 |
| CRITICAL | 2 | 3,239 |
| MUST | 2 | 3,730 |
| DO NOT | 3 | 450 |
| NEVER | 5 | 957 |
| ALWAYS | 6 | 1,297 |

Reading the hits in context, most of the `NEVER` and `ALWAYS` occurrences are
creative-writing prose rather than instructions. **The sample contains exactly one
genuine capitalised directive: `NEVER give a baby honey.`**

Where non-acronym capitals do appear, they mark logical contrast in reasoning
traces, not urgency: AND 552, OR 393, NOT 237, IS 179, BUT 165, IF 91, THEN 29 —
as in `lacks binding force AND establishes common standards`, or
`Centrifugal force IS real in rotating frames, NOT "made up"`.

The tokenizer corroborates this independently. Baguettotron's 65,536-piece
vocabulary holds 3,502 multi-character all-caps alphabetic tokens against 36,597
lowercase. Since the BPE was fitted on SYNTH, the absence of caps merges is the
same fact as the corpus counts, seen from the other side — and it is why
capitalising a directive costs two to three times its lowercase token count.

## The marker the corpus actually uses

Markdown emphasis, measured on the same sample:

| marker | per 1,000 words | % of documents |
|---|---|---|
| `### heading` | 8.27 | 82.7% |
| all caps, all tokens | 8.10 | 61.8% |
| `**bold**` | 7.78 | 52.1% |
| line-initial `**bold**` label | 6.78 | — |
| `*italic*` | 0.67 | 14.1% |
| all caps, non-acronym | 0.56 | — |
| **all caps as emphasis** | **0.34** | — |
| `__bold__` | 0 | 0% |

Bold outnumbers emphatic capitals **23.1 to 1**.

Two qualifications that matter for reading the behavioural results.

**The marker carries it, not the word.** Bolded directive words are as rare as
capitalised ones: `**never**` occurs once in 10.5M words, `**always**` once,
`**do not**` never. What the model has seen is the *construction* — roughly
82,000 `**X**` spans, 169 of 200 sampled being headings or labels, the most
frequent being `**Key insight**` 1,273, `**Conclusion**` 1,111,
`**Final assessment**` 939. The prior is "a salient label follows", not "this
word is stressed".

**Bold is a reasoning-trace convention, not a prompt convention.** 93% of bold
spans sit in `synthetic_reasoning` (14.70 per 1,000) against `synthetic_answer`
1.12 and `query` **0.0013 — one bold span in 763,630 words of user turns**.
Capitals at least occur in queries, at 4.06 per 1,000, mostly as acronyms. So a
bolded directive in a user turn is off-distribution for that register while being
maximally in-distribution for the register the model writes in. That is what the
register arm tests.
