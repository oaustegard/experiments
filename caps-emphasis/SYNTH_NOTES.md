# All-caps in SYNTH — base rate

**Dataset:** `PleIAs/SYNTH`, config `default`, split `train` (the Baguettotron
pretraining corpus).
**Sample:** 22,100 documents, 10,523,198 word tokens, drawn at random offsets via
the HF datasets-server rows API (which exposes a partial index of 1,089,584 of the
79,648,272 rows). Text measured = `query` + `synthetic_reasoning` +
`synthetic_answer`; `query_seed_text` (verbatim Wikipedia) and `constraints`
(generation metadata) measured separately and excluded from headline numbers.

## Headline

| metric | value |
|---|---|
| all-caps tokens (len ≥ 2) | 85,277 |
| **caps per 1,000 word tokens, all** | **8.10** |
| caps per 1,000, excluding curated acronyms + roman numerals | 5.36 |
| caps per 1,000, excluding data-driven acronyms | 0.56 |
| **emphatic caps per 1,000** (non-acronym, mixed-case line ≥5 words, non-shouting doc) | **0.34** |
| docs with ≥1 caps token | 61.8% |
| docs with ≥1 non-acronym caps token (curated-list rule) | 47.5% |
| documents that are majority-uppercase throughout | 12 / 22,100 (0.05%) |

Caps is overwhelmingly *acronyms*. 96% of all-caps tokens are acronym-like;
79% are 2–3 characters long. Only **4.2%** of all-caps occurrences are a real
word being capitalised inside running prose.

## Where caps appears (200 random occurrences, bucketed)

| bucket | count |
|---|---|
| acronym/initialism | 176 |
| heading/label (line-initial, followed by `:`, or whole line caps) | 8 |
| emphasis-in-sentence | 8 |
| structured-marker (code, tables, subscripted symbols, `A₁`/`KE_total`) | 8 |
| other | 0 |

Verbatim examples per bucket are in `synth_caps.json` → `buckets.examples`.

## Directive words in caps — the number that matters

Caps directive forms are effectively absent. Counts are occurrences in the 22,100-doc sample:

| word | CAPS | lowercase | caps share |
|---|---|---|---|
| IMPORTANT | **0** | 3,147 | 0.000 |
| REQUIRED | **0** | 4,459 | 0.000 |
| AVOID | **0** | 916 | 0.000 |
| CAUTION | **0** | 108 | 0.000 |
| ATTENTION | **0** | 958 | 0.000 |
| REMEMBER | **0** | 396 | 0.000 |
| MUST NOT | **0** | 35 | 0.000 |
| SHOULD | 1 | 4,195 | 0.0002 |
| MUST | 2 | 3,730 | 0.0005 |
| CRITICAL | 2 | 3,239 | 0.0006 |
| NOTE | 2 | 808 | 0.0025 |
| WARNING | 1 | 241 | 0.004 |
| DANGER | 1 | 218 | 0.005 |
| ALWAYS | 6 | 1,297 | 0.005 |
| NEVER | 5 | 957 | 0.005 |
| DO NOT | 3 | 450 | 0.007 |
| ONLY | 6 | 5,706 | 0.001 |
| DO | 39 | 8,705 | 0.004 |
| NOT | 256 | 28,337 | 0.009 |
| ALL | 112 | 9,789 | 0.011 |

Reading the contexts: of the 5 `NEVER` and 6 `ALWAYS` hits, most are creative-writing
prose (a character's tinnitus described as `CONSTANT, PAIN, ALWAYS there`), not
instructions. Exactly one reads as a genuine caps directive in the whole sample:
`NEVER give a baby honey.`

## What caps *is* used for, when it isn't an acronym

The non-acronym caps that exist are **logical/contrastive markers inside reasoning
traces**, not instruction emphasis. Top emphatic tokens and their character:

`AND` 552 · `OR` 393 · `NOT` 237 · `IS` 179 · `BUT` 165 · `ALL` 104 · `IF` 91 ·
`CAN` 52 · `TRUE` 49 · `NO` 44 · `ARE` 39 · `DO` 38 · `THEN` 29 · `FALSE` 19

Typical usage:

- `Declaration lacks binding force AND establishes common standards`
- `"dialectos" could mean regional varieties OR systematic linguistic varieties`
- `- But NOT presidential office itself`
- `IF mission launched on schedule → THEN landing site selection followed protocol → THEN … → BUT mission status uncertain`
- `Centrifugal force IS real in rotating frames, NOT "made up"`

That is boolean/contrast marking in the `synthetic_reasoning` field, which carries
almost 3× the caps density of the answer field.

## Per-field and per-exercise

| field | caps / 1,000 words |
|---|---|
| `synthetic_reasoning` | 11.88 |
| `synthetic_answer` | 4.43 |
| `query` | 4.06 |
| `query_seed_text` (Wikipedia seed, not generated) | 6.12 |
| `constraints` (metadata) | 5.15 |

Docs containing any caps, by exercise: creative writing 70.6%, mcq 64.1%,
memorization 63.5%, rag 57.9%, editing 48.8%, cooking 48.6%, constrained writing
39.1%, math mcq 22.3%, math exercise 17.1%.

## Corroborating evidence from the tokenizer

Baguettotron's 65,536-token vocabulary contains 3,502 multi-character all-caps
alphabetic tokens vs 36,597 lowercase and 13,754 Title/mixed — and none of the
directive words got a caps token. Token cost of each surface form
(no-leading-space / with-leading-space):

| word | CAPS | lowercase | Title |
|---|---|---|---|
| never | 2/2 | 2/1 | 1/1 |
| always | 3/3 | 1/1 | 2/1 |
| must | 2/2 | 1/1 | 1/1 |
| important | 3/3 | 1/1 | 2/1 |
| critical | 3/3 | 1/1 | 1/1 |
| required | 3/3 | 1/1 | 1/1 |
| warning | 2/2 | 1/1 | 1/2 |
| avoid | 3/3 | 1/1 | 2/1 |
| note | 1/1 | 1/1 | 1/1 |

Every directive word except `NOTE` (and the function words `DO`, `NOT`, `ALL`,
`ONLY`) costs 2–3 tokens in caps and 1 token in lowercase. The BPE merges were
fitted on SYNTH, so this is an independent confirmation of the corpus counts:
caps directives were too rare to earn a merge.

## Bottom line

Writing `NEVER` or `IMPORTANT` in a Baguettotron prompt puts the model on a
sequence it saw essentially never in training (0–6 occurrences per 10.5M words),
and costs 2–3× the tokens of the lowercase form. Where SYNTH does capitalise a
real word, it means logical contrast (`AND`/`OR`/`NOT`/`BUT`), not urgency.

## Files

- `synth_caps.json` — full measurements, top-60 tokens, bucket examples, tokenizer table
- `synth_caps_emphasis.json` — emphatic-caps isolation pass with contexts
- `baguettotron_vocab_caps.json` — vocabulary analysis
- `fetch_synth.py`, `analyze_caps.py`, `emphasis_pass.py`, `curated_acronyms.txt` — code
