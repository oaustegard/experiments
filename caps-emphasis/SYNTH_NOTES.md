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

---

# Markdown emphasis vs capitalisation

Same 22,100-document sample (10,523,198 word tokens, fields `query` +
`synthetic_reasoning` + `synthetic_answer`), no re-fetch. Full numbers in
`synth_bold.json`.

## Rates side by side

| marker | per 1,000 words | % of docs |
|---|---|---|
| `### heading` (ATX) | **8.27** | 82.7% |
| ALL CAPS, all tokens (mostly acronyms) | 8.10 | 61.8% |
| `**bold**` | **7.78** | 52.1% |
| line-initial `**bold**` label | 6.78 | — |
| `*italic*` | 0.67 | 14.1% |
| ALL CAPS, non-acronym | 0.56 | — |
| **ALL CAPS as emphasis** | **0.34** | — |
| `` `code` `` inline | 0.16 | 0.6% |
| fenced code block | 0.084 | 3.3% |
| `_italic_` | 0.0027 | 0.05% |
| `__bold__` | **0** | 0% |

**`**bold**` outnumbers emphatic ALL CAPS 23.1 : 1** (7.78 vs 0.34 per 1,000
words), and outnumbers all non-acronym caps 13.9 : 1. That ratio lands on top of
the pilot sweep's ~20× suppression gap between `**never**` and `NEVER`.

The complement is **not** null: SYNTH has a strong, well-formed bold prior. It
just is not primarily an *emphasis* prior — see below.

## Where bold appears (200 random `**…**` spans)

| bucket | count |
|---|---|
| heading/label (`**Key insight:**`, `**Confidence levels**:`) | **169** |
| emphasis-in-sentence | 16 |
| term-definition (`**capacity to suffer** = moral consideration`) | 7 |
| structured-marker (bold label + `●`/`◐`/`○` confidence glyph) | 8 |
| other | 0 |

Verbatim examples, one per bucket:

- heading/label — `**Confidence levels**: ⏎ - Regulatory framework evolution: ● ⏎ - Specific fine amounts…`
- emphasis-in-sentence — `Key distinction: **legal positivism** vs **judicial sovereignty**.`
- term-definition — `Jeremy Bentham (1748-1832). Utilitarian philosopher. Key insight: **capacity to suffer** = moral consideration.`
- structured-marker — `**Consensus evidence** ● : ⏎ PET-CT = modality of choice for lymph node metastasis`

Three more per bucket are in `synth_bold.json` → `buckets.examples`.

Split by role, the comparison narrows sharply:

| role | bold /1,000 | caps /1,000 | ratio |
|---|---|---|---|
| heading / label | 6.58 | 0.32 | **20×** |
| mid-sentence emphasis | 0.62 | 0.32 | 1.9× |

So SYNTH's salience convention is `### Heading` and `**Label:**`, not emphasis
markup and not capitals. Bold beats caps ~20× as a *structural* marker and only
~2× as in-sentence stress.

## The most-bolded strings are reasoning-scaffold labels

`**Key insight**` 1,273 (+1,243 with trailing colon) · `**Conclusion**` 1,111 ·
`**Final assessment**` 939 · `**Synthesis:**` 496 · `**Physical constraints:**` 397 ·
`**Final synthesis**` 276 · `**Initial assessment:**` 236 · `**Confidence assessment:**` 219 ·
`**Answer structure:**` 216. These are section headers of the synthetic reasoning
trace, not stressed words.

## Directive-word head-to-head

Counts in the 22,100-doc sample. "in-bold" = the word appears anywhere inside a
`**…**` span (i.e. inside a label like `**Critical gap:**`).

| word | `**word**` exact | in-bold span | ALL CAPS | plain lowercase |
|---|---|---|---|---|
| never | **1** | 7 | 5 | 957 |
| always | **1** | 6 | 6 | 1,297 |
| must | **3** | 18 | 2 | 3,730 |
| important | **4** | 23 | 0 | 3,147 |
| critical | **17** | 1,208 | 2 | 3,239 |
| note | **15** | 160 | 2 | 808 |
| required | **1** | 180 | 0 | 4,459 |
| avoid | **13** | 28 | 0 | 916 |
| warning | **0** | 4 | 1 | 241 |
| caution | **0** | 1 | 0 | 108 |
| remember | **0** | 1 | 0 | 396 |
| should | **0** | 32 | 1 | 4,195 |
| do not | **0** | 3 | 3 | 450 |
| not | 25 | 237 | 256 | 28,337 |
| only | 3 | 52 | 6 | 5,706 |
| all | 0 | 57 | 112 | 9,789 |

Bolded *directive words* are as close to zero as capitalised ones: `**never**`
once, `**always**` once, `**must**` three times, `**do not**` never, in 10.5M
words. `**critical**` (17) and `**note**` (15) are the only ones with any
presence, and their 1,208 / 160 in-bold counts are almost entirely the label
forms `**Critical gap:**`, `**Critical constraint:**`, `**Note:**`.

**So the bold advantage is carried by the marker, not by the word.** The model
has seen `**X**` 82,000 times — always meaning "this is the salient label of what
follows" — and has essentially never seen `NEVER`. Wrapping a directive in `**`
puts it in a heavily-attested salience frame; capitalising it puts it nowhere.

## Field asymmetry — relevant to where the directive is placed

| field | words | `**bold**` /1,000 | `*italic*` /1,000 |
|---|---|---|---|
| `query` (the user turn) | 763,630 | **0.0013** (1 occurrence total) | 0.06 |
| `synthetic_reasoning` | 5,225,974 | **14.70** | 1.02 |
| `synthetic_answer` | 4,533,594 | 1.12 | 0.36 |

Bold is a **reasoning-trace** convention: 93% of all bold spans are in
`synthetic_reasoning`, and the entire sample contains exactly **one** bold span in
a user query. A `**never**` in a user turn is therefore off-distribution for the
query register while being maximally in-distribution for the register the model
generates. ALL CAPS at least occurs in queries (4.06/1,000, mostly acronyms).
Worth controlling for if the pilot's directive sits in the user turn.

## Bottom line

Not a null. `**bold**` is 23× more frequent than emphatic ALL CAPS and appears in
52% of documents, versus a caps-as-emphasis rate of 0.34/1,000. But the prior is
"bold marks the label of a salient section", not "bold marks a stressed word" —
169 of 200 sampled spans are labels/headings. Bolding a directive borrows a
convention the model has seen ~82,000 times; capitalising it borrows one it has
seen almost never. That asymmetry, not any word-level statistic, is the
mechanism consistent with the pilot's 20× gap.
