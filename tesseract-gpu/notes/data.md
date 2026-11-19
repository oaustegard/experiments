# Synthetic printed-text line set (data/lines)

This file describes the synthetic line set in `data/lines/`: 600 real sentences
or verse lines, each rendered in one font and then degraded in eight ways, with
the ground-truth text. The set measures the tesseract-gpu engines against stock
Tesseract, and scores weight packing and image handling. It is built by
`run/make_lines.py` (seed 20261119) and written to `data/lines/labels.jsonl`
(4,800 rows) and one PNG per row.

## Sources

Every base line is a real sentence or verse line from a public corpus. None is
written by hand.

| lang | base lines | source URL | license |
|---|---:|---|---|
| eng | 300 | https://huggingface.co/datasets/wikimedia/wikipedia (config `20231101.en`, via the datasets-server rows API at https://datasets-server.huggingface.co/rows) | CC BY-SA 3.0 and GFDL (dataset card) |
| fra | 150 | https://huggingface.co/datasets/wikimedia/wikipedia (config `20231101.fr`, same API) | CC BY-SA 3.0 and GFDL (dataset card) |
| grc | 150 | https://github.com/PerseusDL/canonical-greekLit, `data/tlg0012/tlg001/tlg0012.tlg001.perseus-grc2.xml` (Homer, *Iliad*, polytonic with breathings and accents) | CC BY-SA 4.0 (repository README) |

Project Gutenberg was the first choice for the English and French sources. Its
mirror returned HTTP 503, so Wikipedia was used.

## Sampling

- **eng and fra:** 40 random 20-row windows per language, drawn with a seeded
  RNG from the split. Paragraphs were split into sentences at `.!?` followed by
  whitespace and a capital. A sentence was kept when it was 20 to 70 characters
  long, ended in `.!?`, held only Latin letters (with accents), spaces and
  common punctuation, and had no digits, markup or brackets. The pool was
  de-duplicated, sorted, and 300 (eng) or 150 (fra) were sampled with a seeded
  RNG.
- **grc:** every `<l>` verse line in the XML was taken, with tags stripped and
  whitespace collapsed. Lines of 20 to 70 characters were kept, then 150 were
  sampled with a seeded RNG.
- Raw responses are cached under `data/lines/_corpus/`, so a rerun does not
  fetch them again.

Text is stored in NFC form. The ground truth in `labels.jsonl` is the `text`
field.

## Fonts

- Candidate fonts are the `.ttf` and `.otf` files that `fc-list` reports on the
  machine that built the set. 108 loaded with a usable cmap. Collections
  (`.ttc`) were skipped.
- For each base line, the candidates are the fonts whose cmap covers every code
  point in the line. One is picked at random with the line's seed.
- No extra font packages were installed. The system fonts covered the polytonic
  Greek block well enough.
- **Greek lines and fonts without an `x` glyph.** The coverage test does not
  check for an `x`, and Greek text has none, so a font without one can be
  picked for a Greek line. The lowres condition reads the x-height from the
  font's `x` glyph (`xheight()` in `run/make_lines.py`), so such a font would
  scale against its fallback glyph. The code does not check for this case. None
  of the 103 fonts used in the set lacks an `x` glyph, so the case did not arise.
- **Fonts used.** The clean rows (one per base line, 600 in all) use 103
  distinct font files in 18 family groups. The groups below come from the
  file-name prefix, so a family's bold and italic files fall in one group.

  | family group (file-name prefix) | clean rows | files |
  |---|---:|---:|
  | Inter | 119 | 18 |
  | InterDisplay | 112 | 18 |
  | JetBrainsMono | 51 | 16 |
  | FreeSerif | 28 | 3 |
  | Loma | 28 | 4 |
  | Caladea | 26 | 4 |
  | LiberationSerif | 25 | 4 |
  | LiberationMono | 24 | 4 |
  | Carlito | 22 | 4 |
  | unifont (`unifont`, `unifont_jp`, and their samples) | 22 | 4 |
  | DejaVuSansMono | 21 | 4 |
  | LiberationSans | 20 | 4 |
  | DejaVuSans | 19 | 2 |
  | FreeMono | 18 | 3 |
  | DejaVuSerif | 18 | 2 |
  | FreeSans | 16 | 3 |
  | FreeSerif, FreeMono, FreeSans bold files | 16 | 3 |
  | Japanese fonts (`fonts-japanese-gothic`, `ipag`, `ipagp`) | 15 | 3 |

## Rendering

- Greyscale (`L`) PNG, 8 bits. Dark ink (0 to 60) on a light background (235 to
  255). Padding 4 to 12 px. Font size 18 to 40 px, recorded as `size`.

## Conditions

Each base line gets one file per condition. Every condition is applied to the
clean render, not to another condition.

| cond | parameters |
|---|---|
| clean | none |
| blur | Gaussian, sigma uniform in [0.6, 1.2] |
| noise | additive Gaussian, sd uniform in [12, 25], clipped to 0 to 255 |
| jpeg | JPEG quality uniform integer in [15, 30], then re-read as PNG |
| lowres | resized so the x-height is 6 to 8 px: target uniform in [6, 8], scale = target / x-height of the font's `x` glyph at that size; Lanczos |
| rotate | rotation uniform in [-2, 2] degrees, bicubic, canvas expanded, filled with the value of the top-left pixel |
| shade | linear illumination ramp from a uniform value in [0.45, 0.7] up to 1.0 (horizontal or vertical, random direction), then contrast scaled to 40% about the image mean |
| invert | 255 minus the clean image (light text on dark) |

Every lowres image is smaller than its clean render. The widest lowres image is
0.98 of its clean width, and the median is 0.46.

## Seeds

Each base line has an id such as `eng0001`. Two generators are seeded from it:

- A Python `random.Random` seeded with the string `"20261119:<id>"`, where
  `20261119` is the seed. It draws, in this order, the font, the font size, the
  padding, the background and ink values, and then the parameters of the
  conditions.
- A NumPy generator, used for the noise condition, seeded with the first 8 hex
  digits of the SHA-256 of `"20261119:<id>"`.

The sentence sample uses its own seeded generators (`"20261119:eng:sample"` and
the same form for the other languages), so the lines do not change when a
condition is edited.

## Row schema

`data/lines/labels.jsonl` has one JSON object per line, with these seven fields:

```json
{"id": "eng0001", "file": "eng/eng0001_clean.png", "lang": "eng", "cond": "clean", "text": "They also did not allow CRPF jawans to enter their village.", "font": "LiberationSerif-Bold.ttf", "size": 26}
```

| field | meaning |
|---|---|
| `id` | base line id: language and a four-digit index, shared by the eight rows of one line |
| `file` | PNG path relative to `data/lines/` |
| `lang` | `eng`, `fra` or `grc` |
| `cond` | one of `clean`, `blur`, `noise`, `jpeg`, `lowres`, `rotate`, `shade`, `invert` |
| `text` | ground-truth text, NFC |
| `font` | font file name, without its directory |
| `size` | font size in pixels, 18 to 40 |

The set has 600 base lines (300 eng, 150 fra, 150 grc) and 8 conditions, so
4,800 rows.

## Tesseract runs

Tesseract was run on every PNG as `tesseract <png> stdout --oem 1 --psm 7`, once
with `tessdata_best` and once with `tessdata_fast`. The outputs are in
`data/lines/tess_best.jsonl` and `data/lines/tess_fast.jsonl`.

## Known limits

- The line pool is sentences of at most 70 characters, so the eng and fra
  lines are short clauses from encyclopedia prose. Wikipedia text is not a
  stand-in for book or newspaper typography.
- The Greek lines come from one epic, so vocabulary and accent patterns are
  narrow.
- The fonts are the system fonts of the machine that built the set. Most are
  Latin sans and serif families, and the Japanese and unifont files cover
  characters the others lack.
- Each condition is applied alone. Real scans combine several degradations, and
  this set does not test that combination.
