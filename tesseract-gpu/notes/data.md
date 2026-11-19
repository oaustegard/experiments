# Line benchmark: data and sources

Built by `run/make_lines.py` (seed 20261119). Output: `data/lines/` with
`labels.jsonl` (4,800 rows) and `<lang>/<id>_<cond>.png`.

## Sources

Every base line is a real sentence or verse line from a public corpus. None is
written by hand.

| lang | base lines | source URL | license |
|---|---:|---|---|
| eng | 300 | https://huggingface.co/datasets/wikimedia/wikipedia (config `20231101.en`, via the datasets-server rows API at https://datasets-server.huggingface.co/rows) | CC BY-SA 3.0 and GFDL (dataset card) |
| fra | 150 | https://huggingface.co/datasets/wikimedia/wikipedia (config `20231101.fr`, same API) | CC BY-SA 3.0 and GFDL (dataset card) |
| grc | 150 | https://github.com/PerseusDL/canonical-greekLit, `data/tlg0012/tlg001/tlg0012.tlg001.perseus-grc2.xml` (Homer, *Iliad*, polytonic with breathings and accents) | CC BY-SA 4.0 (repository README) |

Project Gutenberg was tried first for the English and French sources and the
gutenberg.org mirror returned 503, so it was not used.

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
- Raw responses are cached under `data/lines/_corpus/` so reruns do not refetch.

Text is stored in NFC form. Ground truth in `labels.jsonl` is the `text` field.

## Fonts

- Candidate fonts are the `.ttf` and `.otf` files reported by `fc-list`. 108
  loaded with a usable cmap. Collections (`.ttc`) were skipped.
- For each base line, the candidates are the fonts whose cmap covers every
  code point in the line. One is picked at random with the per-line seed. The
  file name is recorded in `labels.jsonl`.
- No extra font packages were installed. The 76 system fonts that cover the
  polytonic Greek extended block were enough.

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
| noise | additive Gaussian, sd uniform in [12, 25], clipped |
| jpeg | JPEG quality uniform integer in [15, 30], then re-read as PNG |
| lowres | resized so the x-height is 6 to 8 px (target uniform in [6, 8], scale = target / x-height of the `x` glyph at that size); Lanczos |
| rotate | rotation uniform in [-2, 2] degrees, bicubic, canvas expanded, filled with the corner background |
| shade | linear illumination ramp from a uniform value in [0.45, 0.7] up to 1.0 (horizontal or vertical, random direction), then contrast scaled to 40% about the image mean |
| invert | 255 minus the clean image (light text on dark) |

## Known limits

- The line pool is sentences of at most 70 characters, so the eng and fra
  lines are short clauses from encyclopedia prose. Wikipedia text is not a
  stand-in for book or newspaper typography.
- The Greek lines are from one epic, so vocabulary and accent patterns are
  narrow.
- Fonts are whatever this container has installed. Most are sans or serif
  Latin families; coverage of script-specific display faces is thin.
- Conditions are applied one at a time. Real scans combine them, which this
  benchmark does not test.
