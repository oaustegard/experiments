# Real-scan line crops (data/real)

This file describes the real-scan line set in `data/real/`: 800 text-line crops
cut from scanned English forms (FUNSD) and receipts (SROIE), each with its
transcription. It measures the tesseract-gpu engines and Tesseract on scans
rather than on rendered text. It is built by `run/make_real.py`, and the output
is `data/real/labels.jsonl` plus one 8-bit greyscale PNG per line in
`data/real/funsd/` and `data/real/sroie/`.

## Sets used

| set | lang | source | licence | rows |
|---|---|---|---|---:|
| funsd | eng | https://huggingface.co/datasets/nielsr/funsd (parquet copy of FUNSD, https://guillaumejaume.github.io/FUNSD/) | FUNSD's own Terms of Use: "solely for non-commercial, research and educational purposes" (https://guillaumejaume.github.io/FUNSD/work/). Research use is allowed. | 400 |
| sroie | eng | https://huggingface.co/datasets/arvindrajan92/sroie_document_understanding (mirror of ICDAR-2019 SROIE, https://rrc.cvc.uab.es/?ch=13) | The mirror's card says MIT. The upstream SROIE licence was not checked, so the MIT tag is the mirror's claim only. | 400 |

FUNSD is scanned forms. SROIE is scanned receipts.

## Which pages were used

`make_real.py` takes at most 400 lines per set, in document order, and stops
when the cap is reached.

- **FUNSD.** Both the `train` and `test` parquet files are loaded, train first.
  The 400 lines come from the first seven forms of the train split, `funsd-0`
  to `funsd-6`, so no test form is in the set.
- **SROIE.** Only the train shard `data/train-00000-of-00001-66201d4afd6b73ca.parquet`
  is read. The 400 lines come from nine receipts, `sroie-0000` to `sroie-0008`.

## Line construction

- **FUNSD** has word boxes and entity tags only. Words are grouped by the set's
  own entity spans: a new group starts at each `B-` tag, on a change of class,
  and at the start of each run of `O` words. Words inside a group are then
  merged into lines by vertical overlap, which needs more than half the smaller
  height to be shared.
- **SROIE** has one OCR text box per line, as a 4-point polygon. The axis-aligned
  bounds of the polygon are the line box, and the slope of its top edge is kept
  for the filter below.

## Deskewing

Crops are not deskewed. Each crop is the axis-aligned box of the line, padded by
2 px, so a tilted line appears tilted in its PNG. The rotation filter below
keeps only lines whose top edge is within 3 degrees of level, so a kept crop
can still be tilted by up to 3 degrees. FUNSD word boxes are axis-aligned, so
their slope is recorded as 0.

## Filters

Each candidate line is skipped if:

- it has fewer than 3 characters after stripping (`short`);
- its box is under 8 px tall (`short_box`);
- its box is taller than it is wide (`vertical`);
- its top edge is more than 3 degrees from horizontal (`rotated`).

Kept crops get 2 px of padding, clamped to the page, and are saved as 8-bit
greyscale PNGs. Skip counts from the last run were: funsd 20 `short`, 3
`vertical`, 28 `empty_crop` (the word box lies outside the page); sroie 48
`short`.

## Row schema

`data/real/labels.jsonl` has one JSON object per line:

```json
{"id": "funsd-0-000", "file": "data/real/funsd/funsd-0-000.png", "lang": "eng", "cond": "real", "set": "funsd", "text": "R&D"}
```

| field | meaning |
|---|---|
| `id` | `<set>-<document>-<line>`. For FUNSD the document is the form index, for SROIE the receipt index. The line number is the candidate's position in the document, so a gap marks a skipped line. |
| `file` | PNG path relative to the repository root |
| `lang` | `eng` for both sets |
| `cond` | always `real` |
| `set` | `funsd` or `sroie` |
| `text` | the transcription, with surrounding whitespace removed |

## Verification

The row count and the per-set split come from this command, run in
`tesseract-gpu/`:

```
python3 -c 'import json, collections; rows = [json.loads(l) for l in open("data/real/labels.jsonl")]; print(len(rows), collections.Counter(r["set"] for r in rows))'
```

It prints `800 Counter({'funsd': 400, 'sroie': 400})`. Every file named in
`labels.jsonl` exists and opens as a PNG in mode `L`. Tesseract was not run on
this set.

## Not used

- **XFUND French** would supply the `fra` rows. It was left out because its
  licence could not be confirmed: the Hugging Face cards carry no licence text,
  and the upstream `doc-analysis/XFUND` repository and its release archive could
  not be reached. The set therefore has no French rows.
- The `darentang/sroie` mirror has no images, only paths, so it cannot be used.
