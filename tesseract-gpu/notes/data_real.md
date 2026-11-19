# Real-scan printed-text lines (data/real)

Built by `run/make_real.py`. Each row of `data/real/labels.jsonl` is one text line
cropped from a scanned page, with its transcription. `cond` is `real`.

## Sets used

| set | lang | source | licence | rows |
|---|---|---|---|---|
| funsd | eng | https://huggingface.co/datasets/nielsr/funsd (parquet copy of FUNSD, https://guillaumejaume.github.io/FUNSD/) | FUNSD's own Terms of Use: "solely for non-commercial, research and educational purposes" (https://guillaumejaume.github.io/FUNSD/work/). Research use allowed. | 400 |
| sroie | eng | https://huggingface.co/datasets/arvindrajan92/sroie_document_understanding (mirror of ICDAR-2019 SROIE, https://rrc.cvc.uab.es/?ch=13) | Mirror card says MIT. The upstream SROIE licence could not be read from here (GitHub and the RRC site were unreachable), so treat the MIT tag as the mirror's claim only. | 400 |

Both sets are real scans: FUNSD is scanned forms, SROIE is scanned receipts.

## Line construction

- **FUNSD** has word boxes and entity tags only. Words are grouped by the set's
  own entity spans (a new group at each `B-` tag, on a class change, and at the
  start of each run of `O` words). Words inside a group are then merged into
  lines by vertical overlap.
- **SROIE** already has one OCR text box per line (4-point polygon). The axis
  bounds of each polygon are the line box, and the top-edge slope gives the rotation.

## Filters

- Under 3 characters after strip: skipped.
- Vertical (height > width): skipped.
- Rotated (top-edge slope over 3 degrees from horizontal): skipped. This is my
  threshold; the brief did not give one.
- Box under 8 px tall: skipped.
- Crop has 2 px padding, clamped to the page. Saved as 8-bit greyscale PNG.
- At most 400 lines per set, taken in document order.

Skip counts from the last run: funsd 20 short, 3 vertical, 28 empty crop (the
word box lies outside the page); sroie 48 short.

## Not used

- **XFUND French** (`FrancophonIA/XFUND`, `nnul/xfund-multilingual`, 50 French
  val forms, plus train). Not needed to reach the target. The licence could not
  be checked: no licence text in the HF cards, and the upstream
  `doc-analysis/XFUND` GitHub repo and its release archive returned 403/404
  from here. Left out until the licence is confirmed. It would supply the
  `fra` rows.
- The `darentang/sroie` mirror has no images (only paths), so it was not used.

## Verification

```
800 Counter({'funsd': 400, 'sroie': 400})
```

Every file named in labels.jsonl exists and opens. Mode is `L`. Tesseract was not run.
