# Licences and third-party material in tesseract-gpu

The experiment's own code is MIT (the repository's `LICENSE`, copyright Oskar Austegard). Parts of it are ports of
Apache-2.0 code, it loads Apache-2.0 models, and its demo page uses Datalab's Indra OCR models, which have their own
licence with use restrictions.

## Tesseract (Apache-2.0)

These files reimplement Tesseract 5.3.4 (https://github.com/tesseract-ocr/tesseract, Apache License 2.0, copyright
Google and the Tesseract contributors) in Python, PyTorch, JavaScript and WGSL. They follow its source closely, so they
are treated as derivative works of it and are distributed under Apache-2.0 terms as well as MIT; the full Apache text is
in `LICENSE-APACHE-2.0`.

| file | follows |
|---|---|
| `tessgpu/traineddata.py`, `web/traineddata.js` | `ccutil/tessdatamanager`, `lstm/network`, `lstm/weightmatrix`, `lstm/lstm`, `ccutil/unicharcompress` (file format) |
| `tessgpu/ref.py`, `server/torch_engine.py`, `web/prep.js`, `web/kernels/*.js` | `lstm/input`, `lstm/networkio`, `lstm/convolve`, `lstm/maxpool`, `lstm/lstm`, `lstm/fullyconnected`, `lstm/functions`, `ccutil/helpers` (TRand) |
| `tessgpu/beam.py`, `web/beam.js` | `lstm/recodebeam` (no-dictionary path), `ccutil/unicharcompress` |
| `harness/*.cpp` | links Tesseract 5.3.4 (built from source in a sandbox) to dump its outputs |

The changes against Tesseract: the network runs as batched GPU kernels or NumPy/PyTorch tensor code instead of
Tesseract's per-line C++; inverted lines are detected from the image instead of by a second forward pass; nothing from
Tesseract's page layout, dictionaries or legacy engine is included.

The tessdata models (https://github.com/tesseract-ocr/tessdata_best, `tessdata_fast`) are Apache-2.0. The demo page
fetches them from jsDelivr at run time; this repository does not redistribute them (`models/` is git-ignored).

## Indra OCR (code Apache-2.0, weights modified OpenRAIL-M)

The demo page (`web/`) imports `indra-browser/web/indra.js` from this repository at run time. That file and its
neighbours are a JavaScript port of Datalab's indra-ocr package (https://github.com/datalab-to/indra, Apache-2.0); see
`indra-browser/NOTICE.md`.

The Indra model weights (https://huggingface.co/datalab-to/indra-ocr) are Datalab's, under the AI Pubs Open RAIL-M
License (Modified), version 0.1; the full text is `indra-browser/web/models/LICENSE.txt`. indra-browser serves them
rounded to 16-bit floats or 8-bit integers (a Derivative of the Model under that licence); tesseract-gpu's page loads
those files from indra-browser and does not copy them. Anyone using the page uses the models and is bound by the
licence's use restrictions (Attachment A): no use that breaks the law or infringes rights; no use by a person or
organisation with more than $20M in gross revenue in the prior year or more than $20M in total funding, except personal
use or research; and no use at all by anyone who provides a product or service that competes with Datalab's. The
licence also applies itself to the models' output (section 8). Commercial licences: https://www.datalab.to/pricing.

## Benchmark data

The line images and their sources are described in `notes/data.md` and `notes/data_real.md`. None of them are committed
(`data/` is git-ignored). FUNSD is licensed for non-commercial research only. `run/kaggle/server_bench.py` embeds 300
English sentences from Wikipedia, licensed CC BY-SA 3.0; they keep that licence.
