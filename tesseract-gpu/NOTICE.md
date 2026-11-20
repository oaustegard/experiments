# Licences and third-party material in tesseract-gpu

The experiment's own code is MIT (the repository's `LICENSE`, copyright Oskar Austegard). Parts of it are ports of
Apache-2.0 code, and every model it loads (Tesseract's tessdata, PaddlePaddle's PP-OCRv6 detector) is Apache-2.0.

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

## Line finding (Apache-2.0)

The PP-OCRv6 text-line detector weights (https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx and `_small_`,
PaddlePaddle, Apache License 2.0) are converted by `run/export_det.py` into `web/det/models/` (weights unchanged in
value, re-laid out for the WebGPU kernels) and run by `web/det/`. The box post-processing in `tessgpu/layout.py` and
`web/layout.js` follows the algorithm of PaddleOCR's DBPostProcess (Apache-2.0); the column-aware ordering is ported
from anything-to-text in oaustegard.github.io (same owner). `web/tesslayout.js` loads tesseract.js-core
(https://github.com/naptha/tesseract.js-core, Apache-2.0, Tesseract 5.1) from jsDelivr at run time for Tesseract's own
page layout; the server pipeline uses tesserocr (MIT) over Tesseract's own library.

## Benchmark data

The line images and their sources are described in `notes/data.md` and `notes/data_real.md`. None of them are committed
(`data/` is git-ignored). FUNSD is licensed for non-commercial research only. `run/kaggle/server_bench.py` embeds 300
English sentences from Wikipedia, licensed CC BY-SA 3.0; they keep that licence.
