# Licences and third-party material in indra-browser

## Indra OCR code (Apache-2.0)

`web/indra.js`, `web/layout.js`, `web/text.js`, `web/lm.js`, `web/imgproc.js`, `web/deskew.js`, `web/crops.js`,
`web/order.js`, `web/math.js`, `web/nets.js` and `run/ref.py` port the pipeline of Datalab's indra-ocr 1.1.2
(https://github.com/datalab-to/indra, Apache License 2.0, copyright Datalab) to JavaScript, WGSL and NumPy. They follow
its source closely, so they are treated as derivative works and are distributed under Apache-2.0 terms as well as the
repository's MIT licence; the full Apache text is in `LICENSE-APACHE-2.0`. The changes against indra-ocr: the networks
run as WebGPU kernels instead of ONNX Runtime, and image operations (resizing, deskew, line crops) run on the GPU or in
JavaScript instead of OpenCV.

## Indra OCR model weights (AI Pubs Open RAIL-M License, Modified)

The files under `web/models/` are Datalab's model weights (https://huggingface.co/datalab-to/indra-ocr), rounded to
16-bit floats or 8-bit integers, and the word-model values to 2 bits; nothing else in them is changed. They are a
Derivative of the Model under that licence, whose full text ships beside them as `web/models/LICENSE.txt`, and they
remain under it. Use is bound by its Attachment A: no use that breaks the law or infringes rights; no use by a person or
organisation with more than $20M in gross revenue in the prior year or more than $20M in total funding, except personal
use or research; and no use at all by anyone who provides a product or service that competes with Datalab's. The
licence also applies itself to the models' output (section 8). Commercial licences: https://www.datalab.to/pricing.
