# Notice: OCRia-nano-1 weights in this directory

The model files packed under `web/models/` and `web/gpu/models/` are derived from **OCRia-nano-1** by **LlamaIndex, Inc.**
(<https://huggingface.co/llamaindex/OCRia-nano-1>), and are distributed under the **OCRia-nano-1 Model
License**, which is based on the AI Pubs Open RAIL-M License v0.1 as modified by LlamaIndex. A copy is in
[`web/models/LICENSE`](web/models/LICENSE). By downloading or using these files you agree to that licence.
They are not open-source software, and nothing here is endorsed by LlamaIndex.

## Use restrictions (paragraph 5 and Attachment A of the licence)

The licence's use-based restrictions apply to these files, to anything derived from them and to their
output, and anyone who uses them must comply with them. In summary (the licence text governs; "Research
Purposes" is defined in its section 1(n): non-commercial research and teaching, evaluation and benchmarking,
and internal R&D that reaches no product or production workflow):

- no use that breaks the law or infringes anyone's intellectual property;
- commercial use only by an organization (with its affiliates) under US$2 million in gross revenue in the
  prior year **and** under US$2 million in total funding; above either threshold, personal use and
  Research Purposes only;
- no use to provide, or as part of, a document-parsing, OCR or extraction product or service that
  competes with LlamaIndex's (LlamaParse, LlamaCloud), whatever the size of the user, except for Research
  Purposes.

Licensing beyond these terms: <https://www.llamaindex.ai/contact>.

## Changes made to the files

The original files are LlamaIndex's release on Hugging Face, revision
`53d45fe6803a568deedb081550c2fdb97cc99f97`.

- `det.onnx`, `router.onnx`, `rec_world.onnx`, `rec_rtl.onnx`, `rec_cjk.onnx` and `tablenet.bin`: every
  float32 tensor of 1,024 or more elements was rounded to integers in −L..L per row (first axis), with one
  float32 scale per row; L = 127 in the shipped packs. Smaller tensors, the
  graphs and all other bytes are unchanged.
- `rec_latin.onnx` (already int8 in the original), the `*.charset.json` and `*.meta.json` files: unchanged.
- `web/gpu/models/` holds the same networks re-exported by `run/export_gpu.py` as an op program
  (`program.json`) and int8 weights (`weights.bin`, L = 127) for the WebGPU engine in `web/gpu/`;
  `rec_latin`'s own int8 weights are kept as they are.
- The files in `web/models/` are stored in `OCPK` bundles (`run/pack.py`): the rounded tensors as int8 rows with their
  scales, every other byte copied as it is. `web/ocria.js` rebuilds files of the original size and layout
  from them when it loads.

## Other components

- [LiteParse](https://github.com/run-llama/liteparse) (`@llamaindex/liteparse-wasm` 2.21.2), which runs the
  models, is Apache-2.0. It is not copied here; `web/ocria.js` loads it from jsdelivr.
- The code in this directory (`run/`, `web/*.js`, `web/*.html`, `web/test/`) is under this repository's MIT
  licence.
