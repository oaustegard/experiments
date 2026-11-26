# Notices

## Model weights

The packed weights that the browser page in this experiment (`web/index.html`) downloads derive from NVIDIA Nemotron Parse 2.0
(https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
licensed under the OpenMDW License Agreement, version 1.1. A copy of that agreement, with NVIDIA's notices, is in
`web/models/LICENSE.txt` and sits beside the weight files on the download server (Cloudflare R2).

Changes made in this experiment: the image encoder's weight matrices are rounded to 4-bit integers in groups of 32 and
the text decoder's to 8-bit integers per row, each with 16-bit scales (`run/export.py` writes them); norms and biases are stored at 16 bits. The files are split
into gzip shards for download. No weights were retrained.

The tokenizer (`tokenizer.json`) is NVIDIA's, licensed under CC-BY-4.0
(https://creativecommons.org/licenses/by/4.0/), unchanged.

## Code

In the browser page: pdf.js (Apache-2.0, Mozilla) renders PDF pages; marked (MIT) renders Markdown; DOMPurify
(Apache-2.0 / MPL-2.0) sanitizes it. All three load from esm.sh or jsDelivr.
