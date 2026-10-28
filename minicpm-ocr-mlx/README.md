# MiniCPM-OCR 1.0 on MLX

An MLX port of [openbmb/MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0)
(1.3B, released 2026-10-21) that runs on the released mlx-vlm 0.7.10 with no
library changes. `convert.py` writes a checkpoint that carries its own model
file, so stock `mlx_vlm.generate` loads it. In fp32 its greedy transcripts are
token-for-token identical to transformers 5.21 on all four test crops.

## Use it on a Mac

```bash
pip install "mlx-vlm==0.7.10"
```

```bash
python convert.py ~/models/minicpm-ocr-q8 --bits 8 --quantize mlp,attn,linear_attn,embed
```

```bash
mlx_vlm.generate --model ~/models/minicpm-ocr-q8 --image crop.png --prompt "Text Recognition:" --temperature 0 --max-tokens 2048
```

Conversion and the processor need no torch (checked in a clean venv). The prompt picks the task and must match the crop: `Text
Recognition:`, `Table Recognition:` (answers in OTSL, which the MiniCPM-OCR
SDK turns into HTML), `Formula Recognition:`, `Seal Recognition:`, `Chart to
Table:`. The model reads regions, not whole pages: the upstream SDK runs
PP-DocLayoutV3 first and sends each crop separately.

Speed on Metal is unmeasured; every number below comes from MLX's CPU backend
in a Linux container.

## What the port is

MiniCPM-OCR 1.0 is MiniCPM-V 4.6's architecture: a SigLIP vision tower (27
layers, width 1152), a 2×2 merger, and a Qwen3.5 hybrid text model (18 Gated
DeltaNet layers, 6 full-attention). transformers declares the OCR classes as
empty subclasses of the 4.6 ones, run in the `"4x"` downsample mode, with an
image processor that sizes for that mode. mlx-vlm 0.7.10 ships a 4.6 port, so
[`minicpm_ocr.py`](minicpm_ocr.py) reuses it and fixes four things that break
for this checkpoint:

| mlx-vlm 0.7.10 `minicpmv4_6` | Effect on MiniCPM-OCR | Fix in `minicpm_ocr.py` |
|---|---|---|
| Image processor rounds sizes to `patch_size * 4` (the 16x merge) whatever `downsample_mode` says | Slices sized 16×68 patches where transformers uses 14×68; token counts differ | Merge unit from `downsample_mode` |
| Slice-grid search breaks ties differently from transformers | Different grids on some aspect ratios | transformers' search |
| Processor deletes the chat template's empty `<think>\n\n</think>\n\n` block | Prompt differs from what transformers and vLLM send | Keep it |
| CLI prompt wraps the image as `(<image>./</image>)` | Two stray parenthesis tokens around the image | Content-list format, rendered by the checkpoint's own chat template |

Converted checkpoints set `"model_file": "minicpm_ocr.py"` in `config.json`;
mlx-vlm imports that file in place of a built-in package, and importing it
registers the processor and prompt format for `model_type: minicpm_ocr`. One
harmless `ERROR ... Model type minicpm_ocr not supported` line still prints:
mlx-vlm's `load_image_processor` resolves the model without the checkpoint
path, ignores `model_file`, catches its own error and falls through to the
right processor.

## Results

Reference: transformers 5.21, fp32, CPU, greedy, on four synthetic crops in
[`results/`](results/): a paragraph (one slice), a wide page (thumbnail plus
three slices), the paragraph at 11 px (one slice) and a ruled table. Every MLX
checkpoint reproduces transformers' token ids and slice sizes exactly on all
four. The table gives, per checkpoint, teacher-forced top-1 agreement with
transformers' 350 answer tokens, their mean log-probability, and whether a free
greedy decode gives the same text.

| Checkpoint | Quantized | bf16 size | Teacher-forced top-1 (tokens) | Mean log p | Greedy text = transformers |
|---|---|---|---|---|---|
| `fp32` | none (fp32 reference port) | 4.9 GB (fp32) | 350/350 | -0.0025 | 4/4 |
| `q8-all` | 8-bit: MLP, attention, linear attention, embeddings | 1.90 GB | 350/350 | -0.0024 | not run |
| `q4-default` | 4-bit: MLP, attention (mlx-vlm 4.6 default) | 2.16 GB | 350/350 | -0.0055 | 4/4 |
| `q4-linattn` | 4-bit: + linear attention | 1.89 GB | 349/350 | -0.0078 | 3/4 |
| `q4-all` | 4-bit: + embeddings / tied LM head | 1.52 GB | 349/350 | -0.0066 | not run |

fp32 first-step logits against transformers: cosine 0.99958 to 0.999999, top-1
equal on all four ([`results/parity-fp32.json`](results/parity-fp32.json)).

The one disagreement in `q4-linattn` and `q4-all` is on the 11 px crop, where
transformers itself misreads "December" as "DeceMBER"; `q4-linattn`'s greedy
decode reads "December". It counts against the checkpoint here because the
reference is transformers, not the ground truth.

`q8-all` is the one to use: it matches fp32 on every token and is smaller than
mlx-vlm's default 4-bit recipe, which leaves the 254M-parameter embedding table
and the linear-attention layers in bf16 (13.3 bits per weight on average). The
vision tower and mergers (548M parameters) stay in bf16 in every variant,
about 1.1 GB, which sets the floor on size.

## Prior art

`nativ-community` published MLX conversions on 2026-10-27, from bf16 down to
4-bit, mxfp4 and nvfp4 (e.g.
[MiniCPM-OCR-1.0-4bit](https://huggingface.co/nativ-community/MiniCPM-OCR-1.0-4bit)).
Their READMEs say they need "an MLX-VLM build containing the MiniCPM-OCR port",
which is not bundled and is not in the PyPI release (0.7.10 has no
`minicpm_ocr` package), and they report smoke validation against their own
bf16 output (1/3 exact matches for 4-bit). This experiment differs in running
on the released mlx-vlm and in checking token ids and transcripts against
transformers. Whether their port has the processor issues above was not
checked.

## Caveats

- Four synthetic crops in one font: enough to show the port is exact, not a
  measure of OCR quality. The quantization ranking rests on 350 tokens.
- The sweep checkpoints keep their unquantized parts in fp32, because MLX's
  CPU backend has no fast bf16 path; the bf16 checkpoints you would ship (sizes
  in the table) add bf16 rounding on top and were not run.
- Greedy decoding was not run for `q8-all` and `q4-all`: quantized matmuls on
  CPU took about 13 minutes per variant for the page alone.
- `small_image_upsize="adaptive"` (opt-in, needs OpenCV glyph statistics and
  per-slice sizes) is not ported and raises `NotImplementedError`. The default,
  `"always"`, is.
- The DSpark draft model is not supported.
- `minicpm_ocr.py` subclasses mlx-vlm internals and was tested against 0.7.10
  only.
- On Linux, the mlx-cpu wheel bundles the netlib reference BLAS, 30× slower
  than OpenBLAS for fp32 matmul. Running with `LD_PRELOAD` pointing at any
  OpenBLAS (OpenCV's wheel ships one) gave a 10× speedup here.

## Files

| File | Role |
|---|---|
| [`minicpm_ocr.py`](minicpm_ocr.py) | The mlx-vlm model file: config, processor fixes, registration |
| [`convert.py`](convert.py) | HF checkpoint to MLX, with the parts to quantize selectable |
| [`reference.py`](reference.py) | Draws the test crops and records transformers' ids, logits and transcripts |
| [`parity.py`](parity.py) | Token ids, first-step logits and greedy text against the reference |
| [`sweep.py`](sweep.py) | Teacher-forced agreement, LM head applied only at answer positions |
| [`recheck.py`](recheck.py) | Under a minute: processor ids against the reference, and this README's table against `results/` |
| [`ERRORS.md`](ERRORS.md) | What went wrong along the way |
