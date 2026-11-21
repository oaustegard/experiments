# ane-dettin

dEttin-17M (the browser-decider model: fully fine-tuned Ettin-17M, 7 layers, hidden 256) ported to Core ML and run on the Apple Neural Engine. Measured 2026-11-16 on an M1 MacBook Air (8 GB, macOS 27).

## Method
`ane_model.py` rewrites the model in the Apple ane-transformers layout: (1,C,1,S) activations, a 1x1 Conv2d for every Linear, per-head einsum attention, fixed sequence length, RoPE tables and the local-window mask (|i-j| <= 64) baked in as constants. The head runs at every position; logits are read at the [MASK] positions outside the model. `convert.py` produces an fp16 mlprogram with coremltools 8.3.0 (torch 2.5.1, transformers 4.51.3). `parity.py` compares against HF ModernBertModel (max |dp| 7e-07 in torch). `bench.py` times 500 Typed Decisions questions, one per predict call.

## Results (500 questions, token median 307)
| S | CPU | CPU+GPU | CPU+NE |
|---|---|---|---|
| 512 | 6.13 ms | 14.7 ms | 2.92 ms |
| 256 | 2.53 ms | 8.0 ms | 1.21 ms |

Accuracy at S=512 is 0.747 against 0.750 for the fp32 reference. 715 of 716 ops prefer the Neural Engine; only the embedding gather stays on CPU. First load with the Neural Engine takes 3.4-4.2 s (compile).

## Not included
The converted `.mlpackage` files (34 MB each), `pack/weights.bin`, `tokenizer.json` and `cases.json` are left out; they regenerate from the browser-decider pack. `src/dettin.py` is the reference decoder from dettin-68m.

## Not done
Spikes/LoRA (multifunction models), power draw, Swift/native timing, the 68M model, enumerated shapes.
