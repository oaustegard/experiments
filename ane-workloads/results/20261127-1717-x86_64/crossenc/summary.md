# Cross-encoder on Core ML: Linux x86_64, 2026-11-27T17:17:17

Weights: stock encoder, random head. macOS None, AC None, low power None. torch 2.7.0+cpu, transformers 4.57.6, coremltools 9.0, numpy 2.3.5.

- PyTorch, plain norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 1.08e-06 (1.32e-06 against the model run unpadded); the same module in PyTorch fp16: 0.37778711318969727. Largest centred norm input 1150.46.
- PyTorch, prescaled norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 1.08e-06 (1.32e-06 against the model run unpadded); the same module in PyTorch fp16: 0.0031142234802246094. Largest centred norm input 1150.46.

## plain-b1: scorer-stock-plain-b1-s384-fp16.mlpackage

Converted in 4.9 s, 32.8 MB, 706 operations.


## plain-b16: scorer-stock-plain-b16-s384-fp16.mlpackage

Converted in 5.9 s, 32.8 MB, 706 operations.


## prescaled-b1: scorer-stock-prescaled-b1-s384-fp16.mlpackage

Converted in 6.0 s, 32.8 MB, 790 operations.


## prescaled-b16: scorer-stock-prescaled-b16-s384-fp16.mlpackage

Converted in 5.4 s, 32.8 MB, 790 operations.


Not run: prediction, placement and timing need macOS.
