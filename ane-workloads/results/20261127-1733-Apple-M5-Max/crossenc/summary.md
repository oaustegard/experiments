# Cross-encoder on Core ML: Apple M5 Max, 2026-11-27T17:33:48

Weights: stock encoder, random head. macOS 27.2.1, AC True, low power False. torch 2.7.0, transformers 4.57.6, coremltools 9.0, numpy 2.3.5.

- PyTorch, plain norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 2.97e-06 (3.00e-06 against the model run unpadded); the same module in PyTorch fp16: 0.3772985339164734. Largest centred norm input 1150.46.
- PyTorch, prescaled norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 2.97e-06 (3.00e-06 against the model run unpadded); the same module in PyTorch fp16: 0.00616706907749176. Largest centred norm input 1150.46.

## plain-b1: scorer-stock-plain-b1-s384-fp16.mlpackage

Converted in 2.0 s, 32.8 MB, 706 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 702}; cost share {'MLCPUComputeDevice': 0.2666, 'MLNeuralEngineComputeDevice': 0.7334}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 706}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 100, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 16, 'ios18.mul': 219, 'ios18.cast': 1, 'ios18.sub': 1, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 0.956 | 1045.5 | 0.956 (0.985) | 1.699 / 1.692 | 7.54e-04 |  |
| idle | CPU_ONLY | 2.513 | 398.0 | 2.513 (2.566) | 0.165 / 0.148 | 2.09e-04 |  |

## prescaled-b1: scorer-stock-prescaled-b1-s384-fp16.mlpackage

Converted in 2.1 s, 32.8 MB, 790 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 786}; cost share {'MLCPUComputeDevice': 0.2526, 'MLNeuralEngineComputeDevice': 0.7474}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 790}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 112, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 4, 'ios18.mul': 255, 'ios18.cast': 1, 'ios18.sub': 13, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios16.reduce_mean': 24, 'ios18.rsqrt': 12, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 0.948 | 1054.8 | 0.948 (0.977) | 1.758 / 1.764 | 7.24e-04 |  |
| idle | CPU_ONLY | 2.718 | 367.9 | 2.718 (2.927) | 0.197 / 0.178 | 4.53e-04 |  |

