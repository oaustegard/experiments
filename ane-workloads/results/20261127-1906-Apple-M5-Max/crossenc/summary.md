# Cross-encoder on Core ML: Apple M5 Max, 2026-11-27T19:06:05

Weights: stock encoder, random head. macOS 27.2.1, AC True, low power False. torch 2.7.0, transformers 4.57.6, coremltools 9.0, numpy 2.3.5.

- PyTorch, plain norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 2.97e-06 (3.00e-06 against the model run unpadded); the same module in PyTorch fp16: 0.3772985339164734. Largest centred norm input 1150.46.
- PyTorch, prescaled norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 2.97e-06 (3.00e-06 against the model run unpadded); the same module in PyTorch fp16: 0.00616706907749176. Largest centred norm input 1150.46.

## plain-b1: scorer-stock-plain-b1-s384-fp16.mlpackage

Converted in 2.0 s, 32.8 MB, 706 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 702}; cost share {'MLCPUComputeDevice': 0.2666, 'MLNeuralEngineComputeDevice': 0.7334}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 706}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 100, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 16, 'ios18.mul': 219, 'ios18.cast': 1, 'ios18.sub': 1, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 0.958 | 1043.6 | 0.958 (0.987) | 1.678 / 1.683 | 7.54e-04 |  |
| idle | CPU_ONLY | 2.507 | 398.9 | 2.507 (2.539) | 0.167 / 0.162 | 2.09e-04 |  |
| busy | CPU_AND_NE | 1.011 | 989.0 | 1.011 (1.065) | 1.943 / 1.922 | 7.54e-04 | 45.53 -> None (busy 1.0) |
| busy | CPU_ONLY | 3.729 | 268.2 | 3.729 (4.534) | 0.182 / 0.188 | 2.09e-04 | 45.53 -> None (busy 1.0) |

## plain-b16: scorer-stock-plain-b16-s384-fp16.mlpackage

Converted in 2.0 s, 32.8 MB, 706 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 702}; cost share {'MLCPUComputeDevice': 0.0349, 'MLNeuralEngineComputeDevice': 0.9651}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 706}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 100, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 16, 'ios18.mul': 219, 'ios18.cast': 1, 'ios18.sub': 1, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 1.062 | 941.9 | 16.988 (17.026) | 3.098 / 3.067 | 1.34e-03 |  |
| idle | CPU_ONLY | 1.771 | 564.7 | 28.333 (28.9) | 0.173 / 0.168 | 2.19e-03 |  |
| busy | CPU_AND_NE | 1.111 | 900.1 | 17.777 (20.544) | 4.854 / 4.371 | 1.34e-03 | 45.53 -> 40.83 (busy 0.805) |
| busy | CPU_ONLY | 3.594 | 278.2 | 57.506 (61.719) | 0.169 / 0.168 | 2.19e-03 | 45.53 -> None (busy 1.0) |

## prescaled-b1: scorer-stock-prescaled-b1-s384-fp16.mlpackage

Converted in 2.1 s, 32.8 MB, 790 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 786}; cost share {'MLCPUComputeDevice': 0.2526, 'MLNeuralEngineComputeDevice': 0.7474}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 790}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 112, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 4, 'ios18.mul': 255, 'ios18.cast': 1, 'ios18.sub': 13, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios16.reduce_mean': 24, 'ios18.rsqrt': 12, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 0.952 | 1050.1 | 0.952 (0.988) | 1.705 / 1.694 | 7.24e-04 |  |
| idle | CPU_ONLY | 2.754 | 363.1 | 2.754 (2.787) | 0.178 / 0.165 | 4.53e-04 |  |
| busy | CPU_AND_NE | 1.091 | 916.7 | 1.091 (1.266) | 3.15 / 2.634 | 7.24e-04 | 45.53 -> None (busy 1.0) |
| busy | CPU_ONLY | 4.279 | 233.7 | 4.279 (5.005) | 0.205 / 0.199 | 4.53e-04 | 45.53 -> None (busy 1.0) |

## prescaled-b16: scorer-stock-prescaled-b16-s384-fp16.mlpackage

Converted in 2.1 s, 32.8 MB, 790 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 786}; cost share {'MLCPUComputeDevice': 0.0325, 'MLNeuralEngineComputeDevice': 0.9675}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 790}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 112, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 4, 'ios18.mul': 255, 'ios18.cast': 1, 'ios18.sub': 13, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios16.reduce_mean': 24, 'ios18.rsqrt': 12, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 1.112 | 899.3 | 17.791 (17.841) | 3.367 / 3.328 | 1.51e-03 |  |
| idle | CPU_ONLY | 1.965 | 508.9 | 31.439 (31.894) | 0.177 / 0.176 | 2.03e-03 |  |
| busy | CPU_AND_NE | 1.152 | 867.8 | 18.438 (21.18) | 5.2 / 3.443 | 1.51e-03 | 45.53 -> None (busy 1.0) |
| busy | CPU_ONLY | 3.336 | 299.8 | 53.374 (61.659) | 0.199 / 0.196 | 2.03e-03 | 45.53 -> None (busy 1.0) |

