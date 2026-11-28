# Cross-encoder on Core ML: Apple M5 Max, 2026-11-27T19:21:56

Weights: student (private). macOS 27.2.1, AC True, low power False. torch 2.7.0, transformers 4.57.6, coremltools 9.0, numpy 2.3.5.

- PyTorch, plain norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 4.05e-06 (4.05e-06 against the model run unpadded); the same module in PyTorch fp16: 0.6210954189300537. Largest centred norm input 1119.33.
- PyTorch, prescaled norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 4.05e-06 (4.05e-06 against the model run unpadded); the same module in PyTorch fp16: 0.0034517496824264526. Largest centred norm input 1119.33.

## plain-b1: scorer-student-plain-b1-s384-fp16.mlpackage

Converted in 2.0 s, 32.8 MB, 707 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 703}; cost share {'MLCPUComputeDevice': 0.2664, 'MLNeuralEngineComputeDevice': 0.7336}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 707}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 100, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 16, 'ios18.mul': 219, 'ios18.cast': 1, 'ios18.sub': 1, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios18.batch_norm': 1, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 0.959 | 1043.3 | 0.959 (0.985) | 1.672 / 1.684 | 1.63e-04 |  |
| idle | CPU_ONLY | 2.597 | 385.1 | 2.597 (2.635) | 0.162 / 0.147 | 6.91e-04 |  |
| busy | CPU_AND_NE | 1.0 | 1000.2 | 1.0 (1.057) | 1.95 / 1.914 | 1.63e-04 | 45.12 -> None (busy 1.0) |
| busy | CPU_ONLY | 3.713 | 269.3 | 3.713 (4.393) | 0.187 / 0.183 | 6.91e-04 | 45.12 -> None (busy 1.0) |

## plain-b16: scorer-student-plain-b16-s384-fp16.mlpackage

Converted in 2.0 s, 32.8 MB, 707 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 703}; cost share {'MLCPUComputeDevice': 0.0349, 'MLNeuralEngineComputeDevice': 0.9651}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 707}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 100, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 16, 'ios18.mul': 219, 'ios18.cast': 1, 'ios18.sub': 1, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios18.batch_norm': 1, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 1.06 | 943.0 | 16.968 (17.014) | 3.018 / 2.993 | 7.89e-04 |  |
| idle | CPU_ONLY | 1.811 | 552.1 | 28.983 (29.591) | 0.16 / 0.146 | 4.08e-03 |  |
| busy | CPU_AND_NE | 1.1 | 908.9 | 17.604 (19.097) | 4.438 / 4.022 | 7.89e-04 | 45.12 -> 41.68 (busy 0.731) |
| busy | CPU_ONLY | 3.402 | 293.9 | 54.438 (60.936) | 0.187 / 0.182 | 4.08e-03 | 45.12 -> None (busy 1.0) |

## prescaled-b1: scorer-student-prescaled-b1-s384-fp16.mlpackage

Converted in 2.0 s, 32.8 MB, 798 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 794}; cost share {'MLCPUComputeDevice': 0.2513, 'MLNeuralEngineComputeDevice': 0.7487}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 798}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 113, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 3, 'ios18.mul': 258, 'ios18.cast': 1, 'ios18.sub': 14, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios16.reduce_mean': 26, 'ios18.rsqrt': 13, 'ios18.batch_norm': 1, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 0.965 | 1036.0 | 0.965 (0.991) | 1.697 / 1.676 | 1.28e-05 |  |
| idle | CPU_ONLY | 2.856 | 350.1 | 2.856 (2.9) | 0.177 / 0.163 | 8.66e-04 |  |
| busy | CPU_AND_NE | 1.069 | 935.7 | 1.069 (1.147) | 2.969 / 2.576 | 1.28e-05 | 45.12 -> None (busy 1.0) |
| busy | CPU_ONLY | 4.357 | 229.5 | 4.357 (6.044) | 0.201 / 0.199 | 8.66e-04 | 45.12 -> None (busy 1.0) |

## prescaled-b16: scorer-student-prescaled-b16-s384-fp16.mlpackage

Converted in 2.1 s, 32.8 MB, 798 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 794}; cost share {'MLCPUComputeDevice': 0.0324, 'MLNeuralEngineComputeDevice': 0.9676}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 798}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 113, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 3, 'ios18.mul': 258, 'ios18.cast': 1, 'ios18.sub': 14, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios16.reduce_mean': 26, 'ios18.rsqrt': 13, 'ios18.batch_norm': 1, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 1.112 | 899.1 | 17.796 (17.832) | 3.333 / 3.315 | 1.35e-03 |  |
| idle | CPU_ONLY | 2.046 | 488.7 | 32.737 (33.236) | 0.165 / 0.16 | 3.81e-03 |  |
| busy | CPU_AND_NE | 1.157 | 864.2 | 18.515 (20.124) | 5.706 / 3.615 | 1.35e-03 | 45.12 -> None (busy 1.0) |
| busy | CPU_ONLY | 3.439 | 290.8 | 55.017 (62.072) | 0.197 / 0.195 | 3.81e-03 | 45.12 -> None (busy 1.0) |

