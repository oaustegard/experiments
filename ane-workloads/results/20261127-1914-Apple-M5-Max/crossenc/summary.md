# Cross-encoder on Core ML: Apple M5 Max, 2026-11-27T19:14:03

Weights: student (private). macOS 27.2.1, AC True, low power False. torch 2.7.0, transformers 4.57.6, coremltools 9.0, numpy 2.3.5.

- PyTorch, plain norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 4.89e-06 (4.89e-06 against the model run unpadded); the same module in PyTorch fp16: 0.407806396484375. Largest centred norm input 1131.36.
- PyTorch, prescaled norms against the Hugging Face model, 28 pairs of 50 to 384 tokens: max logit difference 4.89e-06 (4.89e-06 against the model run unpadded); the same module in PyTorch fp16: 0.002462923526763916. Largest centred norm input 1131.36.

## plain-b1: scorer-student-plain-b1-s384-fp16.mlpackage

Converted in 2.0 s, 32.8 MB, 707 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 703}; cost share {'MLCPUComputeDevice': 0.2664, 'MLNeuralEngineComputeDevice': 0.7336}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 707}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 100, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 16, 'ios18.mul': 219, 'ios18.cast': 1, 'ios18.sub': 1, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios18.batch_norm': 1, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 0.958 | 1044.2 | 0.958 (0.984) | 1.665 / 1.624 | 3.66e-04 |  |
| idle | CPU_ONLY | 2.589 | 386.3 | 2.589 (2.621) | 0.162 / 0.157 | 2.47e-04 |  |
| busy | CPU_AND_NE | 1.043 | 959.1 | 1.043 (1.087) | 1.923 / 1.951 | 3.66e-04 | 44.97 -> None (busy 1.0) |
| busy | CPU_ONLY | 3.734 | 267.8 | 3.734 (4.535) | 0.19 / 0.184 | 2.47e-04 | 44.97 -> None (busy 1.0) |

## plain-b16: scorer-student-plain-b16-s384-fp16.mlpackage

Converted in 2.0 s, 32.8 MB, 707 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 703}; cost share {'MLCPUComputeDevice': 0.0349, 'MLNeuralEngineComputeDevice': 0.9651}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 707}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 100, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 16, 'ios18.mul': 219, 'ios18.cast': 1, 'ios18.sub': 1, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios18.batch_norm': 1, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 1.062 | 941.4 | 16.995 (17.135) | 2.994 / 3.031 | 8.62e-04 |  |
| idle | CPU_ONLY | 1.813 | 551.6 | 29.009 (29.526) | 0.165 / 0.164 | 3.86e-03 |  |
| busy | CPU_AND_NE | 1.111 | 900.5 | 17.768 (20.288) | 4.422 / 4.174 | 8.62e-04 | 44.97 -> 41.84 (busy 0.712) |
| busy | CPU_ONLY | 3.434 | 291.2 | 54.941 (59.244) | 0.192 / 0.182 | 3.86e-03 | 44.97 -> None (busy 1.0) |

## prescaled-b1: scorer-student-prescaled-b1-s384-fp16.mlpackage

Converted in 2.0 s, 32.8 MB, 798 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 794}; cost share {'MLCPUComputeDevice': 0.2513, 'MLNeuralEngineComputeDevice': 0.7487}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 798}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 113, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 3, 'ios18.mul': 258, 'ios18.cast': 1, 'ios18.sub': 14, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios16.reduce_mean': 26, 'ios18.rsqrt': 13, 'ios18.batch_norm': 1, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 0.956 | 1046.2 | 0.956 (0.986) | 1.747 / 1.738 | 3.07e-04 |  |
| idle | CPU_ONLY | 2.847 | 351.2 | 2.847 (2.884) | 0.177 / 0.174 | 1.87e-04 |  |
| busy | CPU_AND_NE | 1.073 | 932.3 | 1.073 (1.143) | 2.993 / 2.561 | 3.07e-04 | 44.97 -> None (busy 1.0) |
| busy | CPU_ONLY | 4.166 | 240.0 | 4.166 (4.953) | 0.186 / 0.184 | 1.87e-04 | 44.97 -> None (busy 1.0) |

## prescaled-b16: scorer-student-prescaled-b16-s384-fp16.mlpackage

Converted in 2.1 s, 32.8 MB, 798 operations.

- Placement with CPU_AND_NE: {'MLCPUComputeDevice': 4, 'MLNeuralEngineComputeDevice': 794}; cost share {'MLCPUComputeDevice': 0.0324, 'MLNeuralEngineComputeDevice': 0.9676}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 1, 'select': 1, 'ios18.gather': 1}}
- Placement with CPU_ONLY: {'MLCPUComputeDevice': 798}; cost share {'MLCPUComputeDevice': 1.0}; off the Neural Engine: {'MLCPUComputeDevice': {'ios18.greater_equal': 1, 'ios18.add': 113, 'select': 1, 'ios18.gather': 1, 'ios18.transpose': 29, 'ios18.expand_dims': 3, 'ios18.layer_norm': 3, 'ios18.mul': 258, 'ios18.cast': 1, 'ios18.sub': 14, 'ios18.conv': 30, 'split': 35, 'ios18.slice_by_index': 113, 'ios18.concat': 63, 'ios16.einsum': 56, 'ios18.softmax': 28, 'ios18.gelu': 8, 'ios16.reduce_mean': 26, 'ios18.rsqrt': 13, 'ios18.batch_norm': 1, 'ios18.reshape': 1}}

| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | max prob diff vs PyTorch | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|
| idle | CPU_AND_NE | 1.112 | 899.1 | 17.795 (17.846) | 3.35 / 3.328 | 1.11e-03 |  |
| idle | CPU_ONLY | 1.984 | 504.0 | 31.748 (32.273) | 0.175 / 0.177 | 3.27e-03 |  |
| busy | CPU_AND_NE | 1.153 | 867.4 | 18.446 (19.739) | 4.819 / 3.8 | 1.11e-03 | 44.97 -> None (busy 1.0) |
| busy | CPU_ONLY | 3.355 | 298.0 | 53.683 (61.142) | 0.194 / 0.194 | 3.27e-03 | 44.97 -> None (busy 1.0) |

