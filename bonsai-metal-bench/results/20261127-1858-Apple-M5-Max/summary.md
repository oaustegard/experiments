# Bonsai 2.5 27B on Apple M5 Max — 2026-11-27T18:58:36

- memory 128 GiB, GPU cores 40, macOS 27.2.1, AC power True, low power mode False, wired limit 0 MB

Bandwidth probe (best of matvec/sum over 4 GiB): **None GB/s**

## Contention: GPU (llama-bench, PQ2_0) with the ANE and CPU loaded

Synthetic encoder (4.228 GFLOP/call): cpu_only 1.539 ms, cpu_gpu 0.722 ms, cpu_ne 0.503 ms. Op placement with CPU_AND_NE: {'none': 140, 'MLNeuralEngineComputeDevice': 74}

CPU load processes: 6. Baseline is the mean of the first and last runs; their gap shows thermal drift.

| condition | pp2048 tok/s | vs base | tg128 tok/s | vs base | background alone → during the GPU run |
|---|---:|---:|---:|---:|---|
| baseline | 673.2 | -6.3% | 44.0 | -2.4% | – |
| ane | 743.5 | +3.5% | 45.5 | +1.0% | ane: 1884.13 → 1823.51 calls/s |
| cpu_mem | 570.3 | -20.6% | 30.1 | -33.2% | cpu_mem: 407.51 → 305.17 GB/s |
| cpu_compute | 717.7 | -0.1% | 45.2 | +0.3% | cpu_compute: 4575.19 → 3361.27 Melem/s |
| cpu_matmul | 658.0 | -8.4% | 45.7 | +1.3% | cpu_matmul: 2975.09 → 2439.75 GFLOP/s |
| ane+cpu_mem | 483.1 | -32.8% | 28.8 | -36.1% | ane: 1928.71 → 1513.78 calls/s; cpu_mem: 394.43 → 304.1 GB/s |
| baseline_end | 764.1 | +6.3% | 46.1 | +2.4% | – |

The 'during' rate covers llama-bench's whole run, including its model load, when the GPU is idle, so it slightly understates the slowdown of the loads.

