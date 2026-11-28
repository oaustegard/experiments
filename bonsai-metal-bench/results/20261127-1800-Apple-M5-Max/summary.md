# Bonsai 2.5 27B on Apple M5 Max — 2026-11-27T18:00:06

- memory 128 GiB, GPU cores 40, macOS 27.2.1, AC power True, low power mode False, wired limit 0 MB

Bandwidth probe (best of matvec/sum over 4 GiB): **549.6 GB/s**

## llama.cpp (PrismML fork 43f31f400c)

| packing | file GiB | pp512 | pp2048 | pp8192 | tg128 @0 | @8192 | @32768 | ceiling @0 | % of ceiling @0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PQ2_0 | 6.76 | 809.1 | 746.2 | 668.1 | 47.3 | 44.7 | 35.7 | 72.7 | 65% |
| PTQ1_0 | 5.58 | 585.1 | 650.9 | 635.4 | 44.3 | 38.7 | 32.0 | 87.3 | 51% |

Prefill ubatch sweep, PQ2_0, 8192-token prompt: ub 256: 622 tok/s, ub 512: 643 tok/s, ub 1024: 636 tok/s, ub 2048: 619 tok/s

Decode ceiling = bandwidth / (file bytes + GDN state read+write + 64 KiB x depth of fp16 KV). Using the whole file overstates bytes slightly (the embedding table is not read per token).

## Contention: GPU (llama-bench, PQ2_0) with the ANE and CPU loaded

Synthetic encoder (4.228 GFLOP/call): cpu_only 1.543 ms, cpu_gpu 1.504 ms, cpu_ne 0.524 ms. Op placement with CPU_AND_NE: RuntimeError('{\n    NSLocalizedDescription = "The model is not found at URL: file:///var/folders/3j/zrk6d42d15x4vswfldr2b714f7z28s/T/synthetic_A2B84F80-63B6-45FD-910C-1E1EBBB82830.mlmodelc";\n}')

CPU load processes: 6. Baseline is the mean of the first and last runs; their gap shows thermal drift.

| condition | pp2048 tok/s | vs base | tg128 tok/s | vs base | background alone → during the GPU run |
|---|---:|---:|---:|---:|---|
| baseline | 698.8 | -5.0% | 47.6 | +0.3% | – |
| ane | 746.1 | +1.4% | 45.3 | -4.6% | ane: 1902.99 → 1827.49 calls/s |
| cpu_mem | 557.2 | -24.3% | 29.7 | -37.4% | cpu_mem: 406.3 → 305.33 GB/s |
| cpu_compute | 725.6 | -1.4% | 45.4 | -4.3% | cpu_compute: 4536.27 → 3289.28 Melem/s |
| cpu_matmul | 707.3 | -3.9% | 45.5 | -4.2% | cpu_matmul: 2993.84 → 2164.56 GFLOP/s |
| ane+cpu_mem | 551.7 | -25.0% | 29.6 | -37.6% | ane: 1917.08 → 1390.83 calls/s; cpu_mem: 391.15 → 295.49 GB/s |
| baseline_end | 773.0 | +5.0% | 47.3 | -0.3% | – |

The 'during' rate covers llama-bench's whole run, including its model load, when the GPU is idle, so it slightly understates the slowdown of the loads.

