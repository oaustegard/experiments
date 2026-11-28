# Task 2 (rebuild extension): tensor `kernel_mul_mm` tile constants

Fork 43f31f400c, PQ2_0, tensor API on. Baseline = mean of the first and last stock runs: pp2048 700.1, pp8192 616.7 tok/s. TFLOPS = 2 x 24.35e9 x tok/s. Perplexity: `llama-perplexity -c 2048 --chunks 6` on the first 90 kB of METHODS.md.

| variant | constants | pp2048 tok/s (TFLOPS) | vs stock | pp8192 tok/s (TFLOPS) | vs stock | perplexity |
|---|---|---|---|---|---|---|
| stock | stock | 756.9 (36.9) | +8.1% | 635.0 (30.9) | +3.0% | 14.3652 |
| by4 | N_MM_BLOCK_Y=4 | 313.0 (15.2) | -55.3% | 308.4 (15.0) | -50.0% | 14.3652 |
| nk4 | N_MM_NK=4 | 636.0 (31.0) | -9.2% | 570.7 (27.8) | -7.5% | 14.3652 |
| nk8 | N_MM_NK=8 | 631.3 (30.7) | -9.8% | 588.5 (28.7) | -4.6% | 14.3652 |
| by4nk4 | N_MM_BLOCK_Y=4, N_MM_NK=4 | 355.9 (17.3) | -49.2% | 327.3 (15.9) | -46.9% | 14.3652 |
| by4nk8 | N_MM_BLOCK_Y=4, N_MM_NK=8 | 303.1 (14.8) | -56.7% | 321.5 (15.7) | -47.9% | 14.3652 |
| stock-again | stock | 643.3 (31.3) | -8.1% | 598.3 (29.1) | -3.0% | None |
