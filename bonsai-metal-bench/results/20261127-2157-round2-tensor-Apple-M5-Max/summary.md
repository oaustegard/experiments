# Task 2: prefill with the Metal tensor API on and off

Apple chip: Apple M5 Max. Fork commit 43f31f400c; upstream {"tag": "b12823", "commit": "3ab385146cad1a374dadc40e042dcc50edceb046"}. TFLOPS = 2 x 24.35e9 x tok/s.

| build | file | tensor API | pp512 tok/s (TFLOPS) | pp2048 | pp8192 |
|---|---|---|---|---|---|
| fork | PQ2_0 | on | 785.4 (38.2) | 737.7 (35.9) | 657.7 (32.0) |
| fork | PQ2_0 | off | 242.1 (11.8) | 249.4 (12.1) | 243.5 (11.9) |
| fork | PTQ1_0 | on | 540.8 (26.3) | 622.5 (30.3) | 608.9 (29.7) |
| fork | PTQ1_0 | off | 215.6 (10.5) | 217.8 (10.6) | 197.9 (9.6) |
| upstream | Q2_0 | on | 700.6 (34.1) | 652.0 (31.8) | 620.1 (30.2) |
| upstream | Q2_0 | off | 230.2 (11.2) | 237.0 (11.5) | 236.8 (11.5) |

Prefill ratio, tensor on / off:

- fork PQ2_0: pp512 3.24x, pp2048 2.96x, pp8192 2.70x
- fork PTQ1_0: pp512 2.51x, pp2048 2.86x, pp8192 3.08x
- upstream Q2_0: pp512 3.04x, pp2048 2.75x, pp8192 2.62x

Ubatch sweep, fork PQ2_0, 8192-token prompt (tok/s):

| tensor API | ub 256 | ub 512 | ub 1024 | ub 2048 |
|---|---|---|---|---|
| on | 646.6 | 602.0 | 576.7 | 610.7 |
| off | 236.9 | 238.1 | 237.6 | 226.6 |

Matmul pipelines requested (`llama-bench -v`, one 512-token prefill each; `kernels.json`):

- `fork-PQ2_0-tensor_on`: kernel_mul_mm_f32_f32, kernel_mul_mm_pq2_0_f32, kernel_mul_mv_pq2_0_f32
- `fork-PQ2_0-tensor_off`: kernel_mul_mm_f32_f32, kernel_mul_mm_pq2_0_f32, kernel_mul_mv_pq2_0_f32
- `fork-PTQ1_0-tensor_on`: kernel_mul_mm_f32_f32, kernel_mul_mm_ptq1_0_f32, kernel_mul_mv_ptq1_0_f32
- `fork-PTQ1_0-tensor_off`: kernel_mul_mm_f32_f32, kernel_mul_mm_ptq1_0_f32, kernel_mul_mv_ptq1_0_f32
- `upstream-Q2_0-tensor_on`: kernel_mul_mm_tensor_f32_f32, kernel_mul_mm_tensor_q2_0_f32, kernel_mul_mv_q2_0_f32
- `upstream-Q2_0-tensor_off`: kernel_mul_mm_f32_f32, kernel_mul_mm_q2_0_f32, kernel_mul_mv_q2_0_f32
