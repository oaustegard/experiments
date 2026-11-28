# Bonsai 2.5 27B on Apple M5 Max — 2026-11-27T18:30:17

- memory 128 GiB, GPU cores 40, macOS 27.2.1, AC power True, low power mode False, wired limit 0 MB

Bandwidth probe (best of matvec/sum over 4 GiB): **None GB/s**

## llama-server MTP A/B (PQ2_0-mtp, released Metal preset, greedy)

| | decode tok/s (median, 6 prompts) | prefill tok/s (8K prompt) | draft acceptance |
|---|---:|---:|---:|
| mtp_off | 45.79 | 632.3 | None |
| mtp_on | 54.33 | 672.7 | 0.914 |

MTP decode ratio: **1.187**

## MLX pack (mlx 0.32.0)

- 512-token prompt: prefill 836 tok/s, decode 48.8 tok/s, peak 10.3 GB
- 2048-token prompt: prefill 738 tok/s, decode 47.1 tok/s, peak 12.8 GB
- 8192-token prompt: prefill 718 tok/s, decode 46.9 tok/s, peak 13.0 GB
- 32768-token prompt: prefill 639 tok/s, decode 37.0 tok/s, peak 17.3 GB
- MTP head (draft block 2): decode 43.9 -> 48.9 tok/s (1.11x)

