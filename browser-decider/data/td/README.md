---
license: apache-2.0
language:
- en
pretty_name: Typed Decisions
size_categories:
- n<1K
task_categories:
- text-classification
tags:
- structured-decisions
- calibration
- probabilistic-classification
- system-one
- workflow-evaluation
- synthetic
configs:
- config_name: agent_trace_observability
  data_files:
  - split: train
    path: agent_trace_observability/train-*.parquet
  - split: test
    path: agent_trace_observability/test-*.parquet
- config_name: customer_service
  data_files:
  - split: train
    path: customer_service/train-*.parquet
  - split: test
    path: customer_service/test-*.parquet
- config_name: invoice_processing
  data_files:
  - split: train
    path: invoice_processing/train-*.parquet
  - split: test
    path: invoice_processing/test-*.parquet
- config_name: security_incidents
  data_files:
  - split: train
    path: security_incidents/train-*.parquet
  - split: test
    path: security_incidents/test-*.parquet
- config_name: all
  data_files:
  - split: train
    path: all/train-*.parquet
  - split: test
    path: all/test-*.parquet
---

# Typed Decisions

A benchmark for typed probabilistic decisions. A model gets one piece of
unstructured state and answers five typed questions about it at once, and every
answer is a probability distribution, not a single label.

The schema follows the System One primitives (`noul`, `choice`, `score`) used by
[TypeSafe AI](https://typesafe.ai), so a row replays against any API with that
shape. The benchmark is independent: it is not affiliated with TypeSafe and does
not reproduce their Jev model.

![Accuracy against KL divergence and against latency for every scored model](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/assets/leaderboard.png)

## Leaderboard

`test` split: 400 cases, 2,000 decisions. This table lists general models scored
zero-shot: they have never seen these workflows or their question schemas.
Models that were fitted or fine-tuned on `train` are in the second table below;
the two tables are not comparable.

| # | Model | Kind | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ | p50 latency | Price / 1M input |
|---|---|---|---|---|---|---|---|---|
| 1 | **[Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large)** (31B) via [Decision Models](https://decisionmodels.io) (`quyet-1-large`) ([#38](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/38)) | general, zero-shot, open weights | **0.791** | 0.281 | 0.114 | 0.093 | 3,335 ms | $0.035 |
| 2 | [meraGPT Decider 1](https://meragpt.com/models/state-decider-1?utm_source=huggingface&utm_medium=dataset_card&utm_campaign=typed-decisions) (`sd-1`) | general, zero-shot | 0.768 | **0.096** | **0.052** | 0.180 | **526 ms** | $0.03 |
| 3 | [Baseten Mammoth 1](https://openrouter.ai/baseten/mammoth-1) (27B, from Qwen3.5-27B) via OpenRouter (`baseten/mammoth-1:free`) | general, zero-shot | 0.768 | 0.122 | 0.063 | 0.153 | 415 ms | free tier |
| 4 | [Amos 1 medium](https://huggingface.co/mukaddamzaid/amos-1-medium) (9B) ([#31](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/31)) | general, zero-shot, open weights | 0.7645 | 0.151 | 0.080 | 0.103 | 118 ms (H100) | open weights |
| 5 | [Amos 1 large](https://huggingface.co/mukaddamzaid/amos-1-large) (27B) ([#31](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/31)) | general, zero-shot, open weights | 0.761 | 0.126 | 0.069 | 0.097 | 357 ms (H100) | open weights |
| 6 | [pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) (27B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.753 | 0.441 | 0.160 | 0.060 | – | open weights |
| 7 | [FastBrain-9B](https://huggingface.co/TypeScript/FastBrain-9B) (9.4B, Qwen3.5-9B + 116M decision head) ([#57](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/57)) | general, zero-shot, open weights | 0.7525 | 0.151 | 0.080 | 0.085 | 50 ms (RTX PRO 6000) | open weights |
| 8 | [d3](https://huggingface.co/vllm-sr/d3) (27B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.749 | 0.409 | 0.146 | 0.049 | – | open weights |
| 9 | [Baseten Elephant 1](https://openrouter.ai/baseten/elephant-1) (9B, from Qwen3.5-9B) via OpenRouter (`baseten/elephant-1:free`) | general, zero-shot | 0.748 | 0.134 | 0.069 | 0.162 | 352 ms | free tier |
| 10 | [Darwin-27B-ZTC](https://huggingface.co/FINAL-Bench/Darwin-27B-ZTC) (27B) ([#17](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/17)) | general, zero-shot, open weights | 0.743 | 0.204 | 0.097 | – | – | open weights |
| 11 | [d3-mini](https://huggingface.co/vllm-sr/d3-mini) (4B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.743 | 0.319 | 0.123 | 0.047 | – | open weights |
| 12 | [Liquid AI d1](https://docs.liquid.ai/lfm/models/decision-models) (`d1:free`) | general, zero-shot | 0.742 | 0.475 | 0.155 | 0.124 | 525 ms | free tier |
| 13 | [linne-oak 1.0](https://huggingface.co/PNPM244/linne-oak-1-0) (31B) ([#35](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/35)) | general, zero-shot, open weights | 0.739 | 0.709 | 0.191 | – | 1.3 s (8 concurrent) | open weights |
| 14 | [vLLM-SR d3](https://huggingface.co/vllm-sr/d3) (26B, FP8) via [Decision Models](https://decisionmodels.io) (`s1-pro`) ([#38](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/38)) | general, zero-shot, open weights | 0.739 | 0.437 | 0.151 | 0.124 | 3,405 ms | $0.08 |
| 15 | [d3-turbo](https://huggingface.co/vllm-sr/d3-turbo) (26B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.736 | 0.438 | 0.162 | 0.069 | – | open weights |
| 16 | [clef](https://huggingface.co/Cloudflare/clef) (27B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.728 | 0.191 | 0.103 | 0.025 | – | open weights |
| 17 | [openjev](https://huggingface.co/openjev/openjev) (27B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.728 | 0.506 | 0.191 | 0.101 | – | open weights |
| 18 | TypeSafe Jev 1.13.0 | general, zero-shot | 0.727 | 1.442 | 0.148 | 0.144 | 710 ms | $0.042 |
| 19 | [Blink v0.3](https://huggingface.co/PixilabAI/Blink-v0.3-26B-A4B-NVFP4) (26B-A4B, NVFP4) via [Decision Models](https://decisionmodels.io) (`s1-fast`) ([#38](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/38)) | general, zero-shot, open weights | 0.727 | 0.153 | 0.079 | 0.129 | 3,274 ms | $0.019 |
| 20 | [d3-flash](https://huggingface.co/vllm-sr/d3-flash) (9B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.717 | 0.322 | 0.131 | 0.031 | – | open weights |
| 21 | [Featherless Simple Jev](https://simple-jev.featherless.ai) (`Qwen3.6-35B-A3B-classifier`) | general, zero-shot | 0.716 | 0.488 | 0.176 | – | – | free demo |
| 22 | [jev-vision-27b](https://huggingface.co/e12ex2/jev-vision-27b) (Qwen3.8-27B + LoRA) | general, zero-shot, open weights | 0.714 | 0.318 | 0.129 | 0.026 | 199 ms (H100) | open weights |
| 23 | [signpost-d-4b](https://huggingface.co/skamalj/signpost-d-4b) (4.5B, Qwen3.5-4B LoRA soup) ([#61](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/61)) | general, zero-shot, open weights | 0.712 | 0.162 | 0.085 | 0.091 | 62 ms (RTX PRO 6000) | open weights |
| 24 | [clef-flash](https://huggingface.co/Cloudflare/clef-flash) (9B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.710 | 0.209 | 0.111 | 0.016 | – | open weights |
| 25 | [prima-ratio + 12B](https://github.com/andrea-tomassi/prima-ratio) | general, zero-shot | 0.702 | 0.564 | 0.234 | 0.146 | 700 ms | local GGUF |
| 26 | [JEV-35B](https://huggingface.co/autotrust/JEV-35B) (35B MoE) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.700 | 0.526 | 0.169 | 0.062 | – | open weights |
| 27 | [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) (28B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.688 | 0.544 | 0.194 | 0.113 | – | open weights |
| 28 | [clef-omni](https://huggingface.co/Cloudflare/clef-omni) (35B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.673 | 0.247 | 0.123 | 0.030 | – | open weights |
| 29 | [OpenDecider-small](https://huggingface.co/manjunathshiva/opendecider-small) | general, zero-shot, open weights | 0.671 | 0.211 | 0.117 | – | 40 ms | open weights |
| 30 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (27B, base model) ([#60](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/60)) | general, zero-shot, open weights | 0.661 | 0.590 | 0.214 | 0.143 | 381 ms (RTX PRO 6000) | open weights |
| 31 | [d1-3B](https://huggingface.co/LiquidAI/d1-3B) (3B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.656 | 0.286 | 0.144 | 0.037 | – | open weights |
| 32 | [d1-2B-OptiQ-4bit](https://huggingface.co/mlx-community/d1-2B-OptiQ-4bit) (2B, OptiQ 4-bit MLX quant of d1-2B) | general, zero-shot, open weights | 0.641 | 0.283 | 0.139 | 0.126 | 273 ms (M3 Max) | open weights |
| 33 | [d1-2B](https://huggingface.co/LiquidAI/d1-2B) (2B) | general, zero-shot, open weights | 0.637 | 0.282 | 0.137 | 0.124 | – | open weights |
| 34 | [d3-nano](https://huggingface.co/vllm-sr/d3-nano) (2B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.629 | 0.545 | 0.211 | 0.113 | – | open weights |
| 35 | [Bongard-mini](https://huggingface.co/AgentBull/bongard-mini) | general, zero-shot, open weights | 0.594 | 0.256 | 0.132 | 0.067 | 225 ms | open weights |
| 36 | [Jeff-Gemma4-E2B](https://huggingface.co/mstrasser/Jeff-Gemma4-E2B) | general, zero-shot, open weights | 0.561 | 0.403 | 0.219 | 0.188 | 2,272 ms | open weights |
| 37 | [d3-lite](https://huggingface.co/vllm-sr/d3-lite) (0.8B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.531 | 0.518 | 0.244 | 0.130 | – | open weights |
| 38 | [GEV-26B-Decide](https://huggingface.co/autotrust/GEV-26B-Decide) (26B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.522 | 0.423 | 0.215 | 0.066 | – | open weights |
| 39 | [Jeff-Qwen3.5-2B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-2B) | general, zero-shot, open weights | 0.511 | 0.460 | 0.237 | 0.203 | 1,346 ms | open weights |
| 40 | [JEV-27B](https://huggingface.co/autotrust/JEV-27B) (27B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.495 | 0.550 | 0.253 | 0.136 | – | open weights |
| 41 | [Jeff-Qwen3.5-0.8B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B) | general, zero-shot, open weights | 0.483 | 0.679 | 0.313 | 0.251 | 662 ms | open weights |
| 42 | [d3-edge](https://huggingface.co/vllm-sr/d3-edge) (0.6B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.425 | 1.176 | 0.456 | 0.341 | – | open weights |
| 43 | [THX-01](https://huggingface.co/doofz/THX-01) (0.3B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | general, zero-shot, open weights | 0.377 | 0.620 | 0.320 | 0.203 | – | open weights |
| – | Prior (ignores the input) | reference | 0.470 | 0.347 | 0.189 | 0.088 | – | – |
| – | Uniform (same probability on every option) | reference | 0.308 | 0.444 | 0.238 | 0.169 | – | – |

### Fitted or fine-tuned on `train`

These models were trained on this benchmark's `train` split, so they are not
zero-shot and their scores are not comparable with the table above.

| # | Model | Kind | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ | Reported latency |
|---|---|---|---|---|---|---|---|
| 1 | [Decider-4B](https://huggingface.co/roudone/Decider-4B) (Qwen3.5-4B + LoRA) ([#36](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/36)) | general, then fine-tuned on `train` and other decision data, temperature-calibrated | 0.826 | 0.0803 | 0.0430 | 0.160 | 107 ms |
| 2 | [Arbiter-4B](https://huggingface.co/Goddragon/Arbiter-4B) (from NeoHorse-Jev-4B + LoRA) ([#34](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/34)) | general, then fine-tuned on `train` and other decision data, temperature-calibrated | 0.817 | 0.092 | 0.048 | 0.184 | 133 ms (A800) |
| 3 | [Darwin-27B-ZTC-v5](https://huggingface.co/FINAL-Bench/Darwin-27B-ZTC-v5) (27B, full fine-tune) ([#33](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/33)) | general, then fine-tuned on `train` | 0.812 | – | – | – | – |
| 4 | [ministral3-3b-reasoning-typed-decisions](https://huggingface.co/rony/ministral3-3b-reasoning-typed-decisions) (Ministral 3 3B Reasoning, full fine-tune) ([#30](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/30)) | specialist | 0.808 | 0.077 | 0.042 | 0.182 | 373 ms (GB10) |
| 5 | [Amos 1 large](https://huggingface.co/mukaddamzaid/amos-1-large) (27B), per-question priors fitted on `train` ([#31](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/31)) | general, then fitted on `train` (one log-prior bias per question and option; no weights changed) | 0.8025 | 0.122 | – | – | 357 ms |
| 6 | [sysone-typed-decisions-0.4B](https://huggingface.co/forward-metric/sysone-typed-decisions-0.4B) (Ettin encoder 400M, full fine-tune) ([#44](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/44)) | specialist, temperature-calibrated | 0.8025 | 0.080 | 0.0438 | 0.174 | 40 ms (Colab L4) |
| 7 | [OpenDecider-large-td](https://huggingface.co/manjunathshiva/opendecider-large-td) (Qwen3-Next-80B-A3B + LoRA) | general, then fine-tuned on `train` | 0.801 | 0.081 | 0.044 | – | 440 ms |
| 8 | [pramana-2b](https://huggingface.co/apracv/pramana-2b) (Qwen3.5-2B, two LoRA adapters averaged) ([#46](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/46)) | general, then fine-tuned on `train`, temperature-calibrated | 0.8005 | 0.0755 | 0.0415 | 0.1675 | – |
| 9 | [od1-typed-decisions](https://huggingface.co/mvbalaji/od1-typed-decisions) (Qwen3.5-4B) | specialist | 0.797 | 0.082 | 0.045 | 0.156 | – |
| 10 | [pramana-0.8b](https://huggingface.co/apracv/pramana-0.8b) (Qwen3.5-0.8B + LoRA) ([#46](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/46)) | general, then fine-tuned on `train`, temperature-calibrated | 0.7970 | 0.0819 | 0.0446 | 0.1646 | – |
| 11 | [FastBrain-9B](https://huggingface.co/TypeScript/FastBrain-9B) (9.4B), per-question priors and temperature fitted on `train` ([#57](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/57)) | general, then fitted on `train` (no weights changed) | 0.797 | 0.0816 | 0.0437 | 0.167 | 50 ms |
| 12 | [OpenDecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano) (400M encoder) | general, then fine-tuned on `train` | 0.796 | 0.079 | 0.043 | – | 17 ms |
| 13 | Rev-gen1-midi (25.6B) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.7945 | 0.086 | 0.047 | 0.127 | 673 ms (B200, local) |
| 14 | [OpenDecider-small-td](https://huggingface.co/manjunathshiva/opendecider-small-td) (Qwen3-4B + LoRA) | general, then fine-tuned on `train` | 0.792 | 0.080 | 0.043 | – | 40 ms |
| 15 | [Phocinae-Largha-150M-v1](https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1) (mmBERT-small, 144.3M) | specialist | 0.791 | – | – | – | 21.0 ms (RTX 5090) |
| 16 | [nullspan](https://huggingface.co/gnumanth/nullspan) (ModernBERT-large, 395M, v1.3.0) ([#55](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/55)) | specialist, temperature-calibrated | 0.790 | 0.076 | 0.042 | 0.1665 | 48 ms (L4) |
| 17 | Rev-gen1-midi-lite (15.7B) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.7885 | 0.083 | 0.046 | 0.124 | 82 ms (B200, local) |
| 18 | [OpenDecider-medium-td](https://huggingface.co/manjunathshiva/opendecider-medium-td) (Qwen3-30B-A3B + LoRA) | general, then fine-tuned on `train` | 0.788 | 0.081 | 0.044 | – | 214 ms |
| 19 | [SafeDecide-400M](https://huggingface.co/yotambr/safedecide-400m) (Ettin-encoder-400m, 400M) ([#39](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/39)) | fine-tuned on `train`, each question scored on its own | 0.787 | 0.083 | 0.045 | 0.171 | 605 ms (1 CPU thread) |
| 20 | Rev-gen1-mini-lite (2.8B) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.7775 | 0.100 | 0.055 | 0.110 | 221 ms (B200, local) |
| 21 | Rev-gen1-mini (4.2B) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.7760 | 0.096 | 0.052 | 0.121 | 361 ms (B200, local) |
| 22 | [soft-decider-421m](https://huggingface.co/winwinwinbb/soft-decider-421m) | specialist | 0.774 | – | – | 0.141 | 50 ms |
| 23 | [Qwen3.5-4B-typed-decisions](https://huggingface.co/wozniakos10/Qwen3.5-4B-typed-decisions) (Qwen3.5-4B + LoRA, Clef head) | general, then fine-tuned on `train` | 0.7725 | 0.0992 | 0.0543 | 0.139 | – |
| 24 | [lazydecider-0.8b](https://huggingface.co/rahulkumar0612/lazydecider-0.8b) (Qwen3.5-0.8B + LoRA, Clef joint-decision head) ([#47](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/47)) | specialist | 0.770 | 0.0938 | 0.0510 | 0.1454 | – |
| 25 | [embeddinggemma-2-ft-typed-decision](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision) (0.3B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | fine-tuned on `train` | 0.770 | 0.119 | 0.065 | 0.155 | – |
| 26 | [Laya typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | fitted on `train` | 0.766 | – | – | – | – |
| 27 | Rev-gen1-micro-lite (1.3B) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.7645 | 0.106 | 0.058 | 0.107 | 37 ms (B200, local) |
| 28 | Rev-gen1-micro (1.9B) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.7645 | 0.150 | 0.080 | 0.223 | 52 ms (B200, local) |
| 29 | Rev-gen1-UltraBERT-150M (150M) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.7630 | 0.173 | 0.092 | 0.228 | 82 ms (B200, local) |
| 30 | Rev-gen1-UltraBERT-380M (378M) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.7620 | 0.144 | 0.075 | 0.203 | 82 ms (B200, local) |
| 31 | [gemma3-270m-typed-decisions](https://huggingface.co/rony/gemma3-270m-typed-decisions) (Gemma 3 270M, full fine-tune) | specialist | 0.761 | 0.132 | 0.072 | 0.148 | 68 ms |
| 32 | [nanoGentzen-v3](https://huggingface.co/Sagicc/nanoGentzen-v3) (4.93M) ([#59](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/59)) | weights trained on synthetic logic data; rules and calibrated distributions built for this benchmark's workflows | 0.7585 | 0.207 | 0.100 | 0.065 | 18 ms |
| 33 | [SafeDecide-140M](https://huggingface.co/yotambr/safedecide-140m) (mmBERT-small, 140M) ([#39](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/39)) | general, then fine-tuned on `train` | 0.754 | 0.126 | 0.069 | 0.147 | 80 ms (1 CPU thread) |
| 34 | [Seb-9B](https://huggingface.co/ironbcc/seb-9b) (Qwen3.5-9B, v1) ([#13](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/13)) | general, then fitted on `train`, one question per request | 0.754 | 0.349 | 0.131 | 0.031 | – |
| 35 | [tasksource-decider-nano](https://huggingface.co/tasksource/tasksource-decider-nano) (149M cross-encoder, `v5`) ([#45](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/45)) | general; `train` is 0.5% of its training mix | 0.7515 | 0.1125 | 0.0621 | 0.125 | 37 ms (A30, shared) |
| 36 | Rev-gen1-nano (752M) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.7515 | 0.117 | 0.064 | 0.094 | 52 ms (B200, local) |
| 37 | [ProKope-421M-8K](https://huggingface.co/ProKope-AI/ProKope-421M-8K) (421M, 8K context) | specialist | 0.7185 | – | 0.385 | – | 43 ms |
| 38 | [serin-0.8b](https://huggingface.co/lambdasec/serin-0.8b) (Qwen3.5-0.8B, 0.8B) | general, then trained on `train` | 0.7165 | 0.9577 | 0.1502 | 0.1122 | – |
| 39 | [gemma3-270m-typed-decisions](https://huggingface.co/rony/gemma3-270m-typed-decisions) (Gemma 3 270M, full fine-tune) | general, fine-tuned on `train` | 0.702 | 0.156 | 0.091 | 0.128 | 68 ms |
| 40 | SAP RPT 1.6 via the SAP Decisions API (`/api/systemone`), [evaluation](https://huggingface.co/datasets/tomgutt/rpt-1.6-typed-decisions-evaluation) ([#49](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/49)) | in-context: 300 provider-held `train` examples per workflow, no weight updates; closed API | 0.672 | 0.646 | 0.151 | 0.019 | 656 ms (p50, API round trip) |
| 41 | [Bekko System One v0 400M](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m), scored by us | trained on `train` and other data | 0.668 | 0.214 | 0.113 | 0.143 | 808 ms |
| 42 | [vega-08b-public-intents](https://huggingface.co/nandakishorm/vega-08b-public-intents) (0.8B) ([#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37)) | specialist (bundled typed-decisions adapter active) | 0.663 | 0.435 | 0.191 | 0.074 | – |
| 43 | ModernBERT-base (149M), ours | specialist, fitted per workflow | 0.646 | 0.223 | 0.119 | 0.179 | 349 ms |
| 44 | Rev-gen1-EmbeddingGemma-2-Frozen (272M) ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) | general, trained on a mix including filtered `train`; weights and predictions private | 0.6015 | 0.248 | 0.142 | 0.042 | 92 ms (B200, local) |
| 45 | MiniLM-L6 (22M), ours | specialist, fitted per workflow | 0.587 | 0.262 | 0.143 | 0.108 | 22 ms |
| 46 | [Bekko System One v0 68M](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m), scored by us | trained on `train` and other data | 0.537 | 0.293 | 0.160 | 0.116 | 176 ms |
| 47 | [Bekko System One v0 17M](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m), scored by us | trained on `train` and other data | 0.483 | 0.344 | 0.203 | 0.136 | 33 ms |

**Quyet-1.0-Large, served by Decision Models, has the highest accuracy among zero-shot models (0.791). [meraGPT Decider 1](https://meragpt.com/models/state-decider-1?utm_source=huggingface&utm_medium=dataset_card&utm_campaign=typed-decisions) is second on accuracy (0.768) and has the lowest KL (0.096) and Brier (0.052) of any zero-shot model.** It leads Jev on every question type (`noul` 0.840 vs
0.775, `choice` 0.733 vs 0.720, `score` 0.739 vs 0.696), its distributions sit far
closer to the gold (KL 0.096 vs 1.442), it is faster end to end, and it costs less
per token. It answers `POST /v1/systemone` at
[meragpt.com](https://meragpt.com/docs/systemone?utm_source=huggingface&utm_medium=dataset_card&utm_campaign=typed-decisions), so the typesafe-sdk works
against it by setting `TYPESAFE_BASE_URL`.

To add a model, score it on `test` with the full distributions and open a
discussion with the numbers and the mode (specialist or general) it used.

### Benchmarking a model with HF Jobs

One command scores any model that serves vLLM's `/v1/systemone` (any model
without a native server works this way; serve it with a current nightly). It
runs one request per case (state + all five questions) over the `test` split
and prints accuracy, KL from gold, Brier and ECE:

```bash
hf jobs run --flavor a100-large --timeout 2h \\
  --env MODEL_ID=your-org/your-model \\
  --secrets HF_TOKEN \\
  --volume hf://datasets/LocalLLaMA/typed-decisions:/repo:ro \\
  pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel \\
  bash /repo/bench/score_with_jobs.sh
```

- Pick the `--flavor` for the model's size: `l4x1` up to ~2B, `l40sx1` or
  `a100-large` for ~27B, `h200` above.
- `bench/score_with_jobs.sh` installs vLLM nightly, serves the model, scores all
  400 cases, and writes `results.json`.
- The run finishes by creating the eval YAML and opening a PR on the model repo
  with the four scores, so they show on the Hub leaderboard once merged. Add
  `--env MODE=specialist` if you fitted the model on `train`.

### Who scored each row

- **Scored by us:** meraGPT Decider 1, TypeSafe Jev 1.13.0, Liquid AI d1, Baseten Mammoth 1 and Elephant 1, Featherless Simple Jev, the three Jeff models, the three rows served by Decision Models ([#38](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/38)), the Bekko and Adaptive Classifier rows, and the Prior and Uniform references.
- **Scored by Hugging Face:** the rows linked to [#37](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/37), run on Hugging Face Jobs.
- **Reported by an outside evaluator:** Qwen3.8-27B ([#60](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/60)), scored by adcostanza with the dataset's own scorer on SGLang; the full trace is public and rescores identically. SAP RPT 1.6 ([#49](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/49)), a self-run, unofficial evaluation of SAP's public Decisions API. The API answers from 300 `train` examples per workflow that SAP holds on its side, with no weight updates; those examples are SAP's description and could not be checked.
- **Reported by the model's creator:** every other row, in its own discussion. The ten Rev-gen1 rows ([#64](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/64)) come from private checkpoints whose predictions are not public, so they could not be rescored here. [serin-0.8b](https://huggingface.co/lambdasec/serin-0.8b) (lambdasec) has no discussion thread; it is a self-run evaluation with the benchmark's own scorer.

## Reading a score

Gold is the mean of three samples from a teacher of roughly 4B-class capability,
so a score measures agreement with that teacher, not correctness. A better model
can score lower wherever the teacher is wrong (it missed a duplicate invoice whose
ID matched an earlier one).

| Reference | Accuracy | What it is |
|---|---|---|
| Prior | 0.470 | the floor: label frequencies, ignoring the input |
| Perfect factor recovery | 0.704 | a model fitted to the latent factors that generated each case |
| Teacher self-agreement | 0.735 | a fresh teacher sample against gold built from the others |

Scores well above 0.735 mean a model is learning the teacher's quirks. Per-question
ceilings vary from 0.560 (`agent_trace/urgency`) to 0.937
(`customer_service/category`), so read scores per question as well as on average.

**Specialist and general numbers are not comparable.** A specialist is fitted on
`train` for these four workflows and cannot answer anything else. A general model
takes any question schema at request time and has never seen these. Say which
mode you used; the gap between them is the price of generality, not a quality
ranking.

## The data

| Type | Answer | Shape |
|---|---|---|
| `noul` | yes/no | probability that the statement is true |
| `choice` | one of N labels | distribution over labels, plus confidence |
| `score` | an ordered rubric | distribution over levels, plus an expected score |

Every option carries a written description in `criteria`, and the descriptions
are part of the input.

| Workflow | Decision | Train | Test |
|---|---|---|---|
| `agent_trace_observability` | Does an agent run need human review, and how urgently? | 300 | 100 |
| `customer_service` | The right response and action for a customer thread and account. | 300 | 100 |
| `invoice_processing` | Pay, hold or reject a vendor bill against its order and delivery. | 300 | 100 |
| `security_incidents` | Close, investigate or contain a security alert, given machine history. | 300 | 100 |

`state` and `questions` together are exactly the body of a `POST /v1/systemone`
request. `gold` holds the full gold distributions, and flat
`<question>__label` / `__probabilities` / `__score` / `__probability_true` columns
hold the same answers for convenience. `factors` and `label_agreement` describe how
the case was built and are not model input.

```python
from datasets import load_dataset
import json

ds = load_dataset("LocalLLaMA/typed-decisions", "customer_service", split="test")
row = ds[0]
state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
print(gold["urgency"]["probabilities"])   # score against the full distribution
```

Report KL or log loss and Brier next to accuracy; calibration is the point.

## How it was built

Each case starts from independently sampled latent factors (topic, tone,
severity, discrepancy type and so on), which a model renders into free text where
the artefact is textual and keeps structured where it is structured. A teacher
labels each case three times at temperature 0.7, and the gold is the mean of those
distributions, so it stays soft where a decision is genuinely ambiguous. Before
release, an audit checks state diversity, label balance, and that the gold
actually tracks the input. `train` comes from a separate run at a different seed;
packaging refuses to build if any case id or state appears in both splits.

The write-up behind the benchmark, including the two bugs it exposed in the
classifier library: [Typed Decisions on Latent Node](https://latentnode.pages.dev/articles/typed-decisions.html).
