# Whisper large-v3-turbo: Apple M5 Max, 2026-11-27T19:29:52

macOS 27.2.1, AC True, low power False. LibriSpeech test-clean, 40 windows, 982.0 s of audio.

RTF is processing seconds per audio second with the model load excluded (engine-reported; the wall column is the same from wall-clock, all windows minus one window). Load is one invocation's model load: first is this run's first invocation.

| engine | GPU | WER % | RTF | x realtime | RTF wall | load s first | load s later | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|---|
| whisperkit | idle | 2.00 | 0.03157 | 31.7 | 0.03177 | 0.809 | 0.635 |  |
| whisperkit | busy | 2.00 | 0.03657 | 27.3 | 0.036 | 0.809 | 0.71 | 45.36 -> None (busy 1.0) |
| whispercpp | idle | 2.00 | 0.01681 | 59.5 | 0.01633 | 2.834 | 0.587 |  |
| whispercpp | busy | 2.00 | 0.03885 | 25.7 | 0.021 | 2.834 | 1.256 | 45.36 -> 39.09 (busy 0.998) |
| mlx | idle | 2.58 | 0.01545 | 64.7 | 0.01567 | 2.629 | 0.17 |  |
| mlx | busy | 2.33 | 0.0187 | 53.5 | 0.01872 | 2.629 | 0.23 | 45.36 -> None (busy 1.0) |

- whisperkit: binary /opt/homebrew/bin/whisperkit-cli, version v1.1.1, model large-v3-v20240930_turbo, compute_units cpuAndNeuralEngine, models_existed_before True
- whispercpp: commit 79b11911d9d38c4be2fc7dc8b4efeff3324c090f, encoder_compute_units ne, models_existed_before False, encoder_units_line config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
- mlx: model mlx-community/whisper-large-v3-turbo, mlx_whisper mlx-whisper==0.4.3
