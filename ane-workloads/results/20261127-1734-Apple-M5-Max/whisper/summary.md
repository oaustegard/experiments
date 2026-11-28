# Whisper large-v3-turbo: Apple M5 Max, 2026-11-27T17:34:54

macOS 27.2.1, AC True, low power False. LibriSpeech test-clean, 4 windows, 102.7 s of audio.

RTF is processing seconds per audio second with the model load excluded (engine-reported; the wall column is the same from wall-clock, all windows minus one window). Load is one invocation's model load: first is this run's first invocation.

| engine | GPU | WER % | RTF | x realtime | RTF wall | load s first | load s later | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|---|
| whisperkit | idle | 0.70 | 0.03182 | 31.4 | 0.03049 | 64.163 | 0.679 |  |

- whisperkit: binary /opt/homebrew/bin/whisperkit-cli, version v1.1.1, model large-v3-v20240930_turbo, compute_units cpuAndNeuralEngine, models_existed_before False
