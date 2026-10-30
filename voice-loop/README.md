# voice-loop

A phone-first page that chains the browser's own speech recognition, a small
instruction-tuned language model running in the page, and the browser's own
speech synthesis. Built to find out how well that loop works on iOS Safari and
as a Home Screen web app.

Live: https://austegard.com/experiments/voice-loop/web/

## Pieces

- **Speech in:** `webkitSpeechRecognition`. On iOS it runs through Siri dictation, so it needs
  Siri & Dictation enabled and usually a network connection. The audio leaves the device, while
  the model's answer does not.
- **Model:** LFM2.5-350M on the hand-written WebGPU engine in [`lfm-wgsl`](../lfm-wgsl/README.md)
  (`web/llm.js`):
  - The weights stream from Hugging Face straight into GPU buffers. A copy goes into the Cache API,
    so later loads come from disk.
  - The tokenizer and chat template come from transformers.js 4.3.1.
  - Sampling follows Liquid's settings for LFM2: temperature 0.3, min_p 0.15, repetition
    penalty 1.05.
- **Speech out:** Supertonic 2 (`onnx-community/Supertonic-TTS-2-ONNX`, about 260 MB) runs in a
  worker (`web/tts-worker.js`) on onnxruntime-web wasm.
  - Sentences are spoken as they stream from the model. Each is synthesized while the previous one
    plays, then scheduled back to back on one AudioContext.
  - The system voice is the alternative. In the container, Supertonic made 6.0 s of audio in 4.4 s
    on one wasm thread, while Kokoro (kokoro-js 1.2.1, q8) needed 22.7 s, so Supertonic is the
    default.
- **Conversation mode:** after the last sentence finishes, the page starts listening again.

## History

The first version ran transformers.js 3.8.1 with SmolLM2 or Qwen2.5 in a worker and spoke with
`speechSynthesis`. On the iPhone 16 Pro, the tab reloaded as soon as SmolLM2-360M loaded, which
means it ran out of memory. Oskar also found the system voice poor. Both problems led to the
current design.

## Measured

The LFM engine ran on the iPhone 16 Pro through the GPU bridge. In Safari on iOS 27.0.1 it loaded
without the tab being killed, prefilled a 35-token prompt in 107–181 ms and decoded at a median of
14 ms per token, about 71 tokens/s. Its next-token picks agreed with onnxruntime at 13 of 13 steps.
See `lfm-wgsl/README.md`.

Supertonic drops words from short utterances at the model card's speed of 1.05. Each synthesis
below was transcribed with Whisper base.en, three runs per row:

| Text | Speed | Transcripts |
|---|---|---|
| "This is how answers will sound." (31 characters) | 1.05 | "This will sound.", "This answers will sound." |
| same | 0.9 | 1 of 3 verbatim |
| same | 0.8 | 2 of 3 verbatim |
| A 95-character sentence | 0.9 | 3 of 3 verbatim |
| "Hello. This is how my answers will sound when you ask me something." | 0.9 | 3 of 3 verbatim |
| "The sky appears blue because of the way sunlight interacts with Earth's atmosphere." | 0.9 | 3 of 3 verbatim |

The page therefore runs Supertonic at speed 0.9. It holds sentences shorter than 60 characters
until the next one arrives and synthesizes them together. Supertonic predicts one duration for each
utterance, and on a short one the predicted duration leaves no room for some words.

## Pending on a phone

- The whole page on the iPhone: Supertonic's speed, and memory with both models loaded.
- Does `webkitSpeechRecognition` work in Home Screen (standalone) mode? It has returned
  `service-not-allowed` there in past iOS releases.
- Whether speech plays through the speaker or the earpiece after dictation.
