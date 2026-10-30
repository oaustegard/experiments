# voice-loop

A phone-first page that chains the browser's own speech recognition, a small
instruction-tuned language model running in the page, and the browser's own
speech synthesis. Built to find out how well that loop works on iOS Safari and
as a Home Screen web app.

Live: https://austegard.com/experiments/voice-loop/web/

## Pieces

- **Speech in:** `webkitSpeechRecognition`. On iOS it runs through Siri
  dictation, so it needs Siri & Dictation enabled and usually a network
  connection. The audio leaves the device, while the model's answer does not.
- **Model:** transformers.js 3.8.1 in a module worker (`web/worker.js`). WebGPU
  with `q4f16` when the adapter has `shader-f16`, `q4` without it, wasm `q8`
  when there is no adapter or the box is ticked. Choices: SmolLM2 135M (~120 MB),
  SmolLM2 360M (~270 MB, default), Qwen2.5 0.5B (~480 MB). Weights come from
  the Hugging Face hub and stay in the browser cache.
- **Speech out:** `speechSynthesis`. Sentences are spoken as they stream in, so
  the first sentence plays while the rest generates. iOS needs one utterance
  started inside a tap before it will speak, so every button unlocks it.
- **Conversation mode:** after the last sentence finishes, it starts listening
  again.

The page reports first-token latency and tokens per second under each answer,
and recognition time under the talk button.

## Verified

Headless Chromium on this container (no GPU, wasm forced, SmolLM2 135M): the
model loaded in 13.8 s and answered a typed question at 11.7 tok/s, with first
token at 1.0 s and no console errors. Its answer to "Why is the sky blue?" was
nonsense, which is what 135M parameters gets you. Speech recognition and
synthesis exist there, but there is no microphone, so neither was exercised.

## Pending on a phone

- Does `webkitSpeechRecognition` work in Home Screen (standalone) mode? It has
  returned `service-not-allowed` there in past iOS releases.
- WebGPU on iOS 26: the f16 path, load time and tok/s per model.
- After dictation, iOS may route speech to the earpiece at low volume. Check
  whether answers come out of the speaker.
- Memory ceiling: whether Qwen2.5 0.5B loads without the tab being killed.
