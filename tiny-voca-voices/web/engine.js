// The Voice Tuner's engine: loads tinyvoca.js's model and phonemizer and answers the page's messages. The worker
// (worker.js) runs it off the main thread; the page runs the same code itself if the worker fails or stalls,
// which happened on an iPhone where the in-page library (example.html) worked.
// In:  {type: "load"}, {type: "speak", id, text, settings}, {type: "sing", id, score, settings},
//      {type: "chorus", id, score, parts: [{settings, transpose, delayMs, vibratoRate, pan}]},
//      {type: "export", id, settings, name}
// Out: status, ready, sentence (audio as soon as it is made), done (whole clip), voice (a voice file), error
(function (root) {
  "use strict";
  root.makeTunerEngine = function (post) {
    let tts;
    return async function handle(data) {
      try {
        if (data.type === "load") {
          tts = await TinyVoca.load({
            onStatus: (text, progress, bytes) => post({ type: "status", text, progress, bytes }),
          });
          post({ type: "ready" });
        } else if (data.type === "speak") {
          const started = performance.now();
          const voice = tts.voiceFromSettings(data.settings);
          const audio = await tts.speak(data.text, voice, {
            onSentence: (heard, phones, index, total) => post(
              { type: "sentence", id: data.id, index, total, phonemes: phones.text, audio: heard, sampleRate: tts.sampleRate },
              [heard.buffer]),
          });
          post({ type: "done", id: data.id, audio, sampleRate: tts.sampleRate, totalTime: performance.now() - started },
               [audio.buffer]);
        } else if (data.type === "sing") {
          const started = performance.now();
          const voice = tts.voiceFromSettings(data.settings);
          const audio = await tts.sing(data.score, voice, {
            onLine: (heard, index, total) => post(
              { type: "sentence", id: data.id, index, total, phonemes: "", audio: heard, sampleRate: tts.sampleRate },
              [heard.buffer]),
          });
          post({ type: "done", id: data.id, audio, sampleRate: tts.sampleRate, sung: true,
                 totalTime: performance.now() - started }, [audio.buffer]);
        } else if (data.type === "chorus") {
          const started = performance.now();
          const parts = data.parts.map((p) => ({ ...p, voice: tts.voiceFromSettings(p.settings) }));
          const { left, right } = await tts.chorus(data.score, parts, {
            onPart: (i, n) => post({ type: "status", text: `Singing voice ${i + 1} of ${n}` }),
          });
          post({ type: "done", id: data.id, left, right, sampleRate: tts.sampleRate, sung: true, chorus: parts.length,
                 totalTime: performance.now() - started }, [left.buffer, right.buffer]);
        } else if (data.type === "export") {
          post({ type: "voice", id: data.id, voice: tts.voiceFromSettings(data.settings, data.name) });
        }
      } catch (error) {
        const message = String((error && error.message) || error);
        post({ type: "error", id: data.id, message, plain: data.type === "sing" || data.type === "chorus" || /only reads English/.test(message) });
      }
    };
  };
})(typeof self !== "undefined" ? self : this);
