// The Voice Tuner's worker: runs tinyvoca.js (the same library other pages use) off the main thread.
// In:  {type: "load"}, {type: "speak", id, text, settings}, {type: "sing", id, score, settings},
//      {type: "export", id, settings, name}
// Out: status, ready, sentence (audio as soon as it is made), done (whole clip), voice (a voice file), error

// The page passes its build id; asking for tinyvoca.js under it means a browser never pairs a new page with a
// cached old library.
const BUILD = new URL(location.href).searchParams.get("build") || "";
importScripts("tinyvoca.js?build=" + encodeURIComponent(BUILD));

let tts;

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      tts = await TinyVoca.load({ onStatus: (text, progress) => postMessage({ type: "status", text, progress }) });
      postMessage({ type: "ready" });
    } else if (data.type === "speak") {
      const started = performance.now();
      const voice = tts.voiceFromSettings(data.settings);
      const audio = await tts.speak(data.text, voice, {
        onSentence: (heard, phones, index, total) => postMessage(
          { type: "sentence", id: data.id, index, total, phonemes: phones.text, audio: heard, sampleRate: tts.sampleRate },
          [heard.buffer]),
      });
      postMessage({ type: "done", id: data.id, audio, sampleRate: tts.sampleRate, totalTime: performance.now() - started },
                  [audio.buffer]);
    } else if (data.type === "sing") {
      const started = performance.now();
      const voice = tts.voiceFromSettings(data.settings);
      const audio = await tts.sing(data.score, voice, {
        onLine: (heard, index, total) => postMessage(
          { type: "sentence", id: data.id, index, total, phonemes: "", audio: heard, sampleRate: tts.sampleRate },
          [heard.buffer]),
      });
      postMessage({ type: "done", id: data.id, audio, sampleRate: tts.sampleRate, sung: true,
                    totalTime: performance.now() - started }, [audio.buffer]);
    } else if (data.type === "export") {
      postMessage({ type: "voice", id: data.id, voice: tts.voiceFromSettings(data.settings, data.name) });
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error),
                  plain: data.type === "sing" || /only reads English/.test(String(error && error.message)) });
  }
};
