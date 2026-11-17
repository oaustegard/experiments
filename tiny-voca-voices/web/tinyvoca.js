// tinyvoca.js — speak text in a tiny-voca voice file, in a browser page or a Web Worker.
//
//   <script src="https://austegard.com/experiments/tiny-voca-voices/web/tinyvoca.js"></script>
//   const tts = await TinyVoca.load({ onStatus: (text, progress) => console.log(text, progress) });
//   const voice = await (await fetch("my-voice.voice.json")).json();     // exported from the Voice Tuner
//   const audio = await tts.speak("Hello there.", voice);                  // Float32Array, 24 kHz mono
//   const wav = TinyVoca.wav(audio);                                       // Blob, 16-bit PCM
//
// load() fetches onnxruntime-web and Pyodide from jsDelivr, the model and phonemizer source from `base` (this
// file's folder on austegard.com by default), and dizione's dictionary from Hugging Face. About 22 MB transferred
// the first time (8.4 MB model, 11 MB Pyodide, numpy and dictionary, 2.5 MB onnxruntime); kept in Cache Storage after. Self-hosting: copy tinyvoca.js, tiny-voca-tunable.bin,
// dizione-src.bin and voice-space.json to one folder and pass its URL as `base`.
//
// A voice file (format "tiny-voca-voice/1") carries the two 192-value embeddings the model reads and the six
// numeric controls, so it plays without the tuner. `settings` holds the tuner's slider values, for reloading.
//
// Phonemizer, sentence splitting and the model calls follow tiny-voca's own demo (github.com/n4ze3m/tiny-voca,
// web/public/worker.js, Apache-2.0).
(function (root) {
  "use strict";
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const HF = "https://huggingface.co/n4ze3m/tiny-voca/resolve/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14/phonemizer/";
  const DATA = ["lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json"];
  const MODEL = { file: "tiny-voca-tunable.bin", sha256: "4631b3249cb7fe7aa923a23a815a762caeda6576f89ebbd5eccd1294236cfe5d" };
  // Voice files made before singing was added name the earlier export of the same weights; they play unchanged.
  const COMPATIBLE = new Set([MODEL.sha256, "a3dcc3e6317e58621e4c1623992cffd471f67e5156519868dd5a31f69a50619d"]);
  const FORMAT = "tiny-voca-voice/1";
  const SEMITONE = Math.log(2) / 12;
  const PEAK = 0.99;
  const here = (typeof document !== "undefined" && document.currentScript && document.currentScript.src) ||
               (typeof location !== "undefined" ? location.href : "");
  const DEFAULT_BASE = new URL(".", here || "https://austegard.com/experiments/tiny-voca-voices/web/").href;

  function loadScript(url) {
    if (typeof importScripts === "function") { importScripts(url); return Promise.resolve(); }
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = url; s.crossOrigin = "anonymous";
      s.onload = resolve; s.onerror = () => reject(new Error("could not load " + url));
      document.head.appendChild(s);
    });
  }

  async function download(url, label, onStatus) {
    let store = null;
    try {
      store = await caches.open("tinyvoca-1");
      const hit = await store.match(url);
      if (hit) return new Uint8Array(await hit.arrayBuffer());
    } catch (error) { store = null; }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    const total = Number(response.headers.get("content-length")) || 0;
    let bytes;
    if (!response.body || !total) {
      bytes = new Uint8Array(await response.arrayBuffer());
    } else {
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        onStatus(label, received / total);
      }
      bytes = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    }
    if (store) store.put(url, new Response(bytes)).catch(() => {});
    return bytes;
  }

  // The voice space: F2 and M2 rows of both embedding tables, the F2-M2 distance in each, four unit directions.
  // embedding() builds (1 - mix) * F2 + mix * M2 + sum(axis_i * distance * direction_i).
  function embedding(space, table, settings) {
    const t = space.voice_space[table];
    const v = new Array(t.rows[0].length);
    const m = settings.mix;
    for (let i = 0; i < v.length; i++) v[i] = (1 - m) * t.rows[0][i] + m * t.rows[1][i];
    (settings.axes || []).forEach((amount, k) => {
      if (!amount) return;
      for (let i = 0; i < v.length; i++) v[i] += amount * t.distance * t.directions[k][i];
    });
    return v.map((x) => Math.round(x * 1e6) / 1e6);
  }

  // Tuner settings -> voice file. settings: {mix, axes: [a1..a4], pitch (semitones), range, speed, formant, tilt, breath}
  function voiceFromSettings(space, settings, name = "my voice", base = DEFAULT_BASE) {
    return {
      format: FORMAT, name,
      model: { url: new URL(MODEL.file, base).href, sha256: MODEL.sha256 },
      voice: embedding(space, "voice", settings), frame_voice: embedding(space, "frame_voice", settings),
      speed: settings.speed, pitch_shift: settings.pitch * SEMITONE, pitch_range: settings.range,
      formant: settings.formant, tilt: settings.tilt, breath: settings.breath,
      settings,
    };
  }

  function checkVoice(voice) {
    if (!voice || voice.format !== FORMAT) throw new Error(`not a ${FORMAT} voice file`);
    for (const key of ["voice", "frame_voice"]) {
      if (!Array.isArray(voice[key]) || voice[key].length !== 192 || !voice[key].every(Number.isFinite)) {
        throw new Error(`voice file: ${key} must hold 192 numbers`);
      }
    }
    for (const key of ["speed", "pitch_shift", "pitch_range", "formant", "tilt", "breath"]) {
      if (!Number.isFinite(voice[key])) throw new Error(`voice file: ${key} must be a number`);
    }
    if (voice.model && voice.model.sha256 && !COMPATIBLE.has(voice.model.sha256)) {
      throw new Error("voice file was made for a different model file");
    }
    return voice;
  }

  const ABBREVIATIONS = /(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$/i;
  function sentences(text) {
    const parts = [];
    let current = "";
    for (const piece of text.replace(/\s+/g, " ").trim().split(/(?<=[.!?…]["')\]]?)\s+(?=["'(\[]?[A-Z0-9])/)) {
      current = current ? current + " " + piece : piece;
      if (!ABBREVIATIONS.test(current)) { parts.push(current); current = ""; }
    }
    if (current) parts.push(current);
    return parts.filter((part) => /[\p{L}\p{N}]/u.test(part));
  }
  function finished(text) {
    const body = text.replace(/\s+$/, "");
    if (!body) return text;
    const last = body.split(/\s+/).pop();
    const marked = /[\p{L}\p{N}]/u.test(last) ? /[.!?…,;:]["'”’)\]»]*$/.test(last) : /[.!?…]/.test(last) || /^[,;:]$/.test(last);
    return marked ? text : body + ".";
  }
  function limitPeak(samples) {
    let loudest = 0;
    for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
    if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
    return samples;
  }

  function wav(samples, rate = 24000) {
    const data = new DataView(new ArrayBuffer(44 + samples.length * 2));
    const text = (o, s) => [...s].forEach((ch, i) => data.setUint8(o + i, ch.charCodeAt(0)));
    text(0, "RIFF"); data.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVEfmt ");
    data.setUint32(16, 16, true); data.setUint16(20, 1, true); data.setUint16(22, 1, true);
    data.setUint32(24, rate, true); data.setUint32(28, rate * 2, true); data.setUint16(32, 2, true); data.setUint16(34, 16, true);
    text(36, "data"); data.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) data.setInt16(44 + 2 * i, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
    return new Blob([data], { type: "audio/wav" });
  }


  // "D4/1 G4/2 B4/.5~G4/.5 r/1" -> [{notes: [{hz, beats}], rest}]
  function parseNotes(text) {
    return text.trim().split(/\s+/).filter(Boolean).map((entry) => {
      const notes = entry.split("~").map((part) => {
        const m = part.match(/^([A-Ga-g]|r|R)([#b]?)(-?\d)?\/(\d*\.?\d+)$/);
        if (!m) throw new Error(`cannot read note "${part}" (write e.g. D4/1, F#4/.5, r/2)`);
        const beats = Number(m[4]);
        if (/r/i.test(m[1])) return { rest: true, beats };
        if (m[3] === undefined) throw new Error(`note "${part}" needs an octave, e.g. ${m[1].toUpperCase()}4`);
        const step = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[m[1].toUpperCase()] + (m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0);
        const midi = 12 * (Number(m[3]) + 1) + step;
        return { hz: 440 * 2 ** ((midi - 69) / 12), beats };
      });
      if (notes.some((n) => n.rest)) {
        if (notes.length > 1) throw new Error(`a rest stands alone: "${entry}"`);
        return { rest: true, beats: notes[0].beats };
      }
      return { notes };
    });
  }

  async function load({ base = DEFAULT_BASE, onStatus = () => {}, threads = 1 } = {}) {
    if (typeof ort === "undefined") await loadScript(ORT + "ort.wasm.min.js");
    if (typeof loadPyodide === "undefined") await loadScript(PYODIDE + "pyodide.js");
    const space = await (await fetch(new URL("voice-space.json", base))).json();

    async function phonemizer() {
      onStatus("Loading the phonemizer");
      const pyodide = await loadPyodide({ indexURL: PYODIDE });
      await pyodide.loadPackage("numpy");
      const [source, ...files] = await Promise.all([
        download(new URL("dizione-src.bin", base).href, "Loading the phonemizer", onStatus),
        ...DATA.map((name) => download(HF + name, "Loading the dictionary", onStatus)),
      ]);
      onStatus("Starting the phonemizer");
      pyodide.FS.mkdirTree("/dizione-data");
      DATA.forEach((name, index) => pyodide.FS.writeFile("/dizione-data/" + name, files[index]));
      pyodide.unpackArchive(source.buffer, "zip", { extractDir: "/dizione-src" });
      pyodide.runPython(`
import json, os, sys
os.environ["DIZIONE_DATA"] = "/dizione-data"
sys.path.insert(0, "/dizione-src")
import dizione

def phonemize_json(text):
    try:
        result = dizione.phonemize(text)
    except ValueError:
        if any(word for mark, words in dizione.pieces(text) if not mark for word in words):
            raise
        return "null"
    return json.dumps({"ids": list(result.ids), "stress": list(result.stress), "text": str(result)})

phonemize_json("Warm up.")
`);
      return pyodide.globals.get("phonemize_json");
    }

    async function model() {
      const bytes = await download(new URL(MODEL.file, base).href, "Loading the model", onStatus);
      onStatus("Starting the model");
      ort.env.wasm.wasmPaths = ORT;
      ort.env.wasm.numThreads = threads;
      return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    }

    const [phonemize, session] = await Promise.all([phonemizer(), model()]);

    // One sentence -> utterances of at most max_tokens phonemes (cut at commas, then between words).
    function fit(sentence) {
      const result = JSON.parse(phonemize(sentence));
      if (!result) return [];
      if (result.ids.length <= space.max_tokens - 12) return [result];
      const pieces = sentence.split(/(?<=[,;:—])\s+/);
      if (pieces.length > 1) return pieces.flatMap(fit);
      const words = sentence.split(" ");
      if (words.length < 2) return [];
      const half = Math.ceil(words.length / 2);
      return [...fit(words.slice(0, half).join(" ")), ...fit(words.slice(half).join(" "))];
    }

    // Phonemes ({ids, stress}) in one voice -> Float32Array at 24 kHz, as the model made it (may peak above 1).
    // Phonemes ({ids, stress}) in one voice -> {audio, durations}. frames: frames per token (-1 = the model's own),
    // f0: Hz per output frame (0 = the model's own pitch); both optional.
    async function run(phones, voice, { frames = null, f0 = null, pitchShift = voice.pitch_shift } = {}) {
      const count = phones.ids.length;
      const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
      const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
      const track = f0 && f0.length ? Float32Array.from(f0) : new Float32Array(1);
      const output = await session.run({
        tokens: integers(phones.ids), stress: integers(phones.stress),
        voice: new ort.Tensor("float32", Float32Array.from(voice.voice), [1, 192]),
        frame_voice: new ort.Tensor("float32", Float32Array.from(voice.frame_voice), [1, 192]),
        speed: scalar(voice.speed), pitch_shift: scalar(pitchShift), pitch_range: scalar(voice.pitch_range),
        formant: scalar(voice.formant), tilt: scalar(voice.tilt), breath: scalar(voice.breath),
        frames_override: integers(frames || phones.ids.map(() => -1)),
        f0_override: new ort.Tensor("float32", track, [1, track.length]),
      });
      return { audio: output.audio.data, durations: Array.from(output.durations.data, Number) };
    }

    // Phonemes in one voice -> Float32Array at 24 kHz, as the model made it (may peak above 1).
    async function synthesize(phones, voice) { return (await run(phones, voice)).audio; }

    // Text of any length in one voice -> Float32Array at 24 kHz, peak at most 0.99. onSentence(audio, phones,
    // index, total) is called as each piece is made, for playing while the rest is computed.
    async function speak(text, voice, { onSentence } = {}) {
      checkVoice(voice);
      const list = sentences(finished(text));
      const chunks = [];
      for (let index = 0; index < list.length; index++) {
        for (const phones of fit(list[index])) {
          const audio = await synthesize(phones, voice);
          chunks.push(audio);
          if (onSentence) onSentence(limitPeak(audio.slice()), phones, index, list.length);
        }
      }
      if (!chunks.length) throw new Error("tiny-voca only reads English.");
      const audio = new Float32Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
      let offset = 0;
      for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
      return limitPeak(audio);
    }


    // ---- Singing ----------------------------------------------------------------------------------------------
    // score: {lyrics, notes, bpm = 90, transpose = 0, vibrato = 0.35, vibratoRate = 5.5, glide = 0.04}
    //   lyrics  words, with hyphens between syllables ("A-maz-ing grace"); one phrase per line.
    //   notes   one entry per syllable, in order: NOTE/BEATS, e.g. "D4/1 G4/2". Join notes with ~ to hold one
    //           syllable over several ("B4/.5~G4/.5"). A rest is its own entry: "r/1". Lines of notes are optional.
    //   transpose in semitones; vibrato depth in semitones; glide in seconds.
    // The voice's timbre is used as is; its pitch shift and speed are not (the notes set pitch and timing).
    // Returns Float32Array at 24 kHz. onLine(audio, index, total) is called as each lyric line is made.
    async function sing(score, voice, { onLine } = {}) {
      checkVoice(voice);
      const bpm = score.bpm || 90, transpose = score.transpose || 0;
      const depth = score.vibrato ?? 0.35, rate = score.vibratoRate ?? 5.5, glide = score.glide ?? 0.04;
      const fps = space.sample_rate / space.hop;
      const beatFrames = (beats) => Math.max(1, Math.round(beats * 60 / bpm * fps));
      const entries = parseNotes(score.notes);
      const lines = score.lyrics.split("\n").map((l) => l.trim()).filter(Boolean);
      const sym = space.symbols, vowels = new Set(space.vowels);
      const SP = sym.indexOf("<sp>"), BOS = sym.indexOf("<bos>"), EOS = sym.indexOf("<eos>");
      let cursor = 0;
      const nextEntry = () => {
        if (cursor >= entries.length) throw new Error("the notes ran out before the lyrics did");
        return entries[cursor++];
      };
      const chunks = [];
      for (let li = 0; li < lines.length; li++) {
        const words = lines[li].split(/\s+/).map((w) => w.replace(/[^\p{L}\p{N}'\-]/gu, "")).filter((w) => w.replace(/-/g, ""));
        const phones = JSON.parse(phonemize(words.map((w) => w.replace(/-/g, "")).join(" ") + "."));
        if (!phones) throw new Error(`nothing to sing in line ${li + 1}`);
        // Split tokens into words at <sp>; punctuation and edge tokens belong to no word.
        const groups = [[]];
        phones.ids.forEach((id, i) => {
          if (id === SP) groups.push([]);
          else if (sym[id] && !sym[id].startsWith("<") && !/^[,.?!;:"'()—…]$/.test(sym[id])) groups[groups.length - 1].push(i);
        });
        const wordTokens = groups.filter((g) => g.length);
        if (wordTokens.length !== words.length) {
          throw new Error(`line ${li + 1}: the phonemizer read ${wordTokens.length} words where the lyrics have ${words.length}; spell out numbers and symbols`);
        }
        const predicted = (await run(phones, voice)).durations;
        const frames = phones.ids.map(() => 0);
        const notesOf = new Map();       // token index -> [{hz, frames}] for vowels, {hz} for consonants
        // Rests before the first syllable of a word go on the gap token before it.
        const gapBefore = (tokenIndex) => {
          for (let i = tokenIndex - 1; i >= 0; i--) if (phones.ids[i] === SP || phones.ids[i] === BOS) return i;
          return 0;
        };
        const takeRests = () => {
          let total = 0;
          while (cursor < entries.length && entries[cursor].rest) total += beatFrames(entries[cursor++].beats);
          return total;
        };
        words.forEach((word, wi) => {
          const tokens = wordTokens[wi];
          const pieces = word.split("-").filter(Boolean).length;
          const nuclei = tokens.filter((i) => vowels.has(sym[phones.ids[i]]));
          if (!nuclei.length) nuclei.push(tokens.reduce((a, b) => (predicted[b] > predicted[a] ? b : a)));
          // Group nuclei into one group per lyric syllable; extra syllables become extra notes on the last group.
          const k = Math.min(pieces, nuclei.length);
          const groupsN = Array.from({ length: k }, (_, g) => nuclei.slice(Math.floor(g * nuclei.length / k), Math.floor((g + 1) * nuclei.length / k)));
          const rest = takeRests();
          if (rest) frames[gapBefore(tokens[0])] += rest;
          const syllableNotes = groupsN.map(() => nextEntry().notes);
          for (let extra = k; extra < pieces; extra++) syllableNotes[k - 1] = syllableNotes[k - 1].concat(nextEntry().notes);
          // Token ranges per syllable: consonants between syllables split one to the next onset, the rest to the coda.
          const bounds = [tokens[0]];
          for (let g = 1; g < k; g++) {
            const lastPrev = groupsN[g - 1][groupsN[g - 1].length - 1], firstNext = groupsN[g][0];
            const between = firstNext - lastPrev - 1;
            bounds.push(between <= 1 ? lastPrev + 1 : firstNext - (between - Math.floor(between / 2)));
          }
          bounds.push(tokens[tokens.length - 1] + 1);
          for (let g = 0; g < k; g++) {
            const notes = syllableNotes[g].map((n) => ({ hz: n.hz * 2 ** (transpose / 12), frames: beatFrames(n.beats) }));
            const total = notes.reduce((s, n) => s + n.frames, 0);
            const range = [];
            for (let i = bounds[g]; i < bounds[g + 1]; i++) range.push(i);
            const nucleusSet = new Set(groupsN[g]);
            const consonants = range.filter((i) => !nucleusSet.has(i));
            let cons = consonants.map((i) => Math.max(1, Math.min(7, predicted[i] || 2)));
            const room = Math.max(consonants.length, total - 3 * nucleusSet.size);
            let sum = cons.reduce((s, x) => s + x, 0);
            while (sum > room && cons.some((x) => x > 1)) { const j = cons.indexOf(Math.max(...cons)); cons[j]--; sum--; }
            consonants.forEach((i, j) => { frames[i] = cons[j]; });
            const vowelFrames = Math.max(nucleusSet.size, total - sum);
            const firstNucleus = groupsN[g][0];
            consonants.forEach((i) => notesOf.set(i, [{ hz: (i < firstNucleus ? notes[0] : notes[notes.length - 1]).hz, frames: frames[i] }]));
            // Several nuclei in one syllable (rare) share its vowel time; the notes run across them in order.
            const per = groupsN[g].map((_, j) => Math.floor(vowelFrames * (j + 1) / groupsN[g].length) - Math.floor(vowelFrames * j / groupsN[g].length));
            let queue = notes.map((n) => ({ ...n, frames: Math.max(1, Math.round(n.frames * vowelFrames / total)) }));
            groupsN[g].forEach((i, j) => {
              frames[i] = per[j];
              let need = per[j];
              const segs = [];
              while (need > 0 && queue.length) {
                const take = Math.min(need, queue[0].frames);
                segs.push({ hz: queue[0].hz, frames: take });
                queue[0].frames -= take; need -= take;
                if (!queue[0].frames) queue.shift();
              }
              if (need > 0) segs.push({ hz: segs.length ? segs[segs.length - 1].hz : notes[notes.length - 1].hz, frames: need });
              notesOf.set(i, segs);
            });
          }
        });
        // Rests at the end of the line, and a short release, go on the last edge token.
        const eos = phones.ids.lastIndexOf(EOS);
        frames[eos >= 0 ? eos : frames.length - 1] += takeRests() + 6;
        // Pitch track: note targets per frame, a glide between them, vibrato on held notes.
        const f0 = [];
        phones.ids.forEach((_, i) => {
          const segs = notesOf.get(i);
          if (!segs) { for (let n = 0; n < frames[i]; n++) f0.push(0); return; }
          for (const seg of segs) {
            const held = seg.frames / fps;
            for (let n = 0; n < seg.frames; n++) {
              const t = n / fps;
              const swell = held > 0.35 ? Math.min(1, Math.max(0, (t - 0.2) / 0.15)) : 0;
              f0.push(seg.hz * 2 ** (depth * swell * Math.sin(2 * Math.PI * rate * t) / 12));
            }
          }
        });
        const alpha = 1 - Math.exp(-1 / (Math.max(glide, 1e-3) * fps));
        let y = 0;
        for (let n = 0; n < f0.length; n++) {
          if (!f0[n]) { y = 0; continue; }
          const target = Math.log(f0[n]);
          y = y ? y + (target - y) * alpha : target;
          f0[n] = Math.exp(y);
        }
        const { audio } = await run(phones, voice, { frames, f0, pitchShift: 0 });
        chunks.push(audio);
        if (onLine) onLine(limitPeak(audio.slice()), li, lines.length);
      }
      if (cursor < entries.length && entries.slice(cursor).some((e) => !e.rest)) {
        throw new Error(`${entries.slice(cursor).filter((e) => !e.rest).length} notes were left over after the last syllable`);
      }
      const audio = new Float32Array(chunks.reduce((s, c) => s + c.length, 0));
      let offset = 0;
      for (const c of chunks) { audio.set(c, offset); offset += c.length; }
      return limitPeak(audio);
    }

    return { speak, sing, run, synthesize, phonemize: (text) => JSON.parse(phonemize(text)), space, sampleRate: space.sample_rate,
             voiceFromSettings: (settings, name) => voiceFromSettings(space, settings, name, base) };
  }

  root.TinyVoca = { load, voiceFromSettings, checkVoice, parseNotes, wav, FORMAT, MODEL };
})(typeof self !== "undefined" ? self : this);
