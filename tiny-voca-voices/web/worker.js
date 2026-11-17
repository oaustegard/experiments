// Phonemizer (dizione through Pyodide) and the tunable tiny-voca model (onnxruntime-web), off the main thread.
// Adapted from tiny-voca's own demo (github.com/n4ze3m/tiny-voca, web/public/worker.js, Apache-2.0): same
// sentence splitting and phonemizer setup; the model takes voice embeddings and four more controls.
//
// In:  {type: "load"}, {type: "speak", id, text, settings}
// Out: status, ready, sentence (audio as soon as it is made), done (whole clip), error

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
// The phonemizer data, pinned to the model repository's revision of 2026-11-16.
const HF = "https://huggingface.co/n4ze3m/tiny-voca/resolve/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14/phonemizer/";
const DATA = ["lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json"];
const VERSION = "1";

importScripts(PYODIDE + "pyodide.js", ORT + "ort.wasm.min.js");

let pyodide, phonemize, session, space;
const status = (text, progress) => postMessage({ type: "status", text, progress });

async function download(url, label) {
  const key = `${url}${url.includes("?") ? "&" : "?"}v=${VERSION}`;
  let store = null;
  try {
    store = await caches.open("tiny-voca-tuner-" + VERSION);
    const hit = await store.match(key);
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
      status(label, received / total);
    }
    bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  }
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

async function loadPhonemizer() {
  status("Loading the phonemizer");
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  await pyodide.loadPackage("numpy");
  const [source, ...files] = await Promise.all([
    download("dizione.zip", "Loading the phonemizer"),
    ...DATA.map((name) => download(HF + name, "Loading the dictionary")),
  ]);
  status("Starting the phonemizer");
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
  phonemize = pyodide.globals.get("phonemize_json");
}

async function loadModel() {
  space = await (await fetch("voice-space.json")).json();
  const bytes = await download("tiny-voca-tunable.onnx", "Loading the model");
  status("Starting the model");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

// The two embeddings of a voice: (1 - m) * F2 + m * M2, plus each timbre axis times its direction, scaled by
// the F2-M2 distance in that table so that 1.0 on an axis is as far as F2 is from M2.
function embedding(table, settings) {
  const t = space.voice_space[table];
  const v = new Float32Array(t.rows[0].length);
  const m = settings.mix;
  for (let i = 0; i < v.length; i++) v[i] = (1 - m) * t.rows[0][i] + m * t.rows[1][i];
  settings.axes.forEach((amount, k) => {
    if (!amount) return;
    const d = t.directions[k];
    for (let i = 0; i < v.length; i++) v[i] += amount * t.distance * d[i];
  });
  return v;
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

const SEMITONE = Math.log(2) / 12;

async function synthesize(phones, settings, voice, frameVoice) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("float32", voice, [1, voice.length]),
    frame_voice: new ort.Tensor("float32", frameVoice, [1, frameVoice.length]),
    speed: scalar(settings.speed), pitch_shift: scalar(settings.pitch * SEMITONE), pitch_range: scalar(settings.range),
    formant: scalar(settings.formant), tilt: scalar(settings.tilt), breath: scalar(settings.breath),
  });
  return output.audio.data;
}

const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

async function speak({ id, text, settings }) {
  const started = performance.now();
  const voice = embedding("voice", settings);
  const frameVoice = embedding("frame_voice", settings);
  const list = sentences(finished(text));
  const chunks = [];
  let modelTime = 0;
  for (let index = 0; index < list.length; index++) {
    for (const phones of fit(list[index])) {
      const mark = performance.now();
      const audio = await synthesize(phones, settings, voice, frameVoice);
      modelTime += performance.now() - mark;
      chunks.push(audio);
      const heard = audio.slice();
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: phones.text, audio: heard,
                    sampleRate: space.sample_rate }, [heard.buffer]);
    }
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: "tiny-voca only reads English.", plain: true });
    return;
  }
  const audio = new Float32Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: space.sample_rate, modelTime, totalTime: performance.now() - started },
              [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await Promise.all([loadPhonemizer(), loadModel()]);
      postMessage({ type: "ready" });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
};
