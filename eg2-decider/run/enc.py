"""EmbeddingGemma 2 (onnx-community model_q4.onnx, the browser build) on CPU, with an append-only cache.

Every text is embedded once per (prompt, text) and kept in out/cache/{keys.jsonl, vecs.f32}, flushed
after each batch, so a killed job resumes where it stopped. Vectors are 768-d, L2-normalised, the same
graph the austegard.com pages run on onnxruntime-web.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "out" / "hf"
CACHE = ROOT / "out" / "cache"

PROMPTS = {
    "doc": "title: none | text: ",
    "query": "task: search result | query: ",
    "cls": "task: classification | query: ",
    "fact": "task: fact checking | query: ",
    "sts": "task: sentence similarity | query: ",
}


def key(prompt: str, text: str, max_tokens: int) -> str:
    return hashlib.sha1(f"{max_tokens}\x00{prompt}\x00{text}".encode()).hexdigest()


def render(obj, prefix="") -> str:
    """JSON state -> readable 'a · b: value' lines (embedders read these better than raw JSON)."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            return obj
    lines = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [str(k).replace("_", " ")])
        elif isinstance(o, list):
            if all(not isinstance(x, (dict, list)) for x in o):
                lines.append(f"{' · '.join(path)}: {', '.join(map(str, o))}")
            else:
                for i, v in enumerate(o, 1):
                    if isinstance(v, dict) and "role" in v and "text" in v and len(v) == 2:
                        lines.append(f"{v['role']}: {v['text']}")
                    else:
                        walk(v, path + [str(i)])
        else:
            lines.append(f"{' · '.join(path)}: {o}")

    walk(obj, [prefix] if prefix else [])
    return "\n".join(lines)


class EG2:
    def __init__(self, model="model_q4.onnx", threads=None, tag="", hf=None):
        """hf: directory holding onnx/<model> and tokenizer.json (default the full EG2 download).
        The 4-layer cut (out/cut4, from austegard.com/bsky/image-search/model/ort) outputs 'embedding'."""
        hf = Path(hf) if hf else HF
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or int(os.environ.get("EG2_THREADS", "4"))
        self.sess = ort.InferenceSession(str(hf / "onnx" / model), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.output = self.sess.get_outputs()[0].name if "sentence_embedding" not in {
            o.name for o in self.sess.get_outputs()} else "sentence_embedding"
        self.tok = Tokenizer.from_file(str(hf / "tokenizer.json"))
        t = json.loads((hf / "tokenizer.json").read_text())
        self.specials = sorted((a["content"] for a in t["added_tokens"] if a.get("special")), key=len, reverse=True)
        self.name = model
        CACHE.mkdir(parents=True, exist_ok=True)
        # one writer per cache: a second process embedding at the same time passes its own tag
        stem = f"{model}{'.' + tag if tag else ''}"
        self.kpath, self.vpath = CACHE / f"{stem}.keys.jsonl", CACHE / f"{stem}.vecs.f32"
        self.index = {}
        if self.kpath.exists():
            keys = [json.loads(l)["k"] for l in self.kpath.read_text().splitlines() if l.strip()]
            vecs = np.fromfile(self.vpath, np.float32).reshape(-1, 768)
            n = min(len(keys), len(vecs))  # a crash between the two writes leaves at most one extra row
            self.vecs = [vecs[:n]]
            self.index = {k: i for i, k in enumerate(keys[:n])}
        else:
            self.vecs = []
        self._flat = None

    def clean(self, text):
        for sp in self.specials:
            if sp in text:
                text = text.replace(sp, " ")
        return text

    def _run(self, ids_list):
        S = max(len(e) for e in ids_list)
        ids = np.zeros((len(ids_list), S), np.int64)
        attn = np.zeros((len(ids_list), S), np.int64)
        for b, e in enumerate(ids_list):
            ids[b, :len(e)] = e
            attn[b, :len(e)] = 1
        feed = {"input_ids": ids, "attention_mask": attn}
        empty = np.zeros((0, 512), np.float32)
        for extra in ("image_features", "video_features", "audio_features"):
            if extra in self.inputs:
                feed[extra] = empty
        e = self.sess.run([self.output], feed)[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def embed(self, texts, prompt="doc", max_tokens=1024, tok_budget=8192, log_every=0):
        p = PROMPTS.get(prompt, prompt)
        keys = [key(p, t, max_tokens) for t in texts]
        todo = sorted({k: t for k, t in zip(keys, texts) if k not in self.index}.items())
        if todo:
            enc = [(k, self.tok.encode(p + self.clean(t)).ids[:max_tokens]) for k, t in todo]
            enc.sort(key=lambda x: len(x[1]))
            t0, done = time.time(), 0
            with open(self.kpath, "a") as kf, open(self.vpath, "ab") as vf:
                i = 0
                while i < len(enc):
                    L = len(enc[i][1])
                    j = i + 1
                    while j < len(enc) and (j - i + 1) * len(enc[j][1]) <= tok_budget and j - i < 64:
                        j += 1
                    batch = enc[i:j]
                    v = self._run([e for _, e in batch]).astype(np.float32)
                    vf.write(v.tobytes()); vf.flush()
                    for k, _ in batch:
                        kf.write(json.dumps({"k": k}) + "\n")
                    kf.flush()
                    base = sum(len(x) for x in self.vecs)
                    self.vecs.append(v)
                    for n, (k, _) in enumerate(batch):
                        self.index[k] = base + n
                    done += len(batch)
                    i = j
                    if log_every and done // log_every != (done - len(batch)) // log_every:
                        el = time.time() - t0
                        print(f"  {self.name} {prompt}: {done}/{len(enc)} ({done / el:.1f}/s, len {L})", flush=True)
            self._flat = None
        if self._flat is None or len(self._flat) != len(self.index):
            self._flat = np.concatenate(self.vecs) if self.vecs else np.zeros((0, 768), np.float32)
        return self._flat[[self.index[k] for k in keys]]


ENCODERS = {"full": ("model_q4.onnx", HF), "cut4": ("model_t3w.onnx", ROOT / "out" / "cut4")}


def encoder(tag=""):
    """The encoder named by $EG2_ENCODER (full EG2 q4 by default, or the 4-layer cut 'cut4')."""
    model, hf = ENCODERS[os.environ.get("EG2_ENCODER", "full")]
    return EG2(model, tag=tag, hf=hf)


def results_dir():
    name = os.environ.get("EG2_ENCODER", "full")
    return ROOT / "results" / ("" if name == "full" else name)


if __name__ == "__main__":
    m = EG2()
    t0 = time.time()
    v = m.embed(sys.argv[1:] or ["hello world", "a cat sat on the mat"], prompt="query")
    print(v.shape, v[:, :4], f"{time.time() - t0:.2f}s")
