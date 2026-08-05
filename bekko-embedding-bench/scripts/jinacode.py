"""jinaai/jina-embeddings-v2-base-code — a code-trained encoder, for the
cross-modal half of the benchmark.

Part A is natural-language bug report -> source file, i.e. **NL query against
code**. bekko is trained for text retrieval and jina v5 nano for general text;
this model is trained on (docstring, code) pairs over 30 languages, which is
exactly that pairing. If a code-tuned encoder is going to beat a general one
anywhere, this task is where.

BERT-family, 12 layers x 768 hidden, mean pooling, ALiBi (so no position cap),
61k code-aware vocab, no query/document prefixes.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(os.environ.get("BEKKO_BENCH_MODELS", "/home/user/models")) / "jina-code"


class JinaCodeEncoder:
    full_dim = 768
    model_id = "jinaai/jina-embeddings-v2-base-code"

    def __init__(self, max_len: int = 512, threads: int | None = None) -> None:
        self.tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="<pad>")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(ROOT / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def model_bytes(self) -> int:
        return (ROOT / "model.onnx").stat().st_size

    def encode(
        self, texts: list[str], *, batch_size: int = 16, sort_by_length: bool = True,
        progress: bool = False,
    ) -> np.ndarray:
        if sort_by_length and len(texts) > batch_size:
            order = np.argsort([len(t) for t in texts], kind="stable")
            packed = self.encode([texts[i] for i in order], batch_size=batch_size,
                                 sort_by_length=False, progress=progress)
            out = np.empty_like(packed)
            out[order] = packed
            return out
        out = np.empty((len(texts), self.full_dim), dtype=np.float32)
        t0 = time.time()
        for s in range(0, len(texts), batch_size):
            batch = texts[s : s + batch_size]
            encs = self.tok.encode_batch(batch)
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            feed = {k: v for k, v in feed.items() if k in self.input_names}
            hidden = self.sess.run(None, feed)[0]
            m = mask.astype(np.float32)[..., None]
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out[s : s + len(batch)] = pooled
            if progress and (s // batch_size) % 20 == 0:
                print(f"  jina-code: {s + len(batch)}/{len(texts)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.clip(n, 1e-9, None)).astype(np.float32)
