"""Official jina-embeddings-v5-text-nano q4 encoder — the current remax_kb default.

Pinned to the same repo/revision remax_kb pins (`embedders.py`):
``jinaai/jina-embeddings-v5-text-nano-retrieval`` @ ac5d898c. Split ONNX
(graph + external-data weights).

Two differences from bekko that must be honoured or the comparison is invalid:
  * jina uses **query/document prefixes** ("Query: " / "Document: "); bekko
    uses none.
  * jina pools **last-token**; the q4 export exposes a ``sentence_embedding``
    output that already does this, so we read that rather than re-pooling.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(os.environ.get("BEKKO_BENCH_MODELS", "/home/user/models")) / "jina-q4"

PROMPTS = {"query": "Query: ", "document": "Document: "}


class JinaQ4Encoder:
    full_dim = 768
    model_id = "jinaai/jina-embeddings-v5-text-nano-retrieval@ac5d898c"

    def __init__(self, max_len: int = 512, threads: int | None = None) -> None:
        self.tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="<|end_of_text|>")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(
            str(ROOT / "model_q4.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def model_bytes(self) -> int:
        return sum(
            (ROOT / f).stat().st_size for f in ("model_q4.onnx", "model_q4.onnx_data")
        )

    def encode(
        self, texts: list[str], *, prompt: str = "document", batch_size: int = 8,
        dim: int | None = None,
    ) -> np.ndarray:
        pre = PROMPTS[prompt]
        out = np.empty((len(texts), self.full_dim), dtype=np.float32)
        for s in range(0, len(texts), batch_size):
            batch = [pre + t for t in texts[s : s + batch_size]]
            encs = self.tok.encode_batch(batch)
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            res = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})
            emb = res[self.out_names.index("sentence_embedding")]
            out[s : s + len(batch)] = emb
        n = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.clip(n, 1e-9, None)
        if dim:
            v = out[:, :dim]
            return (v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)).astype(
                np.float32
            )
        return out
