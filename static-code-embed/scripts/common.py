"""Shared pieces for the static-code-embed experiment.

Reuses the mini-CTXBench harness from ../bekko-embedding-bench (instances, rg
arm, dense arm, RRF, recall) so every number here is paired against the same
59 issues on the same tree. Adds one thing: a static-table encoder with a
hybrid tokenizer (whole-word vocabulary first, subword fallback), which is the
Model2Vec inference recipe (lookup, mean, L2) plus the corpus-vocabulary lever.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parents[1]
BENCH = HERE.parent / "bekko-embedding-bench"
WORK = HERE.parent / ".work"
MODELS = Path(os.environ.setdefault("BEKKO_BENCH_MODELS", "/home/user/models"))
os.environ.setdefault("BEKKO_BENCH_REPO", str(WORK / "sklearn"))
sys.path.insert(0, str(BENCH / "scripts"))

from eval_search import (  # noqa: E402
    approx_tokens, arm_dense, arm_rg, extract_identifiers, recall_at, rrf,
)

WORD = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")  # BERT pre-token runs, underscores kept


def load_instances() -> list[dict]:
    return json.load(open(BENCH / "instances.json"))


def load_chunks(mode: str = "ast") -> list[dict]:
    return json.load(open(BENCH / f"chunks_{mode}.json"))


class HybridTok:
    """Whole-word vocabulary first, base subword tokenizer as fallback.

    With an empty word vocabulary this is exactly the base tokenizer, so the
    vanilla potion arm goes through the same code path as the adapted ones.
    Ids >= base vocab size index the word table appended after the base table.
    """

    def __init__(self, base: Tokenizer, words: dict[str, int] | None = None,
                 drop_ids: set[int] | None = None) -> None:
        self.base = base
        self.words = words or {}
        self.drop = drop_ids or set()
        self.n_base = base.get_vocab_size()
        # potion's tokenizer.json carries truncation (512, right); the word
        # path must honour the same cap or the two paths see different text.
        self.max_len = base.truncation["max_length"] if base.truncation else None

    def __call__(self, text: str) -> list[int]:
        if not self.words:
            return [i for i in self.base.encode(text).ids if i not in self.drop]
        out: list[int] = []
        for w in WORD.findall(text):
            j = self.words.get(w)
            if j is not None:
                out.append(self.n_base + j)
            else:
                out.extend(i for i in self.base.encode(w).ids if i not in self.drop)
            if self.max_len and len(out) >= self.max_len:
                break
        return out[: self.max_len] if self.max_len else out


class StaticTable:
    """Model2Vec-style encoder: token lookup, mean pool, L2 normalize."""

    def __init__(self, table: np.ndarray, tok: HybridTok) -> None:
        self.table = np.asarray(table, dtype=np.float32)
        self.tok = tok
        self.dim = self.table.shape[1]

    @classmethod
    def from_model2vec(cls, path: Path, words_path: Path | None = None) -> "StaticTable":
        from safetensors.numpy import load_file
        table = load_file(str(path / "model.safetensors"))["embeddings"]
        base = Tokenizer.from_file(str(path / "tokenizer.json"))
        drop = {i for t, i in base.get_vocab().items() if t in ("[UNK]", "[PAD]", "<unk>", "<pad>")}
        words = None
        if words_path is not None:
            d = json.load(open(words_path))
            words = d["words"]
            table = np.concatenate([table.astype(np.float32), np.load(words_path.with_suffix(".npy"))])
        return cls(table, HybridTok(base, words, drop))

    @classmethod
    def from_dir(cls, path: Path) -> "StaticTable":
        """Our own saved format: table.npy + tokenizer.json + words.json."""
        table = np.load(path / "table.npy")
        base = Tokenizer.from_file(str(path / "tokenizer.json"))
        meta = json.load(open(path / "words.json"))
        drop = set(meta.get("drop_ids", []))
        return cls(table, HybridTok(base, meta["words"], drop))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "table.npy", self.table)
        self.tok.base.save(str(path / "tokenizer.json"))
        json.dump({"words": self.tok.words, "drop_ids": sorted(self.tok.drop)},
                  open(path / "words.json", "w"))

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for k, t in enumerate(texts):
            ids = self.tok(t)
            if ids:
                v = self.table[ids].mean(0)
                n = np.linalg.norm(v)
                out[k] = v / n if n > 0 else v
        return out


def evaluate_arm(name: str, enc_query, corpus_mat: np.ndarray, chunks: list[dict],
                 rg_cache: dict, inst: list[dict]) -> list[dict]:
    rows = []
    for it in inst:
        q = it["title"] + "\n" + it["body"]
        qv = enc_query(q)
        dranked, backing = arm_dense(qv, corpus_mat, chunks)
        rranked, rwall, rchars, idents = rg_cache[it["issue"]]
        fused = rrf(rranked, dranked)
        gold = it["gold"]
        rows.append({
            "arm": name, "issue": it["issue"], "n_idents": len(idents), "n_gold": len(gold),
            "rg_r5": recall_at(rranked, gold, 5), "rg_r10": recall_at(rranked, gold, 10),
            "dense_r5": recall_at(dranked, gold, 5), "dense_r10": recall_at(dranked, gold, 10),
            "rrf_r5": recall_at(fused, gold, 5), "rrf_r10": recall_at(fused, gold, 10),
            "dense_tokens": approx_tokens(sum(len(chunks[i]["text"]) for i in backing[:10])),
            "rg_tokens": approx_tokens(rchars),
        })
    return rows


def rg_all(inst: list[dict]) -> dict:
    cache = {}
    for it in inst:
        idents = extract_identifiers(it["title"] + "\n" + it["body"])
        ranked, wall, chars = arm_rg(idents)
        cache[it["issue"]] = (ranked, wall, chars, idents)
    return cache
