"""Builds the fixed 256-sequence corpus for the modernbert-bidirectionality probe.

Walks wikitext-103-raw-v1 validation lines in order, drops blanks and headings,
accumulates a buffer until its tokenization reaches >= 256 tokens, truncates to
exactly 256 ids (forcing [SEP] at position 255), and starts the next buffer
from the following line. Deterministic: no randomness is used here (the
masking draw lives in probe.py). Output is fixed once written.

Run: python3 data.py
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

TOKENIZER_NAME = "answerdotai/ModernBERT-base"
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "validation"
N_SEQUENCES = 256
SEQ_LEN = 256
# Not used for sampling here (data.py is deterministic); recorded in the
# metadata because it is the seed the rest of the experiment is keyed to.
GLOBAL_SEED = 20260920


def is_dropped_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("="):
        return True
    return False


def build_corpus(tokenizer, lines, n_sequences: int, seq_len: int):
    sequences: list[list[int]] = []
    buffer = ""
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    for line in lines:
        if len(sequences) >= n_sequences:
            break
        if is_dropped_line(line):
            continue

        buffer = f"{buffer} {line.strip()}" if buffer else line.strip()
        ids = tokenizer(buffer, add_special_tokens=True)["input_ids"]

        if len(ids) >= seq_len:
            truncated = ids[:seq_len]
            truncated[0] = cls_id
            truncated[-1] = sep_id
            sequences.append(truncated)
            buffer = ""  # remainder of this line/buffer is discarded

    return sequences


def dataset_revision(ds) -> str | None:
    try:
        version = ds.info.version
        if version is None:
            return None
        return str(version)
    except Exception:
        return None


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    lines = ds["text"]

    sequences = build_corpus(tokenizer, lines, N_SEQUENCES, SEQ_LEN)
    if len(sequences) < N_SEQUENCES:
        raise RuntimeError(
            f"only built {len(sequences)}/{N_SEQUENCES} sequences from the corpus"
        )
    for seq in sequences:
        assert len(seq) == SEQ_LEN
        assert seq[0] == tokenizer.cls_token_id
        assert seq[-1] == tokenizer.sep_token_id

    ids_path = DATA_DIR / "corpus_ids.json"
    tmp_path = ids_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(sequences, f)
    tmp_path.replace(ids_path)

    meta = {
        "n": len(sequences),
        "seed": GLOBAL_SEED,
        "tokenizer": TOKENIZER_NAME,
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": dataset_revision(ds),
        "seq_len": SEQ_LEN,
    }
    meta_path = DATA_DIR / "corpus_meta.json"
    tmp_meta_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    tmp_meta_path.replace(meta_path)

    print("done -> data/corpus_ids.json")


if __name__ == "__main__":
    main()
