"""Ground truth for the JS engine: HF tokenizer ids and ORT fp32 embeddings.

Usage: python3 -I run/ref.py

Writes out/ref/:
  ids.jsonl       {"t": text, "ids": [...]} for every text the eval uses: the 61,686 posts
                  (document side, no prompt) and the 2,498 test queries with leaf-ir's query
                  prompt, plus EDGE. HF tokenizers, truncation 128, no padding.
  fixtures.json   [{"t", "ids", "emb"}] -- 400 sampled texts + EDGE, embedded by
                  onnx/model.onnx (fp32, MongoDB's own export, so independent of
                  export_weights.py) via its sentence_embedding output.
"""
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "out" / "hf"
REF = ROOT / "out" / "ref"
NATIVE = ROOT.parent / "eg2-native-small" / "out"
POSTS = ROOT.parent / "eg2-query-student" / "out" / "posts.jsonl"
QP = "Represent this sentence for searching relevant passages: "
EDGE = [
    "", " ", "a", "Hello, World!", "naïve café résumé Ångström", "ÀÉÎÕÜ ÇÑ", "straße", "İstanbul ǅ ﬁ",
    "日本語のテキスト and 中文 mixed", "한국어 텍스트", "emoji 🤖🧠✨ and 👩‍💻", "tabs\tand\nnewlines\r\nhere",
    "zero​width‍joiner﻿bom", "control\x00\x01\x7fchars�", "[CLS] literal [SEP] tokens [MASK] [UNK] [PAD]",
    "https://example.com/path?q=1&x=2#frag", "@handle.bsky.social #hashtag $AAPL 100% 3.14 1,000,000",
    "don't won't it's l'été", "C++ C# F# .NET node.js", "—–-‐ “quotes” «guillemets» ‹›", "…⋯·•°±×÷",
    "x" * 150, "supercalifragilisticexpialidociousantidisestablishmentarianism", "Ⅻ ① ² ½ ℃ ™ ©",
    "Ελληνικά Русский עברית العربية हिन्दी", "word " * 200, "ab́c è ñ", "　ideographic　space",
    "ÅÆØ åæø ÿ œ ß ð þ", "𝐁𝐨𝐥𝐝 𝕯𝖔𝖚𝖇𝖑𝖊 ｆｕｌｌｗｉｄｔｈ",
]


def main():
    REF.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(128)
    posts = [p["text"] or "" for p in map(json.loads, POSTS.open())]
    queries = [QP + json.loads(l)["q"] for f in ("synth_test", "real_test") for l in (NATIVE / "q" / f"{f}.jsonl").open()]
    texts = posts + queries + EDGE + [QP + e for e in EDGE]
    enc = tok.encode_batch(texts)
    with (REF / "ids.jsonl").open("w") as f:
        for t, e in zip(texts, enc):
            f.write(json.dumps({"t": t, "ids": e.ids}, ensure_ascii=False) + "\n")
    print("ids", len(texts), flush=True)

    rng = random.Random(0)
    pick = rng.sample(range(len(posts)), 250) + [len(posts) + i for i in rng.sample(range(len(queries)), 150)]
    fx = [texts[i] for i in pick] + EDGE + [QP + e for e in EDGE]
    sess = ort.InferenceSession(str(HF / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])
    out = []
    for t in fx:
        ids = tok.encode(t).ids
        a = np.array([ids], np.int64)
        e = sess.run(["sentence_embedding"], {"input_ids": a, "attention_mask": np.ones_like(a),
                                              "token_type_ids": np.zeros_like(a)})[0][0]
        out.append({"t": t, "ids": ids, "emb": [float(x) for x in e / np.linalg.norm(e)]})
    (REF / "fixtures.json").write_text(json.dumps(out, ensure_ascii=False))
    print("fixtures", len(out))


if __name__ == "__main__":
    main()
