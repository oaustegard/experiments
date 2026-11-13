"""Bake what the page needs besides the model: phonemes for each line, the vocabulary, and voice packs.

The page runs only the model. Text -> phonemes stays in Python (misaki), so every line is stored as KPipeline would
hand it to the model: chunks of at most 510 phonemes, each with its tokens (text, phonemes, whitespace) for the
word-timing rule. Voice packs are written one file per voice as raw float16, [510, 256] (the 510 rows are style
vectors indexed by phoneme count; fp16 keeps every word time exact, kokoro-onnx-kinescript round 5).

Run with a Python that has misaki, spaCy's en_core_web_sm and onnxruntime:

    python bake.py MODEL.onnx VOICES.bin REQUESTS.json OUTDIR VOICE ...
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt

HERE = Path(__file__).resolve().parent
# en_tokenize from kinescript-kokoro's onnx engine (a port of kokoro/pipeline.py), loaded by path, numpy only
_path = HERE.parents[1] / "kinescript" / "plugins" / "kinescript-kokoro" / "kinescript_kokoro" / "onnx.py"
_spec = importlib.util.spec_from_file_location("kokoro_onnx_engine", _path)
K = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)


def g2p_for(lang, cache={}):
    if lang not in cache:
        from misaki import en, espeak
        cache[lang] = en.G2P(trf=False, british=lang == "b", fallback=espeak.EspeakFallback(british=lang == "b"),
                             unk="")
    return cache[lang]


def bake_line(req):
    _, tokens = g2p_for(req["speaker"][0])(req["text"])
    chunks = [{"ps": ps, "tokens": [{"text": t.text, "phonemes": t.phonemes or "", "ws": bool(t.whitespace)}
                                     for t in tks]}
              for ps, tks in K.en_tokenize(tokens) if ps]
    return {"id": req["id"], "text": req["text"], "speaker": req["speaker"], "rate": req.get("rate", 1),
            "chunks": chunks}


if __name__ == "__main__":
    model, voices, requests, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    (out / "voices").mkdir(parents=True, exist_ok=True)
    sess = rt.InferenceSession(model, providers=["CPUExecutionProvider"])
    vocab = json.loads(sess.get_modelmeta().custom_metadata_map["kokoro_config"])["vocab"]
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    lines = [bake_line(r) for r in json.loads(requests.read_text())]
    (out / "lines.json").write_text(json.dumps(lines, ensure_ascii=False, indent=1))
    packs = np.load(voices)
    for name in sys.argv[5:]:
        packs[name].reshape(510, 256).astype("<f2").tofile(out / "voices" / f"{name}.f16.bin")
    print(len(lines), "lines,", sum(len(x["chunks"]) for x in lines), "chunks;", len(vocab), "symbols;",
          len(sys.argv[5:]), "voices")
