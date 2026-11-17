"""Reference outputs of Python dizione, for run/dizione_check.mjs. Runs only inside the jail (it imports dizione):

    sh scripts/jail.sh -C /tmp/jail/tv -e PYTHONPATH=/tmp/jail/tv/diz/src -e DIZIONE_DATA=/tmp/jail/tv/w/phonemizer \
        -- python3 port/dizione_reference.py sentences CORPUS.json OUT.jsonl [DEBUG.jsonl]
    ... -- python3 port/dizione_reference.py g2p WORDS.json OUT.jsonl

sentences: one JSON line per corpus entry, in order, with what tinyvoca.js got from Pyodide: "ok" with ids, stress
and text; "null" when dizione.phonemize raised ValueError and dizione.pieces has no words; "throw" with the error
otherwise. The optional DEBUG file holds the normaliser's tokens and every word's source, for finding a mismatch.
g2p: the unknown-word model alone, Model.predict for each word, with the smallest gap between the two best
log-probabilities over the frames (how close the argmax came to a tie).
"""

import json
import sys
import time

import numpy as np

import dizione
from dizione import normalize
from dizione.convert import to_text


def outcome(text):
    try:
        try:
            result = dizione.phonemize(text)
        except ValueError:
            if any(word for mark, words in dizione.pieces(text) if not mark for word in words):
                raise
            return {"kind": "null"}
        return {"kind": "ok", "ids": list(result.ids), "stress": list(result.stress), "text": str(result)}
    except Exception as error:                      # noqa: BLE001 - every failure is a recorded outcome
        return {"kind": "throw", "error": f"{type(error).__name__}: {error}"[:300]}


def debug(text):
    phonemizer = dizione.get()
    try:
        tokens = phonemizer.normalizer.tokens(text)
        said = [t.mark if isinstance(t, normalize.Punct) else
                ("+" if t.attach else "") + ("[" + t.text + "]" if t.spell else t.text) for t in tokens]
        units = [[u.word, u.source, to_text(u.phones)] for kind, value in phonemizer.analyse(text) if kind == "clause"
                 for u in value]
        return {"said": " ".join(said), "units": units}
    except Exception as error:                      # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"[:300]}


def sentences(corpus, out, debug_out=None):
    texts = json.load(open(corpus, encoding="utf-8"))
    start = time.perf_counter()
    with open(out, "w", encoding="utf-8") as sink:
        for index, text in enumerate(texts):
            sink.write(json.dumps({"i": index, **outcome(text)}, ensure_ascii=False) + "\n")
    elapsed = time.perf_counter() - start
    if debug_out:
        with open(debug_out, "w", encoding="utf-8") as sink:
            for index, text in enumerate(texts):
                sink.write(json.dumps({"i": index, **debug(text)}, ensure_ascii=False) + "\n")
    print(f"{len(texts)} sentences in {elapsed:.1f} s ({1000 * elapsed / max(len(texts), 1):.2f} ms each)")


def g2p(words, out):
    model = dizione.get().model
    items = json.load(open(words, encoding="utf-8"))
    start = time.perf_counter()
    with open(out, "w", encoding="utf-8") as sink:
        for word in items:
            spelling = model.spellable(word)
            gap = None
            if spelling:
                logp = model.log_probabilities(spelling)
                top = np.sort(logp, axis=1)
                gap = float((top[:, -1] - top[:, -2]).min())
            sink.write(json.dumps({"w": word, "p": model.predict(word), "gap": gap}) + "\n")
    print(f"{len(items)} words in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sentences":
        sentences(*sys.argv[2:])
    elif mode == "g2p":
        g2p(*sys.argv[2:])
    else:
        sys.exit(__doc__)
