"""Build the data file dizione.js reads: dizione's tables and tiny-voca's built phonemizer data in one .bin.

    python3 -I run/dizione_bundle.py --data PHONEMIZER_DIR --tables DIZIONE_SRC/dizione/data [--out web]

PHONEMIZER_DIR holds the built files of tiny-voca's phonemizer (lexicon.tsv, align.json, g2p.json, g2p.npz,
homographs.json: huggingface.co/n4ze3m/tiny-voca, phonemizer/), the tables are dizione's data/*.txt. This
script only reads those files (json, numpy) and never imports dizione. It writes web/dizione-<sha12>.bin, removes
older dizione-*.bin bundles there and points DATA_FILE in web/dizione.js at the new name.

Layout: "DIZ1", u32 header length, header JSON (tables, alignment table, homograph weights, model settings,
section offsets), then binary sections on 4-byte boundaries:
  lexicon.word_prefix   u8 per word: letters shared with the previous word (words sorted)
  lexicon.word_suffix   the rest of each word, 0-terminated
  lexicon.phone_prefix  u8 per word: phone codes shared with the previous word's pronunciation
  lexicon.phone_suffix  the rest of the codes (1 byte per phone with its stress), 0-terminated
  g2p.<array>           the unknown-word model's weights, float32 little-endian, exactly as in g2p.npz
"""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

TABLES = ("abbreviations.txt", "chat.txt", "endings.txt", "exceptions.txt", "function_words.txt", "homograph_rules.txt",
          "letters.txt", "phrases.txt", "units.txt")
ORDER = ("slot", "embed__weight", "gru__weight_ih_l0", "gru__weight_hh_l0", "gru__bias_ih_l0", "gru__bias_hh_l0",
         "gru__weight_ih_l0_reverse", "gru__weight_hh_l0_reverse", "gru__bias_ih_l0_reverse", "gru__bias_hh_l0_reverse",
         "gru__weight_ih_l1", "gru__weight_hh_l1", "gru__bias_ih_l1", "gru__bias_hh_l1",
         "gru__weight_ih_l1_reverse", "gru__weight_hh_l1_reverse", "gru__bias_ih_l1_reverse", "gru__bias_hh_l1_reverse",
         "out__weight", "out__bias")


def front_code(items):
    """(shared prefix length with the previous item, rest) for each item of a sequence of sequences."""
    previous, out = (), []
    for item in items:
        n = 0
        while n < min(len(previous), len(item), 255) and previous[n] == item[n]:
            n += 1
        out.append((n, item[n:]))
        previous = item
    return out


def lexicon_sections(path):
    rows = [line.split("\t", 1) for line in path.read_text(encoding="utf-8").splitlines() if line]
    words = [word for word, _ in rows]
    assert words == sorted(words) and len(set(words)) == len(words), "lexicon.tsv must be sorted with unique words"
    assert all(word.isascii() and "\0" not in word for word in words)
    pronunciations = [text.split() for _, text in rows]
    assert all(" ".join(p) == text for p, (_, text) in zip(pronunciations, rows)), "single spaces between phones"
    counts = collections.Counter(symbol for p in pronunciations for symbol in p)
    symbols = [s for s, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    assert len(symbols) < 255
    code = {s: k + 1 for k, s in enumerate(symbols)}
    word_prefix, word_suffix = bytearray(), bytearray()
    for n, rest in front_code(words):
        word_prefix.append(n)
        word_suffix += rest.encode("ascii") + b"\0"
    phone_prefix, phone_suffix = bytearray(), bytearray()
    for n, rest in front_code([tuple(code[s] for s in p) for p in pronunciations]):
        phone_prefix.append(n)
        phone_suffix += bytes(rest) + b"\0"
    meta = {"count": len(words), "codes": sum(map(len, pronunciations)), "symbols": [""] + symbols}
    return meta, {"lexicon.word_prefix": bytes(word_prefix), "lexicon.word_suffix": bytes(word_suffix),
                  "lexicon.phone_prefix": bytes(phone_prefix), "lexicon.phone_suffix": bytes(phone_suffix)}


def model_sections(directory):
    settings = json.loads((directory / "g2p.json").read_text(encoding="utf-8"))
    with np.load(directory / "g2p.npz", allow_pickle=False) as arrays:
        assert sorted(arrays.files) == sorted(ORDER), arrays.files
        weights = {name: arrays[name] for name in ORDER}
    specs, sections = {}, {}
    for name, array in weights.items():
        assert array.dtype == np.float32
        specs[name] = {"shape": list(array.shape), "dtype": "f32"}
        sections["g2p." + name] = np.ascontiguousarray(array, dtype="<f4").tobytes()
    return {"settings": settings, "arrays": specs}, sections


def build(data, tables):
    lexicon, sections = lexicon_sections(data / "lexicon.tsv")
    g2p, more = model_sections(data)
    sections.update(more)
    header = {
        "version": 1,
        "tables": {name: (tables / name).read_text(encoding="utf-8") for name in TABLES},
        "align": json.loads((data / "align.json").read_text(encoding="utf-8")),
        "homographs": json.loads((data / "homographs.json").read_text(encoding="utf-8")),
        "g2p": g2p,
        "lexicon": lexicon,
        "sources": {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                    for name in ("lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json")},
    }
    # Offsets depend on the header's length and the header holds the offsets: lay out until it is stable.
    offsets = {}
    while True:
        header["sections"] = offsets
        blob = json.dumps(header, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        position = (8 + len(blob) + 3) // 4 * 4
        layout = {}
        for name, payload in sections.items():
            layout[name] = [position, len(payload)]
            position = (position + len(payload) + 3) // 4 * 4
        if layout == offsets:
            break
        offsets = layout
    out = bytearray(b"DIZ1" + struct.pack("<I", len(blob)) + blob)
    for name, payload in sections.items():
        out += b"\0" * (offsets[name][0] - len(out)) + payload
    out += b"\0" * ((-len(out)) % 4)
    return bytes(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="tiny-voca's built phonemizer directory")
    parser.add_argument("--tables", type=Path, required=True, help="dizione's data/ directory (the .txt tables)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "web")
    args = parser.parse_args()
    bundle = build(args.data, args.tables)
    name = f"dizione-{hashlib.sha256(bundle).hexdigest()[:12]}.bin"
    for old in args.out.glob("dizione-*.bin"):
        if old.name != name and old.name != "dizione-src.bin":
            old.unlink()
    (args.out / name).write_bytes(bundle)
    script = args.out / "dizione.js"
    text = script.read_text(encoding="utf-8")
    start = text.index('const DATA_FILE = "') + len('const DATA_FILE = "')
    script.write_text(text[:start] + name + text[text.index('"', start):], encoding="utf-8")
    print(f"{name}: {len(bundle):,} bytes, {len(gzip.compress(bundle, 9)):,} gzip -9")


if __name__ == "__main__":
    main()
