"""Per-character Unicode properties the pipeline reads (Python's unicodedata and str
predicates), for every character of every recognizer vocabulary, as one JSON object:
{char: flags}. Bits: 1 isalpha, 2 isdigit, 4 isupper, 8 islower, 16 isspace,
32 bidi R/AL, 64 bidi L, 128 bidi EN/AN, 256 category Mn/Me, 512 category Mc,
1024 category Lo, 2048 east-asian wide (W/F).
    python3 run/unicode_props.py /tmp/indra-models/meta.json web/models/PACK/uprops.json"""
import json
import sys
import unicodedata

meta = json.load(open(sys.argv[1]))
chars = set()
for k, v in meta.items():
    if k.startswith("vocab"):
        for t in json.loads(v)["chars"]:
            chars.update(t)
# Hangul syllables composed from jamo by NFC
chars.update(chr(c) for c in range(0xAC00, 0xD7A4))
out = {}
for c in sorted(chars):
    b = unicodedata.bidirectional(c)
    cat = unicodedata.category(c)
    f = (c.isalpha() * 1 | c.isdigit() * 2 | c.isupper() * 4 | c.islower() * 8 | c.isspace() * 16
         | (b in ("R", "AL")) * 32 | (b == "L") * 64 | (b in ("EN", "AN")) * 128
         | (cat in ("Mn", "Me")) * 256 | (cat == "Mc") * 512 | (cat == "Lo") * 1024
         | (unicodedata.east_asian_width(c) in "WF") * 2048)
    if f != 64 | 1 | 8:  # a plain lowercase Latin-like letter is the default, left out
        out[c] = f
json.dump({"default": 64 | 1 | 8, "props": out}, open(sys.argv[2], "w"), ensure_ascii=False, separators=(",", ":"))
print(len(chars), "chars,", len(out), "stored")
