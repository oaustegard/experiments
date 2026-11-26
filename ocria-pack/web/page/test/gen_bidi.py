"""Print the Bidi_Class run table ocr.js embeds as BIDI_RUNS (non-L runs only).

    python3 test/gen_bidi.py > /tmp/bidi.txt
"""
import unicodedata

CODE = {"R": "R", "AL": "A", "EN": "E", "ES": "S", "ET": "T", "AN": "N", "CS": "C", "NSM": "M", "BN": "Z",
        "B": "B", "S": "G", "WS": "W", "ON": "O", "LRE": "P", "RLE": "Q", "PDF": "U", "LRO": "V", "RLO": "X",
        "LRI": "H", "RLI": "I", "FSI": "F", "PDI": "D"}

runs, prev, start = [], None, 0
for cp in range(0x110000):
    c = unicodedata.bidirectional(chr(cp)) or "L"
    if c != prev:
        if prev is not None:
            runs.append((start, cp - 1, prev))
        prev, start = c, cp
runs.append((start, 0x10FFFF, prev))
out, last = [], 0
for s, e, c in runs:
    if c == "L":
        continue
    out.append(f"{s - last:x}{CODE[c]}{e - s + 1:x}")
    last = e + 1
print(" ".join(out), end="")
