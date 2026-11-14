"""Reference phonemes for checking web/g2p.js: misaki 0.9.4, British, no espeak fallback (unknown words come back
empty, as in the port), on a corpus of the narration lines, prose sentences from this repo's writeups, and
hand-written edge cases. Writes results/g2p_reference.json.

    python g2p_corpus.py      (a Python with misaki and spaCy's en_core_web_sm)
"""
import json
import random
import re
from pathlib import Path

from misaki import en

HERE = Path(__file__).resolve().parent
EDGE = [
    "I read the book yesterday, and I'll read it again tomorrow.",
    "Don't worry: it's only 3.14 percent of the total, isn't it?",
    "The meeting is at 9:30 on March 3rd, 2024, in room 101.",
    "It cost $5.50, or about £4 and 20 pence, last year.",
    "Dr. Smith and Mr. Jones met the CEO of NASA in the U.S. last week.",
    "We sold 1,250 units in 1999 and 12,000 in 2005; that's 10x more.",
    "A state-of-the-art, well-known model can't do everything — yet.",
    "\"Quoted text,\" she said, \"ends here.\" (Brackets, too.)",
    "The 21st century began in 2001, not 2000.",
    "Muninn says [Kokoro](/kˈQkəɹQ/) is [really](+2) small.",
    "Is it 50% or 0.5? We'll see after the 2nd test.",
    "An hour, a unicorn, an apple, the end, the apple.",
    "They used to live here; we live there now.",
    "E.g. this, i.e. that, etc. and so on.",
    "He was born on 4 July 1776 and died in 1826.",
    "Version 2.5.1 shipped with 3 bugs and 42 fixes.",
    "Hello, world! How's it going? Fine, thanks.",
    "The quick brown fox jumps over the lazy dog.",
    "Kinescript renders JSON scenes to MP4 with ffmpeg.",
    "Running, jumped, cats, boxes, studies, stopped, planning.",
]


def prose(n=160):
    sents = []
    for f in sorted((HERE.parent).glob("*/RESULTS.md")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"<!--.*?-->|`[^`]*`|\[([^\]]*)\]\([^)]*\)|[*_#|>]", lambda m: m.group(1) or " ", text)
        for s in re.split(r"(?<=[.!?])\s+", text):
            s = " ".join(s.split())
            if 40 <= len(s) <= 200 and re.fullmatch(r"[A-Za-z0-9 ,.;:'\"()%$£\-—?!/]+", s):
                sents.append(s)
    random.Random(0).shuffle(sents)
    return sents[:n]


narration = [r["text"] for r in json.loads((HERE / "requests.json").read_text())]
g2p = en.G2P(trf=False, british=True, fallback=None, unk="")
out = []
for kind, texts in (("narration", narration), ("edge", EDGE), ("prose", prose())):
    for t in texts:
        ps, tokens = g2p(t)
        out.append({"kind": kind, "text": t, "ps": ps,
                    "tokens": [{"text": x.text, "phonemes": x.phonemes or "", "ws": bool(x.whitespace)} for x in tokens]})
(HERE / "results" / "g2p_reference.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(len(out), "sentences")
