"""Phoneme ids and stress for test sentences, for checks run outside the jail."""
import json
from tiny_voca.infer import finished, sentences
from tiny_voca.text import phonemize, set_frontend
set_frontend("own")
TEXTS = [
    "She had never seen the sea before, and for a long while she stood at the edge of the water without saying a word.",
    "The meeting moved to three thirty on Thursday, so please bring the quarterly numbers and a working laptop.",
    "Did you really think I would forget your birthday?",
    "Hello! This is tiny voca, a small text to speech model that runs on a phone or in a browser.",
    "The ravens flew north over the fjord as the light went out of the sky.",
    "Turn left at the second traffic light, then keep going until you see the bakery.",
]
out = []
for text in TEXTS:
    for s in sentences(finished(text)):
        p = phonemize(s)
        out.append({"text": s, "ids": list(p.ids), "stress": list(p.stress)})
json.dump(out, open("out/phones.json", "w"), indent=0)
print(len(out))
