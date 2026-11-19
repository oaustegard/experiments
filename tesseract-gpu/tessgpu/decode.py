"""CTC decoding of Tesseract LSTM outputs into text.

greedy(): best code per timestep, repeats merged, null dropped, codes mapped back to
unichars through the recoder. Tesseract's own RecodeBeamSearch (no dictionary) usually
returns the same path for alphabetic scripts; beam.py ports it for the rest.
"""

import numpy as np


class Codec:
    def __init__(self, model):
        self.null = model.null_char
        self.chars = model.unicharset
        self.by_codes = {}
        for uid, codes in enumerate(model.recoder):
            if uid < 3 and uid != 0:      # UNICHAR_JOINED / BROKEN share codes with others
                continue
            self.by_codes.setdefault(codes, uid)
        self.single = all(len(c) == 1 for c in model.recoder)
        self.max_len = max(len(c) for c in model.recoder)

    def codes_to_text(self, codes):
        out = []
        i = 0
        while i < len(codes):
            for n in range(min(self.max_len, len(codes) - i), 0, -1):
                uid = self.by_codes.get(tuple(codes[i:i + n]))
                if uid is not None:
                    out.append(self.chars[uid])
                    i += n
                    break
            else:
                i += 1  # an incomplete code sequence: drop it
        return "".join(out)


def best_path(probs, null):
    """Argmax per timestep, merge repeats, drop null. Returns (codes, timesteps)."""
    best = probs.argmax(axis=1)
    codes, steps = [], []
    prev = -1
    for t, c in enumerate(best):
        if c != prev and c != null:
            codes.append(int(c))
            steps.append(t)
        prev = c
    return codes, steps


def clean(text):
    return " ".join(text.split())


def greedy(probs, codec):
    codes, _ = best_path(probs, codec.null)
    return clean(codec.codes_to_text(codes))
