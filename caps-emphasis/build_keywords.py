"""Bin prohibitive keywords by what capitalising them costs in tokens.

The pilot used one keyword per token-delta level, which makes the delta a
relabelling of "which word" (adversarial review finding D3). This fills each bin
with several keywords so the case effect can be estimated within a bin and the
delta slope estimated across bins with keyword as a random factor.
"""
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C

# frame -> the directive sentence; {K} is the keyword, {W} the forbidden word.
PROHIBITIVE = [
    ("never",      "{K} mention the word {W}."),
    ("avoid",      "{K} mentioning the word {W}."),
    ("omit",       "{K} the word {W} from your answer."),
    ("exclude",    "{K} the word {W} from your answer."),
    ("skip",       "{K} the word {W} entirely."),
    ("suppress",   "{K} the word {W} in your answer."),
    ("withhold",   "{K} the word {W} from your answer."),
    ("do not",     "{K} mention the word {W}."),
    ("must not",   "You {K} mention the word {W}."),
    ("cannot",     "You {K} mention the word {W}."),
    ("refuse to",  "You must {K} mention the word {W}."),
    ("decline to", "You must {K} mention the word {W}."),
    ("refrain from", "{K} mentioning the word {W}."),
    ("abstain from", "{K} mentioning the word {W}."),
    ("on no account", "{K} mention the word {W}."),
    ("at no point",   "{K} mention the word {W}."),
    ("by no means",   "{K} mention the word {W}."),
    ("under no circumstances", "{K} mention the word {W}."),
    ("it is forbidden to", "{K} mention the word {W}."),
    ("it is prohibited to", "{K} mention the word {W}."),
    ("you are banned from", "{K} mentioning the word {W}."),
    ("you are barred from", "{K} mentioning the word {W}."),
    ("drop",       "{K} the word {W} from your answer."),
    ("leave out",  "{K} the word {W} of your answer."),
    ("steer clear of", "{K} the word {W}."),
    ("keep away from", "{K} the word {W}."),
]


def main():
    bins = collections.defaultdict(list)
    for kw, frame in PROHIBITIVE:
        lo = C.ntok(" " + kw)
        up = C.ntok(" " + kw.upper())
        bins[up - lo].append(dict(keyword=kw, frame=frame, lower_tokens=lo,
                                  caps_tokens=up, delta=up - lo))
    out = {str(k): v for k, v in sorted(bins.items())}
    C.dump(os.path.join(HERE, "keywords.json"), out)
    for d, rows in sorted(bins.items()):
        print(f"delta=+{d}  n={len(rows):2d}  " +
              ", ".join(r["keyword"] for r in rows))


if __name__ == "__main__":
    main()
