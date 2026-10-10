"""yake 0.7.3 keyword extraction with the post-tokenization pipeline in Mojo.

Python does what yake does before term statistics: newline/tab normalisation,
sentence splitting and word tokenisation with segtok (the same calls yake
makes), and lowercasing. It then flattens the tokens into UTF-32 buffers and
hands them to `_kernel.extract`, which computes term statistics, the
co-occurrence graph, candidates, scores and deduplication. Python turns the
returned token spans back into keyword strings.
"""
import os
import re
import string
import unicodedata

import numpy as np
from segtok.segmenter import split_multi
from segtok.tokenizer import split_contractions, web_tokenizer

from . import _kernel

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW_DIR = os.path.join(_HERE, "data", "StopwordsList")
_CAP = re.compile(r"^(\s*([A-Z]))")
_DEDUP_MODE = {"jaro_winkler": 3, "jaro": 3, "sequencematcher": 1, "seqm": 1}
_OPTION_KEYS = ["lan", "n", "dedup_lim", "dedup_func", "window_size", "top", "features"]


def _pre_filter(text):
    """yake.data.utils.pre_filter."""
    buffer = ""
    for part in text.split("\n"):
        sep = "\n\n" if _CAP.match(part) else " "
        buffer += sep + part.replace("\t", " ")
    return buffer


def _tokenize_sentences(text):
    """yake.data.utils.tokenize_sentences: one token list per sentence."""
    out = []
    for s in list(split_multi(text)):
        if len(s.strip()) > 0:
            out.append([
                w
                for w in split_contractions(web_tokenizer(s))
                if w and (len(w) == 1 or w[0] != "'")
            ])
    return out


def _load_stopwords_file(lan):
    """yake's KeywordExtractor._load_stopwords for the bundled lists."""
    path = os.path.join(_SW_DIR, f"stopwords_{lan[:2].lower()}.txt")
    if not os.path.exists(path):
        path = os.path.join(_SW_DIR, "stopwords_noLang.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except UnicodeDecodeError:
        print("Warning: reading stopword list as ISO-8859-1")
        with open(path, encoding="ISO-8859-1") as fh:
            text = fh.read()
    return set(text.lower().split("\n"))


def _is_extend(ch):
    """Grapheme Extend plus ZWJ (marks, variation selectors, skin tones)."""
    cp = ord(ch)
    return (
        unicodedata.category(ch) in ("Mn", "Me", "Mc")
        or cp == 0x200D
        or 0xFE00 <= cp <= 0xFE0F
        or 0x1F3FB <= cp <= 0x1F3FF
    )


def _is_pictographic(cp):
    """Approximation of Extended_Pictographic (regional indicators excluded)."""
    if 0x1F1E6 <= cp <= 0x1F1FF:
        return False
    return (
        0x1F000 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or cp in (0xA9, 0xAE, 0x203C, 0x2049, 0x2122, 0x2139, 0x2194, 0x2195, 0x2196,
                  0x2197, 0x2198, 0x2199, 0x21A9, 0x21AA, 0x231A, 0x231B, 0x2328,
                  0x23CF, 0x2B50, 0x2B55)
    )


def _char_flags(ch):
    """Per-character bits. Bits 0-4 are Python's own str predicates; the
    grapheme bits (5-8) drive the Jaro cluster split in the kernel."""
    cp = ord(ch)
    return (
        (1 if ch.isdigit() else 0)
        | (2 if ch.isalpha() else 0)
        | (4 if ch.isupper() else 0)
        | (8 if ch.islower() else 0)
        | (16 if unicodedata.category(ch) == "Lt" else 0)
        | (32 if _is_extend(ch) else 0)
        | (64 if cp == 0x200D else 0)
        | (128 if _is_pictographic(cp) else 0)
        | (256 if 0x1F1E6 <= cp <= 0x1F1FF else 0)
    )


_FLAG_CACHE = {}


def _flags_of(ch):
    flags = _FLAG_CACHE.get(ch)
    if flags is None:
        flags = _FLAG_CACHE[ch] = _char_flags(ch)
    return flags


def _utf32(strings):
    return np.frombuffer("".join(strings).encode("utf-32-le"), dtype=np.uint32)


def _offsets(strings):
    off = np.zeros(len(strings) + 1, dtype=np.int64)
    np.cumsum(np.fromiter(map(len, strings), dtype=np.int64, count=len(strings)), out=off[1:])
    return off


def _addr(arr):
    return arr.__array_interface__["data"][0]


class KeywordExtractor:
    """yake.KeywordExtractor with the same constructor and extract_keywords."""

    def __init__(
        self,
        lan="en",
        n=3,
        dedup_lim=0.9,
        dedup_func="seqm",
        window_size=1,
        top=20,
        features=None,
        stopwords=None,
        lemmatize=False,
        lemma_aggregation="min",
        lemmatizer="spacy",
        **kwargs,
    ):
        if lemmatize:
            raise NotImplementedError("lemmatize is not supported by this port")
        self.config = {
            "lan": lan,
            "n": n,
            "dedup_lim": dedup_lim,
            "dedup_func": dedup_func,
            "window_size": window_size,
            "top": top,
            "features": features,
        }
        for key in _OPTION_KEYS:
            if key in kwargs:
                self.config[key] = kwargs[key]
        given = stopwords or kwargs.get("stopwords")
        self.stopword_set = set(given) if given is not None else _load_stopwords_file(lan)
        self._dedup_mode = _DEDUP_MODE.get(str(dedup_func).lower(), 2)
        words = sorted(self.stopword_set)
        self._sw_words = words
        self._sw_buf = _utf32(words)
        self._sw_off = _offsets(words)

    def extract_keywords(self, text):
        if not text:
            return []
        cfg = self.config
        text = text.replace("\n", " ")
        sentences = _tokenize_sentences(_pre_filter(text))
        toks = [w for sent in sentences for w in sent]
        T = len(toks)
        if T == 0:
            return []
        NS = len(sentences)
        lows = [w.lower() for w in toks]

        obuf = _utf32(toks)
        ooff = _offsets(toks)
        lbuf = _utf32(lows)
        loff = _offsets(lows)
        slen = np.fromiter(map(len, sentences), dtype=np.int64, count=NS)
        sent = np.repeat(np.arange(NS, dtype=np.int64), slen)
        sstart = np.repeat(np.cumsum(slen) - slen, slen).astype(np.int64)

        uniq = set("".join(toks)) | set("".join(lows))
        cls = np.zeros(max(ord(c) for c in uniq) + 1, dtype=np.uint16)
        for ch in uniq:
            cls[ord(ch)] = _flags_of(ch)

        nmax = int(cfg["n"])
        top = int(cfg["top"])
        dedup_lim = float(cfg["dedup_lim"])
        no_dedup = 1 if dedup_lim >= 1.0 else 0
        out_cap = T * max(nmax, 1) + 1
        out_first = np.zeros(out_cap, dtype=np.int64)
        out_len = np.zeros(out_cap, dtype=np.int64)
        out_h = np.zeros(out_cap, dtype=np.float64)
        ctl = np.array(
            [
                T, NS, nmax, int(cfg["window_size"]), self._dedup_mode, no_dedup, top, cls.shape[0],
                _addr(obuf), _addr(ooff), _addr(lbuf), _addr(loff), _addr(self._sw_buf),
                _addr(cls), _addr(sent), _addr(sstart), _addr(out_first), _addr(out_len), _addr(out_h),
                _addr(self._sw_off), len(self._sw_words),
            ],
            dtype=np.int64,
        )
        fl = np.array([dedup_lim], dtype=np.float64)
        count = _kernel.extract(ctl, fl)

        results = []
        for i in range(count):
            first = int(out_first[i])
            k = int(out_len[i])
            if no_dedup:
                kw = " ".join(toks[first:first + k]).lower()
            else:
                kw = " ".join(toks[first:first + k])
            results.append((kw, float(out_h[i])))
        return results
