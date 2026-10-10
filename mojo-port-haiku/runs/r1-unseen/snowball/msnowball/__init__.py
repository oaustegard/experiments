"""snowballstemmer's english, german, russian and french stemmers, in Mojo.

The algorithm runs in Mojo (msnowball/kernel.mojo, built to _kernel.so). This
shim only moves words across the boundary as flat UTF-32-LE buffers, one
UInt32 per code point, which is how Python indexes str. The buffers are stdlib
arrays; their addresses are handed to the kernel.
"""
import sys
from array import array

from . import _kernel  # noqa: F401

_LANGS = ['armenian', 'basque', 'catalan', 'dutch_porter', 'english', 'estonian', 'french', 'german', 'hungarian', 'indonesian', 'irish', 'italian', 'nepali', 'norwegian', 'persian', 'portuguese', 'romanian', 'russian', 'serbian', 'sesotho', 'spanish', 'swedish', 'yiddish']
_NEW = {l: getattr(_kernel, "new_" + l) for l in _LANGS}
_STEM = {l: getattr(_kernel, "stem_" + l) for l in _LANGS}
# One heap-allocated Mojo stemmer per language, built on first use (the
# Among tables are built once, not per call).
_HANDLES = {}

_U32 = "I"
assert array(_U32).itemsize == 4


def _handle(lang):
    handle = _HANDLES.get(lang)
    if handle is None:
        handle = _NEW[lang]()
        _HANDLES[lang] = handle
    return handle


def _utf32(text):
    buf = array(_U32)
    buf.frombytes(text.encode("utf-32-le", "surrogatepass"))
    if sys.byteorder == "big":
        buf.byteswap()
    return buf


def _stem_batch(lang, words):
    n = len(words)
    if n == 0:
        return []
    handle = _handle(lang)
    stem = _STEM[lang]
    lens = array(_U32, map(len, words))
    src = _utf32("".join(words))
    cap = 2 * len(src) + 16 * n + 64
    while True:
        # The kernel overwrites its length buffer with output lengths, so a
        # retry needs a fresh copy of the input lengths.
        work = array(_U32, lens)
        dst = array(_U32, [0]) * cap
        try:
            total = stem(
                handle, n, src.buffer_info()[0], work.buffer_info()[0],
                dst.buffer_info()[0], cap,
            )
        except Exception as exc:  # Mojo raises a plain Exception
            if "output buffer too small" not in str(exc):
                raise
            cap *= 4
            continue
        break
    if sys.byteorder == "big":
        dst = array(_U32, dst[:total])
        dst.byteswap()
    text = dst[:total].tobytes().decode("utf-32-le", "surrogatepass")
    out = []
    pos = 0
    for length in work:
        out.append(text[pos:pos + length])
        pos += length
    return out


class _Stemmer:
    def __init__(self, lang):
        self._lang = lang

    def stemWord(self, word):
        return _stem_batch(self._lang, [word])[0]

    def stemWords(self, words):
        return _stem_batch(self._lang, list(words))


def stemmer(lang):
    """Return a stemmer for `lang` (english, german, russian or french)."""
    if lang not in _NEW:
        raise ValueError(f"no Mojo stemmer for {lang!r}; have {sorted(_NEW)}")
    return _Stemmer(lang)
