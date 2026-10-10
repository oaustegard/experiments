# Port snowballstemmer's stemmers to Mojo

Reference: snowballstemmer 3.1.1, pure Python, installed at
`/usr/local/lib/python3.13/dist-packages/snowballstemmer/`. The runtime is
`basestemmer.py` and `among.py`; each language is a generated module,
`<lang>_stemmer.py`. Use the pure-Python classes as the reference (for
example `snowballstemmer.english_stemmer.EnglishStemmer`), never
`snowballstemmer.stemmer()`, which hands off to the C PyStemmer when it is
installed.

## Deliverable

A package `msnowball/` in this directory:

- `msnowball.stemmer(lang)` for `lang` in `english`, `german`, `russian`,
  `french`, returning an object with
  - `stemWord(word: str) -> str`
  - `stemWords(words: list[str]) -> list[str]`
- Output identical to the reference stemmer for every input string,
  including uppercase, punctuation, digits, empty strings and non-ASCII.
- The stemming runs in Mojo. At runtime the package may import numpy but not
  `snowballstemmer` or `Stemmer`.
- `build.sh` builds everything from source.

## Performance

On a `stemWords` batch, pure Python takes about 30 µs per word and PyStemmer
(the C build of the same algorithms) about 3 µs. Aim to be much faster than
pure Python on `stemWords`; report how you compare to C. Measure with
`python3 BENCH --target snowball --pkg .`.

## Hints

The four language modules are generated code with a regular shape. Writing a
small Python script that translates them to Mojo can be less error-prone
than translating 2,000+ lines by hand; your call. Python indexes `str` by
code point, and the stemmers compare code points.
