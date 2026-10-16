# Tiny Little Office link format v2

A proposal for [aboard-io/tinylittleoffice](https://github.com/aboard-io/tinylittleoffice): shorter links that survive more of the places people paste them. Every number below is measured on documents from that repository and rechecked by `recheck.py`.

<!-- recheck {"test_docs": 35, "train_docs": 193, "v1_chars": 10676, "v2_chars": 8800, "v2_vs_v1_pct": -17.6, "median_doc_pct": -14.3, "dict_gz_kb_added": 13.2, "b62_overhead_pct": 0.78} -->

## Findings

- **A new deflate dictionary shortens links by 17.6%.** On the 35 held-out documents in `samples/`, the payloads go from 10,676 to 8,800 characters in total; the median document shrinks 14.3%. The compressor stays fflate's raw deflate, which v1 already ships. Only the dictionary changes, under a new flag.
- **The dictionary costs bundle bytes.** 32 KB of dictionary adds 13.2 KB gzipped. A smaller one keeps most of the gain:

  | dictionary | links vs v1 | bundle, gzipped |
  |---|---|---|
  | 8 KB | −11.6% | +2.8 KB |
  | 16 KB | −14.5% | +6.0 KB |
  | 32 KB | −18.3% | +13.2 KB |

  (This table is base64 payload length from `bench3.py`. The −17.6% headline also includes the base62 change below.)
- **Documents unlike the training set gain less.** The deck written in this session shrinks 7.7%, a plain-prose letter 9.0%, a Norwegian shopping list 2.2%. The writer still picks the shortest of the dictionaries, so no document gets longer.
- **The other compressors tested don't justify a new decoder.**
  - zstd at level 22 with a 32 KB trained dictionary is 20.8% shorter than v1, only 2.5 points better than deflate with the same dictionary size. fflate has no zstd decoder, so this would add a new dependency.
  - Brotli at quality 11, with its built-in web dictionary, is 0.5% shorter than v1.
- **Base62 removes every punctuation character from the payload for 0.78%.** A v2 payload contains letters, digits and the `+` breaks, nothing else. Base64url's two extra characters each have a known failure (see the table under Platform behaviour). A base62 block encoding (32 bytes in 43 characters) costs 0.78% over base64.
- **A forgiving reader makes the rest of the mangling harmless.** A v2 reader percent-decodes until nothing changes, then drops every character outside the alphabet. `test_roundtrip.py` runs 229 corpus documents through five manglings:
  - percent-encoded twice, as Webex does;
  - a curly quote glued to the end;
  - wrapped at 76 columns by a mail client;
  - a zero-width space inserted;
  - no mangling.

  All 2,290 Python decodes come back unchanged, and so do all 1,145 Node decodes (`codec_v2.mjs` with `node:zlib`).
- **QR codes are smaller with an uppercase base32 body.** QR alphanumeric mode spends 5.5 bits on a character from `0-9 A-Z $%*+-./:` and byte mode spends 8. The session's deck as a base64url link needs QR version 26 (121×121 modules). As uppercase base32 in an alphanumeric segment, it needs version 23 (109×109), even though the link text is 19% longer. Upper-case flags select this body, so the app's QR export can use it while chat links stay base62.
- **One character cut from a link can silently decode to the wrong document, in v1 as in v2.** Cutting the last character of each of the 229 links made one decode cleanly to the wrong text. Raw deflate has no checksum, and the stream's end marker only catches most cuts. The same lack of a checksum is why the reader doesn't trim letters glued to the end of a link: a trimmed stream can also end cleanly on the wrong document. A CRC-16 in 3 base62 characters (0.2% of a 1,300-character link) would turn both into errors. It is listed as an option below, not measured.

## Platform behaviour

| character | what happens to it | source |
|---|---|---|
| `~` | Webex percent-encodes it twice (`%257E`), and the v1 router doesn't percent-decode, so the link opens empty | observed by Oskar, 2026-10-15 |
| `~` `_` at the end | GitHub's autolinker leaves trailing `? ! . , : * _ ~` out of the link | GFM spec, extended autolinks |
| `-` | a line-break opportunity, so mail clients wrap long links there | UAX #14 class HY; v1's own reason for `~` (`codec.ts`) |
| `_` pairs | Markdown-ish chat renderers can italicize the text between two underscores when a link isn't recognised as one | inferred, not tested |
| letter-and-digit runs over 301 | iMessage and Signal break the link | v1's `withBreaks` comment |
| `+` | survived Webex. `URLSearchParams` would read it as a space, but the payload is not parsed that way | observed; v1 already strips `+`/`%2B` |
| `’` after the link | Webex pulled the closing quote into the link (`%E2%80%99`) | observed by Oskar, 2026-10-15 |

Only the Webex rows were observed. Testing Slack, Teams, Outlook SafeLinks, WhatsApp, iMessage and a physical QR scan would confirm or retire the rest.

## Proposed format

```
#/v2/<app>[.<view>]/<flag><body>[?show|?d]
```

| flag | bytes | body |
|---|---|---|
| `w` | raw deflate, v2 dictionary | base62, `+` after every 300 characters |
| `y` | raw deflate, v1 dictionary | base62 |
| `z` | raw deflate, no dictionary | base62 |
| `W` `Y` `Z` | the same | uppercase base32, for QR codes |
| `E` | sealed (password) | as v1, base62 |

- Base62 uses the alphabet `0-9A-Za-z` in big-endian blocks. A full block is 32 bytes in 43 characters. A last block of k bytes uses the fewest characters c with 62^c ≥ 256^k, and that length is unique for each k, so the reader knows k from the character count.
- `r` (raw text) is dropped. A deflate stream that ends cleanly is the only cut-link check the format has.
- The reader:
  1. percent-decodes until stable;
  2. skips leading non-alphanumerics;
  3. takes the flag;
  4. keeps only alphabet characters;
  5. decodes;
  6. requires the inflater to reach end of stream.
- Writers emit v2. Readers keep v1, so every existing link still opens.
- The v2 dictionary is frozen the day it ships, as v1's is. It can be its own lazy-loaded chunk, fetched on the first v2 link, so first paint doesn't pay for it.
- Optional: 3 characters of CRC-16 after the flag, so that a link that was cut, or had letters glued to its end, gives an error rather than the wrong document.

`codec_v2.py` is the reference implementation of the whole format. `codec_v2.mjs` is the text layer (base62 and the forgiving reader) in the shape `codec.ts` could take.

## Method

- **Corpus.** Train: 193 documents from `test/doc/corpus/**/*.json` (templates and per-app examples). Test: the 35 `samples/*.tlo` files. Any text found in both sets was removed from train. `corpus.py` builds `data/corpus.json` from a checkout.
- **Dictionary.**
  - `zstandard.train_dictionary(32768, …)`, zstd's COVER trainer, built the dictionary.
  - Its raw-content tail was cut from the zstd header: the first offset where every 8-byte window over the next 64 bytes appears in the training text.
  - The v1 dictionary was appended last, so the commonest TLO words sit at the shortest distances. Then the whole thing was truncated to deflate's 32 KB window.
- **Compression.**
  - The baseline uses zlib at level 9, memLevel 9, in place of fflate.
  - On the README's ten sample links, zlib came out 1 to 8 characters shorter than the site's own output, so the baseline flatters v1 slightly.
  - Trying fixed-Huffman blocks as well changed nothing at all.
- **QR versions.** The `qrcode` package with error correction L sized the codes: the URL prefix as a byte segment and the payload as a byte or alphanumeric segment.
- **Limits.** The training and test documents were written by the same people, in the same house style. The out-of-distribution numbers (2–9%) are the better guide for documents written by others.

## Files

| file | what it is |
|---|---|
| `codec_v2.py` | reference encoder and forgiving reader |
| `codec_v2.mjs` | base62 and reader in JS (BigInt, no dependencies) |
| `test_roundtrip.py`, `test_node.mjs` | the mangling tests, both languages |
| `corpus.py` | builds `data/corpus.json` from a tinylittleoffice checkout |
| `bench.py`, `bench2.py`, `bench3.py` | compressor, dictionary and QR comparisons |
| `recheck.py` | checks this file's quoted numbers against the data |
| `data/dict-v2.bin` | the proposed 32 KB dictionary |
| `data/dict-v1.txt` | v1's dictionary, copied from `codec.ts` (MIT) |

To reproduce from a checkout of tinylittleoffice:

```
python3 corpus.py <checkout> data/corpus.json
python3 bench2.py data data/dict-v1.txt
python3 bench3.py data data/dict-v1.txt data/deck.tlo
python3 test_roundtrip.py && python3 recheck.py
```
