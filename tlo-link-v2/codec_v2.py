"""Reference codec for a proposed Tiny Little Office link v2 payload.

    #/v2/<app>/<flag><body>

flag (one letter) says how the bytes were packed and which alphabet the body uses:
    lower case: body is base62 (0-9 A-Z a-z), for links typed, pasted and chatted
    upper case: body is base32 (A-Z 2-7), for QR codes, where it fits the
                alphanumeric mode (5.5 bits a character instead of 8)
    w / W  raw deflate against the v2 dictionary (data/dict-v2.bin, 32 KB)
    y / Y  raw deflate against the v1 dictionary (unchanged, for continuity)
    z / Z  raw deflate, no dictionary

base62 works in blocks: 32 bytes are 43 characters (62^43 >= 2^256), and a last
block of k bytes is the fewest characters c with 62^c >= 256^k, so a decoder
knows each block's length from the count alone. A writer puts "+" after every
300 characters (iMessage and Signal break links at longer letter-and-digit runs).

A reader is deliberately forgiving: it percent-decodes until nothing changes
(chat clients encode once or twice), then keeps only the body alphabet's
characters, so a "+", a space or line break a mail client wrapped in, a
zero-width space, or a quote glued to the end cannot change the document. Letters
glued to the end are not recoverable without a checksum (see RESULTS.md).
"""
import base64
import zlib
from pathlib import Path
from urllib.parse import unquote

DATA = Path(__file__).with_name("data")
DICTS = {
    "w": (DATA / "dict-v2.bin").read_bytes(),
    "y": (DATA / "dict-v1.txt").read_bytes(),
    "z": None,
}
B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
B62_REV = {c: i for i, c in enumerate(B62)}
BLOCK, BLOCK_CHARS, BREAK_EVERY = 32, 43, 300


def chars_for(k: int) -> int:
    """Fewest base62 characters that hold k bytes."""
    c = 0
    while 62 ** c < 256 ** k:
        c += 1
    return c


TAIL = {chars_for(k): k for k in range(1, BLOCK)}  # last-block length -> bytes


def b62encode(data: bytes) -> str:
    out = []
    for i in range(0, len(data), BLOCK):
        blk = data[i:i + BLOCK]
        n, width = int.from_bytes(blk, "big"), chars_for(len(blk))
        s = []
        for _ in range(width):
            n, r = divmod(n, 62)
            s.append(B62[r])
        out.append("".join(reversed(s)))
    return "".join(out)


def b62decode(s: str) -> bytes:
    out = bytearray()
    full, rest = divmod(len(s), BLOCK_CHARS)
    if rest and rest not in TAIL:
        raise ValueError(f"a base62 body of {len(s)} characters is cut short")
    for i in range(full + (1 if rest else 0)):
        chunk = s[i * BLOCK_CHARS:(i + 1) * BLOCK_CHARS]
        k = BLOCK if len(chunk) == BLOCK_CHARS else TAIL[len(chunk)]
        n = 0
        for ch in chunk:
            n = n * 62 + B62_REV[ch]
        if n >= 256 ** k:
            raise ValueError("base62 block out of range")
        out += n.to_bytes(k, "big")
    return bytes(out)


def deflate(raw: bytes, zdict: bytes | None) -> bytes:
    c = zlib.compressobj(9, zlib.DEFLATED, -15, 9, **({"zdict": zdict} if zdict else {}))
    return c.compress(raw) + c.flush()


def inflate(data: bytes, zdict: bytes | None) -> bytes:
    d = zlib.decompressobj(-15, **({"zdict": zdict} if zdict else {}))
    out = d.decompress(data) + d.flush()
    if not d.eof:
        raise ValueError("the link was cut off: the compressed document does not reach its end")
    return out


def encode(doc: str, qr: bool = False) -> str:
    """TLO text -> flag + body, the shortest of w / y / z; breaks added for base62."""
    raw = doc.encode("utf-8")
    flag, data = min(((f, deflate(raw, d)) for f, d in DICTS.items()), key=lambda x: len(x[1]))
    if qr:
        return flag.upper() + base64.b32encode(data).decode().rstrip("=")
    body = flag + b62encode(data)
    return "+".join(body[i:i + BREAK_EVERY] for i in range(0, len(body), BREAK_EVERY))


def clean(payload: str) -> str:
    for _ in range(4):
        nxt = unquote(payload)
        if nxt == payload:
            break
        payload = nxt
    return payload


def decode(payload: str) -> str:
    p = clean(payload).strip()
    p = p.lstrip("+ \t\r\n")
    flag = p[0]
    if flag.lower() not in DICTS:
        raise ValueError(f"unknown flag {flag!r}")
    if flag.isupper():
        body = "".join(ch for ch in p[1:].upper() if ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
        unpack = lambda b: base64.b32decode(b + "=" * (-len(b) % 8))
    else:
        body = "".join(ch for ch in p[1:] if ch in B62_REV)
        unpack = b62decode
    # Letters glued to the end ("…link.Thanks") are in the alphabet and are not
    # removed: raw deflate has no checksum, and trimming a character can yield a
    # different stream that also ends cleanly (a wrong document, worse than an error).
    try:
        return inflate(unpack(body), DICTS[flag.lower()]).decode("utf-8")
    except (zlib.error, UnicodeDecodeError, base64.binascii.Error) as e:
        raise ValueError(str(e)) from e
