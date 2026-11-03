"""Fetch a few DIV2K validation images (2K-resolution PNGs) without downloading the 449 MB zip.

    python3 data/fetch_div2k.py [0801 0802 ...]

DIV2K (Agustsson and Timofte, NTIRE 2017) is the standard high-resolution set for super-resolution. The
zip's central directory is read with HTTP range requests and only the named members are fetched.
Default: the first six validation images, 0801 to 0806. Writes data/div2k/<id>.png.
"""
from __future__ import annotations

import struct
import sys
import urllib.request
import zlib
from pathlib import Path

URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
OUT = Path(__file__).resolve().parent / "div2k"


def get(a: int, b: int) -> bytes:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}", "User-Agent": "image-to-wgsl"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def size() -> int:
    req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "image-to-wgsl"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def main(ids: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    n = size()
    tail = get(n - 65536, n - 1)
    e = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[e + 12:e + 20])
    cd = get(cd_off, cd_off + cd_size - 1)
    members, i = {}, 0
    while i < len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[i + 10:i + 12])
        csize, usize = struct.unpack("<II", cd[i + 20:i + 28])
        nl, xl, cl = struct.unpack("<HHH", cd[i + 28:i + 34])
        off, = struct.unpack("<I", cd[i + 42:i + 46])
        name = cd[i + 46:i + 46 + nl].decode()
        members[Path(name).stem] = (meth, csize, usize, off)
        i += 46 + nl + xl + cl
    for k in ids:
        dst = OUT / f"{k}.png"
        if dst.exists():
            continue
        meth, csize, usize, off = members[k]
        hdr = get(off, off + 29)
        nl, xl = struct.unpack("<HH", hdr[26:30])
        data = get(off + 30 + nl + xl, off + 30 + nl + xl + csize - 1)
        raw = data if meth == 0 else zlib.decompress(data, -15)
        assert len(raw) == usize, (k, len(raw), usize)
        dst.write_bytes(raw)
        print(k, usize)


if __name__ == "__main__":
    main(sys.argv[1:] or [f"08{i:02d}" for i in range(1, 7)])
