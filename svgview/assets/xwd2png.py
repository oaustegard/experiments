#!/usr/bin/env python3
"""Convert an Xvfb -fbdir framebuffer dump (XWD format) to PNG.

Used by the smoke test to screenshot the real window without needing
ImageMagick or x11grab in the container.

    python3 assets/xwd2png.py /tmp/fb/Xvfb_screen0 shot.png
"""

import struct
import sys
import zlib


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    raw = open(sys.argv[1], "rb").read()

    # 25 big-endian u32s of header, then the window name.
    f = struct.unpack(">25I", raw[:100])
    header_size, version, _fmt, depth = f[0], f[1], f[2], f[3]
    width, height = f[4], f[5]
    bits_per_pixel, bytes_per_line = f[11], f[12]
    red_mask, green_mask, blue_mask = f[14], f[15], f[16]
    ncolors = f[19]

    if version != 7:
        sys.exit(f"unexpected XWD version {version}")
    if bits_per_pixel not in (24, 32):
        sys.exit(f"unsupported bits_per_pixel {bits_per_pixel}")

    offset = header_size + ncolors * 12
    pixels = raw[offset:]

    def shift(mask: int) -> int:
        return (mask & -mask).bit_length() - 1

    rs, gs, bs = shift(red_mask), shift(green_mask), shift(blue_mask)
    step = bits_per_pixel // 8

    rows = []
    for y in range(height):
        base = y * bytes_per_line
        row = bytearray(b"\x00")  # PNG filter type 0
        for x in range(width):
            i = base + x * step
            word = int.from_bytes(pixels[i : i + step], "little")
            row += bytes(
                (
                    (word & red_mask) >> rs,
                    (word & green_mask) >> gs,
                    (word & blue_mask) >> bs,
                )
            )
        rows.append(bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + chunk(b"IEND", b"")
    )
    open(sys.argv[2], "wb").write(png)
    print(f"{sys.argv[2]} ({width}x{height}, depth {depth})")


if __name__ == "__main__":
    main()
