"""Full-resolution stills from the HEVC stream: the parameter sets and the frames since the last
keyframe, rewritten from length-prefixed NAL units (hvcC style) to Annex B, decoded by ffmpeg."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

START = b"\x00\x00\x00\x01"


def to_annexb(access_unit: bytes, nal_len: int) -> bytes:
    out, i = bytearray(), 0
    while i + nal_len <= len(access_unit):
        n = int.from_bytes(access_unit[i:i + nal_len], "big")
        i += nal_len
        if n == 0 or i + n > len(access_unit):
            raise ValueError("truncated NAL unit")
        out += START + access_unit[i:i + n]
        i += n
    if i != len(access_unit):
        raise ValueError("trailing bytes after the last NAL unit")
    return bytes(out)


def elementary_stream(parameter_sets: tuple[int, list[bytes]], gop: list[bytes]) -> bytes:
    nal_len, sets = parameter_sets
    return b"".join(START + s for s in sets) + b"".join(to_annexb(f, nal_len) for f in gop)


def still(parameter_sets, gop: list[bytes], out_path: str | Path, ffmpeg: str = "ffmpeg") -> Path | None:
    """Decode the GOP and write its last picture to out_path (PNG or JPEG by suffix). None when there
    is no keyframe yet or ffmpeg fails."""
    if not parameter_sets or not gop:
        return None
    out_path = Path(out_path)
    with tempfile.NamedTemporaryFile(suffix=".hevc") as f:
        f.write(elementary_stream(parameter_sets, gop))
        f.flush()
        r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "hevc", "-i", f.name,
                            "-update", "1", "-frames:v", str(len(gop)), str(out_path)],
                           capture_output=True, text=True)
    return out_path if r.returncode == 0 and out_path.exists() else None
