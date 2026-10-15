"""Sill's wire format: a 14-byte big-endian header, then a JSON or binary payload.

Transcribed from Saffsanity/sill `Sources/StreamProtocol/StreamMessage.swift` (Apache-2.0).
Header: kind(1) timestamp(8, IEEE double) isKeyframe(1) payloadLength(4).
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from enum import IntEnum

HEADER = struct.Struct(">BdBI")
assert HEADER.size == 14

# Caps the host enforces (StreamMessage.max*); a reader refuses more than it could legitimately get.
MAX_CLIENT_PAYLOAD = 1 << 20
MAX_PAIRING_PAYLOAD = 4096
MAX_FRAME_PAYLOAD = 32 << 20
MAX_OTHER_HOST_PAYLOAD = 4 << 20


class Kind(IntEnum):
    PARAMETER_SETS = 0
    FRAME = 1
    WINDOW_LIST = 2
    THUMBNAIL = 3
    APP_ICON = 4
    APP_LIST = 5
    SELECT_SOURCE = 6
    LAUNCH_APP = 7
    INPUT = 8
    VIEWPORT = 9
    PING = 10
    PONG = 11
    CLIENT_STATS = 12
    TICK = 13
    CURSOR_SHAPE = 14
    WINDOW_COMMAND = 15
    HOST_SETTINGS = 16
    CHANGE_SETTINGS = 17
    MAC_INFO = 18
    PAIR_REQUEST = 19
    PAIR_RESULT = 20
    PAIRING_WANTED = 21
    GOODBYE = 22
    HELLO = 23
    MAC_MENU = 24
    PRESS_MENU_ITEM = 25
    MAC_POINTER = 26
    FETCH_MENU = 27
    GESTURE = 28


# Kinds a device sends that change something on the Mac. Everything else a device sends is
# bookkeeping (hello, ping, stats) or a read (a menu subscription). sill-agent refuses these unless
# the session was opened with control armed; the host-side fork refuses them for observer devices.
CONTROL_KINDS = frozenset({
    Kind.SELECT_SOURCE, Kind.LAUNCH_APP, Kind.INPUT, Kind.VIEWPORT, Kind.WINDOW_COMMAND,
    Kind.CHANGE_SETTINGS, Kind.PRESS_MENU_ITEM, Kind.GESTURE, Kind.PAIRING_WANTED,
})


@dataclass(frozen=True)
class Message:
    kind: int          # an int, not Kind: a newer host may send a kind this build does not know
    timestamp: float
    keyframe: bool
    payload: bytes

    def json(self):
        return json.loads(self.payload)


def encode(kind: int, payload: bytes = b"", keyframe: bool = False, timestamp: float | None = None) -> bytes:
    ts = time.time() if timestamp is None else timestamp
    return HEADER.pack(int(kind), ts, 1 if keyframe else 0, len(payload)) + payload


def encode_json(kind: int, value) -> bytes:
    return encode(kind, json.dumps(value, separators=(",", ":")).encode())


def parse_header(b: bytes) -> tuple[int, float, bool, int]:
    kind, ts, key, n = HEADER.unpack(b)
    return kind, ts, key == 1, n


def host_cap(kind: int) -> int:
    return MAX_FRAME_PAYLOAD if kind == Kind.FRAME else MAX_OTHER_HOST_PAYLOAD


class Reader:
    """Incremental decoder over a byte stream. `cap(kind)` bounds what a header may announce."""

    def __init__(self, cap=host_cap):
        self.buf = bytearray()
        self.cap = cap

    def feed(self, data: bytes) -> list[Message]:
        self.buf += data
        out = []
        while len(self.buf) >= HEADER.size:
            kind, ts, key, n = parse_header(bytes(self.buf[:HEADER.size]))
            if n > self.cap(kind):
                raise ValueError(f"kind {kind} announces {n} bytes, over the cap {self.cap(kind)}")
            if len(self.buf) < HEADER.size + n:
                break
            payload = bytes(self.buf[HEADER.size:HEADER.size + n])
            del self.buf[:HEADER.size + n]
            out.append(Message(kind, ts, key, payload))
        return out


# --- Swift Codable shapes. Enums with associated values encode as {"case": {"_0": v, label: v}}.

def source_none():
    return {"none": {}}


def source_desktop():
    return {"desktop": {}}


def source_window(window_id: int):
    return {"window": {"_0": int(window_id)}}


def pointer(action: str, x: float, y: float):
    assert action in ("move", "leftDown", "leftUp", "rightDown", "rightUp"), action
    return {"pointer": {"_0": action, "x": float(x), "y": float(y)}}


def scroll(x: float, y: float, dx: float, dy: float):
    return {"scroll": {"x": float(x), "y": float(y), "dx": float(dx), "dy": float(dy)}}


def text(s: str):
    return {"text": {"_0": s}}


def key(hid_usage: int, down: bool, modifiers: int = 0):
    return {"key": {"hidUsage": int(hid_usage), "down": bool(down), "modifiers": int(modifiers)}}


# UIKeyModifierFlags bits, which match CGEventFlags for these four (Input.swift).
SHIFT, CONTROL, OPTION, COMMAND = 1 << 17, 1 << 18, 1 << 19, 1 << 20


def decode_thumbnail(payload: bytes) -> tuple[int, bytes] | None:
    if len(payload) <= 4:
        return None
    return int.from_bytes(payload[:4], "big"), payload[4:]


def decode_icon(payload: bytes) -> tuple[str, bytes] | None:
    if len(payload) <= 2:
        return None
    n = int.from_bytes(payload[:2], "big")
    if len(payload) <= 2 + n:
        return None
    return payload[2:2 + n].decode(), payload[2 + n:]


def decode_parameter_sets(payload: bytes) -> tuple[int, list[bytes]] | None:
    """nalLen(1) count(1) then per set: length(2) bytes."""
    if len(payload) < 2:
        return None
    nal_len, count, i, sets = payload[0], payload[1], 2, []
    for _ in range(count):
        if len(payload) < i + 2:
            return None
        n = int.from_bytes(payload[i:i + 2], "big")
        i += 2
        if len(payload) < i + n:
            return None
        sets.append(payload[i:i + n])
        i += n
    return nal_len, sets


def encode_parameter_sets(nal_len: int, sets: list[bytes]) -> bytes:
    out = bytearray([nal_len, len(sets)])
    for s in sets:
        out += len(s).to_bytes(2, "big") + s
    return bytes(out)
