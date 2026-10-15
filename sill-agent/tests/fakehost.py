"""A stand-in Sill host for loopback tests: mutual TLS 1.3 with both ALPNs, the typed-code pairing
window, and a session that sends the catalog, signed Mac info, thumbnails, menus and real HEVC
frames (libx265 via ffmpeg), and records every client message.

It is written from the protocol source, not from a capture of the real host, so it proves that this
client agrees with this reading of the protocol. Agreement with Sill itself needs a run against a Mac.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sill_agent import crypto, wire  # noqa: E402
from sill_agent.wire import Kind  # noqa: E402


def hevc_clip(n_frames=12, keyint=6, size="320x240") -> tuple[list[bytes], list[tuple[bool, list[bytes]]]]:
    """Encode a test pattern; return (VPS/SPS/PPS, [(keyframe, [NAL units])] per access unit)."""
    with tempfile.TemporaryDirectory() as t:
        out = os.path.join(t, "clip.hevc")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"testsrc=size={size}:rate=10", "-frames:v", str(n_frames), "-c:v", "libx265",
                        "-x265-params", f"keyint={keyint}:min-keyint={keyint}:scenecut=0:log-level=error:bframes=0",
                        "-pix_fmt", "yuv420p", "-f", "hevc", out], check=True, capture_output=True)
        data = open(out, "rb").read()
    nals = [n for n in _split_annexb(data) if n]
    params, aus, cur = [], [], None
    for n in nals:
        t = (n[0] >> 1) & 0x3F
        if t in (32, 33, 34):
            if t not in [(p[0] >> 1) & 0x3F for p in params]:
                params.append(n)
            continue
        if t < 32 and n[2] & 0x80:            # first_slice_segment_in_pic_flag: a new picture
            cur = [16 <= t <= 21, []]
            aus.append(cur)
        if cur is not None and t < 41:
            cur[1].append(n)
    return params, [(k, us) for k, us in aus]


def _split_annexb(data: bytes) -> list[bytes]:
    out, i, start = [], 0, None
    while i < len(data) - 2:
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            if start is not None:
                out.append(data[start:i].rstrip(b"\x00"))
            i += 3
            start = i
        else:
            i += 1
    if start is not None:
        out.append(data[start:])
    return out


def length_prefixed(nals: list[bytes]) -> bytes:
    return b"".join(len(n).to_bytes(4, "big") + n for n in nals)


class FakeHost:
    WINDOWS = [
        {"id": 101, "title": "notes.md", "appName": "TextEdit", "bundleID": "com.apple.TextEdit", "width": 800, "height": 600},
        {"id": 202, "title": "Inbox", "appName": "Mail", "bundleID": "com.apple.mail", "width": 1200, "height": 800},
    ]

    def __init__(self, device_cert_pem: Path, *, require_pairing=True):
        self.dir = tempfile.mkdtemp()
        # Reuse the client's Identity builder for the host's own certificate: same shape as Sill's.
        host_id = crypto.Identity(os.path.join(self.dir, "host"))
        self.key = serialization.load_pem_private_key(host_id.key_path.read_bytes(), None)
        self.fingerprint = host_id.fingerprint
        self.ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.ctx.minimum_version = self.ctx.maximum_version = ssl.TLSVersion.TLSv1_3
        self.ctx.load_cert_chain(host_id.cert_path, host_id.key_path)
        self.ctx.set_alpn_protocols(["sill/1", "sill-pair/1"])
        # Python's ssl has no verify callback, so the device's own self-signed certificate is the
        # trust anchor (partial chain); trust is still decided by the fingerprint below.
        self.ctx.verify_mode = ssl.CERT_REQUIRED
        self.ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
        self.ctx.load_verify_locations(cafile=str(device_cert_pem))
        self.require_pairing = require_pairing
        self.paired: set[bytes] = set()
        self.code: str | None = None
        self.tries_left = 5
        self.received: list[tuple[int, bytes]] = []   # every session message from a device
        self.title_suffix = ""
        self.active = {"none": {}}
        self.menu_version = 7
        self.pressed: list[str] = []
        self.params, self.aus = hevc_clip()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def open_pairing_window(self) -> str:
        d = [int(b) % 10 for b in os.urandom(11)]
        self.code = "".join(map(str, d)) + str(crypto.damm("".join(map(str, d))))
        self.tries_left = 5
        return self.code

    def stop(self):
        self._stop.set()
        self.sock.close()

    # --- plumbing

    def _accept(self):
        while not self._stop.is_set():
            try:
                raw, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(raw,), daemon=True).start()

    def _serve(self, raw):
        try:
            s = self.ctx.wrap_socket(raw, server_side=True)
        except (ssl.SSLError, OSError):
            raw.close()
            return
        fp = crypto.fingerprint_of_cert_der(s.getpeercert(binary_form=True))
        try:
            if s.selected_alpn_protocol() == "sill-pair/1":
                self._pairing(s, fp)
            elif fp in self.paired or not self.require_pairing:
                self._session(s, fp)
        except (OSError, ValueError, ssl.SSLError):
            pass
        finally:
            s.close()

    def _send(self, s, kind, payload=b"", key=False):
        s.sendall(wire.encode(kind, payload, keyframe=key))

    def _sendj(self, s, kind, value):
        self._send(s, kind, json.dumps(value).encode())

    # --- pairing

    def _pairing(self, s, device_fp):
        r = wire.Reader(cap=lambda k: wire.MAX_PAIRING_PAYLOAD)
        s.settimeout(10)
        msgs = []
        while not msgs:
            c = s.recv(4096)
            if not c:
                return
            msgs = r.feed(c)
        req = msgs[0].json()
        if self.code is None:
            return self._sendj(s, Kind.PAIR_RESULT, {"ok": False, "reason": "closed"})
        k = crypto.code_key(self.code, self.fingerprint)
        want = crypto.device_proof(k, device_fp, self.fingerprint)
        if not crypto.hmac.compare_digest(crypto.b64u_decode(req["proof"]), want):
            self.tries_left -= 1
            return self._sendj(s, Kind.PAIR_RESULT, {"ok": False, "reason": "code", "triesLeft": self.tries_left})
        self.paired.add(device_fp)
        self.code = None
        self._sendj(s, Kind.PAIR_RESULT, {
            "ok": True, "proof": crypto.b64u(crypto.mac_proof(k, self.fingerprint, device_fp)),
            "macID": crypto.mac_id(self.fingerprint), "name": "Test Mac",
            "recognitionKey": crypto.b64u(os.urandom(32))})

    # --- session

    def _windows(self):
        ws = [dict(w) for w in self.WINDOWS]
        ws[0]["title"] += self.title_suffix
        return {"macName": "Test Mac", "windows": ws, "active": self.active, "hostVersion": "0.5.1",
                "protocol": 1, "gestures": 1}

    def _mac_info(self):
        info = json.dumps({"macID": crypto.mac_id(self.fingerprint), "name": "Test Mac",
                           "issuedAt": time.time()}).encode()
        point = self.key.public_key().public_bytes(serialization.Encoding.X962,
                                                   serialization.PublicFormat.UncompressedPoint)
        sig = self.key.sign(info, ec.ECDSA(hashes.SHA256()))
        return {"info": base64.b64encode(info).decode(), "key": base64.b64encode(point).decode(),
                "sig": base64.b64encode(sig).decode()}

    def _session(self, s, fp):
        lock = threading.Lock()

        def send(kind, payload=b"", key=False):
            with lock:
                self._send(s, kind, payload, key)

        def sendj(kind, v):
            send(kind, json.dumps(v).encode())

        sendj(Kind.WINDOW_LIST, self._windows())
        sendj(Kind.HOST_SETTINGS, {"settings": {"maxFPS": 60}})
        sendj(Kind.MAC_INFO, self._mac_info())
        sendj(Kind.APP_LIST, [{"name": "TextEdit", "bundleID": "com.apple.TextEdit"}])
        for w in self.WINDOWS:
            send(Kind.THUMBNAIL, w["id"].to_bytes(4, "big") + b"\xff\xd8JPEG-" + str(w["id"]).encode())
        streaming = threading.Event()

        def stream():
            try:
                send(Kind.PARAMETER_SETS, wire.encode_parameter_sets(4, self.params))
                for key, nals in self.aus:
                    if self._stop.is_set():
                        return
                    send(Kind.FRAME, length_prefixed(nals), key)
                    time.sleep(0.01)
            except OSError:
                return              # the device left mid-stream

        r = wire.Reader(cap=lambda k: wire.MAX_CLIENT_PAYLOAD)
        s.settimeout(0.02)       # recv holds the lock that send takes: OpenSSL is one thread at a time
        while not self._stop.is_set():
            try:
                with lock:
                    c = s.recv(65536)
            except (socket.timeout, ssl.SSLWantReadError):
                time.sleep(0.005)
                continue
            if not c:
                return
            for m in r.feed(c):
                self.received.append((m.kind, m.payload))
                if m.kind == Kind.PING:
                    send(Kind.PONG, m.payload)
                elif m.kind == Kind.SELECT_SOURCE:
                    self.active = m.json()
                    sendj(Kind.WINDOW_LIST, self._windows())
                    if not streaming.is_set():
                        streaming.set()
                        threading.Thread(target=stream, daemon=True).start()
                elif m.kind == Kind.INPUT:
                    ev = m.json()
                    if "text" in ev:          # a visible effect, so a test can tell delivery from silence
                        self.title_suffix += ev["text"]["_0"]
                        sendj(Kind.WINDOW_LIST, self._windows())
                elif m.kind == Kind.FETCH_MENU:
                    d = m.json()
                    if "id" not in d:
                        sendj(Kind.MAC_MENU, {"version": self.menu_version, "app": "TextEdit",
                                              "menus": [{"id": "1", "title": "File", "submenu": True},
                                                        {"id": "2", "title": "Edit", "submenu": True}]})
                    else:
                        sendj(Kind.MAC_MENU, {"version": self.menu_version, "answering": d["token"], "menu": d["id"],
                                              "items": [{"id": d["id"] + ".0", "title": "Save", "key": "⌘S"}]})
                elif m.kind == Kind.PRESS_MENU_ITEM:
                    d = m.json()
                    self.pressed.append(d["id"])
                    sendj(Kind.MAC_MENU, {"version": self.menu_version, "answering": d["token"], "pressed": True})
