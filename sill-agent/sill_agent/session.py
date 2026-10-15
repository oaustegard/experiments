"""A Sill device in Python: pair once, then hold a session that keeps the Mac's catalog current.

The session is observe-only unless opened with `control=True`. That gate lives in this process,
so it polices the agent's own code, not the agent: the enforcement that counts is on the Mac
(the observer patch in host-patch/). Treat this one as a seatbelt against bugs.
"""
from __future__ import annotations

import json
import select
import socket
import ssl
import struct
import threading
import time
from dataclasses import dataclass, field

from . import crypto, wire
from .wire import Kind

SESSION_ALPN = "sill/1"
PAIRING_ALPN = "sill-pair/1"
REMOTE_PORT = 7455
DEVICE_NAME = "sill-agent"
APP_VERSION = "0.5.1"   # what the hello claims; a host refuses devices below its floor with kind 22 "update"


class PinMismatch(Exception):
    pass


class ControlRefused(PermissionError):
    pass


def tls_connect(host: str, port: int, identity: crypto.Identity, alpn: str, pin: bytes | None,
                timeout: float = 10) -> tuple[ssl.SSLSocket, bytes]:
    """Mutual TLS 1.3 with one ALPN. The Mac's key is checked against `pin` before a byte is sent;
    pin None accepts any P-256 key (the typed pairing path, whose proofs bind both keys)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ctx.maximum_version = ssl.TLSVersion.TLSv1_3
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE          # trust is the pin, checked below
    ctx.load_cert_chain(identity.cert_path, identity.key_path)
    ctx.set_alpn_protocols([alpn])
    raw = socket.create_connection((host, port), timeout=timeout)
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s = ctx.wrap_socket(raw, server_hostname="sill")
    fp = crypto.fingerprint_of_cert_der(s.getpeercert(binary_form=True))
    if fp is None or (pin is not None and fp != pin):
        s.close()
        raise PinMismatch(f"the Mac presented {crypto.b64u(fp)[:10] if fp else 'a non-P-256 key'}, "
                          f"expected {crypto.b64u(pin)[:10] if pin else '?'}; nothing sent")
    if s.selected_alpn_protocol() != alpn:
        s.close()
        raise ConnectionError(f"ALPN {s.selected_alpn_protocol()!r}, wanted {alpn!r}")
    return s, fp


def _read_one(s: ssl.SSLSocket, timeout: float) -> wire.Message | None:
    s.settimeout(timeout)
    r = wire.Reader()
    while True:
        chunk = s.recv(65536)
        if not chunk:
            return None
        msgs = r.feed(chunk)
        if msgs:
            return msgs[0]


def pair(host: str, port: int, identity: crypto.Identity, *, code: str | None = None,
         link: str | None = None, name: str = DEVICE_NAME) -> dict:
    """One pairing connection: kind 19 out, kind 20 back. On ok, checks proof_M and the Mac ID, then
    saves the Mac's pin in the identity. Returns the kind 20 dict; raises on anything not ok."""
    if (code is None) == (link is None):
        raise ValueError("give exactly one of code= or link=")
    if link is not None:
        lk = crypto.parse_link(link)
        s, mac_fp = tls_connect(host, port or lk.port, identity, PAIRING_ALPN, lk.fingerprint)
        key, method = lk.secret, "qr"
    else:
        digits = crypto.check_code(code)
        s, mac_fp = tls_connect(host, port, identity, PAIRING_ALPN, None)
        key, method = crypto.code_key(digits, mac_fp), "code"
    proof = crypto.device_proof(key, identity.fingerprint, mac_fp)
    req = {"v": 1, "method": method, "proof": crypto.b64u(proof), "name": name, "model": "sill-agent"}
    try:
        s.sendall(wire.encode_json(Kind.PAIR_REQUEST, req))
        m = _read_one(s, 15)
    finally:
        s.close()
    if m is None or m.kind != Kind.PAIR_RESULT:
        raise ConnectionError(f"no kind 20 ({m.kind if m else 'EOF'})")
    r = m.json()
    if not r.get("ok"):
        raise PermissionError(f"pairing refused: {r.get('reason')} {r.get('message') or ''}".strip())
    want = crypto.mac_proof(key, mac_fp, identity.fingerprint)
    if not crypto.hmac.compare_digest(crypto.b64u_decode(r.get("proof") or ""), want):
        raise PermissionError("proof_M does not check: not the Mac that showed the code")
    if r.get("macID") != crypto.mac_id(mac_fp):
        raise PermissionError("macID is not the Mac's key")
    identity.save_mac(mac_fp, r.get("name"), r.get("recognitionKey"), f"{host}:{port}")
    return r


@dataclass
class Catalog:
    """What the Mac has told this device, newest of each."""
    window_list: dict | None = None
    app_list: list | None = None
    host_settings: dict | None = None
    mac_info: dict | None = None        # verified kind 18 only
    menu_top: dict | None = None
    menu_answers: dict = field(default_factory=dict)   # token -> kind 24 with `answering`
    pointer: dict | None = None
    goodbye: dict | None = None
    thumbnails: dict = field(default_factory=dict)     # window id -> (received at, jpeg)
    icons: dict = field(default_factory=dict)          # bundle id -> png
    parameter_sets: tuple | None = None                # (nal length size, [VPS, SPS, PPS])
    gop: list = field(default_factory=list)            # frames since the last keyframe, keyframe first
    frames: int = 0
    rtt_ms: float | None = None
    unknown_kinds: dict = field(default_factory=dict)


class Session:
    MAX_GOP = 900   # frames kept since a keyframe; past it the GOP is dropped until the next one

    def __init__(self, host: str, port: int, identity: crypto.Identity, *, control: bool = False,
                 pin: bytes | None = None, device: str = DEVICE_NAME):
        self.host, self.port, self.identity, self.control, self.device = host, port, identity, control, device
        self.pin = pin or identity.mac_pin()
        if self.pin is None:
            raise ValueError("no paired Mac in this identity: pair first")
        self.catalog = Catalog()
        self.sent: list[tuple[float, int, bytes]] = []    # an audit of every control message sent
        self._token = 0
        self._lock = threading.Condition()
        # One OpenSSL connection must not be read and written from two threads at once, so a single
        # I/O thread owns the socket; other threads queue bytes and wake it through a socketpair.
        self._out = bytearray()
        self._out_lock = threading.Lock()
        self._wake_r, self._wake_w = socket.socketpair()
        self._sock = None
        self._closed = threading.Event()
        self.error: BaseException | None = None

    # --- lifecycle

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self, hello: bool = True) -> "Session":
        self._sock, _ = tls_connect(self.host, self.port, self.identity, SESSION_ALPN, self.pin)
        if hello:
            self._send(Kind.HELLO, json.dumps({"appVersion": APP_VERSION, "protocol": 1,
                                               "device": self.device}).encode())
        self._sock.setblocking(False)
        self._io_thread = threading.Thread(target=self._io_loop, name="sill-io", daemon=True)
        self._io_thread.start()
        threading.Thread(target=self._ping_loop, name="sill-ping", daemon=True).start()
        return self

    def close(self, drain: float = 2.0):
        """Flush what is queued (up to `drain` seconds), then close."""
        deadline = time.time() + drain
        while self._out and self.alive and time.time() < deadline:
            time.sleep(0.01)
        self._closed.set()
        self._wake()
        if getattr(self, "_io_thread", None) is not None:
            self._io_thread.join(2)
        for x in (self._sock, self._wake_r, self._wake_w):
            try:
                x and x.close()
            except OSError:
                pass
        with self._lock:
            self._lock.notify_all()

    def _wake(self):
        try:
            self._wake_w.send(b"\0")
        except OSError:
            pass

    @property
    def alive(self) -> bool:
        return not self._closed.is_set()

    # --- sending

    def _send(self, kind: int, payload: bytes = b""):
        if kind in wire.CONTROL_KINDS:
            if not self.control:
                raise ControlRefused(f"kind {Kind(kind).name} changes the Mac; session is observe-only")
            self.sent.append((time.time(), int(kind), payload))
        self._enqueue(wire.encode(kind, payload))

    def _enqueue(self, data: bytes):
        if self._closed.is_set():
            raise ConnectionError("session closed" + (f": {self.error}" if self.error else ""))
        with self._out_lock:
            self._out += data
        self._wake()

    def _send_json(self, kind: int, value):
        self._send(kind, json.dumps(value, separators=(",", ":")).encode())

    def _next_token(self) -> int:
        self._token += 1
        return self._token

    # --- reading

    def _io_loop(self):
        r = wire.Reader()
        s = self._sock
        want_write_to_read = False      # a TLS write that needs the socket readable first (rare)
        try:
            while not self._closed.is_set():
                with self._out_lock:
                    pending_out = bool(self._out)
                wlist = [s] if pending_out and not want_write_to_read else []
                if not s.pending():
                    readable, writable, _ = select.select([s, self._wake_r], wlist, [], 0.5)
                else:
                    readable, writable = [s], wlist
                if self._wake_r in readable:
                    self._wake_r.recv(4096)
                if s in readable or s.pending():
                    want_write_to_read = False
                    while True:
                        try:
                            chunk = s.recv(1 << 18)
                        except ssl.SSLWantReadError:
                            break
                        except ssl.SSLWantWriteError:
                            break
                        if not chunk:
                            return
                        for m in r.feed(chunk):
                            self._apply(m)
                        if not s.pending():
                            break
                if s in writable or (pending_out and not want_write_to_read):
                    with self._out_lock:
                        buf = bytes(self._out[:1 << 16])
                    if buf:
                        try:
                            n = s.send(buf)
                        except ssl.SSLWantWriteError:
                            n = 0
                        except ssl.SSLWantReadError:
                            n, want_write_to_read = 0, True
                        with self._out_lock:
                            del self._out[:n]
        except (OSError, ValueError) as e:
            if not self._closed.is_set():
                self.error = e
        finally:
            self._closed.set()
            with self._lock:
                self._lock.notify_all()

    def _ping_loop(self):
        while not self._closed.wait(1.0):
            try:
                self._send(Kind.PING, struct.pack(">d", time.time()))
                c = self.catalog
                self._send_json(Kind.CLIENT_STATS, {"fps": 0, "frameAgeMs": -1,
                                                    "rttMs": int(c.rtt_ms) if c.rtt_ms is not None else -1,
                                                    "device": self.device})
            except (OSError, ConnectionError):
                return

    def _apply(self, m: wire.Message):
        c = self.catalog
        k = m.kind
        with self._lock:
            if k == Kind.WINDOW_LIST:
                c.window_list = m.json()
            elif k == Kind.THUMBNAIL:
                t = wire.decode_thumbnail(m.payload)
                if t:
                    c.thumbnails[t[0]] = (time.time(), t[1])
            elif k == Kind.APP_ICON:
                t = wire.decode_icon(m.payload)
                if t:
                    c.icons[t[0]] = t[1]
            elif k == Kind.APP_LIST:
                c.app_list = m.json()
            elif k == Kind.HOST_SETTINGS:
                c.host_settings = m.json()
            elif k == Kind.MAC_INFO:
                info = crypto.verify_mac_info(m.payload, self.pin)
                if info is not None:
                    c.mac_info = info
            elif k == Kind.MAC_MENU:
                d = m.json()
                if d.get("answering") is not None:
                    c.menu_answers[d["answering"]] = d
                else:
                    c.menu_top = d
            elif k == Kind.MAC_POINTER:
                c.pointer = m.json()
            elif k == Kind.GOODBYE:
                c.goodbye = m.json()
            elif k == Kind.PONG:
                if len(m.payload) == 8:
                    sent = struct.unpack(">d", m.payload)[0]
                    c.rtt_ms = (time.time() - sent) * 1000
            elif k == Kind.PARAMETER_SETS:
                c.parameter_sets = wire.decode_parameter_sets(m.payload)
                c.gop = []
            elif k == Kind.FRAME:
                c.frames += 1
                if m.keyframe:
                    c.gop = [m.payload]
                elif c.gop and len(c.gop) < self.MAX_GOP:
                    c.gop.append(m.payload)
                else:
                    c.gop = []
            elif k in (Kind.TICK, Kind.CURSOR_SHAPE):
                pass
            else:
                c.unknown_kinds[k] = c.unknown_kinds.get(k, 0) + 1
            self._lock.notify_all()

    def wait_for(self, predicate, timeout: float = 10.0):
        """Block until predicate(catalog) is truthy (returns it), the session ends, or timeout (None)."""
        deadline = time.time() + timeout
        with self._lock:
            while True:
                v = predicate(self.catalog)
                if v:
                    return v
                left = deadline - time.time()
                if left <= 0 or self._closed.is_set():
                    return None
                self._lock.wait(left)

    # --- observing (never changes the Mac)

    def windows(self, timeout: float = 10.0) -> list[dict]:
        wl = self.wait_for(lambda c: c.window_list, timeout)
        return wl["windows"] if wl else []

    def active_source(self):
        wl = self.catalog.window_list
        return wl.get("active") if wl else None

    def thumbnail(self, window_id: int, timeout: float = 5.0, newer_than: float = 0.0) -> bytes | None:
        t = self.wait_for(lambda c: (c.thumbnails.get(window_id) or (0, None))[0] > newer_than
                          and c.thumbnails[window_id], timeout)
        return t[1] if t else None

    def subscribe_menus(self, timeout: float = 5.0) -> dict | None:
        """Kind 27 without an id: the host reads the top level over Accessibility; nothing is opened."""
        self._send_json(Kind.FETCH_MENU, {"token": self._next_token()})
        return self.wait_for(lambda c: c.menu_top, timeout)

    # --- control (refused unless control=True)

    def select(self, source):
        self._send_json(Kind.SELECT_SOURCE, source)

    def select_window(self, window_id: int):
        self.select(wire.source_window(window_id))

    def launch(self, bundle_id: str):
        self._send_json(Kind.LAUNCH_APP, {"bundleID": bundle_id})

    def window_command(self, window_id: int, action: str):
        assert action in ("close", "minimize", "fullScreen")
        self._send_json(Kind.WINDOW_COMMAND, {"id": int(window_id), "action": action})

    def input(self, event: dict):
        self._send_json(Kind.INPUT, event)

    def click(self, x: float, y: float, button: str = "left"):
        for e in (wire.pointer("move", x, y), wire.pointer(f"{button}Down", x, y), wire.pointer(f"{button}Up", x, y)):
            self.input(e)

    def type_text(self, s: str):
        self.input(wire.text(s))

    def press_key(self, hid_usage: int, modifiers: int = 0):
        self.input(wire.key(hid_usage, True, modifiers))
        self.input(wire.key(hid_usage, False, modifiers))

    def scroll(self, x: float, y: float, dy: float, dx: float = 0.0):
        self.input(wire.scroll(x, y, dx, dy))

    def fetch_menu(self, menu_id: str, timeout: float = 5.0) -> dict | None:
        """Opens one menu's items on the Mac (a read the host does through the menu itself), so it
        counts as control."""
        top = self.catalog.menu_top or {}
        title = _title_of(top, menu_id, self.catalog.menu_answers)
        tok = self._next_token()
        if not self.control:
            raise ControlRefused("fetching a submenu opens it on the Mac; session is observe-only")
        self.sent.append((time.time(), int(Kind.FETCH_MENU), menu_id.encode()))
        self._enqueue(wire.encode_json(Kind.FETCH_MENU, {"version": top.get("version"), "id": menu_id,
                                                         "title": title, "token": tok}))
        return self.wait_for(lambda c: c.menu_answers.get(tok), timeout)

    def press_menu(self, item_id: str, timeout: float = 5.0) -> dict | None:
        top = self.catalog.menu_top or {}
        tok = self._next_token()
        self._send_json(Kind.PRESS_MENU_ITEM, {"version": top.get("version"), "id": item_id,
                                               "title": _title_of(top, item_id, self.catalog.menu_answers),
                                               "token": tok})
        return self.wait_for(lambda c: c.menu_answers.get(tok), timeout)


def _title_of(top: dict, item_id: str, answers: dict) -> str | None:
    for d in [top, *answers.values()]:
        for it in (d.get("menus") or []) + (d.get("items") or []):
            if it.get("id") == item_id:
                return it.get("title")
    return None
