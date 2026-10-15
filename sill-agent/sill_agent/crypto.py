"""Sill's identities, pins and pairing proofs (Sources/StreamProtocol/RemoteIdentity.swift, Pairing.swift).

Trust is the key: each side pins the SHA-256 of the other's SubjectPublicKeyInfo (P-256). Names,
dates and chains in the certificates mean nothing.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

PAIR_LABEL = b"sill-pair-v1"
PBKDF2_ROUNDS = 600_000
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
DAMM = [
    [0, 3, 1, 7, 5, 9, 8, 6, 4, 2], [7, 0, 9, 2, 1, 5, 4, 8, 6, 3], [4, 2, 0, 6, 8, 7, 1, 3, 5, 9],
    [1, 7, 5, 0, 9, 8, 3, 4, 2, 6], [6, 1, 2, 3, 0, 4, 5, 9, 7, 8], [3, 6, 7, 4, 2, 0, 9, 5, 8, 1],
    [5, 8, 6, 9, 7, 2, 0, 1, 3, 4], [8, 9, 4, 5, 3, 6, 2, 0, 1, 7], [9, 4, 3, 8, 6, 1, 7, 2, 0, 5],
    [2, 5, 8, 1, 4, 3, 6, 7, 9, 0],
]
# DER prefix of a P-256 SPKI before the 65-byte uncompressed point.
SPKI_PREFIX = bytes.fromhex("3059301306072a8648ce3d020106082a8648ce3d030107034200")


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def b64u_decode(t: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_-]*", t or "") or len(t) % 4 == 1:
        raise ValueError(f"not base64url: {t!r}")
    return base64.urlsafe_b64decode(t + "=" * (-len(t) % 4))


def mac_id(fp: bytes) -> str:
    """Crockford base32 of fp[0:10], 16 characters."""
    n = int.from_bytes(fp[:10], "big")
    return "".join(CROCKFORD[(n >> (75 - 5 * i)) & 31] for i in range(16))


def damm(digits: str) -> int:
    i = 0
    for d in digits:
        i = DAMM[i][int(d)]
    return i


def check_code(text: str) -> str:
    """The typed code with spaces and hyphens removed; ValueError when it is not 12 digits with a
    valid Damm check digit (so a typo never spends one of the Mac's five tries)."""
    kept = re.sub(r"[ \- ]", "", text)
    if not re.fullmatch(r"[0-9]{12}", kept):
        raise ValueError("a code has 12 digits")
    if damm(kept) != 0:
        raise ValueError("that code has a typo; check it against the Mac")
    return kept


def spki_fingerprint_of_point(point: bytes) -> bytes:
    return hashlib.sha256(SPKI_PREFIX + point).digest()


def fingerprint_of_cert_der(der: bytes) -> bytes | None:
    """SHA-256 of the certificate's SPKI when its key is P-256, else None (no pin matches None)."""
    cert = x509.load_der_x509_certificate(der)
    key = cert.public_key()
    if not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(key.curve, ec.SECP256R1):
        return None
    return hashlib.sha256(key.public_bytes(serialization.Encoding.DER,
                                           serialization.PublicFormat.SubjectPublicKeyInfo)).digest()


def code_key(code: str, mac_fp: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", code.encode(), PAIR_LABEL + mac_fp, PBKDF2_ROUNDS, 32)


def device_proof(key: bytes, device_fp: bytes, mac_fp: bytes) -> bytes:
    return hmac.new(key, PAIR_LABEL + b" device\x00" + device_fp + mac_fp, hashlib.sha256).digest()


def mac_proof(key: bytes, mac_fp: bytes, device_fp: bytes) -> bytes:
    return hmac.new(key, PAIR_LABEL + b" mac\x00" + mac_fp + device_fp, hashlib.sha256).digest()


def recognition_tag(recognition_key: bytes, nonce: bytes) -> str:
    mac = hmac.new(recognition_key, b"sill-tag-v1" + nonce, hashlib.sha256).digest()[:6]
    return b64u(nonce + mac)


def verify_mac_info(payload: bytes, pin: bytes | None) -> dict | None:
    """Kind 18 (SignedMacInfo): the MacInfo dict when the ECDSA signature verifies over the exact info
    bytes, the key is P-256, macID is that key's Mac ID, and the key equals `pin` when one is given."""
    try:
        d = json.loads(payload)
        info = base64.b64decode(d["info"])
        point = base64.b64decode(d["key"])
        sig = base64.b64decode(d["sig"])
        pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), point)
        pub.verify(sig, info, ec.ECDSA(hashes.SHA256()))
    except (ValueError, KeyError, TypeError, InvalidSignature):
        return None
    fp = spki_fingerprint_of_point(point)
    i = json.loads(info)
    if i.get("macID") != mac_id(fp) or (pin is not None and fp != pin):
        return None
    return i


@dataclass(frozen=True)
class PairLink:
    fingerprint: bytes
    secret: bytes
    port: int
    name: str
    addresses: tuple[str, ...]

    @property
    def mac_id(self) -> str:
        return mac_id(self.fingerprint)


def parse_link(url: str) -> PairLink:
    """sill://pair?v=1&m=<macID>&k=<fp>&s=<secret>&n=<name>&p=<port>&a=<address>..."""
    u = urllib.parse.urlsplit(url.strip())
    if u.scheme.lower() != "sill" or u.netloc.lower() != "pair":
        raise ValueError("not a sill://pair link")
    q = urllib.parse.parse_qs(u.query, keep_blank_values=True)

    def one(k):
        v = q.get(k, [])
        if len(v) != 1 or not v[0]:
            raise ValueError(f"link field {k} missing or repeated")
        return v[0]

    if one("v") != "1":
        raise ValueError("link version is not 1")
    fp, secret = b64u_decode(one("k")), b64u_decode(one("s"))
    if len(fp) != 32 or len(secret) != 16:
        raise ValueError("link k or s malformed")
    if one("m") != mac_id(fp):
        raise ValueError("link m is not the Mac ID of k")
    p = one("p")
    if not re.fullmatch(r"[0-9]{1,5}", p) or not 1 <= int(p) <= 65535:
        raise ValueError("link p malformed")
    return PairLink(fp, secret, int(p), (q.get("n") or ["Mac"])[0], tuple(q.get("a", [])[:5]))


class Identity:
    """This client's P-256 key and a self-signed certificate, kept in `directory` (0700), plus the
    paired Mac once there is one (mac.json). Shaped like Sill's own: a random CN that names nothing."""

    def __init__(self, directory: str | os.PathLike):
        self.dir = Path(directory)
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.key_path, self.cert_path, self.mac_path = (self.dir / n for n in ("key.pem", "cert.pem", "mac.json"))
        if not self.key_path.exists():
            self._create()
        der = x509.load_pem_x509_certificate(self.cert_path.read_bytes()).public_bytes(serialization.Encoding.DER)
        self.fingerprint = fingerprint_of_cert_der(der)

    def _create(self):
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, os.urandom(8).hex())])
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(int.from_bytes(os.urandom(16), "big") >> 1 | 1)
                .not_valid_before(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
                .not_valid_after(dt.datetime(9999, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc))
                .sign(key, hashes.SHA256()))
        fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
        self.cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    def saved_mac(self) -> dict | None:
        return json.loads(self.mac_path.read_text()) if self.mac_path.exists() else None

    def save_mac(self, fingerprint: bytes, name: str | None, recognition_key: str | None, address: str | None):
        self.mac_path.write_text(json.dumps({"fingerprint": b64u(fingerprint), "macID": mac_id(fingerprint),
                                             "name": name, "recognitionKey": recognition_key,
                                             "address": address}, indent=1))

    def mac_pin(self) -> bytes | None:
        m = self.saved_mac()
        return b64u_decode(m["fingerprint"]) if m else None
