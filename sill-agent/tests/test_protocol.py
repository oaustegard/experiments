"""The protocol pieces against the vectors Sill's own plan publishes (docs/remote-access-plan.md
§3.4–3.6) and the JSON its Swift checks assert."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sill_agent import crypto, video, wire  # noqa: E402

FP_MAC = hashlib.sha256(b"mac").digest()
FP_DEV = hashlib.sha256(b"device").digest()


def test_vector_fingerprint_stand_ins():
    assert FP_MAC.hex() == "348a629f5ceed032c3e8706ec47d9bfafb00fb4250b018dd965435ca50cb836e"
    assert FP_DEV.hex() == "263a4dbe41488fb87214b0032339dbb9f0c8da14c16dfcf13084bf3c2552eca5"


def test_vector_mac_id():
    assert crypto.mac_id(hashlib.sha256(b"example").digest()) == "A3C5HR4RBV67YR21"


def test_vector_recognition_tag():
    assert crypto.recognition_tag(b"\x11" * 32, bytes.fromhex("a1a2a3a4a5a6")) == "oaKjpKWmL4V47Ctm"


def test_vector_code_path():
    k = crypto.code_key("482913557208", FP_MAC)
    assert k.hex() == "e9baf879a1cc4ac8b21554df480b6c1621671319107c91e7cb05d154f649d31d"
    assert crypto.b64u(crypto.device_proof(k, FP_DEV, FP_MAC)) == "dk7PZj4vHsAxj--1u9Vg7ehSn8a1tWc0RCG46YpFVO8"
    assert crypto.b64u(crypto.mac_proof(k, FP_MAC, FP_DEV)) == "6BmGWC9XT8J8OY8phhq9bASgoSnoWpRxBeqZtvRrWmk"


def test_vector_qr_path():
    k = b"\x22" * 16
    assert crypto.b64u(crypto.device_proof(k, FP_DEV, FP_MAC)) == "AyZAJqNTU5nSKEaHx6RZOf0K5dOeygR_8JSrCV_gNOo"
    assert crypto.b64u(crypto.mac_proof(k, FP_MAC, FP_DEV)) == "XAX-uZPsq_Ax8fvI_sV7Ex0a-ROi19rI-FW-ykPh9TE"


def test_damm_example_and_typos():
    assert crypto.check_code("4829 1355 7208") == "482913557208"
    with pytest.raises(ValueError, match="typo"):
        crypto.check_code("482913557209")
    with pytest.raises(ValueError, match="typo"):
        crypto.check_code("428913557208")          # adjacent transposition
    with pytest.raises(ValueError, match="12 digits"):
        crypto.check_code("48291355720")


def test_link_round_trip_and_refusals():
    fp, s = FP_MAC, b"\x22" * 16
    url = f"sill://pair?v=1&m={crypto.mac_id(fp)}&k={crypto.b64u(fp)}&s={crypto.b64u(s)}&n=Oskar%27s%20Mac&p=7455&a=100.64.0.7"
    lk = crypto.parse_link(url)
    assert (lk.fingerprint, lk.secret, lk.port, lk.name, lk.addresses) == (fp, s, 7455, "Oskar's Mac", ("100.64.0.7",))
    with pytest.raises(ValueError, match="Mac ID"):
        crypto.parse_link(url.replace(crypto.mac_id(fp), crypto.mac_id(FP_DEV)))
    with pytest.raises(ValueError, match="version"):
        crypto.parse_link(url.replace("v=1", "v=2"))


def test_json_shapes_match_swift_codable():
    # Tests/checks/pointer-control/main.swift asserts this exact payload is "the wire's own move".
    assert json.dumps(wire.pointer("move", 0.25, 0.5), separators=(",", ":")) == '{"pointer":{"_0":"move","x":0.25,"y":0.5}}'
    assert wire.source_window(42) == {"window": {"_0": 42}}
    assert wire.source_desktop() == {"desktop": {}}
    assert wire.text("hi") == {"text": {"_0": "hi"}}


def test_header_round_trip_and_cap():
    b = wire.encode(wire.Kind.FRAME, b"abc", keyframe=True, timestamp=1.5)
    assert len(b) == 17 and b[0] == 1 and b[9] == 1 and b[10:14] == b"\x00\x00\x00\x03"
    r = wire.Reader()
    assert r.feed(b[:5]) == []
    (m,) = r.feed(b[5:])
    assert (m.kind, m.timestamp, m.keyframe, m.payload) == (1, 1.5, True, b"abc")
    with pytest.raises(ValueError, match="cap"):
        wire.Reader().feed(wire.HEADER.pack(2, 0.0, 0, (4 << 20) + 1))


def test_parameter_sets_round_trip():
    sets = [b"\x40\x01vps", b"\x42\x01sps", b"\x44\x01pps"]
    assert wire.decode_parameter_sets(wire.encode_parameter_sets(4, sets)) == (4, sets)


def test_annexb_rewrite():
    au = (3).to_bytes(4, "big") + b"abc" + (2).to_bytes(4, "big") + b"de"
    assert video.to_annexb(au, 4) == b"\x00\x00\x00\x01abc\x00\x00\x00\x01de"
    with pytest.raises(ValueError):
        video.to_annexb(au[:-1], 4)


def test_identity_is_p256_and_pins(tmp_path):
    i = crypto.Identity(tmp_path / "id")
    assert len(i.fingerprint) == 32
    assert (tmp_path / "id" / "key.pem").stat().st_mode & 0o777 == 0o600
    assert crypto.Identity(tmp_path / "id").fingerprint == i.fingerprint     # stable across loads
