"""End to end over loopback against tests/fakehost.py: pair by typed code, pin, observe, control."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fakehost import FakeHost  # noqa: E402
from sill_agent import crypto, session, wire  # noqa: E402
from sill_agent.session import ControlRefused, PinMismatch, Session  # noqa: E402


@pytest.fixture
def rig(tmp_path):
    ident = crypto.Identity(tmp_path / "agent")
    host = FakeHost(ident.cert_path)
    yield ident, host
    host.stop()


def test_unpaired_session_is_refused(rig):
    ident, host = rig
    with pytest.raises(ValueError, match="pair first"):
        Session("127.0.0.1", host.port, ident)
    # A forged pin: the host's key does not match, and nothing is sent.
    with pytest.raises(PinMismatch):
        Session("127.0.0.1", host.port, ident, pin=b"\x00" * 32).open()
    assert host.received == []


def test_wrong_code_spends_a_try_and_saves_nothing(rig):
    ident, host = rig
    code = host.open_pairing_window()
    wrong = code[:10] + next(str(d) for d in range(10) if d != int(code[10]))
    wrong += str(crypto.damm(wrong))                       # a valid code, just not this one
    with pytest.raises(PermissionError, match="code"):
        session.pair("127.0.0.1", host.port, ident, code=wrong)
    assert host.tries_left == 4 and ident.saved_mac() is None


def test_pair_observe_and_control(rig, tmp_path):
    ident, host = rig
    code = host.open_pairing_window()
    shown = " ".join(code[i:i + 4] for i in (0, 4, 8))         # as the Mac shows it: "4829 1355 7208"
    r = session.pair("127.0.0.1", host.port, ident, code=shown)
    assert r["ok"] and ident.mac_pin() == host.fingerprint
    assert ident.saved_mac()["macID"] == crypto.mac_id(host.fingerprint)

    with Session("127.0.0.1", host.port, ident) as s:          # observe-only by default
        ws = s.windows()
        assert [w["id"] for w in ws] == [101, 202]
        assert s.thumbnail(202).endswith(b"202")
        assert s.wait_for(lambda c: c.mac_info, 5)["name"] == "Test Mac"   # signature verified against the pin
        assert s.subscribe_menus()["menus"][0]["title"] == "File"
        assert s.wait_for(lambda c: c.rtt_ms is not None, 3) is not None
        for act in (lambda: s.select_window(101), lambda: s.type_text("x"), lambda: s.click(0.5, 0.5),
                    lambda: s.press_menu("1.0"), lambda: s.launch("com.apple.TextEdit"),
                    lambda: s.fetch_menu("1")):
            with pytest.raises(ControlRefused):
                act()
        time.sleep(0.2)
    sent_kinds = {k for k, _ in host.received}
    assert not sent_kinds & set(wire.CONTROL_KINDS), sent_kinds
    assert wire.Kind.HELLO in sent_kinds

    with Session("127.0.0.1", host.port, ident, control=True) as s:
        s.windows()
        s.select_window(101)
        # The host's own effect, not just "something arrived": the active source and the title change.
        assert s.wait_for(lambda c: c.window_list and c.window_list["active"] == {"window": {"_0": 101}}, 5)
        s.type_text(" edited")
        assert s.wait_for(lambda c: c.window_list["windows"][0]["title"] == "notes.md edited", 5)
        s.subscribe_menus()
        assert s.fetch_menu("1")["items"][0]["title"] == "Save"
        assert s.press_menu("1.0")["pressed"] is True and host.pressed == ["1.0"]
        assert s.wait_for(lambda c: c.frames >= len(host.aus), 10)
        from sill_agent import video
        png = video.still(s.catalog.parameter_sets, list(s.catalog.gop), tmp_path / "still.png")
        assert png is not None and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert [k for _, k, _ in s.sent][:2] == [wire.Kind.SELECT_SOURCE, wire.Kind.INPUT]
