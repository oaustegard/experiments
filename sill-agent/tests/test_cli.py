"""The CLI in-process against the fake host."""
import base64
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fakehost import FakeHost  # noqa: E402
import sill  # noqa: E402
from sill_agent import crypto, session  # noqa: E402


def test_cli_pair_windows_refuse_then_export(tmp_path, capsys):
    d = tmp_path / "id"
    ident = crypto.Identity(d)
    host = FakeHost(ident.cert_path)
    try:
        base = ["--host", "127.0.0.1", "--port", str(host.port), "--identity", str(d)]
        code = host.open_pairing_window()
        assert sill.main(base + ["pair", code[:4], code[4:8], code[8:]]) == 0
        assert sill.main(base + ["windows"]) == 0
        out = capsys.readouterr().out
        assert "notes.md" in out and "Inbox" in out
        with pytest.raises(session.ControlRefused):
            sill.main(base + ["type", "hello"])
        assert sill.main(base + ["--control", "type", "hello"]) == 0
        assert host.title_suffix == "hello"
        env = tmp_path / "agent.env"
        assert sill.main(base + ["export", str(env)]) == 0
        assert "SILL_AGENT_KEY" not in capsys.readouterr().out          # never on stdout
        assert stat.S_IMODE(env.stat().st_mode) == 0o600
        # A fresh container restores the same device from the env lines and needs no re-pairing.
        vals = dict(line.split("=", 1) for line in env.read_text().splitlines())
        os.environ.update(vals)
        try:
            assert sill.main(["--host", "127.0.0.1", "--port", str(host.port), "--identity",
                              str(tmp_path / "fresh"), "windows"]) == 0
        finally:
            for k in vals:
                os.environ.pop(k, None)
        assert crypto.Identity(tmp_path / "fresh").fingerprint == ident.fingerprint
    finally:
        host.stop()
