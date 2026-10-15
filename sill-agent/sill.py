#!/usr/bin/env python3
"""sill-agent CLI: an agent as a paired Sill device.

  sill.py pair CODE                     pair by the Mac's 12-digit code (Sill › Pair iPhone or iPad…)
  sill.py pair --link 'sill://pair?…'   pair by the QR link
  sill.py windows                       the Mac's windows, and what streams now
  sill.py thumb ID OUT.jpg              a window's latest thumbnail (the host refreshes them every 1–2 s)
  sill.py still OUT.png                 a full-resolution frame of whatever streams now
  sill.py menus                         the streamed app's menu bar (read over Accessibility; opens nothing)
  sill.py --control select ID|desktop   stream that window (in regular mode the Mac raises it)
  sill.py --control click X Y           frame fractions 0..1 of the streamed source
  sill.py --control type TEXT
  sill.py --control key HID [MODS]      a USB HID usage; MODS: shift,ctrl,opt,cmd
  sill.py --control press ITEM_ID       choose a menu item (ids from `menus` / `submenu`)
  sill.py --control submenu MENU_ID     open one menu and list its items
  sill.py --control launch BUNDLE_ID

Every command but pair needs a paired identity. --host/--port name the Mac's remote door (default
port 7455): its Tailscale address, or 127.0.0.1 with a TCP tunnel (`cloudflared access tcp`) in
front. The identity lives in --identity (default ~/.sill-agent), or comes from SILL_AGENT_KEY and
SILL_AGENT_CERT (PEM, base64) and SILL_AGENT_MAC (JSON) when those are set, so a session container
can be thrown away without re-pairing.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sill_agent import crypto, session, video, wire  # noqa: E402

MODS = {"shift": wire.SHIFT, "ctrl": wire.CONTROL, "opt": wire.OPTION, "cmd": wire.COMMAND}


def identity(args) -> crypto.Identity:
    d = Path(args.identity).expanduser()
    if os.environ.get("SILL_AGENT_KEY") and not (d / "key.pem").exists():
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(d / "key.pem", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(os.environ["SILL_AGENT_KEY"]))
        (d / "cert.pem").write_bytes(base64.b64decode(os.environ.get("SILL_AGENT_CERT", "")))
        if os.environ.get("SILL_AGENT_MAC"):
            (d / "mac.json").write_text(os.environ["SILL_AGENT_MAC"])
    return crypto.Identity(d)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("SILL_AGENT_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("SILL_AGENT_PORT", session.REMOTE_PORT)))
    p.add_argument("--identity", default=os.environ.get("SILL_AGENT_IDENTITY", "~/.sill-agent"))
    p.add_argument("--control", action="store_true", help="arm commands that change the Mac")
    p.add_argument("command")
    p.add_argument("rest", nargs="*")
    p.add_argument("--link")
    a = p.parse_args(argv)
    ident = identity(a)
    cmd, rest = a.command, a.rest

    if cmd == "pair":
        r = session.pair(a.host, a.port, ident, code=None if a.link else " ".join(rest), link=a.link)
        print(f"paired with {r.get('name')} ({r['macID']}); pin saved in {ident.mac_path}")
        return 0
    if cmd == "export":
        # The identity as three env lines, for the environment's secrets. Written to a 0600 file and
        # never printed: the key is the device's whole authority over the Mac.
        out = Path(rest[0]).expanduser()
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write("SILL_AGENT_KEY=" + base64.b64encode(ident.key_path.read_bytes()).decode() + "\n")
            f.write("SILL_AGENT_CERT=" + base64.b64encode(ident.cert_path.read_bytes()).decode() + "\n")
            f.write("SILL_AGENT_MAC=" + json.dumps(ident.saved_mac() or {}, separators=(",", ":")) + "\n")
        print(f"wrote {out} (0600); do not cat it into a transcript")
        return 0

    with session.Session(a.host, a.port, ident, control=a.control) as s:
        if cmd == "windows":
            ws = s.windows()
            print(f"active: {json.dumps(s.active_source())}")
            for w in ws:
                print(f"{w['id']:>8}  {w['appName']} — {w['title']}  ({w['width']}×{w['height']})")
        elif cmd == "thumb":
            jpeg = s.thumbnail(int(rest[0]))
            if jpeg is None:
                print("no thumbnail for that window", file=sys.stderr)
                return 1
            Path(rest[1]).write_bytes(jpeg)
            print(rest[1])
        elif cmd == "still":
            s.wait_for(lambda c: c.gop and c.parameter_sets, 10)
            time.sleep(0.5)                 # a few frames past the keyframe, so the still is current
            out = video.still(s.catalog.parameter_sets, list(s.catalog.gop), rest[0])
            if out is None:
                print("no keyframe received: is anything streaming? (`select` needs --control)", file=sys.stderr)
                return 1
            print(out)
        elif cmd == "menus":
            top = s.subscribe_menus()
            print(json.dumps(top, indent=1, ensure_ascii=False) if top else "no menus")
        elif cmd == "submenu":
            s.subscribe_menus()
            print(json.dumps(s.fetch_menu(rest[0]), indent=1, ensure_ascii=False))
        elif cmd == "select":
            s.windows()
            s.select(wire.source_desktop() if rest[0] == "desktop" else wire.source_window(int(rest[0])))
            ok = s.wait_for(lambda c: c.window_list and c.window_list["active"] != {"none": {}}, 5)
            print(f"active: {json.dumps(s.active_source())}" if ok else "the Mac did not switch")
        elif cmd == "click":
            s.windows()
            s.click(float(rest[0]), float(rest[1]))
        elif cmd == "type":
            s.windows()
            s.type_text(" ".join(rest))
        elif cmd == "key":
            s.windows()
            mods = sum(MODS[m] for m in rest[1].split(",")) if len(rest) > 1 else 0
            s.press_key(int(rest[0], 0), mods)
        elif cmd == "press":
            s.subscribe_menus()
            r = s.press_menu(rest[0])
            print(json.dumps(r, ensure_ascii=False))
        elif cmd == "launch":
            s.windows()
            s.launch(rest[0])
        else:
            p.error(f"unknown command {cmd}")
        time.sleep(0.3)                     # let the last writes leave before the socket closes
        if s.catalog.goodbye:
            print(f"goodbye: {json.dumps(s.catalog.goodbye)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except session.ControlRefused as e:
        print(f"refused: {e} (add --control)", file=sys.stderr)
        sys.exit(2)
    except (session.PinMismatch, PermissionError, ConnectionError, ValueError, OSError) as e:
        print(f"sill-agent: {e}", file=sys.stderr)
        sys.exit(1)
