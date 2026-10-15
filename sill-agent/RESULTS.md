# sill-agent: an agent as a paired Sill device

## Answer

Can a Claude Code session use [Sill](https://github.com/Saffsanity/sill) (open-source Mac → iPhone/iPad
window streaming) to see and operate Oskar's Mac? The client half works against a stand-in host.
`sill_agent/` pairs by the Mac's 12-digit code over mutual TLS 1.3 with pinned P-256 keys. It then
holds a `sill/1` session that keeps the window list, thumbnails, signed Mac info, menu bar and HEVC
frames current, and turns a frame into a full-resolution PNG through ffmpeg. Control (select, click,
type, keys, menu presses, launch) is refused unless the session is opened with `control=True`.

It has not run against a real Mac. Its crypto matches every pairing vector Sill publishes, and its
JSON matches the payload Sill's own Swift check asserts. 16 tests pass, 12 consecutive suite runs
were clean, and 5 of 5 seeded mutants went red. The fake host shares this reading of the protocol,
so agreement with Sill itself is unproven.

Two pieces are missing:
1. **A path from a session to the Mac.** The container has no UDP, and its egress is an HTTPS
   proxy, so it needs a TCP tunnel on port 443, such as `cloudflared access tcp` to a tunnel on the
   Mac. Untested.
2. **Enforcement on the Mac.** Sill gives every paired device full input. The planned fork adds an
   observer list checked in `StreamCoordinator.handle`. The fork was not created: auto mode refused
   it as "Create Public Surface".

## Findings

1. Sill's protocol is agent-shaped. It sends the window list as JSON, each app's menu bar read
   through Accessibility with presses by id, input as 0–1 frame fractions, and app launches by
   bundle id (`Sources/StreamProtocol/*.swift`, about 2,200 lines).
2. Thumbnails are not enough to see a window. The host makes them 208×132 px at JPEG quality 0.6,
   refreshing a static window every 6 s (`WindowCatalog.refreshThumbnails`). Readable text needs
   the HEVC stream, and Sill has one stream for all devices. So a strict observer sees only what a
   device already picked. Letting it pick also raises that window on the Mac and switches every
   device's view.
3. The host already supports the session. It pins keys at both doors, and it requests a keyframe
   whenever a device joins (`StreamCoordinator`), so `still` works at once if anything streams.
4. Upstream ships a test client. `Scripts/sillclient.py` (Apache-2.0) speaks pairing, TLS and
   sessions for the developers' own host tests. `sill_agent` is a library rewrite of the parts an
   agent needs, using `cryptography` in place of `/usr/bin/openssl` subprocesses.
5. One full-suite run in about 30 failed with `assert None` and a host-thread `EBADF`. Both the
   client and the fake host read and wrote one `SSLSocket` from several threads, which OpenSSL does
   not support. Moving the client to a single I/O thread, and putting the fake host's recv and send
   under one lock, removed the thread warnings. The failure never reproduced before or after the
   fix, so the race is the likely cause, not a proven one.

## Method

- **Vectors.** `tests/test_protocol.py` checks against `docs/remote-access-plan.md` §3.4–3.6 and the
  JSON in `Tests/checks/pointer-control/main.swift`.
- **Fake host.** `tests/fakehost.py` is a stand-in Mac on loopback: a Python `ssl` server with both
  ALPNs, a code pairing window with tries, a session that sends the catalog, signed kind 18,
  thumbnails and menus, and a libx265 test pattern (keyint 6) as length-prefixed NAL units.
- **Effects, not arrivals.** Typed text changes a window title, and a pick changes the active
  source. Assertions check those effects, so a test cannot pass on an unchanged frame (METHODS.md,
  screenshot assertions).
- **Mutants.** Five seeded by hand: the proof label, the source JSON key, the control gate, the pin
  comparison, and kind 18 verified without the pin. Each failed the suite.
- **Run it:** `python3 -m pytest -q tests` (about 7 s; needs ffmpeg with libx265 and `cryptography`).

## Log

### 2026-10-15 — client, fake host, CLI

Asked to start the experiment after comparing Petal and Sill as tools for an agent. Built `wire`,
`crypto`, `session` and `video`, plus the `sill.py` CLI: `pair`, `windows`, `thumb`, `still`,
`menus`, and with `--control` also `select`, `click`, `type`, `key`, `press`, `submenu` and
`launch`. `export FILE` writes the identity as three env lines to a 0600 file and never to stdout,
so a session can restore it from environment secrets without pairing again (`test_cli`).

Planned next and not done:
- **Mac-side observer patch.** `ObserverPolicy` in StreamProtocol (pure; allows hello, ping,
  stats, the menu subscription, and selectSource only when `ObserverMaySelect` is set), checked at
  the top of `StreamCoordinator.handle` against `ObserverDevices` (base64url fingerprints) in
  `me.saffer.sill.mac`'s defaults, with a `Tests/checks/observer` check and mutants. The fork it
  needs was refused by auto mode as "[Create Public Surface]". To continue, Oskar forks
  Saffsanity/sill himself or approves the fork. Upstream CI runs on a `runs-on: xcode-27` label,
  which a fork may not have.
- **Tunnel test.** Run `cloudflared` on the Mac exposing `tcp://localhost:7455`, and
  `cloudflared access tcp` in the session. Whether the egress proxy lets it through is the open
  question.
- **A run against a real Sill for Mac 0.5.x.**
