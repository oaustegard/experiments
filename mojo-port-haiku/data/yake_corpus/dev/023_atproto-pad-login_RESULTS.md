# atproto-pad-login

Drive `austegard.com/bsky/pad.html` end-to-end as **muninn.austegard.com** via
Playwright/Chromium: open the shared-pad URL, click *Join*, fill the login
dialog, wait for `rtc.login()` to establish a PDS session and `rtc.connect()`
to send its WebRTC knock to the peer DID.

**Result:** login succeeds; status flips from `Local only · 0 peers` to
`Connected · 0 peers`. The knock reaches the peer DID via the ATProto signal
records (console: `[atproto-rtc][knock] sent to did:plc:r2whjvupgfw55mllpksnombn`).
Confirmed live with the host actively Sharing on the other end — muninn still
stays at `0 peers`. The data channel cannot come up from this container even
when the peer is present: **CCotw has no UDP egress**, so RTCPeerConnection has
no reachable ICE candidates to publish. Both `/dev/udp/stun.l.google.com/19302`
and `/dev/tcp/stun.l.google.com/19302` time out (proxy is HTTPS-CONNECT only).
Signaling ✓ (ATProto records via HTTPS), datachannel ✗ (UDP blocked).

To actually complete a join from here you'd need either (a) to run
`pad_login.mjs` from a machine with normal network egress, or (b) a
TURN-over-TLS relay reachable on 443/TCP, which requires a small pad tweak —
`AtprotoRTC`'s `iceServers` default is hard-coded at `atproto-rtc.js:104` with
no URL knob to override.

Test URL: `https://austegard.com/bsky/pad.html?d=sml6btszpyk3uiskko7o&peer=did%3Aplc%3Ar2whjvupgfw55mllpksnombn`

## Files

- `pad_login.mjs` — the Playwright driver
- `01_before_join.png` — pad open, *Join* button visible
- `02_login_filled.png` — login dialog with muninn's handle + app password
- `03_after_login_submit.png` — dialog closed, session established
- `04_connected.png` — status = `Connected · 0 peers`
- `result.json` — machine-readable summary of the run
- `console.log` — page-side `[debug] atproto-rtc` traces

Screenshot and `result.json` are regenerated on each run; committed as the
proof-of-work artifact for this experiment. The Playwright browsers install
(`node_modules/`) is gitignored.

## Finding: Chromium via the agent proxy needs `--ssl-version-max=tls1.2`

The interesting friction wasn't the pad or ATProto — it was routing headless
Chromium through the session's HTTPS_PROXY. Curl and Node's fetch worked
against `austegard.com` immediately; Chromium consistently returned
`net::ERR_CONNECTION_RESET` on `page.goto()`.

Repro trail (all with `--proxy-server=$HTTPS_PROXY`):

| Config                                                    | Result           |
|-----------------------------------------------------------|------------------|
| Default (TLS 1.3, ECH on)                                 | `CONNECTION_RESET` |
| `--ignore-certificate-errors` + `ignoreHTTPSErrors: true` | `CONNECTION_RESET` |
| Above + proxy CA installed into NSS DB (`certutil -A`)    | `CONNECTION_RESET` |
| Above + `--disable-features=EncryptedClientHello`         | `CONNECTION_RESET` |
| Above + `--ssl-version-max=tls1.2`                        | ✅ 200            |
| **Only** `--ssl-version-max=tls1.2` (nothing else)        | ✅ 200            |

NetLog analysis (`--log-net-log=/tmp/netlog.json`) showed the CONNECT tunnel
came back `200 Connection Established`, Chromium sent a ~1700-byte TLS 1.3
ClientHello into the tunnel, and the proxy closed the socket (`SOCKET_READ_ERROR
{net_error: -101, os_error: 104}` — ECONNRESET). Curl doesn't reproduce
because it doesn't offer the TLS 1.3 extensions Chromium ships (in particular
ECH; disabling ECH alone was *not* enough — the full 1.3 handshake was still
rejected).

So the session's egress gateway (identity chain: leaf → *Egress Gateway SDS
Issuing CA (production)* → *sandbox-egress-gateway-production Egress Gateway
CA*) MITM-terminates HTTPS but breaks on modern Chromium's TLS 1.3 handshake.
Pinning the browser to TLS 1.2 with `--ssl-version-max=tls1.2` is the minimal
workaround; `ignoreHTTPSErrors` alone won't rescue you because the connection
is reset before the cert layer is even reached.

Side finding: the pre-provisioned NSS db at `/root/.pki/nssdb` was **empty**
(`certutil -L` returned only the header row), contradicting the
`/root/.ccr/README.md` claim that "the browser NSS store … [is] already set up".
Adding `/root/.ccr/agent-proxy-ca.crt` with `certutil -A ... -t 'C,,'` is needed
before Chromium can verify the proxy-terminated cert — TLS 1.2 pinning alone
gets past the RESET but the cert would still be untrusted without the CA.

## How to reproduce

```bash
cd ~/claude-workspace/experiments/atproto-pad-login
npm install                 # brings in playwright
# One-time: install the proxy CA into the browser trust store
sudo apt-get install -y libnss3-tools
certutil -A -d sql:$HOME/.pki/nssdb \
  -n 'CCR Agent Proxy CA' -t 'C,,' -i /root/.ccr/agent-proxy-ca.crt
# muninn credentials are already in the session env
node pad_login.mjs
```

Pass an alternate pad URL as `argv[2]` to test another shared pad.
