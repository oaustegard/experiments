// Drive austegard.com/bsky/pad.html as muninn: click Join, fill login,
// wait for the peer connection to open, screenshot the connected state.
//
// Env:
//   MUNINN_BSKY_HANDLE       muninn.austegard.com
//   MUNINN_BSKY_APP_PASSWORD app password
//
// Args: [url]  — full pad URL (defaults to the one in the task).

import { chromium } from 'playwright';
import { writeFileSync } from 'node:fs';

const URL_ARG = process.argv[2]
  || 'https://austegard.com/bsky/pad.html?d=sml6btszpyk3uiskko7o&peer=did%3Aplc%3Ar2whjvupgfw55mllpksnombn';
const HANDLE = process.env.MUNINN_BSKY_HANDLE;
const PASS   = process.env.MUNINN_BSKY_APP_PASSWORD;

if (!HANDLE || !PASS) {
  console.error('MUNINN_BSKY_HANDLE / MUNINN_BSKY_APP_PASSWORD not set');
  process.exit(2);
}

const outDir = new URL('.', import.meta.url).pathname;
const shot   = (name) => outDir + name;
const log    = (...a) => console.log('[pad]', ...a);

const PROXY = process.env.HTTPS_PROXY || process.env.https_proxy;
// The agent proxy's TLS termination fails Chromium's TLS 1.3 handshake
// (peer resets after ClientHello — repro: `--ssl-version-max=tls1.3` → RESET,
//  `tls1.2` → 200). Curl works because it doesn't offer ECH. Pinning to TLS 1.2
// is the smallest workaround; the browser NSS db still needs the proxy CA
// (`/root/.ccr/agent-proxy-ca.crt`) for HTTPS to verify.
const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
  headless: true,
  args: ['--no-sandbox', '--ssl-version-max=tls1.2'],
  proxy: PROXY ? { server: PROXY, bypass: process.env.NO_PROXY || '' } : undefined,
});
const ctx = await browser.newContext({
  viewport: { width: 1200, height: 800 },
  userAgent: 'Mozilla/5.0 (Muninn Playwright) Chromium/136',
  ignoreHTTPSErrors: true,
});
const page = await ctx.newPage();

// Surface JS console + failures. Also mirror atproto-rtc debug to our stdout
// so we can watch the offer/answer/ICE flow in real time.
const consoleLines = [];
page.on('console', m => {
  const line = `[${m.type()}] ${m.text()}`;
  consoleLines.push(line);
  if (/atproto-rtc|ice|offer|answer|knock/i.test(line)) console.log('  · ' + m.text());
});
page.on('pageerror', e => consoleLines.push(`[pageerror] ${e.message}`));
page.on('requestfailed', r => consoleLines.push(`[reqfail] ${r.url()} :: ${r.failure()?.errorText}`));

log('opening', URL_ARG);
await page.goto(URL_ARG, { waitUntil: 'domcontentloaded' });

// Peer link should reveal the Join button.
await page.waitForSelector('#joinBtn:not([hidden])', { timeout: 10_000 });
log('join button visible; status=', await page.textContent('#statusText'));
await page.screenshot({ path: shot('01_before_join.png') });

// Click Join — triggers the login dialog (no session yet).
await page.click('#joinBtn');
await page.waitForSelector('#loginDialog[open]', { timeout: 10_000 });
log('login dialog opened');

await page.fill('#loginHandle', HANDLE);
await page.fill('#loginPassword', PASS);
await page.screenshot({ path: shot('02_login_filled.png') });

await page.click('#loginSubmitBtn');
log('submitted login as', HANDLE);

// Wait for either success (dialog's [open] attribute is removed) or an
// inline error from rtc.login. `waitForSelector(...:not([open]))` won't work
// here because a closed <dialog> is display:none, so Playwright's default
// visibility check never fires. Poll via waitForFunction instead.
let loginResult;
try {
  loginResult = await page.waitForFunction(
    () => {
      const dlg = document.getElementById('loginDialog');
      const err = document.getElementById('loginError');
      if (err && err.textContent && err.textContent.trim().length > 0) {
        return { ok: false, err: err.textContent.trim() };
      }
      if (dlg && !dlg.hasAttribute('open')) {
        return { ok: true, me: window.rtc?.me?.did || null };
      }
      return null;
    },
    { timeout: 20_000 }
  ).then(h => h.jsonValue());
} catch (e) {
  loginResult = { ok: false, err: 'timeout:' + e.message.split('\n')[0] };
}
log('login result:', JSON.stringify(loginResult));
await page.screenshot({ path: shot('03_after_login_submit.png') });

if (!loginResult.ok) {
  writeFileSync(shot('console.log'), consoleLines.join('\n'));
  console.error('login failed; see console.log');
  await browser.close();
  process.exit(1);
}

// Login succeeded. Give the join real time to knock, offer/answer, and ICE.
// Prior peers value observed in the DOM.
log('waiting up to 60s for a peer to attach…');
let statusText = '';
try {
  await page.waitForFunction(
    () => /· [1-9]\d* peer/.test(document.getElementById('statusText')?.textContent || ''),
    { timeout: 60_000 }
  );
  statusText = await page.textContent('#statusText');
} catch (e) {
  statusText = await page.textContent('#statusText');
  log('no peer attached in 60s; status is', JSON.stringify(statusText));
}
log('final status:', statusText);
log('join button label:', await page.textContent('#joinBtn'));

// `rtc` is module-scoped in pad.html so window.rtc doesn't exist — the
// mirrored [atproto-rtc] console lines above are the real signal for what
// happened in the WebRTC layer.

await page.screenshot({ path: shot('04_connected.png') });
writeFileSync(shot('console.log'), consoleLines.join('\n'));
writeFileSync(shot('result.json'), JSON.stringify({
  url: URL_ARG,
  loginResult,
  statusText,
}, null, 2));

log('done; screenshots + result.json written');
await browser.close();
