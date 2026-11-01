// The bot's half of the force-touch run: join the phone's room as an observer and write every record it sends to
// results/raw-<ROOM>.jsonl, until the phone says it is done (or the time runs out).
//
//   node force-touch/observe.mjs ROOM [minutes=30] [lobby=wss://rally-lobby.austegard.workers.dev]
//
// Run it as a background job from a cloud session. The WebSocket runs inside headless Chromium because that can use
// the session's HTTPS proxy; a cloud container has no other route out. The proxy drops long-lived sockets now and then
// (one closed after 8 minutes on 2026-11-01), so the observer reconnects, and the phone resends its hello on rejoin.
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";

const [codeArg, minutes = "30", lobby = "wss://rally-lobby.austegard.workers.dev"] = process.argv.slice(2);
if (!codeArg) { console.error("usage: node force-touch/observe.mjs ROOM [minutes] [lobby]"); process.exit(2); }
const code = codeArg.toUpperCase();
if (!/^[A-Z0-9]{4,8}$/.test(code)) { console.error("room codes are 4-8 characters, A-Z and 0-9 only (the phone strips anything else, so the two would not meet)"); process.exit(2); }
const HERE = path.dirname(fileURLToPath(import.meta.url)), OUT = path.join(HERE, "results", `raw-${code}.jsonl`);
fs.mkdirSync(path.dirname(OUT), { recursive: true });
// one file per attempt: an earlier file for the same code is set aside, never mixed in
if (fs.existsSync(OUT)) { const old = OUT.replace(/\.jsonl$/, `.before-${Date.now()}.jsonl`); fs.renameSync(OUT, old); console.log(`moved the earlier ${path.basename(OUT)} to ${path.basename(old)}`); }
const sink = fs.createWriteStream(OUT, { flags: "a" }), stamp = () => new Date().toISOString().slice(11, 19);
const srv = http.createServer((q, r) => { r.writeHead(200, { "content-type": "text/html" }); r.end("<!doctype html><title>observer</title>"); });
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));          // an origin the room server accepts
const proxy = process.env.HTTPS_PROXY || process.env.https_proxy;
const b = await chromium.launch({ ...(proxy && lobby.startsWith("wss:") ? { proxy: { server: proxy, bypass: "127.0.0.1,localhost" } } : {}), args: ["--ignore-certificate-errors"] });
const page = await b.newPage({ ignoreHTTPSErrors: true });
let finished = null; const fin = new Promise((r) => { finished = r; });
await page.exposeFunction("record", (line) => {
  const m = JSON.parse(line);
  if (m.t?.startsWith("ft-")) sink.write(line + "\n");
  if (m.t === "ft-hello") console.log(`${stamp()} phone: ${m.dev?.ua}`);
  else if (m.t === "ft-press") {
    const last = (a, k) => (a?.length ? a[a.length - 1][k] : null);
    console.log(`${stamp()} press ${m.i + 1} ${m.cond}#${m.rep}: ${(m.t1 - m.t0) | 0} ms, ${m.ptr.length} pointer / ${m.tch.length} touch / ${m.mot.length} motion samples; `
      + `pressure ${last(m.ptr, 1)} width ${last(m.ptr, 2)} force ${last(m.tch, 1)} radiusX ${last(m.tch, 2)}${m.thin ? " (thinned 1/" + m.thin + ")" : ""}`);
  } else if (m.t === "ft-done") { console.log(`${stamp()} phone finished: ${m.presses} presses`); setTimeout(finished, 5000); }   // resent presses may follow
  else if (m.t === "log") console.log(`${stamp()} ${m.text}`);
});
await page.goto(`http://127.0.0.1:${srv.address().port}/`);
await page.evaluate((url) => {
  const go = () => { const ws = new WebSocket(url);
    ws.onopen = () => { window.record(JSON.stringify({ t: "log", text: "joined " + url })); ws.keep = setInterval(() => ws.send('{"t":"ping"}'), 20000); };
    ws.onmessage = (e) => window.record(e.data);
    ws.onclose = (e) => { clearInterval(ws.keep); window.record(JSON.stringify({ t: "log", text: `closed (${e.code}); reconnecting` })); setTimeout(go, 2000); }; };
  go();
}, `${lobby}/room/${code}?name=observer:claude`);
console.log(`${stamp()} watching room ${code} for up to ${minutes} min; writing ${OUT}`);
await Promise.race([fin, new Promise((r) => setTimeout(r, Number(minutes) * 60000))]);
sink.end(); await b.close(); srv.close();
console.log(`${stamp()} stopped. Next: python3 force-touch/analyze.py ${OUT}`);
