/**
 * Is hysnappy a practical in-browser compression library?
 *
 * The head-to-head in hysnappy_bench.mjs asks whether it is fast. This asks
 * whether it is the right tool, which turns on four things that benchmark does
 * not touch: which snappy format it speaks, what happens when input or
 * outputLength is wrong, how it compares to the codec every browser already
 * ships, and whether the 4 KB constraint its design is built around still
 * exists.
 *
 * Runs the browser half in real Chromium via playwright-core. Writes
 * practicality_results.json.
 *
 *   npm install && npm install --no-save playwright-core && node practicality.mjs
 */
import { snappyUncompressor, snappyCompressor } from 'hysnappy'
import { readFileSync, writeFileSync } from 'node:fs'
import http from 'node:http'

const CHROMIUM = process.env.CHROMIUM_PATH
  ?? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
const enc = new TextEncoder()
const hyC = snappyCompressor()
const hyU = snappyUncompressor()

const apiJson = enc.encode(JSON.stringify(Array.from({ length: 400 }, (_, i) => ({
  id: `usr_${i.toString(36)}`, name: `User ${i}`, email: `user${i}@example.com`,
  role: ['admin', 'member', 'viewer'][i % 3], active: i % 7 !== 0,
}))))
const prose = enc.encode(('The quick brown fox jumps over the lazy dog. '
  + 'Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump. ').repeat(400))

const out = { generated_by: 'practicality.mjs', env: { node: process.version } }

// ── 1. Which snappy format ──────────────────────────────────────────────────
// The framed (.sz) format starts with a stream identifier and carries a CRC32C
// per chunk. The block format has no header and no checksum at all.
const STREAM_ID = [0xff, 0x06, 0x00, 0x00, 0x73, 0x4e, 0x61, 0x50, 0x70, 0x59]
const sample = hyC(enc.encode('hyperparam '.repeat(8)))
out.format = {
  framed_sz: STREAM_ID.every((b, i) => sample[i] === b),
  first_bytes: [...sample.slice(0, 6)].map(b => '0x' + b.toString(16).padStart(2, '0')),
  note: 'Block format: no stream header, no CRC32C, no framing, so no streaming decode.',
}

// ── 2. outputLength is unchecked in both directions ─────────────────────────
const packed = hyC(apiJson)
out.output_length_discipline = []
for (const delta of [-1000, -100, -1, 0, 1, 100]) {
  let row
  try {
    const r = hyU(packed, apiJson.length + delta)
    const correct = r.length === apiJson.length && r.every((v, i) => v === apiJson[i])
    row = { delta, threw: false, returned_bytes: r.length, bytes_correct: correct }
  } catch (e) {
    row = { delta, threw: true, message: e.message }
  }
  out.output_length_discipline.push(row)
}

// ── 3. Corruption detection: block-format snappy carries no checksum ────────
{
  let threw = 0, silentWrong = 0, silentRight = 0
  const TRIALS = 400
  for (let t = 0; t < TRIALS; t++) {
    const c = packed.slice()
    c[1 + Math.floor(t / TRIALS * (c.length - 1))] ^= 1 << (t % 8)
    try {
      const r = hyU(c, apiJson.length)
      if (r.length === apiJson.length && r.every((v, i) => v === apiJson[i])) silentRight++
      else silentWrong++
    } catch { threw++ }
  }
  out.corruption = {
    method: 'single-bit flip at an even stride through the compressed bytes',
    compressed_bytes: packed.length,
    trials: TRIALS,
    threw,
    returned_wrong_bytes: silentWrong,
    returned_correct_bytes: silentRight,
    silent_corruption_pct: Math.round(100 * silentWrong / TRIALS),
  }
}

// ── 4. Against the codec every browser already ships ────────────────────────
async function streamCodec(bytes, format, dir = 'c') {
  const s = dir === 'c' ? new CompressionStream(format) : new DecompressionStream(format)
  const w = s.writable.getWriter()
  w.write(bytes); w.close()
  return new Uint8Array(await new Response(s.readable).arrayBuffer())
}
out.ratio = []
for (const [name, data] of [['API JSON', apiJson], ['English prose', prose]]) {
  const snap = hyC(data)
  const gz = await streamCodec(data, 'gzip')
  const raw = await streamCodec(data, 'deflate-raw')
  out.ratio.push({
    corpus: name,
    input_bytes: data.length,
    snappy_bytes: snap.length,
    snappy_pct: Number((100 * snap.length / data.length).toFixed(1)),
    gzip_bytes: gz.length,
    gzip_pct: Number((100 * gz.length / data.length).toFixed(1)),
    deflate_raw_bytes: raw.length,
    snappy_over_gzip: Number((snap.length / gz.length).toFixed(2)),
  })
}

// At 8 MB the per-call CompressionStream setup no longer dominates, so this
// compares codecs rather than API shapes.
{
  const big = new Uint8Array(8 * 1024 * 1024)
  for (let i = 0; i < big.length; i++) big[i] = apiJson[i % apiJson.length]
  const snap = hyC(big)
  const gz = await streamCodec(big, 'gzip')
  const time = async (fn, n) => {
    for (let i = 0; i < 3; i++) await fn()
    const t = performance.now()
    for (let i = 0; i < n; i++) await fn()
    return (performance.now() - t) / n
  }
  const mb = 8
  out.large_payload = {
    input_bytes: big.length,
    snappy_bytes: snap.length,
    gzip_bytes: gz.length,
    snappy_decode_mbps: Math.round(mb / (await time(() => hyU(snap, big.length), 10) / 1000)),
    gzip_decode_mbps: Math.round(mb / (await time(() => streamCodec(gz, 'gzip', 'd'), 10) / 1000)),
    snappy_encode_mbps: Math.round(mb / (await time(() => hyC(big), 5) / 1000)),
    gzip_encode_mbps: Math.round(mb / (await time(() => streamCodec(big, 'gzip'), 5) / 1000)),
  }
}

// ── 5. In a real browser, and does the 4 KB rule still exist? ───────────────
let browserOut = { skipped: 'playwright-core or chromium unavailable' }
try {
  const { chromium } = await import('playwright-core')
  const files = {
    '/index.js': readFileSync('node_modules/hysnappy/js/index.js', 'utf8'),
    '/uncompress.js': readFileSync('node_modules/hysnappy/js/uncompress.js', 'utf8'),
    '/compress.js': readFileSync('node_modules/hysnappy/js/compress.js', 'utf8'),
  }
  const server = http.createServer((req, res) => {
    const body = files[req.url] ?? '<!doctype html><meta charset=utf-8>'
    res.writeHead(200, { 'content-type': req.url.endsWith('.js') ? 'text/javascript' : 'text/html' })
    res.end(body)
  })
  await new Promise(r => server.listen(0, r))
  const port = server.address().port
  const browser = await chromium.launch({ executablePath: CHROMIUM })
  const page = await browser.newPage()
  await page.goto(`http://127.0.0.1:${port}/`)
  browserOut = await page.evaluate(async (port) => {
    const res = { ua: navigator.userAgent }
    const mod = await import(`http://127.0.0.1:${port}/index.js`)
    const t0 = performance.now()
    const uncompress = mod.snappyUncompressor()
    res.first_instantiate_ms = Number((performance.now() - t0).toFixed(2))
    const compress = mod.snappyCompressor()

    const enc2 = new TextEncoder()
    const data = enc2.encode(JSON.stringify(Array.from({ length: 400 }, (_, i) => ({
      id: `usr_${i.toString(36)}`, name: `User ${i}`, email: `user${i}@example.com`,
      role: ['admin', 'member', 'viewer'][i % 3], active: i % 7 !== 0 }))))
    const p = compress(data)
    const back = uncompress(p, data.length)
    res.round_trip_ok = back.length === data.length && back.every((v, i) => v === data[i])
    res.input_bytes = data.length
    res.snappy_bytes = p.length

    const bench = (fn, n) => { for (let i = 0; i < 50; i++) fn()
      const s = performance.now(); for (let i = 0; i < n; i++) fn(); return performance.now() - s }
    const N = 500
    res.snappy_decode_mbps = Math.round(data.length * N / 1048576 / (bench(() => uncompress(p, data.length), N) / 1000))
    res.CompressionStream = typeof CompressionStream !== 'undefined'
    if (res.CompressionStream) {
      const sd = async (b) => { const s = new DecompressionStream('gzip'); const w = s.writable.getWriter()
        w.write(b); w.close(); return new Uint8Array(await new Response(s.readable).arrayBuffer()) }
      const sc = async (b) => { const s = new CompressionStream('gzip'); const w = s.writable.getWriter()
        w.write(b); w.close(); return new Uint8Array(await new Response(s.readable).arrayBuffer()) }
      const gz = await sc(data)
      res.gzip_bytes = gz.length
      const M = 200
      const t = performance.now()
      for (let i = 0; i < M; i++) await sd(gz)
      res.gzip_decode_mbps = Math.round(data.length * M / 1048576 / ((performance.now() - t) / 1000))
    }

    // Take the real module and pad it past 4096 bytes with a VALID custom
    // section, so the only rule that can reject it is the size rule.
    const wasm64 = (await (await fetch(`http://127.0.0.1:${port}/uncompress.js`)).text())
      .match(/const wasm64 = ['"]([^'"]*)['"]/)[1]
    const bin = Uint8Array.from(atob(wasm64), c => c.charCodeAt(0))
    res.real_module_bytes = bin.length
    function padTo(b, target) {
      const body = new Uint8Array(2 + (target - b.length - 8))
      body[0] = 1; body[1] = 0x70 // custom section name: length 1, "p"
      const leb = []; let v = body.length
      do { let x = v & 0x7f; v >>>= 7; if (v) x |= 0x80; leb.push(x) } while (v)
      const o = new Uint8Array(b.length + 1 + leb.length + body.length)
      o.set(b, 0); o[b.length] = 0x00
      o.set(leb, b.length + 1); o.set(body, b.length + 1 + leb.length)
      return o
    }
    res.sync_module_limit = {}
    for (const n of [4096, 8000, 1_000_000, 8_388_608, 8_388_700, 16_000_000]) {
      const padded = padTo(bin, n)
      try {
        new WebAssembly.Instance(new WebAssembly.Module(padded))
        res.sync_module_limit[padded.length] = 'accepted'
      } catch (e) { res.sync_module_limit[padded.length] = e.message.slice(0, 100) }
    }
    return res
  }, port)
  await browser.close()
  server.close()
} catch (e) {
  browserOut = { skipped: String(e.message).slice(0, 200) }
}
out.browser = browserOut

writeFileSync('./practicality_results.json', JSON.stringify(out, null, 2) + '\n')
console.log(JSON.stringify(out, null, 2))
