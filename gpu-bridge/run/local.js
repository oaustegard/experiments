// Run one job locally on SwiftShader (correctness only): node local.js <job.js> <inputs dir> [out.json]
// <inputs dir>/inputs.json maps name -> {file, dtype} (raw little-endian .bin) or {json: value}; a list of
// such maps runs the job once per map in the same worker, in order (to test ctx.cache across jobs).
// Writes outputs with typed arrays as {dtype, b64}. Uses the page's own runtime.js.
const { chromium } = require('playwright'); const http = require('http'), fs = require('fs'), path = require('path');
const [jobFile, inDir, outFile = path.join(inDir, 'out.json')] = process.argv.slice(2);
const RT = path.join(__dirname, '..', 'web', 'runtime.js');
const srv = http.createServer((q, r) => {
  const u = decodeURIComponent(new URL(q.url, 'http://x').pathname);
  const f = u === '/' ? null : u === '/runtime.js' ? RT : u === '/job.js' ? jobFile : path.join(inDir, u.slice(1));
  if (!f) { r.writeHead(200, { 'content-type': 'text/html' }); return r.end('<!doctype html><title>local</title>'); }
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': 'application/octet-stream' }); r.end(fs.readFileSync(f));
}).listen(0, async () => {
  const port = srv.address().port;
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader'] });
  const p = await b.newPage(); p.on('pageerror', (e) => console.error('pageerror', e.message));
  await p.goto(`http://localhost:${port}/`);
  const out = await p.evaluate(async () => {
    const C = { f32: Float32Array, u32: Uint32Array, i32: Int32Array, u8: Uint8Array, u16: Uint16Array, f16: Uint16Array };
    const raw = await (await fetch('/inputs.json')).json(), specs = Array.isArray(raw) ? raw : [raw], sets = [];
    for (const spec of specs) { const inputs = {}; for (const [k, v] of Object.entries(spec)) inputs[k] = v.file ? new C[v.dtype](await (await fetch('/' + v.file)).arrayBuffer()) : v.json; sets.push(inputs); }
    const rt = await (await fetch('/runtime.js')).text(), code = await (await fetch('/job.js')).text();
    const w = new Worker(URL.createObjectURL(new Blob([rt], { type: 'text/javascript' })));
    const enc = (v) => { if (ArrayBuffer.isView(v)) { let s = ''; const u = new Uint8Array(v.buffer, v.byteOffset, v.byteLength); for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return { dtype: v.constructor.name, b64: btoa(s) }; }
      if (Array.isArray(v)) return v.map(enc); if (v && typeof v === 'object') { const o = {}; for (const [k, x] of Object.entries(v)) o[k] = enc(x); return o; } return v; };
    const runs = [];
    for (const [k, inputs] of sets.entries()) runs.push(await new Promise((res) => { const logs = []; w.onmessage = (e) => { if (e.data.t === 'log') logs.push(e.data.s); else res({ t: e.data.t, s: e.data.s, logs, result: enc(e.data.result) }); };
      w.onerror = (e) => res({ t: 'error', s: e.message, logs });
      w.postMessage({ t: 'run', id: 'local' + k, src: `self.__jobs["local${k}"] = async function (ctx) {\n${code}\n};`, inputs }); }));
    return Array.isArray(raw) ? runs : runs[0];
  });
  fs.writeFileSync(outFile, JSON.stringify(out)); for (const o of [].concat(out)) console.log(o.t, o.s ? String(o.s).slice(0, 400) : '', JSON.stringify(o.logs));
  await b.close(); srv.close();
});
