// Renders the film offscreen at chosen times and writes PNGs: node run/shots.mjs [query] [seconds,...] [WxH]
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { serve, open, png } from './lib.mjs';
const query = process.argv[2] || 'test=1&n=160&ar=1.6&iters=10';
const times = (process.argv[3] || '0,2,6').split(',').map(Number);
const [W, H] = (process.argv[4] || '400x640').split('x').map(Number);
const out = path.resolve(import.meta.dirname, '../out'); await mkdir(out, { recursive: true });
const { srv, url } = await serve();
const { browser, page, errors } = await open(url, query);
let t = 0;
for (const until of times) {
  const steps = Math.round((until - t) * 60); t = until;
  const t0 = Date.now();
  if (steps > 0) await page.evaluate(n => window.__soap.step(n), steps);
  const b64 = await page.evaluate(([w, h]) => window.__soap.shot(w, h), [W, H]);
  await writeFile(path.join(out, `t${String(until).padStart(3, '0')}.png`), png(Buffer.from(b64, 'base64'), W, H));
  console.log(`t=${until}s  ${steps} steps in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
if (errors.length) console.log('ERRORS', errors.slice(0, 8));
await browser.close(); srv.close();
