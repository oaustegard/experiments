// Renders the 1200x630 link-preview frame: node run/og.mjs  ->  out/og.png
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { serve, open, png } from './lib.mjs';
const out = path.resolve(import.meta.dirname, '../out'); await mkdir(out, { recursive: true });
const { srv, url } = await serve();
const { browser, page, errors } = await open(url, 'test=1&n=448&ar=0.5&iters=14&seed=' + (process.argv[2] || 21));
await page.evaluate(async () => {
  const S = window.__soap, { NX, NY } = S.cfg;
  await S.step(260);
  for (let k = 0; k < 40; k++) { const t = k / 40; S.blow(NX * (0.2 + 0.6 * t), NY * (0.55 - 0.25 * Math.sin(Math.PI * t)), NX * 0.9, -NY * 1.2 * Math.cos(Math.PI * t)); await S.step(1); }
  S.blow(); await S.step(150);
});
await writeFile(path.join(out, 'og.png'), png(Buffer.from(await page.evaluate(() => window.__soap.shot(1200, 630)), 'base64'), 1200, 630));
if (errors.length) console.log(errors);
await browser.close(); srv.close();
