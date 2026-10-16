import { readFileSync } from 'node:fs';
import { inflateRawSync } from 'node:zlib';
import { b62decode, cleanBody, b62encode } from './codec_v2.mjs';
const here = new URL('.', import.meta.url);
const dicts = { w: readFileSync(new URL('data/dict-v2.bin', here)), y: readFileSync(new URL('data/dict-v1.txt', here)), z: null };
const cases = JSON.parse(readFileSync(0, 'utf8'));
let ok = 0;
for (const c of cases) {
  const { flag, body } = cleanBody(c.payload);
  const bytes = b62decode(body);
  if (b62encode(bytes) !== body) throw new Error('b62 not canonical');
  const opts = dicts[flag] ? { dictionary: dicts[flag] } : {};
  const text = inflateRawSync(bytes, opts).toString('utf8');
  if (text === c.text) ok++; else console.log('node FAIL', c.doc, c.how);
}
console.log(`node: ${ok}/${cases.length} decodes correct`);
process.exit(ok === cases.length ? 0 : 1);
