// Reference for the v2 payload's text layer, in the shape codec.ts could take.
// Compression stays fflate's deflate (a new dictionary under flag "w"); only
// the alphabet and the reader's tolerance change. See codec_v2.py for the spec.

const B62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
const B62_REV = Object.fromEntries([...B62].map((c, i) => [c, BigInt(i)]));
const BLOCK = 32, BLOCK_CHARS = 43, BREAK_EVERY = 300;

/** Fewest base62 characters that hold k bytes. */
function charsFor(k) {
  let c = 0;
  while (62n ** BigInt(c) < 256n ** BigInt(k)) c++;
  return c;
}
const TAIL = new Map();
for (let k = 1; k < BLOCK; k++) TAIL.set(charsFor(k), k);

export function b62encode(bytes) {
  let out = '';
  for (let i = 0; i < bytes.length; i += BLOCK) {
    const blk = bytes.subarray(i, i + BLOCK);
    let n = 0n;
    for (const b of blk) n = (n << 8n) | BigInt(b);
    let s = '';
    for (let w = charsFor(blk.length); w > 0; w--) {
      s = B62[Number(n % 62n)] + s;
      n /= 62n;
    }
    out += s;
  }
  return out;
}

export function b62decode(s) {
  const rest = s.length % BLOCK_CHARS;
  if (rest && !TAIL.has(rest)) throw new Error('cut short');
  const out = [];
  for (let i = 0; i < s.length; i += BLOCK_CHARS) {
    const chunk = s.slice(i, i + BLOCK_CHARS);
    const k = chunk.length === BLOCK_CHARS ? BLOCK : TAIL.get(chunk.length);
    let n = 0n;
    for (const ch of chunk) n = n * 62n + B62_REV[ch];
    if (n >= 256n ** BigInt(k)) throw new Error('block out of range');
    const blk = new Array(k);
    for (let j = k - 1; j >= 0; j--) { blk[j] = Number(n & 255n); n >>= 8n; }
    out.push(...blk);
  }
  return new Uint8Array(out);
}

export const withBreaks = (body) => body.match(new RegExp(`.{1,${BREAK_EVERY}}`, 'g')).join('+');

/** Undo what travel does to a payload: percent-encoding (once or twice), then
 *  every character outside the alphabet: "+" breaks, spaces and line breaks a
 *  mail client wrapped in, zero-width spaces, a quote glued to the end. */
export function cleanBody(payload) {
  let p = payload;
  for (let i = 0; i < 4; i++) {
    let next;
    try { next = decodeURIComponent(p); } catch { break; }
    if (next === p) break;
    p = next;
  }
  p = p.replace(/^[^A-Za-z0-9]+/, '');
  return { flag: p[0], body: p.slice(1).replace(/[^0-9A-Za-z]/g, '') };
}
