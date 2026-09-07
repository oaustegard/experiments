// node verify_tokenizer.mjs  -- compares bpe.js against the python reference
import fs from 'fs';
import { BPETokenizer } from './bpe.js';

const spec = JSON.parse(fs.readFileSync('./model/tokenizer.json', 'utf8'));
const ref = JSON.parse(fs.readFileSync('./tok_ref.json', 'utf8'));
const tok = new BPETokenizer(spec);
let ok = 0, bad = 0;
for (const r of ref) {
  const got = tok.encode(r.text);
  const same = JSON.stringify(got) === JSON.stringify(r.ids);
  const dec = tok.decode(r.ids);
  const decSame = dec === r.decoded;
  if (same && decSame) ok++;
  else {
    bad++;
    console.log('MISMATCH', JSON.stringify(r.text));
    if (!same) console.log('  py ', r.ids, '\n  js ', got);
    if (!decSame) console.log('  decode py', JSON.stringify(r.decoded), 'js', JSON.stringify(dec));
  }
}
console.log(`tokenizer agreement ${ok}/${ok + bad}`);
process.exit(bad ? 1 : 0);
